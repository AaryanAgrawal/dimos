#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pluggable relocalization priors: candidate proposers feeding the shared fine-ICP judge in relocalize.py (``refine_candidates``)."""

from __future__ import annotations

import threading
from typing import Annotated, Literal, Protocol

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
from pydantic import Field

from dimos.mapping.relocalization.relocalize import (
    generate_ransac_candidates,
    refine_candidates,
)
from dimos.msgs.vision_msgs.Detection3D import Detection3D
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.perception.fiducial.marker_aggregation import matrix_from_pose7
from dimos.perception.fiducial.marker_map import (
    MARKER_MAP_SUFFIX,
    load_marker_map,
    marker_length_m_from_map,
)
from dimos.perception.fiducial.marker_tf_module import MarkerTfModule
from dimos.protocol.service.spec import BaseConfig
from dimos.utils.data import resolve_named_path
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# One pydantic config per prior, keyed by a Literal ``type`` into a discriminated union. Pattern from dimos/manipulation/planning/kinematics/config.py:26-57.


class PriorConfigBase(BaseConfig):
    """Fields every prior shares: the on/off toggle plus its accept bar."""

    enabled: bool = True
    # Per-prior accept gate: min wall fitness (dimensionless, 0-1) this prior's fix must clear. 0.6 because the trial's office survey produced sub-0.6 fixes that were meters off while still scoring as "fit".
    fitness_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class RansacPriorConfig(PriorConfigBase):
    """Multi-scale FPFH+RANSAC global search (``RansacPrior``); search knobs live in relocalize.py, this entry owns the accept bar, cadence and geometry floor."""

    type: Literal["ransac"] = "ransac"
    # s between RANSAC fires; one FPFH+RANSAC search costs 4.4-23 s of CPU on the trial's go2/Orin recordings, so the sweep is paced, not per-frame.
    interval_s: float = Field(default=2.0, gt=0.0)
    # Min local-map points (post VoxelGridMapper) before this search fires; below this FPFH matching + the wall-only rerank have too little geometry, so the frame is skipped.
    min_local_points: int = Field(default=50_000, ge=0)


class FiducialPriorConfig(PriorConfigBase):
    """Marker sightings Huber-aggregated into one world->map candidate per tag (``FiducialPrior``). Owns the whole fiducial parameter surface."""

    type: Literal["fiducial"] = "fiducial"
    # Surveyed marker map (map_T_marker per id), a .json path resolved via resolve_named_path; required -- start() no-ops the prior without it.
    marker_map_file: str | None = None


# Discriminated on ``type`` (kinematics/config.py:54 is the exemplar).
PriorConfig = Annotated[
    RansacPriorConfig | FiducialPriorConfig,
    Field(discriminator="type"),
]


class RelocPrior(Protocol):
    """A relocalization candidate proposer; the module owns the trigger. A prior must not self-select a winner (``refine_candidates``'s job); zero candidates is a valid response."""

    name: str

    def propose(
        self,
        global_map: o3d.geometry.PointCloud,
        local_map: o3d.geometry.PointCloud,
    ) -> list[np.ndarray]: ...


class RansacPrior:
    """Wraps relocalize.py's FPFH+RANSAC global search; a pure candidate source the module polls on a paced interval (the pace bounds an eventless search's cost)."""

    name = "ransac"

    def propose(
        self,
        global_map: o3d.geometry.PointCloud,
        local_map: o3d.geometry.PointCloud,
    ) -> list[np.ndarray]:
        return generate_ransac_candidates(global_map, local_map)


class FiducialPrior:
    """Aggregated fiducial tag poses -> ONE map_T_world candidate per tag, ``map_T_marker @ inv(world_T_marker_aggregated)`` (local_map->global_map, as RANSAC)."""

    name = "fiducial"

    def __init__(self, marker_map: dict[int, np.ndarray]) -> None:
        # marker_id -> map_T_marker (4x4); the surveyed marker map.
        self._map_T_marker = marker_map
        # marker_id -> map_T_world (4x4) awaiting its ONE trip past the judge.
        self._pending: dict[int, np.ndarray] = {}
        # observe() and propose() run on different transport threads; the lock makes the read-modify-write of _pending one critical section -- see propose() for the tear.
        self._pending_lock = threading.Lock()

    @classmethod
    def from_config(cls, config: FiducialPriorConfig) -> FiducialPrior | None:
        """Load the surveyed marker map this prior composes against; ``None`` when the config names none."""
        # A constructor cannot decline to exist, and without a map every candidate would be dropped at observe(); None keeps the prior out of the live list and off the detections stream.
        if not config.marker_map_file:
            logger.warning(
                "relocalize: fiducial prior enabled but no marker_map_file; fiducial prior disabled"
            )
            return None
        marker_map_path = resolve_named_path(config.marker_map_file, MARKER_MAP_SUFFIX)
        marker_map = {
            marker_id: transform.to_matrix()
            for marker_id, transform in load_marker_map(marker_map_path).items()
        }
        logger.info(
            "fiducial prior enabled",
            marker_map_file=config.marker_map_file,
            surveyed_marker_length_m=marker_length_m_from_map(marker_map_path),
            n_markers=len(marker_map),
        )
        return cls(marker_map)

    @staticmethod
    def _marker_id_from_detection(detection: Detection3D) -> int | None:
        """Numeric marker id off the wire, via the same parse MarkerTfModule publishes TF from."""
        raw = MarkerTfModule._marker_id_from_detection(detection)
        return int(raw) if raw is not None and raw.isdigit() else None

    def observe_detections(self, msg: Detection3DArray) -> None:
        """Compose every aggregated tag pose in this burst into that tag's world->map fix."""
        for detection in msg.detections[: msg.detections_length]:
            marker_id = self._marker_id_from_detection(detection)
            if marker_id is None:
                continue
            center = detection.bbox.center  # world_T_marker_aggregated (frame_id == world)
            self.observe(
                marker_id,
                matrix_from_pose7(
                    (
                        center.position.x,
                        center.position.y,
                        center.position.z,
                        center.orientation.x,
                        center.orientation.y,
                        center.orientation.z,
                        center.orientation.w,
                    )
                ),
            )

    def observe(self, marker_id: int, world_T_marker_aggregated: np.ndarray) -> str | None:
        """Compose this tag's map_T_world fix from one aggregated pose; returns ``unmapped_id`` or ``None``."""
        map_T_marker = self._map_T_marker.get(marker_id)
        if map_T_marker is None:
            return "unmapped_id"
        with self._pending_lock:
            self._pending[marker_id] = map_T_marker @ np.linalg.inv(world_T_marker_aggregated)
        return None

    @property
    def has_pending(self) -> bool:
        """A composed fix is waiting for the judge -- the module's fire signal."""
        return bool(self._pending)

    def propose(
        self,
        global_map: o3d.geometry.PointCloud,
        local_map: o3d.geometry.PointCloud,
    ) -> list[np.ndarray]:
        # Consume on use (re-offering a drained fix scores worse, world has drifted); swap under the lock or observe()'s read-modify-write tears here into a "dict changed size during iteration" dropped cycle.
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        return list(pending.values())


def relocalize_with_prior(
    global_map: o3d.geometry.PointCloud,
    local_map: o3d.geometry.PointCloud,
    prior: RelocPrior,
) -> tuple[np.ndarray, float] | None:
    """Judge this prior's candidates through the shared fine-ICP tail; ``None`` when it proposed none."""
    transforms = prior.propose(global_map, local_map)
    if not transforms:
        return None
    return refine_candidates(global_map, local_map, transforms)
