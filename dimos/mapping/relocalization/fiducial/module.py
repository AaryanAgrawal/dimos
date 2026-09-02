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

"""Fiducial relocalization: one aggregated tag sighting against the surveyed marker map.

What is here is what only a tag matcher needs - the ``aggregated_detections``
input, the marker map, and the accept bar on the detector's score. Publishing
what comes back is the base module's. No pointcloud premap is needed: the tag's
surveyed pose is the map.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.stream import In
from dimos.mapping.relocalization.module import (
    FRAME_MAP,
    FRAME_WORLD,
    Config,
    RelocalizationModule,
)
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.vision_msgs.Detection3D import Detection3D
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.perception.fiducial.marker_aggregation import matrix_from_pose7
from dimos.perception.fiducial.marker_map import MARKER_MAP_SUFFIX, load_marker_map
from dimos.utils.data import resolve_named_path
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class FiducialConfig(Config):
    # Surveyed marker map, `marker_id -> map_T_marker`; `.json` is appended if
    # absent. Without one the module runs but never attempts a fix.
    marker_map_file: str | None = None
    # Accept bar on the aggregated sighting's score, 0-1 (the detector's
    # min(1, 1/noise_scale)). 0 accepts every aggregated sighting; the bar is unmeasured.
    min_score: float = 0.0


class FiducialRelocalization(RelocalizationModule):
    """``world_T_map = world_T_marker @ inv(map_T_marker)`` from one aggregated tag sighting."""

    config: FiducialConfig
    aggregated_detections: In[Detection3DArray]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._map_T_marker: dict[int, np.ndarray] = {}

    @rpc
    def start(self) -> None:
        super().start()
        if not self.config.marker_map_file:
            logger.info("fiducial relocalization disabled (no marker_map_file configured)")
            return
        path = resolve_named_path(self.config.marker_map_file, MARKER_MAP_SUFFIX)
        self._map_T_marker = {
            marker_id: tf.to_matrix() for marker_id, tf in load_marker_map(path).items()
        }
        logger.info(
            "fiducial relocalization started",
            marker_map_file=str(path),
            n_markers=len(self._map_T_marker),
        )
        unsub = self.aggregated_detections.subscribe(self._on_aggregated)
        self.register_disposable(Disposable(unsub) if callable(unsub) else unsub)

    def _on_aggregated(self, msg: Detection3DArray) -> None:
        for det in msg.detections[: msg.detections_length]:
            if not self.keep_relocalizing():
                return
            tf = self._fix(det)
            if tf is not None:
                self.submit(tf, "fiducial")

    def _fix(self, det: Detection3D) -> Transform | None:
        """This sighting's ``world -> map``, or ``None`` for an unsurveyed tag or one below the bar."""
        marker_id = _marker_id(det)
        map_T_marker = None if marker_id is None else self._map_T_marker.get(marker_id)
        if map_T_marker is None:
            return None
        score = det.results[0].hypothesis.score if det.results_length else 0.0
        if score < self.config.min_score:
            logger.info(
                "relocalize fiducial: refused",
                marker_id=marker_id,
                score=round(score, 2),
                min_score=self.config.min_score,
            )
            return None
        c = det.bbox.center  # world_T_marker, aggregated by the detector
        world_T_marker = matrix_from_pose7(
            (
                c.position.x,
                c.position.y,
                c.position.z,
                c.orientation.x,
                c.orientation.y,
                c.orientation.z,
                c.orientation.w,
            )
        )
        logger.info("relocalize fiducial", marker_id=marker_id, score=round(score, 2))
        return Transform.from_matrix(
            world_T_marker @ np.linalg.inv(map_T_marker),
            frame_id=FRAME_WORLD,
            child_frame_id=FRAME_MAP,
        )


def _marker_id(det: Detection3D) -> int | None:
    """The tag id the detector stamped on ``id``; ``None`` for anything else."""
    raw = str(det.id).strip()
    return int(raw) if raw.isdigit() else None
