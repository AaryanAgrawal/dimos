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

"""EXPERIMENTAL: build an Embodiment from a robot description instead of hand-typed numbers.

    URDF + joint config -> collision AABB in the base frame -> Embodiment scalars

What a URDF CAN give: the body box at a pose, its height above the feet, and where that box sits
relative to the base origin. What it CANNOT give is `envelope` -- the swept box per drift angle --
because that is where the legs actually go, and the LOCOMOTION POLICY decides that, not the model.
The static box is therefore a LOWER bound: the go2's measured union is 0.593 wide against a 0.31 m
trunk, so a walking body sweeps far more than it occupies standing.

`envelope_rows` closes that gap without a simulator: replay real joint angles recorded off
rt/lowstate through this same forward kinematics, bucket by drift angle, and union each bucket.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from pathlib import Path
from typing import Any

import numpy as np

from dimos.navigation.motion.embodiment import Embodiment

# Ivan's lattice drift angles, so a produced row lands on an edge the planner actually generates.
DRIFT_DEG: tuple[float, ...] = (0.0, 26.6, 45.0, 63.4, 90.0, 116.6, 135.0, 153.4, 180.0)


def load_robot(path: Path | str) -> Any:
    """Load a URDF with its collision geometry; meshes are required or the box is empty."""
    import yourdfpy  # type: ignore[import-untyped]  # heavy (trimesh); deferred to keep import cheap

    robot = yourdfpy.URDF.load(
        str(path),
        load_meshes=False,
        load_collision_meshes=True,
        build_scene_graph=True,
        build_collision_scene_graph=True,
    )
    if robot.collision_scene is None or not robot.collision_scene.geometry:
        raise ValueError(f"{path} has no collision geometry -- are its meshes present?")
    return robot


def box_at(robot: Any, cfg: Mapping[str, float] | None = None) -> np.ndarray:
    """Axis-aligned collision bounds at a joint configuration, as ((xmin,ymin,zmin),(xmax,...))."""
    if cfg is not None:
        robot.update_cfg(dict(cfg))
    return np.asarray(robot.collision_scene.bounds, dtype=float)


def embodiment_from_urdf(
    path: Path | str,
    tag: str,
    cfg: Mapping[str, float] | None = None,
    strafe: float = 1.8,
    reverse: float = 1.5,
    yaw_w: float = 0.25,
    comfort: float = 0.4,
    precision: float = 0.05,
    steppable: float = 0.0,
) -> Embodiment:
    """Embodiment scalars from a URDF pose. `envelope` stays empty: it is policy-dependent."""
    robot = load_robot(path)
    lo, hi = box_at(robot, cfg)
    length, width = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    # The feet are the lowest geometry, so the support plane sits at zmin and the base rides above it.
    base_height = float(-lo[2])
    return Embodiment(
        tag=tag,
        length=length,
        width=width,
        center_off=float((hi[0] + lo[0]) / 2.0),
        comfort=comfort,
        precision=precision,
        strafe=strafe,
        reverse=reverse,
        yaw_w=yaw_w,
        steppable=steppable,
        height=float(hi[2] - lo[2]),
        base_height=base_height,
    )


def _drift_bucket(cmd_vx: float, cmd_vy: float) -> float | None:
    """The lattice drift angle nearest this command, or None when the command is a stand-still."""
    if math.hypot(cmd_vx, cmd_vy) < 1e-6:
        return None
    deg = abs(math.degrees(math.atan2(cmd_vy, cmd_vx)))
    return min(DRIFT_DEG, key=lambda d: abs(d - deg))


def envelope_rows(
    path: Path | str,
    samples: Iterable[tuple[float, float, Mapping[str, float]]],
) -> tuple[tuple[float, float, float, float, float], ...]:
    """Swept (deg, length, width, off_x, off_y) rows from REAL joint angles, one per drift bucket.

    Each sample is (cmd_vx, cmd_vy, joint config) as logged beside the velocities during a drift
    sweep, so the rows describe the policy that was actually flying the robot.
    """
    robot = load_robot(path)
    unions: dict[float, np.ndarray] = {}
    for cmd_vx, cmd_vy, cfg in samples:
        bucket = _drift_bucket(cmd_vx, cmd_vy)
        if bucket is None:
            continue
        lo, hi = box_at(robot, cfg)
        prev = unions.get(bucket)
        unions[bucket] = (
            np.array([lo, hi])
            if prev is None
            else np.array([np.minimum(prev[0], lo), np.maximum(prev[1], hi)])
        )
    rows = []
    for deg in sorted(unions):
        lo, hi = unions[deg]
        rows.append(
            (
                deg,
                float(hi[0] - lo[0]),
                float(hi[1] - lo[1]),
                float((hi[0] + lo[0]) / 2.0),
                float((hi[1] + lo[1]) / 2.0),
            )
        )
    return tuple(rows)
