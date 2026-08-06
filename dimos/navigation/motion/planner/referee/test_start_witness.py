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

"""A pose the robot actually occupies may always be departed.

The SE(2) search names its seed node by the cell the start snaps to, and used
to read that cell's clearance to decide whether the start was feasible at all.
The snap moves the body by up to half a cell diagonal (~85 mm), which is enough
to veto a start whose real pose is fine — `door_side` is the world that caught
it (true pose 0.083 m of union clearance against a 0.05 m margin, snapped cell
0.043 m) and gold refused a route that exists.

Both sides of the rule are pinned here: the pose decides, and a pose that is
genuinely not feasible still refuses.
"""

from __future__ import annotations

import math

import numpy as np

from dimos.navigation.motion.embodiment import GO2
from dimos.navigation.motion.scenarios import SCENARIOS, Box, se2_path

DOOR_SIDE = next(sc for sc in SCENARIOS if sc.name == "door_side")


def union_clear(boxes: list[Box], pose: tuple[float, float, float]) -> float:
    """Exact all-gait-union clearance at a pose — no grid, no lattice snap."""
    off = GO2.offsets()
    c, s = math.cos(pose[2]), math.sin(pose[2])
    pts = np.column_stack(
        [
            pose[0] + c * off[:, 0] - s * off[:, 1],
            pose[1] + s * off[:, 0] + c * off[:, 1],
            np.zeros(len(off)),
        ]
    )
    return float(np.min([b.sdf2d(pts) for b in boxes]))


def test_a_pose_the_robot_occupies_may_be_departed() -> None:
    """door_side: the cell would have refused, the pose the robot is in did not."""
    sc = DOOR_SIDE
    gold = se2_path(sc.boxes, sc.start, sc.goal, sc.emb)
    assert gold is not None, "door_side has a route and gold must find it"
    # gold's first vertex IS the cell the start snapped to.
    snapped = (float(gold[0][0]), float(gold[0][1]), sc.start[2])
    assert union_clear(sc.boxes, sc.start) > sc.emb.precision
    assert union_clear(sc.boxes, snapped) < sc.emb.precision, (
        "door_side no longer witnesses the snap: pick a world where it still does"
    )


def test_a_start_inside_an_obstacle_still_refuses() -> None:
    """Negative true clearance is not a pose the robot occupies — it refuses."""
    boxes = [Box(0.0, 0.0, 1.0, 1.0)]
    start = (0.0, 0.0, 0.0)
    assert union_clear(boxes, start) < 0.0
    assert se2_path(boxes, start, (3.0, 0.0), GO2) is None


def test_a_start_under_the_margin_still_refuses() -> None:
    """Clearance below control precision is fiction, and the seed reads it so."""
    half = GO2.width / 2.0 + 0.02  # 0.02 m of room per side: real, but not trusted
    boxes = [Box(0.0, half + 0.5, 4.0, 1.0), Box(0.0, -half - 0.5, 4.0, 1.0)]
    start = (0.0, 0.0, 0.0)
    assert 0.0 < union_clear(boxes, start) < GO2.precision
    assert se2_path(boxes, start, (1.5, 0.0), GO2) is None
