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

import math
from types import SimpleNamespace

import numpy as np

from dimos.navigation.motion.adapter import planner as planner_module
from dimos.navigation.motion.adapter.planner import (
    MotionPlanner,
    MotionPlannerConfig,
    carrot_along,
    to_nav_path,
)
from dimos.navigation.motion.control.laws.seed import PursuitController
from dimos.navigation.motion.planner.autoresearch.planners.gold import pose_stamped
from dimos.navigation.motion.planner.autoresearch.types import Path as RefereePath


def test_to_nav_path_preserves_positions_and_yaw():
    ref = RefereePath(
        frame_id="world", poses=[pose_stamped(0.0, 0.0, 0.0), pose_stamped(1.0, 2.0, math.pi / 2)]
    )
    nav = to_nav_path(ref, ts=3.0, frame_id="odom")
    assert nav.frame_id == "odom"
    assert len(nav.poses) == 2
    assert nav.poses[1].position.x == 1.0
    assert nav.poses[1].position.y == 2.0
    assert abs(nav.poses[1].yaw - math.pi / 2) < 1e-9


def test_to_nav_path_empty():
    assert len(to_nav_path(RefereePath(frame_id="world", poses=[])).poses) == 0


def test_carrot_walks_arc_from_closest_waypoint():
    path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 4.0]])
    # closest waypoint to (2.1, 0.5) is (2, 0); 1.5 m of arc up the second leg
    assert carrot_along(path, (2.1, 0.5), 1.5) == (2.0, 1.5)


def test_carrot_interpolates_within_segment():
    path = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert carrot_along(path, (0.0, 0.0), 5.0) == (5.0, 0.0)


def test_carrot_clamps_to_path_end():
    path = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert carrot_along(path, (0.9, 0.0), 5.0) == (1.0, 0.0)


def test_carrot_single_waypoint():
    assert carrot_along(np.array([[3.0, 4.0]]), (0.0, 0.0), 5.0) == (3.0, 4.0)


def _holding_planner():
    """A MotionPlanner with just enough wired up to call _hold."""
    planner = object.__new__(MotionPlanner)
    planner._stale = False
    planner.config = MotionPlannerConfig()
    planner._viz_at = 0.0
    published, drawn = [], []
    planner.path = SimpleNamespace(publish=published.append)
    planner.plan_body = SimpleNamespace(publish=drawn.append)
    return planner, published, drawn


def test_hold_publishes_single_pose_stub_at_the_current_pose():
    planner, published, _drawn = _holding_planner()
    planner._hold((1.5, -2.0, math.pi / 2), age=7.0)
    assert len(published) == 1
    path = published[0]
    assert path.frame_id == "odom"
    # a single pose is the planner's refusal shape: "hold, no safe route"
    assert len(path.poses) == 1
    assert path.poses[0].position.x == 1.5
    assert path.poses[0].position.y == -2.0
    assert abs(path.poses[0].yaw - math.pi / 2) < 1e-9


def test_hold_stub_stops_the_controller():
    planner, published, _drawn = _holding_planner()
    planner._hold((1.5, -2.0, 0.0), age=7.0)
    pose = pose_stamped(1.5, -2.0, 0.0)
    twist = PursuitController().update(pose, published[0], t=0.0)
    assert (twist.linear.x, twist.linear.y, twist.angular.z) == (0.0, 0.0, 0.0)


def test_hold_warns_once_per_stale_episode(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        planner_module.logger, "warning", lambda msg, **kw: warnings.append(msg), raising=False
    )
    planner, _published, _drawn = _holding_planner()
    for _ in range(3):
        planner._hold((0.0, 0.0, 0.0), age=7.0)
    assert planner._stale
    # edge-triggered: replan_hz would otherwise warn 5x a second for as long
    # as the link stays down
    assert len(warnings) == 1


def test_hold_draws_the_veto_so_it_is_not_mistaken_for_a_dead_module():
    """A refusal must reach the viewer: an empty viewport looks like a crash."""
    planner, published, drawn = _holding_planner()
    planner._hold((1.0, 2.0, 0.0), age=7.0)
    assert len(drawn) == 1
    assert drawn[0] is published[0]
    assert len(drawn[0].poses) == 1
