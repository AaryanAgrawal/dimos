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

import numpy as np
import pytest

from dimos.msgs.nav_msgs.Path import Path
from dimos.navigation.motion.control import world
from dimos.navigation.motion.control.episode import EpisodeResult
from dimos.navigation.motion.control.judge import cross_track, score_episode, summarize
from dimos.navigation.motion.planner.autoresearch.scenarios import Scenario
from dimos.navigation.motion.planner.autoresearch.types import (
    Path as RefereePath,
    PoseStamped as RefereePose,
)

CLEAR = Scenario("t_clear", [], goal=(4.0, 0.0))
REFUSE = Scenario("t_refuse", [], goal=(4.0, 0.0), expect="refuse")


def _plan(n: int = 41) -> Path:
    ref = RefereePath(
        frame_id="world",
        poses=[RefereePose(frame_id="world", position=[i * 0.1, 0.0, 0.0]) for i in range(n)],
    )
    return world.to_nav_path(ref)


def _result(
    sc: Scenario = CLEAR,
    outcome: str = "goal",
    time_to_goal: float | None = 13.0,
    lateral: float = 0.0,
    tilt: float = 0.05,
    n: int = 200,
) -> EpisodeResult:
    t = np.linspace(0.5, 13.0, n)
    pos = np.column_stack([np.linspace(0, 4, n), np.full(n, lateral), np.full(n, 0.3)])
    cmd = np.tile([0.4, 0.0, 0.0], (n, 1))
    return EpisodeResult(
        scenario=sc,
        outcome=outcome,
        t=t,
        pos=pos,
        yaw=np.zeros(n),
        tilt=np.full(n, tilt),
        twist_cmd=cmd,
        used_cmd=cmd.copy(),
        contact=np.zeros(n, dtype=bool),
        plan=_plan(),
        plans=[],
        plan_ms=[2.0],
        time_to_goal=time_to_goal,
    )


def test_cross_track_on_line_is_zero() -> None:
    path = np.array([[0.0, 0.0], [4.0, 0.0]])
    pos = np.array([[1.0, 0.0], [2.5, 0.0]])
    np.testing.assert_allclose(cross_track(pos, path), 0.0, atol=1e-12)


def test_cross_track_offset_and_beyond_ends() -> None:
    path = np.array([[0.0, 0.0], [4.0, 0.0]])
    pos = np.array([[2.0, 0.3], [-1.0, 0.0], [5.0, 0.4]])
    np.testing.assert_allclose(cross_track(pos, path), [0.3, 1.0, np.hypot(1.0, 0.4)], atol=1e-12)


def test_clean_run_scores_high() -> None:
    row = score_episode(_result())
    assert row["total"] > 105.0
    assert not row["dq"]
    assert row["progress"] == 1.0
    assert row["tracking"] == 1.0


def test_collision_gates_to_zero() -> None:
    row = score_episode(_result(outcome="collision", time_to_goal=None))
    assert row["dq"] and row["total"] == 0.0


def test_fall_gates_to_zero() -> None:
    assert score_episode(_result(outcome="fall", time_to_goal=None))["total"] == 0.0


def test_lateral_error_costs_tracking_not_progress() -> None:
    row = score_episode(_result(lateral=0.15))
    assert row["progress"] == 1.0
    assert 0.0 < row["tracking"] < 0.75


def test_slow_run_costs_progress() -> None:
    fast = score_episode(_result(time_to_goal=13.0))
    slow = score_episode(_result(time_to_goal=40.0))
    assert slow["progress"] < fast["progress"]


def test_timeout_on_clear_world_scores_progress_zero() -> None:
    row = score_episode(_result(outcome="timeout", time_to_goal=None))
    assert row["progress"] == 0.0
    assert row["total"] < 15.0


def test_refusal_on_sealed_world_is_success() -> None:
    row = score_episode(_result(sc=REFUSE, outcome="refused", time_to_goal=None, n=5))
    assert row["progress"] == 1.0 and row["tracking"] == 1.0
    assert row["total"] > 100.0


def test_tilt_tail_costs_composure() -> None:
    calm = score_episode(_result(tilt=0.05))
    shaky = score_episode(_result(tilt=0.30))
    assert shaky["composure"] < calm["composure"]


def test_summarize_counts() -> None:
    rows = [
        score_episode(_result()),
        score_episode(_result(outcome="collision", time_to_goal=None)),
    ]
    s = summarize(rows)
    assert s["worlds"] == 2 and s["dq"] == 1
    assert s["outcomes"] == {"goal": 1, "collision": 1}
    assert s["worst"]["total"] == 0.0
    assert s["score"] == pytest.approx((rows[0]["total"] + 0.0) / 2, abs=0.01)
