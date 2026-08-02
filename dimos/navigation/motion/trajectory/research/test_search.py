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

"""Search-space definition and the multi-objective grouping."""

from __future__ import annotations

import pytest

from dimos.navigation.motion.trajectory.research.evaluate import LEG_STATS, NOT_COMPARABLE
from dimos.navigation.motion.trajectory.research.metrics import Summary
from dimos.navigation.motion.trajectory.research.search import (
    OBJECTIVES,
    SPACE,
    _objective_values,
    format_front,
)


def test_space_brackets_the_menagerie_defaults():
    """A search that cannot reach the current values cannot tell you they are right."""
    for name, default in (
        ("armature", 0.01),
        ("damping", 2.0),
        ("frictionloss", 0.2),
        ("foot_friction", 0.8),
        ("foot_friction_torsional", 0.02),
        ("trunk_com_x", 0.0),
        ("leg_mass_scale", 1.0),
    ):
        lo, hi, _log = SPACE[name]
        assert lo < default < hi, name


def test_log_scale_only_where_the_range_spans_decades():
    for name, (lo, hi, log) in SPACE.items():
        if log:
            assert lo > 0, f"{name}: log scale needs a positive floor"
            assert hi / lo >= 10, f"{name}: log scale on a narrow range"


def test_objectives_cover_every_comparable_statistic():
    """A statistic outside every group is silently unoptimized."""
    grouped = {k for keys in OBJECTIVES.values() for k in keys}
    comparable = (set(Summary.__dataclass_fields__) - set(NOT_COMPARABLE)) | set(LEG_STATS)
    assert grouped == comparable


def test_objective_groups_do_not_overlap():
    seen: set[str] = set()
    for keys in OBJECTIVES.values():
        assert not (seen & set(keys))
        seen |= set(keys)


def test_objective_values_are_rms_within_each_group():
    snr = {"gait_hz": 3.0, "height_std": 4.0, "speed": 0.0, "speed_gain": 0.0, "speed_lag": 0.0}
    snr |= {"yaw_rate_gain": 1.0, "yaw_lag": 1.0, "front_lift": 2.0, "rear_lift": 2.0}
    gait, translation, rotation, legs = _objective_values(snr, OBJECTIVES)
    assert gait == pytest.approx(3.5355, abs=1e-3)  # rms(3, 4)
    assert translation == pytest.approx(0.0)
    assert rotation == pytest.approx(1.0)
    assert legs == pytest.approx(2.0)


def test_objective_values_ignore_missing_statistics():
    """height_mean is not scored, so a group must not blow up on its absence."""
    assert _objective_values({"gait_hz": 2.0}, {"gait": ("gait_hz", "height_mean")}) == [2.0]


def test_format_front_lists_objectives_and_params():
    result = {
        "groups": ["gait", "rotation"],
        "n_trials": 10,
        "front": [
            {"objectives": {"gait": 1.0, "rotation": 5.0}, "params": {"armature": 0.02}, "snr": {}},
            {"objectives": {"gait": 4.0, "rotation": 1.0}, "params": {"armature": 0.05}, "snr": {}},
        ],
    }
    text = format_front(result)
    assert "gait" in text and "rotation" in text
    assert "armature=0.02" in text
    assert "2 of 10 trials" in text


def test_format_front_truncates_and_says_so():
    front = [
        {"objectives": {"gait": float(i), "rotation": 1.0}, "params": {}, "snr": {}}
        for i in range(20)
    ]
    text = format_front({"groups": ["gait", "rotation"], "n_trials": 50, "front": front}, limit=5)
    assert "... 15 more" in text
