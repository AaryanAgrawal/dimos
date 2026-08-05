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

"""The diagnosis tool's measurement primitives."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dimos.navigation.motion.adapter.diagnose import (
    Crop,
    Instant,
    Window,
    arclen,
    classify,
    divergence,
    host_setup,
    is_hold,
    parse_instant,
    resample,
    voxel_centers,
    voxel_keys,
)


def test_resample_walks_even_arc_length():
    line = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    out = resample(line, step=0.5)
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(steps, 0.5)
    assert abs(arclen(line) - 2.0) < 1e-9


def test_divergence_is_the_offset_between_parallel_plans():
    a = np.column_stack([np.linspace(0, 2, 21), np.zeros(21)])
    assert divergence(a, a) == 0.0
    assert abs(divergence(a, a + np.array([0.0, 0.3])) - 0.3) < 1e-6


def test_divergence_compares_only_the_shared_arc():
    a = np.column_stack([np.linspace(0, 4, 41), np.zeros(41)])
    assert divergence(a, a[:11]) < 1e-6  # a prefix of the same route has not changed its mind


def test_single_pose_path_is_a_hold():
    assert is_hold(np.zeros((1, 2)))
    assert not is_hold(np.zeros((2, 2)))


def test_voxel_keys_round_trip_through_centres():
    pts = np.array([[0.01, 0.01, 0.30], [0.05, 0.05, 0.31], [-1.0, 2.0, 0.0]])
    keys = voxel_keys(pts, 0.08)
    assert len(keys) == 2  # the first two land in the same 0.08 m voxel
    centres = voxel_centers(keys, 0.08)
    for p in pts:
        assert np.abs(centres - p).max(axis=1).min() <= 0.0401


def test_crop_margin_excludes_the_window_edge():
    crop = Crop(centre=np.array([0.0, 0.0]), radius=2.0, z_lo=-1.0, z_hi=1.0)
    pts = np.array([[0.0, 0.0, 0.0], [1.95, 0.0, 0.0], [0.0, 0.0, 0.99]])
    assert list(crop.inside(pts, margin=0.16)) == [True, False, False]


# --- the follower pass's classifier


def test_a_tick_under_the_threshold_matches_on_every_component():
    verdict, gap = classify((0.30, 0.0, 0.5), (0.35, 0.02, 0.44), boundary=False, threshold=0.15)
    assert verdict == "match"
    assert abs(gap - 0.06) < 1e-9  # the WORST component, not the first one over


def test_one_component_over_the_threshold_is_the_whole_tick():
    # a twist that agrees on speed and disagrees on turn rate is a disagreement
    verdict, _ = classify((0.30, 0.0, 0.5), (0.30, 0.0, 0.9), boundary=False, threshold=0.15)
    assert verdict == "MISMATCH"


def test_a_plan_landing_inside_a_control_period_is_unpairable():
    # which plan the module held is genuinely ambiguous there, so it is neither
    # a match nor a finding
    verdict, _ = classify((0.30, 0.0, 0.5), (0.90, 0.0, 0.5), boundary=True, threshold=0.15)
    assert verdict == "boundary"


def test_a_hold_is_held_against_zero_rather_than_the_law():
    # the module never reached its law -- deadman, latch, or a refusal stub
    assert classify((0.0, 0.0, 0.0), None, boundary=False, threshold=0.15)[0] == "hold"
    assert classify((0.0, 0.0, 0.0), None, boundary=True, threshold=0.15)[0] == "hold"


def test_driving_through_a_hold_is_a_finding_not_a_hold():
    verdict, gap = classify((0.40, 0.0, 0.0), None, boundary=False, threshold=0.15)
    assert verdict == "MISMATCH"
    assert gap == 0.40


# --- the time filters


def test_a_bare_number_is_seconds_into_the_recording():
    assert parse_instant("6.9") == Instant(6.9, absolute=False)
    assert parse_instant("6.9").resolve(1000.0) == 1006.9


def test_a_clock_time_is_utc_time_of_day():
    assert parse_instant("06:34:35.4") == Instant(6 * 3600 + 34 * 60 + 35.4, absolute=True)
    midnight = 1_800_000_000.0 - 1_800_000_000.0 % 86400
    assert parse_instant("01:00:00").resolve(midnight + 3000) == midnight + 3600


def test_a_clock_time_takes_the_occurrence_nearest_the_start():
    # a recording that crosses midnight must not be dated by its own start
    midnight = 1_800_000_000.0 - 1_800_000_000.0 % 86400
    late = midnight - 60.0  # 23:59:00 the previous day
    assert parse_instant("23:59:30").resolve(late) == late + 30.0


def test_a_time_that_is_neither_form_is_refused():
    with pytest.raises(ValueError):
        parse_instant("06:34")


def test_an_unset_window_holds_nothing_out():
    w = Window()
    assert not w.bounded
    assert 1e12 in w and -1e12 in w
    assert Window.between(None, None, 100.0) == w


def test_a_window_is_inclusive_at_both_ends():
    w = Window.between(parse_instant("1.0"), parse_instant("3.0"), 100.0)
    assert (w.lo, w.hi) == (101.0, 103.0)
    assert w.bounded and 101.0 in w and 103.0 in w and 100.9 not in w
    assert list(w.mask(np.array([100.5, 101.0, 102.0, 103.5]))) == [False, True, True, False]


# --- the deployed config, as one JSON


def test_the_host_blob_is_read_off_the_follower_section(tmp_path):
    blob = {
        "modules": {
            "trajectory_follower": {
                "topics": {},
                "config": {
                    "track": "blind",
                    "controller_config": {"max_speed": 0.95, "min_speed": 0.45},
                    "max_path_age_s": 2.5,
                    "obstacle_model": "raw_band",
                },
            }
        }
    }
    path = tmp_path / "motion-host.json"
    path.write_text(json.dumps(blob))
    setup = host_setup(str(path))
    assert setup.track == "blind"
    assert setup.controller.max_speed == 0.95
    assert setup.obstacle_model == "raw_band"
    assert setup.max_path_age_s == 2.5
    # the keys the blob leaves out keep the module's own defaults
    assert setup.control_frequency == 10.0
    assert setup.period == 0.1
    assert setup.controller.max_yaw_rate == 1.4


def test_a_blob_with_no_follower_in_it_says_so(tmp_path):
    path = tmp_path / "motion-host.json"
    path.write_text(json.dumps({"modules": {"motion_planner": {"config": {}}}}))
    with pytest.raises(SystemExit, match="trajectory_follower"):
        host_setup(str(path))
