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

"""Filtering and the chaos-tolerant statistics built on it."""

from __future__ import annotations

import numpy as np
import pytest

from dimos.navigation.motion.trajectory.research.metrics import (
    Summary,
    _gain,
    chaos_spread,
    resample,
    summarize,
    velocity,
    yaw_of,
)


def test_resample_lands_on_a_uniform_grid():
    t = np.array([0.0, 0.1, 0.35, 0.4, 0.9])  # deliberately jittery
    grid, x = resample(t, t * 2.0, rate=100.0)
    np.testing.assert_allclose(np.diff(grid), 0.01, atol=1e-12)
    np.testing.assert_allclose(x, grid * 2.0, atol=1e-9)


def test_resample_handles_multiple_columns():
    t = np.linspace(0, 1, 37)
    grid, x = resample(t, np.stack([t, -t, 2 * t], 1), rate=50.0)
    assert x.shape == (len(grid), 3)
    np.testing.assert_allclose(x[:, 1], -grid, atol=1e-9)


def test_velocity_recovers_a_constant_speed():
    t = np.linspace(0, 4, 400)
    pos = np.stack([0.5 * t, np.zeros_like(t), np.zeros_like(t)], 1)
    _grid, v = velocity(t, pos)
    mid = slice(len(v) // 4, 3 * len(v) // 4)  # ignore edge effects of the window
    np.testing.assert_allclose(v[mid, 0], 0.5, atol=1e-3)


def test_velocity_is_insensitive_to_sample_rate():
    """A 253 Hz recording and a 50 Hz rollout must give the same answer.

    Using a fixed *sample* window instead of a time window is what made an
    earlier pass report a Go2 walking at 3.9 m/s.
    """
    speeds = []
    for n in (200, 1012):
        t = np.linspace(0, 4, n)
        pos = np.stack([0.4 * t, np.zeros_like(t), np.zeros_like(t)], 1)
        _g, v = velocity(t, pos)
        mid = slice(len(v) // 4, 3 * len(v) // 4)
        speeds.append(v[mid, 0].mean())
    assert abs(speeds[0] - speeds[1]) < 1e-3


def test_velocity_rejects_jitter_noise():
    """Irregular sampling must not inflate the speed estimate."""
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.001, 0.008, 2000))
    pos = np.stack([0.4 * t, np.zeros_like(t), np.zeros_like(t)], 1)
    pos += rng.normal(0, 1e-4, pos.shape)  # tracker noise
    _g, v = velocity(t, pos)
    mid = slice(len(v) // 4, 3 * len(v) // 4)
    assert np.linalg.norm(v[mid, :2], axis=1).mean() == pytest.approx(0.4, abs=0.05)


def test_yaw_of_reads_rotation_about_z():
    half = np.pi / 4  # 90 deg yaw
    q = np.array([[np.cos(half), 0.0, 0.0, np.sin(half)], [1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(yaw_of(q), [np.pi / 2, 0.0], atol=1e-9)


def test_gain_is_a_slope_not_a_mean_of_ratios():
    """Sign-flipping commands must not cancel — the bug this replaced."""
    cmd = np.array([1.0, -1.0, 0.8, -0.8, 0.02])
    achieved = 0.6 * cmd
    assert _gain(achieved, cmd, 0.2) == pytest.approx(0.6)

    # a mean of ratios survives this case too, but not near-zero commands
    noisy = achieved.copy()
    noisy[-1] = 5.0  # tiny command, large achieved -> ratio explodes
    assert _gain(noisy, cmd, 0.2) == pytest.approx(0.6)


def test_gain_returns_zero_when_nothing_is_commanded():
    assert _gain(np.ones(5), np.zeros(5), 0.2) == 0.0


def test_summarize_on_a_synthetic_walk():
    t = np.linspace(0, 20, 2000)
    pos = np.stack([0.4 * t, np.zeros_like(t), 0.30 + 0.02 * np.sin(2 * np.pi * 2.0 * t)], 1)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (len(t), 1))
    cmd = np.tile([0.5, 0.0, 0.0], (len(t), 1))
    s = summarize(t, pos, quat, cmd)

    assert s.speed == pytest.approx(0.4, abs=0.02)
    assert s.speed_gain == pytest.approx(0.8, abs=0.05)
    assert s.height_mean == pytest.approx(0.30, abs=0.01)
    assert s.height_std == pytest.approx(0.02 / np.sqrt(2), abs=0.005)
    assert s.gait_hz == pytest.approx(2.0, abs=0.2)


def test_summarize_reads_turn_rate_with_the_right_sign():
    t = np.linspace(0, 10, 1000)
    rate = 0.5
    yaw = rate * t
    quat = np.stack([np.cos(yaw / 2), np.zeros_like(t), np.zeros_like(t), np.sin(yaw / 2)], axis=1)
    pos = np.stack([np.zeros_like(t), np.zeros_like(t), np.full_like(t, 0.3)], 1)
    cmd = np.tile([0.0, 0.0, 1.0], (len(t), 1))
    assert summarize(t, pos, quat, cmd).yaw_rate_gain == pytest.approx(rate, abs=0.05)


def test_chaos_spread_is_peak_to_peak_per_statistic():
    def mk(speed):
        return Summary(
            speed=speed,
            speed_gain=1.0,
            yaw_rate_gain=0.0,
            height_mean=0.3,
            height_std=0.01,
            gait_hz=2.0,
        )

    spread = chaos_spread([mk(0.40), mk(0.44), mk(0.42)])
    assert spread["speed"] == pytest.approx(0.04)
    assert spread["gait_hz"] == 0.0
