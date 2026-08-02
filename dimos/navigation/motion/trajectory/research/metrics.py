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

"""Trajectory statistics that survive chaos, and the filtering they need.

Matching sim to hardware trajectory-by-trajectory does not work: a legged gait
with contacts amplifies a 3-degree initial-pose difference into more than a
metre of divergence in 12 s, non-monotonically, and by a 10 s horizon the
sim-vs-real gap is *smaller* than the spread between two identical simulators.
See ``FINDINGS.md``.

What does survive is distributional -- speed gain, gait frequency, body height
statistics. Chaos scrambles the phase, not the distribution. :func:`summarize`
computes those; :func:`chaos_spread` measures how much each one moves under
perturbation, so a statistic is only trusted when the sim-real difference
clearly exceeds its own noise.

Filtering matters more than it looks. Vive samples arrive at a nominal 253 Hz
but with dt jitter as large as the interval itself (mean 3.96 ms, std 3.88 ms,
gaps to 15 ms), so differentiating raw samples amplifies noise enormously --
and doing it with a fixed *sample* window applies wildly different smoothing to
a 253 Hz recording and a 50 Hz rollout. Everything here resamples onto a
uniform grid first and smooths by a window in *seconds*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RESAMPLE_HZ = 100.0
VELOCITY_WINDOW_S = 0.4

# Gait band. HIMLoco free_walk has no clocked gait -- the only explicit rate in
# the fleet is 1.5 Hz on an experimental trot-clock policy (go2web
# policies/experimental/jun05.rs) -- so this brackets a plausible trot rather
# than targeting a known value. The floor matters: below ~1 Hz the FFT locks
# onto the robot's slow drift around the room instead of its bob.
GAIT_BAND_HZ = (1.0, 6.0)

# Longest policy->body lag to search for when fitting a command gain.
MAX_LAG_S = 0.6


def resample(
    t: np.ndarray, x: np.ndarray, rate: float = RESAMPLE_HZ
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate ``x(t)`` onto a uniform grid at ``rate`` Hz."""
    grid = np.arange(float(t[0]), float(t[-1]), 1.0 / rate)
    if x.ndim == 1:
        return grid, np.interp(grid, t, x)
    return grid, np.stack([np.interp(grid, t, x[:, i]) for i in range(x.shape[1])], axis=1)


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n < 2:
        return x
    pad = n // 2
    ker = np.ones(n) / n
    if x.ndim == 1:
        return np.convolve(np.pad(x, pad, mode="edge"), ker, mode="same")[pad : pad + len(x)]
    return np.stack([_moving_average(x[:, i], n) for i in range(x.shape[1])], axis=1)


def velocity(
    t: np.ndarray,
    pos: np.ndarray,
    *,
    window_s: float = VELOCITY_WINDOW_S,
    rate: float = RESAMPLE_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed derivative of ``pos(t)``, on a uniform grid.

    The window is in seconds so a 253 Hz recording and a 50 Hz rollout get the
    same treatment -- the mistake that made an earlier pass report a Go2
    walking at 3.9 m/s.
    """
    grid, p = resample(t, pos, rate)
    p = _moving_average(p, max(2, round(window_s * rate)))
    return grid, np.gradient(p, 1.0 / rate, axis=0)


def yaw_of(quat: np.ndarray) -> np.ndarray:
    """Heading angle from (n, 4) wxyz quaternions."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw: np.ndarray = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw


def _best_lag(achieved: np.ndarray, commanded: np.ndarray, rate: float) -> int:
    """Samples of delay that best aligns the response with the command.

    The body does not turn the instant the command changes -- the policy has
    its own history and the robot has inertia. Regressing at zero lag against
    a command that alternates faster than that delay reads as no response at
    all, which is how the simulator appeared to turn backwards.
    """
    a = achieved - achieved.mean()
    c = commanded - commanded.mean()
    if not np.any(c) or not np.any(a):
        return 0
    span = int(MAX_LAG_S * rate)
    scores = [float(c[: len(c) - k] @ a[k:]) for k in range(span)]
    return int(np.argmax(scores))


def _gain(
    achieved: np.ndarray, commanded: np.ndarray, threshold: float, *, rate: float = RESAMPLE_HZ
) -> tuple[float, float]:
    """Least-squares slope of achieved against commanded, at the best lag.

    Returns ``(gain, lag_seconds)``. Not ``mean(achieved / commanded)``: in a
    real recording the command flips sign constantly, so per-sample ratios
    blow up near zero and opposite-sign turns cancel.
    """
    lag = _best_lag(achieved, commanded, rate)
    a = achieved[lag:] if lag else achieved
    c_all = commanded[: len(commanded) - lag] if lag else commanded
    m = np.abs(c_all) > threshold
    if not m.any():
        return 0.0, 0.0
    c = c_all[m]
    return float((c @ a[m]) / (c @ c)), lag / rate


@dataclass
class Summary:
    """Chaos-tolerant description of one run."""

    speed: float  # mean planar speed while commanded to move, m/s
    speed_gain: float  # achieved / commanded speed
    yaw_rate_gain: float  # achieved / commanded turn rate
    height_mean: float  # mean base height, m -- NOT comparable sim-to-real:
    # the recorded value is set by the unknown tracker offset and the anchor
    height_std: float  # body bob amplitude, m
    gait_hz: float  # dominant frequency of the detrended vertical bob
    speed_lag: float  # policy->body delay fitted for the speed gain, s
    yaw_lag: float  # same, for the turn gain

    def as_dict(self) -> dict[str, float]:
        return {
            "speed": self.speed,
            "speed_gain": self.speed_gain,
            "yaw_rate_gain": self.yaw_rate_gain,
            "height_mean": self.height_mean,
            "height_std": self.height_std,
            "gait_hz": self.gait_hz,
            "speed_lag": self.speed_lag,
            "yaw_lag": self.yaw_lag,
        }


def summarize(
    t: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    cmd: np.ndarray,
    *,
    moving_threshold: float = 0.25,
) -> Summary:
    """Distributional statistics for one run.

    ``cmd`` is the vx/vy/vyaw in force at each sample of ``t``.
    """
    grid, vel = velocity(t, pos)
    _, c = resample(t, cmd)
    _, z = resample(t, pos[:, 2])

    speed = np.linalg.norm(vel[:, :2], axis=1)
    cmd_speed = np.linalg.norm(c[:, :2], axis=1)
    moving = cmd_speed > moving_threshold

    yaw = np.unwrap(yaw_of(quat))
    grid_y, yaw_u = resample(t, yaw)
    yaw_rate = np.gradient(
        _moving_average(yaw_u, max(2, int(VELOCITY_WINDOW_S * RESAMPLE_HZ))), 1.0 / RESAMPLE_HZ
    )
    n = min(len(yaw_rate), len(c))

    # High-pass by subtracting a 1 s moving average: the raw signal is dominated
    # by the robot drifting up and down the room, not by its gait.
    bob = z - _moving_average(z, int(RESAMPLE_HZ))
    bob = bob * np.hanning(len(bob))
    freqs = np.fft.rfftfreq(len(bob), 1.0 / RESAMPLE_HZ)
    power = np.abs(np.fft.rfft(bob))
    band = (freqs >= GAIT_BAND_HZ[0]) & (freqs <= GAIT_BAND_HZ[1])

    speed_gain, speed_lag = _gain(speed, cmd_speed, moving_threshold)
    yaw_gain, yaw_lag = _gain(yaw_rate[:n], c[:n, 2], 0.2)

    return Summary(
        speed=float(speed[moving].mean()) if moving.any() else 0.0,
        speed_gain=speed_gain,
        yaw_rate_gain=yaw_gain,
        height_mean=float(z.mean()),
        height_std=float(z.std()),
        gait_hz=float(freqs[band][np.argmax(power[band])]) if band.any() else 0.0,
        speed_lag=speed_lag,
        yaw_lag=yaw_lag,
    )


def chaos_spread(summaries: list[Summary]) -> dict[str, float]:
    """Peak-to-peak of each statistic across repeated perturbed runs.

    A statistic is only usable as a fitting target when the sim-vs-real
    difference is comfortably larger than this.
    """
    keys = summaries[0].as_dict()
    return {
        k: float(max(s.as_dict()[k] for s in summaries) - min(s.as_dict()[k] for s in summaries))
        for k in keys
    }
