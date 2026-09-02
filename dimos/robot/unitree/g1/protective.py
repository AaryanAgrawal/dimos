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

"""Protective stops the G1 hardware connection and its MuJoCo adapter both run."""

from __future__ import annotations

from collections.abc import Sequence
import math

# A standing G1 reads about 2 deg; a fall passes 45 deg on the way down.
MAX_TILT_DEG = 45.0
# SIMULATED GR00T, steady state: walking puts at most 2 joints past 4 rad/s, a lift 12, a fall 17.
FLAIL_JOINT_SPEED_RAD_S = 4.0
# One clear of walking, nine clear of a lift; the split is flat from 3 to 6 rad/s.
FLAIL_JOINT_COUNT = 3
# SIMULATED: a lift or fall holds the reason for 0.5 s and more; the arming snap lasts 15 ms.
STOP_HOLD_S = 0.05


def tilt_deg(quaternion_wxyz: Sequence[float]) -> float:
    """Angle between the body z axis and gravity."""
    _w, x, y, _z = quaternion_wxyz
    return math.degrees(math.acos(min(1.0, max(-1.0, 1.0 - 2.0 * (x * x + y * y)))))


def stop_reason(
    quaternion_wxyz: Sequence[float],
    velocities_rad_s: Sequence[float],
    *,
    max_tilt_deg: float = MAX_TILT_DEG,
    flail_joint_speed_rad_s: float = FLAIL_JOINT_SPEED_RAD_S,
    flail_joint_count: int = FLAIL_JOINT_COUNT,
) -> str:
    """Why the robot must be damped now, or empty."""
    tilt = tilt_deg(quaternion_wxyz)
    if tilt > max_tilt_deg:
        return f"fallen, tilt {tilt:.0f} deg"
    fast = sum(abs(v) > flail_joint_speed_rad_s for v in velocities_rad_s)
    if fast >= flail_joint_count:
        return f"flailing, {fast} joints past {flail_joint_speed_rad_s:g} rad/s"
    return ""


class Hold:
    """A reason once it has held for hold_s of wall time; a clean sample resets the clock."""

    def __init__(self, hold_s: float = STOP_HOLD_S) -> None:
        self.hold_s = hold_s
        self._since: float | None = None

    def __call__(self, reason: str, now_s: float) -> str:
        if not reason:
            self._since = None
            return ""
        if self._since is None:
            self._since = now_s
        return reason if now_s - self._since >= self.hold_s else ""
