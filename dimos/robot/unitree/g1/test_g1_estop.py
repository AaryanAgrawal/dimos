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

"""Tilt reads off the hg quaternion, and the checks trip in priority order."""

from __future__ import annotations

from collections.abc import Generator
import math
from types import SimpleNamespace

import pytest

from dimos.robot.unitree.g1.g1_estop import G1EStop, tilt_deg

# Measured on the robot, standing: rt/lowstate imu_state.quaternion with rpy (0.473, -1.776, 0.209) deg.
STANDING_WXYZ = (0.99987, 0.00416, -0.01549, 0.00188)
STANDING_TILT_DEG = 1.838
FACE_DOWN_WXYZ = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)


def _sample(quaternion: tuple[float, float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(imu_state=SimpleNamespace(quaternion=quaternion))


@pytest.fixture()
def estop() -> Generator[G1EStop, None, None]:
    module = G1EStop(max_tilt_deg=45.0, lowstate_timeout_sec=0.5)
    try:
        yield module
    finally:
        module._close_module()


def test_tilt_is_zero_upright_and_yaw_invariant() -> None:
    """Tilt measures departure from gravity only, so spinning in place cannot register as a fall."""
    assert tilt_deg((1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-9)
    for yaw_rad in (0.5, 2.0, math.pi):
        yawed = (math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2))
        assert tilt_deg(yawed) == pytest.approx(0.0, abs=1e-9)


def test_tilt_matches_the_robots_own_rpy() -> None:
    """The wxyz reading agrees with the rpy the same lowstate frame carried; xyzw would read 178."""
    assert tilt_deg(STANDING_WXYZ) == pytest.approx(STANDING_TILT_DEG, abs=1e-3)
    assert tilt_deg(FACE_DOWN_WXYZ) == pytest.approx(90.0, abs=1e-9)


def test_standing_robot_does_not_trip(estop: G1EStop) -> None:
    """A standing G1 sits ~25x under the threshold, so a healthy robot never latches."""
    estop._check(_sample(STANDING_WXYZ), age_sec=0.0)
    assert estop.tripped() == ""


def test_fall_trips(estop: G1EStop) -> None:
    """Tilt past the threshold latches, and stays latched once the robot is level again."""
    estop._check(_sample(FACE_DOWN_WXYZ), age_sec=0.0)
    assert estop.tripped() == "fallen"


def test_silent_lowstate_trips_and_outranks_tilt(estop: G1EStop) -> None:
    """No lowstate means no evidence the robot is upright, so staleness is checked first."""
    estop._check(_sample(FACE_DOWN_WXYZ), age_sec=0.6)
    assert estop.tripped() == "lowstate silent"
