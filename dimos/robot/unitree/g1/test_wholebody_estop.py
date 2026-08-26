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

"""A fallen G1 stops the robot once, through whatever the blueprint wired as its E-STOP."""

from __future__ import annotations

from collections.abc import Generator
import math

import pytest

from dimos.robot.unitree.g1.wholebody_connection import (
    _NUM_MOTORS,
    G1LowStateSnapshot,
    G1WholeBodyConnection,
)


def _sample(roll_deg: float, yaw_deg: float = 0.0) -> G1LowStateSnapshot:
    """A sample rolled by roll_deg and spun by yaw_deg, as the (w,x,y,z) IMU reports it."""
    r, y = math.radians(roll_deg) / 2, math.radians(yaw_deg) / 2
    zeros = [0.0] * _NUM_MOTORS
    return G1LowStateSnapshot(
        positions=zeros,
        velocities=zeros,
        efforts=zeros,
        quaternion=(
            math.cos(r) * math.cos(y),
            math.sin(r) * math.cos(y),
            math.sin(r) * math.sin(y),
            math.cos(r) * math.sin(y),
        ),
        gyroscope=(0.0, 0.0, 0.0),
        accelerometer=(0.0, 0.0, 9.81),
    )


class _Stop:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def set_estop(self, estopped: bool) -> bool:
        self.calls.append(estopped)
        return True


@pytest.fixture()
def connection() -> Generator[G1WholeBodyConnection, None, None]:
    module = G1WholeBodyConnection()
    module._estop = _Stop()  # type: ignore[assignment]
    try:
        yield module
    finally:
        module._close_module()


def test_upright_at_any_yaw_never_stops(connection: G1WholeBodyConnection) -> None:
    """Tilt is measured off gravity, so spinning in place is not a fall."""
    for yaw_deg in (0.0, 90.0, 180.0, 270.0):
        connection._check_upright(_sample(roll_deg=2.0, yaw_deg=yaw_deg))
    assert connection._estop.calls == []  # type: ignore[union-attr]


def test_fall_stops_once_and_stays_stopped(connection: G1WholeBodyConnection) -> None:
    """Past max_tilt_deg it stops, and coming back upright does not un-stop it."""
    for roll_deg in (80.0, 80.0, 0.0):
        connection._check_upright(_sample(roll_deg=roll_deg))
    assert connection._estop.calls == [True]  # type: ignore[union-attr]
