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

"""A fallen or flailing G1 damps itself once, drops later commands, and tells the coordinator."""

from __future__ import annotations

from collections.abc import Generator
import math

import pytest

from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.robot.unitree.g1.protective import stop_reason, tilt_deg
from dimos.robot.unitree.g1.wholebody_connection import (
    _NUM_MOTOR_SLOTS,
    _NUM_MOTORS,
    G1LowStateSnapshot,
    G1WholeBodyConnection,
)


def _quat(roll_deg: float, yaw_deg: float = 0.0) -> tuple[float, float, float, float]:
    """A roll then a yaw, as the (w,x,y,z) IMU reports it."""
    r, y = math.radians(roll_deg) / 2, math.radians(yaw_deg) / 2
    return (
        math.cos(r) * math.cos(y),
        math.sin(r) * math.cos(y),
        math.sin(r) * math.sin(y),
        math.cos(r) * math.sin(y),
    )


def _sample(
    roll_deg: float = 0.0, yaw_deg: float = 0.0, fast_joints: int = 0
) -> G1LowStateSnapshot:
    """A sample rolled by roll_deg with fast_joints joints moving at 6 rad/s."""
    return G1LowStateSnapshot(
        positions=[0.1] * _NUM_MOTORS,
        velocities=[6.0] * fast_joints + [0.0] * (_NUM_MOTORS - fast_joints),
        efforts=[0.0] * _NUM_MOTORS,
        quaternion=_quat(roll_deg, yaw_deg),
        gyroscope=(0.0, 0.0, 0.0),
        accelerometer=(0.0, 0.0, 9.81),
    )


class _MotorCmd:
    def __init__(self) -> None:
        self.mode, self.q, self.dq, self.kp, self.kd, self.tau = 0x01, 0.0, 0.0, 0.0, 2.0, 0.0


class _LowCmd:
    def __init__(self) -> None:
        self.mode_machine, self.crc = 5, 0
        self.motor_cmd = [_MotorCmd() for _ in range(_NUM_MOTOR_SLOTS)]


class _Publisher:
    """Records every frame as (q, kp, kd, tau) per motor."""

    def __init__(self) -> None:
        self.frames: list[list[tuple[float, float, float, float]]] = []

    def Write(self, cmd: _LowCmd) -> None:
        self.frames.append([(m.q, m.kp, m.kd, m.tau) for m in cmd.motor_cmd[:_NUM_MOTORS]])

    def Close(self) -> None:
        return None


class _Crc:
    def Crc(self, cmd: _LowCmd) -> int:
        return 0


class _Stop:
    def __init__(self) -> None:
        self.calls = 0

    def estop(self) -> bool:
        self.calls += 1
        return True

    def clear_estop(self) -> bool:
        return True


@pytest.fixture()
def connection() -> Generator[G1WholeBodyConnection, None, None]:
    module = G1WholeBodyConnection()
    module._estop = _Stop()  # type: ignore[assignment]
    module._publisher, module._crc, module._low_cmd = _Publisher(), _Crc(), _LowCmd()  # type: ignore[assignment]
    module._mode_machine = 5
    try:
        yield module
    finally:
        module._close_module()


def test_stop_reason_names_the_condition() -> None:
    """A fall reports its tilt, a flail how many joints; upright, spinning and walking report nothing."""
    assert stop_reason(_quat(80.0), [0.0] * _NUM_MOTORS) == "fallen, tilt 80 deg"
    assert stop_reason(_quat(0.0), [6.0] * 3 + [0.0] * 26) == "flailing, 3 joints past 5 rad/s"
    assert stop_reason(_quat(2.0), [7.5, -7.5] + [0.0] * 27) == ""
    assert tilt_deg(_quat(0.0, yaw_deg=135.0)) == 0.0


def test_upright_at_any_yaw_never_stops(connection: G1WholeBodyConnection) -> None:
    """Tilt is measured off gravity, so spinning in place is not a fall."""
    for yaw_deg in (0.0, 90.0, 180.0, 270.0):
        connection._check_protective(_sample(roll_deg=2.0, yaw_deg=yaw_deg))
    assert connection._publisher.frames == []  # type: ignore[union-attr]
    assert connection._estop.calls == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize("sample", [_sample(roll_deg=80.0), _sample(fast_joints=3)])
def test_a_trip_damps_once_then_drops_commands(
    connection: G1WholeBodyConnection, sample: G1LowStateSnapshot
) -> None:
    """One frame goes out holding q with kp and tau zero and kd kept; later commands never do."""
    for s in (sample, sample, _sample()):
        connection._check_protective(s)
    connection._on_motor_command(
        MotorCommandArray(q=[0.0] * _NUM_MOTORS, kp=[60.0] * _NUM_MOTORS, kd=[3.0] * _NUM_MOTORS)
    )
    assert connection._publisher.frames == [[(0.1, 0.0, 2.0, 0.0)] * _NUM_MOTORS]  # type: ignore[union-attr]
    assert connection._estop.calls == 1  # type: ignore[attr-defined]
