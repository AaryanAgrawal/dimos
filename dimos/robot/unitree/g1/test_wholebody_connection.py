# Copyright 2025-2026 Dimensional Inc.
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

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.robot.unitree.g1.wholebody_connection import (
    _NUM_MOTOR_SLOTS,
    _NUM_MOTORS,
    G1WholeBodyConnection,
    G1WholeBodyConnectionConfig,
)


def _deferred_connection(events: list[str]) -> G1WholeBodyConnection:
    connection = G1WholeBodyConnection.__new__(G1WholeBodyConnection)
    connection.config = G1WholeBodyConnectionConfig(defer_sport_mode_release=True)
    connection._lock = threading.Lock()
    connection._sport_release_lock = threading.Lock()
    connection._sport_mode_released = False
    connection._mode_machine = 5
    connection._low_cmd = SimpleNamespace(
        mode_machine=5,
        motor_cmd=[
            SimpleNamespace(q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0) for _ in range(_NUM_MOTOR_SLOTS)
        ],
        crc=0,
    )
    connection._crc = MagicMock()
    connection._crc.Crc.return_value = 123
    connection._publisher = MagicMock()
    connection._publisher.Write.side_effect = lambda _cmd: events.append("write")
    connection._release_sport_mode = lambda: events.append("release")
    return connection


def test_deferred_release_happens_once_immediately_before_first_command() -> None:
    events: list[str] = []
    connection = _deferred_connection(events)
    command = MotorCommandArray(
        q=[0.1 * i for i in range(_NUM_MOTORS)],
        kp=[20.0] * _NUM_MOTORS,
        kd=[2.0] * _NUM_MOTORS,
    )

    connection._on_motor_command(command)
    connection._on_motor_command(command)

    assert events == ["release", "write", "write"]
    assert [slot.q for slot in connection._low_cmd.motor_cmd[:_NUM_MOTORS]] == command.q


def test_invalid_command_does_not_release_sport_mode() -> None:
    events: list[str] = []
    connection = _deferred_connection(events)

    connection._on_motor_command(MotorCommandArray(q=[0.0]))

    assert events == []


def test_command_before_connection_ready_does_not_release_sport_mode() -> None:
    events: list[str] = []
    connection = _deferred_connection(events)
    connection._publisher = None

    connection._on_motor_command(MotorCommandArray(q=[0.0] * _NUM_MOTORS))

    assert events == []
