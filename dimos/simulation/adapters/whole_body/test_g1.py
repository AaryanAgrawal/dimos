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

"""The MuJoCo G1 adapter passes the adapter Protocol and damps itself like the hardware connection."""

from __future__ import annotations

from dimos.hardware.whole_body.spec import MotorCommand, WholeBodyAdapter
from dimos.robot.unitree.g1.protective import Hold
from dimos.simulation.adapters.whole_body.g1 import _NUM_MOTORS, SimMujocoG1WholeBodyAdapter

_UPRIGHT = (1.0, 0.0, 0.0, 0.0)
_FALLEN = (0.5, 0.866, 0.0, 0.0)  # 120 deg of roll as (w,x,y,z)


class _Shm:
    """Answers reads with a still robot at the given orientation; records every write."""

    def __init__(self) -> None:
        self.quaternion = _UPRIGHT
        self.writes: list[tuple[list[float], list[float], list[float], list[float]]] = []

    def read_positions(self, n: int) -> list[float]:
        return [0.1] * n

    def read_velocities(self, n: int) -> list[float]:
        return [0.0] * n

    def read_imu(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, float, float], tuple[float, float, float]]:
        return self.quaternion, (0.0, 0.0, 0.0), (0.0, 0.0, 9.81)

    def write_pd_tau_command(
        self, q: list[float], kp: list[float], kd: list[float], tau: list[float]
    ) -> None:
        self.writes.append((q, kp, kd, tau))


def _adapter() -> tuple[SimMujocoG1WholeBodyAdapter, _Shm]:
    adapter, shm = SimMujocoG1WholeBodyAdapter(address="g1.xml"), _Shm()
    adapter._hold = Hold(0.0)  # these tests cover the damp path, not the hold
    adapter._shm, adapter._connected = shm, True  # type: ignore[assignment]
    return adapter, shm


def test_adapter_satisfies_the_protocol() -> None:
    """get_limits exists, so the coordinator's isinstance check accepts the sim adapter."""
    adapter, _ = _adapter()
    assert isinstance(adapter, WholeBodyAdapter) and adapter.get_limits() is None


def test_a_fall_damps_with_the_last_kd_and_drops_later_commands() -> None:
    """One damping write with kp and tau zero and the last commanded kd; later commands never reach SHM."""
    adapter, shm = _adapter()
    command = [MotorCommand(q=0.0, kp=60.0, kd=2.0, tau=1.0)] * _NUM_MOTORS
    assert adapter.write_motor_commands(command)
    shm.quaternion = _FALLEN
    for _ in range(2):
        adapter.read_imu()
    assert adapter.write_motor_commands(command)
    zeros = [0.0] * _NUM_MOTORS
    assert shm.writes[1:] == [([0.1] * _NUM_MOTORS, zeros, [2.0] * _NUM_MOTORS, zeros)]
