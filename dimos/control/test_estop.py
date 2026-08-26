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

"""E-STOP latches at the coordinator and reaches every task, including ones that arrive later."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from dimos.control.coordinator import ControlCoordinator
from dimos.control.task import CoordinatorState, JointStateSnapshot


class _Task:
    """A task that records the latch, like the real ones do."""

    def __init__(self, name: str = "t") -> None:
        self.name, self.estopped = name, False

    def set_estop(self, estopped: bool) -> None:
        self.estopped = estopped


class _Deaf:
    """A task with no set_estop at all: it must not silence the tasks after it."""

    name = "deaf"


class _Angry:
    """A task that raises: it must not swallow the stop for the tasks after it."""

    name = "angry"

    def set_estop(self, estopped: bool) -> None:
        raise RuntimeError("boom")


def _state(speed: float) -> CoordinatorState:
    """One tick with a single joint moving at speed rad/s."""
    return CoordinatorState(
        joints=JointStateSnapshot(
            joint_positions={"j": 0.0},
            joint_velocities={"j": speed},
            joint_efforts={"j": 0.0},
            timestamp=0.0,
        ),
        imu={},
        t_now=0.0,
        dt=0.01,
    )


@pytest.fixture()
def armed() -> Generator[ControlCoordinator, None, None]:
    module = ControlCoordinator(max_joint_speed_rad_s=20.0)
    try:
        yield module
    finally:
        module._close_module()


def test_unconfigured_never_trips() -> None:
    """The limit defaults to None, so existing stacks keep their behaviour exactly."""
    module = ControlCoordinator()
    try:
        assert module._unsafe(_state(speed=999)) == ""
    finally:
        module._close_module()


@pytest.mark.parametrize(("speed", "expected"), [(19.0, ""), (25.0, "rad/s")])
def test_flailing_joint(armed: ControlCoordinator, speed: float, expected: str) -> None:
    """A joint past the limit names itself; one under it is fine."""
    assert expected in armed._unsafe(_state(speed=speed))


def test_one_bad_task_cannot_swallow_the_stop(armed: ControlCoordinator) -> None:
    """A task that raises or has no handler must not stop the latch reaching the rest."""
    good = _Task()
    armed._tasks.update({"angry": _Angry(), "deaf": _Deaf(), "good": good})  # type: ignore[dict-item]
    armed.set_estop(True)
    assert good.estopped


def test_a_task_registered_after_the_trip_comes_up_stopped(armed: ControlCoordinator) -> None:
    """The latch lives on the coordinator, so a late task cannot come up live."""
    armed.set_estop(True)
    late = _Task("late")
    armed._apply_estop("late", late)  # type: ignore[arg-type]
    assert late.estopped
