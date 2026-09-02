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


@pytest.fixture()
def coordinator() -> Generator[ControlCoordinator, None, None]:
    module = ControlCoordinator()
    try:
        yield module
    finally:
        module._close_module()


def test_estop_reaches_every_task_and_clear_releases_them(coordinator: ControlCoordinator) -> None:
    """estop() latches a task; clear_estop() releases the latch without starting anything."""
    task = _Task()
    coordinator._tasks["t"] = task  # type: ignore[assignment]
    assert coordinator.estop() and task.estopped
    assert coordinator.clear_estop() and not task.estopped


def test_one_bad_task_cannot_swallow_the_stop(coordinator: ControlCoordinator) -> None:
    """A task that raises or has no handler must not stop the latch reaching the rest."""
    good = _Task()
    coordinator._tasks.update({"angry": _Angry(), "deaf": _Deaf(), "good": good})  # type: ignore[dict-item]
    coordinator.estop()
    assert good.estopped


def test_a_task_registered_after_the_trip_comes_up_stopped(coordinator: ControlCoordinator) -> None:
    """The latch lives on the coordinator, so a late task cannot come up live."""
    coordinator.estop()
    late = _Task("late")
    coordinator._apply_estop("late", late)  # type: ignore[arg-type]
    assert late.estopped
