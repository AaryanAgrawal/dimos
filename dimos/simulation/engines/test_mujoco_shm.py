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

"""Tests for MuJoCo shared-memory startup."""

from __future__ import annotations

from typing import Any

import pytest

from dimos.simulation.engines import mujoco_shm


class _AttachedBuffer:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_attach_retries_partial_shared_memory_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty buffer is startup-in-progress, and partial attachments are closed."""
    attached = _AttachedBuffer()
    calls = 0

    def attach(**_kwargs: Any) -> _AttachedBuffer:
        nonlocal calls
        calls += 1
        if calls == 1:
            return attached
        raise ValueError("cannot mmap an empty file")

    monkeypatch.setattr(mujoco_shm, "SharedMemory", attach)
    monkeypatch.setattr(mujoco_shm, "_unregister", lambda buffer: buffer)

    with pytest.raises(FileNotFoundError):
        mujoco_shm.ManipShmSet.attach("cold-start")

    assert attached.closed is True
