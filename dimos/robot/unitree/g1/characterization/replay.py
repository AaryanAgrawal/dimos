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

"""Replay recorded G1 twists through the live GR00T input seam."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.memory.cli.dataset import open_store
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.std_msgs.Bool import Bool


class TwistRecordingReplayConfig(ModuleConfig):
    recording: Path = Field(description="Source mem2 DB")
    stream_name: str = "cmd_vel"
    seek_s: float | None = None
    duration_s: float | None = None
    lead_in_s: float = 2.0


class TwistRecordingReplay(Module):
    """Publish one recorded Twist stream at its original timing."""

    config: TwistRecordingReplayConfig
    cmd_vel: Out[Twist]
    done: Out[Bool]

    @rpc
    def start(self) -> None:
        super().start()
        if self.config.g.simulation != "mujoco":
            raise RuntimeError("TwistRecordingReplay only runs with --simulation mujoco")
        store = open_store(self.config.recording)
        self.register_disposable(store)
        replay = store.replay(seek=self.config.seek_s, duration=self.config.duration_s)
        stream = replay.stream(self.config.stream_name)
        timer = threading.Timer(self.config.lead_in_s, self._subscribe, args=(stream,))
        timer.daemon = True
        timer.start()
        self.register_disposable(Disposable(timer.cancel))

    def _subscribe(self, stream: Any) -> None:
        subscription = stream.observable().subscribe(
            on_next=self.cmd_vel.publish,
            on_error=self._on_error,
            on_completed=self._on_completed,
        )
        self.register_disposable(subscription)

    def _on_error(self, error: Exception) -> None:
        self.cmd_vel.publish(Twist())
        raise RuntimeError(f"twist replay failed: {error}") from error

    def _on_completed(self) -> None:
        self.cmd_vel.publish(Twist())
        self.done.publish(Bool(True))
