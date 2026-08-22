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

"""Latch an e-stop when the G1 is in a state it must not keep moving in.

    rt/lowstate (DDS, 500 Hz) -> [silent? fallen?] -> estop: Out[Bool] -> MovementManager gate

EXPERIMENTAL, and not a substitute for the physical e-stop. It reads the robot rather than the LIO
pose because scan matching degrades during the very event being detected, and the pelvis IMU is
gravity-referenced so it needs no mount correction. This will be spun out to feed
ControlCoordinator.set_estop, where the stop becomes a highest-priority task instead of a veto on
one publisher, and the checks below become an ordered set of failure states.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from dimos_lcm.std_msgs import Bool  # type: ignore[import-untyped]
from pydantic import Field

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def tilt_deg(quaternion: Any) -> float:
    """Angle between the pelvis z axis and gravity, from the IMU's fused quaternion (w, x, y, z)."""
    w, x, y, z = (float(v) for v in quaternion)
    world_r_pelvis = Quaternion(x, y, z, w).to_rotation_matrix()
    return math.degrees(math.acos(min(1.0, max(-1.0, float(world_r_pelvis[2, 2])))))


class G1EStopConfig(ModuleConfig):
    network_interface: str = "eth0"
    # A standing G1 measures 1.7-1.9 deg over 3002 samples, so 45 deg clears gait by ~25x.
    max_tilt_deg: float = Field(default=45.0, gt=0.0)
    # lowstate publishes at 500 Hz: half a second of silence means the low-level controller is gone.
    lowstate_timeout_sec: float = Field(default=0.5, gt=0.0)
    poll_hz: float = Field(default=50.0, gt=0.0)


class G1EStop(Module):
    """Watch rt/lowstate and latch estop when the robot has fallen or gone silent."""

    config: G1EStopConfig

    estop: Out[Bool]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._subscriber: Any = None
        self._tripped = ""
        self._stop_event = threading.Event()
        self._watch_thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._stop_event.clear()
        self._subscriber = self._init_lowstate_subscriber()
        self._watch_thread = threading.Thread(target=self._watch_loop, name="g1-estop", daemon=True)
        self._watch_thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._watch_thread is not None and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._watch_thread = None
        if self._subscriber is not None:
            try:
                self._subscriber.Close()
            except (OSError, RuntimeError) as e:
                logger.warning("ChannelSubscriber Close raised", error=str(e))
        self._subscriber = None
        super().stop()

    @rpc
    def tripped(self) -> str:
        """Why this e-stop latched, empty while it is clear."""
        return self._tripped

    def _init_lowstate_subscriber(self) -> Any:
        # Lazy SDK imports - file must import cleanly outside the [unitree-dds] extra.
        try:
            from unitree_sdk2py.core.channel import (  # type: ignore[import-not-found]
                ChannelFactoryInitialize,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # type: ignore[import-not-found]
                LowState_,
            )
        except ImportError as e:
            # Degrading to no e-stop would be worse than not starting: it reads as protection.
            raise RuntimeError("G1EStop needs the [unitree-dds] extra for rt/lowstate") from e
        if self.config.network_interface:
            ChannelFactoryInitialize(0, self.config.network_interface)
        else:
            ChannelFactoryInitialize(0)
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(None, 0)
        return subscriber

    def _watch_loop(self) -> None:
        period = 1.0 / self.config.poll_hz
        last_rx = time.monotonic()
        while not self._stop_event.is_set():
            sample = self._subscriber.Read(period)
            now = time.monotonic()
            if sample is not None:
                last_rx = now
            if not self._tripped:
                self._check(sample, now - last_rx)
            self.estop.publish(Bool(data=bool(self._tripped)))
            self._stop_event.wait(period)

    def _check(self, sample: Any, age_sec: float) -> None:
        """Trip on the first failure state that holds, highest priority first."""
        if age_sec > self.config.lowstate_timeout_sec:
            self._trip("lowstate silent", age_sec=round(age_sec, 2))
        elif sample is not None:
            tilt = tilt_deg(sample.imu_state.quaternion)
            if tilt > self.config.max_tilt_deg:
                self._trip("fallen", tilt_deg=round(tilt, 1))

    def _trip(self, reason: str, **detail: Any) -> None:
        # Latching: nothing clears this in process, because a fallen robot must not resume itself.
        self._tripped = reason
        logger.error("E-STOP tripped", reason=reason, **detail)
