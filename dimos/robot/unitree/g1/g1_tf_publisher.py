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

"""Live sensor-mount tf for the G1.

Publishes the mount tree from g1.urdf onto tf. The mid360 and d435 are fixed
to the torso, but base_link (the pelvis) sits across the three waist joints,
so that edge is computed live from the rt/lowstate waist angles. Until the
first LowState arrives the rest pose is published.

The published tree is rooted at mid360_link so the edges stay off the entity
the live odom -> mid360_link edge writes. The tf buffer composes either
direction.

The lowstate subscriber is read-only. It never opens the rt/lowcmd path, so
it coexists with high-level AI-mode control.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from pydantic import Field

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.protocol.tf.static_tf_publisher import FrameSpec, frames_to_edge_transforms
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

MID360_PITCH = 0.04014257279586953
D435_PITCH = 0.8307767239493009

# g1.urdf fixed sensor mounts on torso_link.
FRAMES: list[FrameSpec] = [
    ("torso_link", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("mid360_link", "torso_link", (0.0002835, 0.00003, 0.41618), (0.0, MID360_PITCH, 0.0)),
    ("d435_link", "torso_link", (0.0576235, 0.01753, 0.42987), (0.0, D435_PITCH, 0.0)),
]

# waist_roll_joint origin, the only nonzero offset in the pelvis -> torso chain.
_WAIST_ROLL_ORIGIN = (-0.0039635, 0.0, 0.044)

# rt/lowstate motor indices, ordering from make_humanoid_joints("g1").
_WAIST_YAW_IDX = 12
_WAIST_ROLL_IDX = 13
_WAIST_PITCH_IDX = 14


def base_to_torso(waist_yaw: float, waist_roll: float, waist_pitch: float) -> Transform:
    """base_link -> torso_link through the g1.urdf waist chain."""
    yaw = Transform(
        rotation=Quaternion.from_euler(Vector3(0.0, 0.0, waist_yaw)),
        frame_id="base_link",
        child_frame_id="waist_yaw_link",
    )
    roll = Transform(
        translation=Vector3(*_WAIST_ROLL_ORIGIN),
        rotation=Quaternion.from_euler(Vector3(waist_roll, 0.0, 0.0)),
        frame_id="waist_yaw_link",
        child_frame_id="waist_roll_link",
    )
    pitch = Transform(
        rotation=Quaternion.from_euler(Vector3(0.0, waist_pitch, 0.0)),
        frame_id="waist_roll_link",
        child_frame_id="torso_link",
    )
    return yaw + roll + pitch


def mount_transforms(
    waist_yaw: float = 0.0, waist_roll: float = 0.0, waist_pitch: float = 0.0
) -> list[Transform]:
    """The mount tree as published: rooted at mid360_link."""
    edges = {t.child_frame_id: t for t in frames_to_edge_transforms(FRAMES)}
    return [
        -edges["mid360_link"],
        -base_to_torso(waist_yaw, waist_roll, waist_pitch),
        edges["d435_link"],
    ]


class G1TfPublisherConfig(ModuleConfig):
    network_interface: str = ""
    publish_hz: float = Field(default=20.0, gt=0.0)


class G1TfPublisher(Module):
    """Publishes the G1 sensor mount tree onto tf, waist edge live from lowstate."""

    config: G1TfPublisherConfig

    tf: Out[TFMessage]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._running = False
        self._subscriber: Any = None
        self._waist = (0.0, 0.0, 0.0)
        self._waist_lock = threading.Lock()
        self._waist_live = False
        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._subscriber = self._init_lowstate_subscriber()
        if self._subscriber is not None:
            self._reader_thread = threading.Thread(
                target=self._reader_loop, name="g1-tf-lowstate", daemon=True
            )
            self._reader_thread.start()
        self._running = True
        self.spawn(self._publish_loop())
        logger.info(
            "G1TfPublisher publishing at %.1f Hz (waist %s)",
            self.config.publish_hz,
            "live" if self._subscriber is not None else "rest pose",
        )

    @rpc
    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._reader_thread = None
        if self._subscriber is not None:
            try:
                self._subscriber.Close()
            except (OSError, RuntimeError) as e:
                logger.warning(f"ChannelSubscriber Close raised: {e}")
        self._subscriber = None
        super().stop()

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
        except ImportError:
            logger.warning("unitree_sdk2py unavailable - publishing rest-pose waist only")
            return None
        try:
            if self.config.network_interface:
                ChannelFactoryInitialize(0, self.config.network_interface)
            else:
                ChannelFactoryInitialize(0)
        except Exception as e:
            # Idempotent - already initialized by a sibling participant is fine.
            logger.debug(f"ChannelFactoryInitialize raised (likely already init'd): {e}")
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(None, 0)
        return subscriber

    def _reader_loop(self) -> None:
        period = 1.0 / self.config.publish_hz
        while not self._stop_event.is_set():
            sample = self._subscriber.Read()
            if sample is not None:
                waist = (
                    float(sample.motor_state[_WAIST_YAW_IDX].q),
                    float(sample.motor_state[_WAIST_ROLL_IDX].q),
                    float(sample.motor_state[_WAIST_PITCH_IDX].q),
                )
                with self._waist_lock:
                    self._waist = waist
                if not self._waist_live:
                    self._waist_live = True
                    logger.info("First LowState received - waist edge is live")
            self._stop_event.wait(period)

    async def _publish_loop(self) -> None:
        period = 1.0 / self.config.publish_hz
        while self._running:
            with self._waist_lock:
                waist_yaw, waist_roll, waist_pitch = self._waist
            transforms = mount_transforms(waist_yaw, waist_roll, waist_pitch)
            now = time.time()
            for transform in transforms:
                transform.ts = now
            self.tf.publish(TFMessage(*transforms))
            await asyncio.sleep(period)
