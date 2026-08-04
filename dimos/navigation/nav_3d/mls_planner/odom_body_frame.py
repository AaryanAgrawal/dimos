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

from __future__ import annotations

from typing import Any

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.native_module import NativeModule, NativeModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry


class OdomBodyFrameConfig(ModuleConfig):
    # base_link from sensor mount rotation, xyzw.
    mount_rotation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    # base_link -> sensor translation (xyz, metres, in base_link's own axes) --
    # the lever arm. LIO reports where the SENSOR is, so without this the body
    # pose is the lidar's: on the Go2 that is 0.30 m ahead of the robot and
    # 0.16 m above it, and every downstream clearance is judged for a body
    # that is not where the robot is.
    mount_translation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    body_frame_id: str = "base_link"


class OdomBodyFrame(Module):
    """Re-express tilted-sensor LIO odometry in the level robot body frame.

    Composes out the fixed mount rotation and subtracts the mount's lever arm,
    so the result really is base_link and not the sensor wearing its name.
    Twist passes through.
    """

    config: OdomBodyFrameConfig

    odometry: In[Odometry]
    body_odometry: Out[Odometry]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._mount_inv = Quaternion(*self.config.mount_rotation).inverse()
        self._lever = Vector3(*self.config.mount_translation)

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odometry.subscribe(self._on_odometry)))

    def _on_odometry(self, msg: Odometry) -> None:
        leveled = msg.orientation * self._mount_inv
        # p_sensor = p_body + R_body * lever, so the body is the sensor less the
        # arm rotated into the world by the BODY's attitude -- the levelled one,
        # not the sensor's, or the arm gets swung by the mount tilt twice.
        offset = leveled.rotate_vector(self._lever)
        body = Vector3(
            msg.position.x - offset.x,
            msg.position.y - offset.y,
            msg.position.z - offset.z,
        )
        self.body_odometry.publish(
            Odometry(
                ts=msg.ts,
                frame_id=msg.frame_id,
                child_frame_id=self.config.body_frame_id,
                pose=Pose(body, leveled),
                twist=msg.twist,
            )
        )


class OdomBodyFrameNativeConfig(NativeModuleConfig):
    cwd: str | None = "rust"
    executable: str = "target/release/odom_body_frame"
    build_command: str | None = "cargo build --release"
    stdin_config: bool = True

    # base_link from sensor mount rotation, xyzw. Identity says the sensor is
    # already level, which is true of no robot that tilts its lidar: the Go2's
    # value is _mount_rotation() in the go2 zenoh blueprints, and a baked host
    # has to be handed it, since --emit-config emits this default instead.
    mount_rotation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    # base_link -> sensor translation (xyz, metres, base_link axes). Zero says
    # the sensor sits exactly on the body origin, which is as unlikely as the
    # identity above; the same --emit-config caveat applies.
    mount_translation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    body_frame_id: str = "base_link"


class OdomBodyFrameNative(NativeModule):
    """Rust-backed odometry leveling: the mount's rotation and its lever arm."""

    config: OdomBodyFrameNativeConfig

    odometry: In[Odometry]
    body_odometry: Out[Odometry]
