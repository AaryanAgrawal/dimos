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

"""Records G1 commands, state, and real or simulated odometry into mem2."""

from __future__ import annotations

from pydantic import Field

from dimos.core.stream import In
from dimos.hardware.sensors.lidar.pointlio.recorder import PointlioRecorder, PointlioRecorderConfig
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


class G1RecorderConfig(PointlioRecorderConfig):
    # don't compress
    stream_codecs: dict[str, str] = Field(
        default_factory=lambda: {"realsense_depth_image": "lz4+lcm"}
    )
    poseless_streams: list[str] = Field(
        default_factory=lambda: [
            "tele_cmd_vel",
            "cmd_vel",
            "motor_states",
            "imu",
            "motor_command",
            "sim_odom",
            "sim_pointcloud",
        ]
    )


class G1Recorder(PointlioRecorder):
    config: G1RecorderConfig

    color_image: In[Image]
    realsense_depth_image: In[Image]
    realsense_camera_info: In[CameraInfo]
    realsense_depth_camera_info: In[CameraInfo]
    tele_cmd_vel: In[Twist]
    cmd_vel: In[Twist]
    motor_states: In[JointState]
    imu: In[Imu]
    motor_command: In[MotorCommandArray]
    sim_odom: In[PoseStamped]
    sim_pointcloud: In[PointCloud2]

    async def _resolve_pose(self, name: str, msg: object, ts: float) -> Pose | None:
        if name in self.config.poseless_streams:
            return None
        return await super()._resolve_pose(name, msg, ts)
