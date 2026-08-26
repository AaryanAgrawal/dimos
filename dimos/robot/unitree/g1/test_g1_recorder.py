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

"""The G1 recorder captures the velocity command beside Point-LIO odometry."""

import pytest_mock

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.unitree.g1.g1_recorder import G1Recorder, G1RecorderConfig


def test_cmd_vel_is_a_recorded_input() -> None:
    recorder = G1Recorder()
    try:
        assert recorder.inputs["cmd_vel"].type is Twist
        assert recorder.inputs["tele_cmd_vel"].type is Twist
        assert recorder.inputs["motor_states"].type is JointState
        assert recorder.inputs["imu"].type is Imu
        assert recorder.inputs["motor_command"].type is MotorCommandArray
        assert recorder.inputs["sim_odom"].type is PoseStamped
        assert recorder.inputs["sim_pointcloud"].type is PointCloud2
        assert "pointlio_odometry" in recorder.inputs
        assert set(recorder.config.poseless_streams) == {
            "tele_cmd_vel",
            "cmd_vel",
            "motor_states",
            "imu",
            "motor_command",
            "sim_odom",
            "sim_pointcloud",
        }
    finally:
        recorder.stop()


async def test_poseless_stream_skips_pose_lookup(mocker: pytest_mock.MockerFixture) -> None:
    recorder = mocker.MagicMock(spec=G1Recorder)
    recorder.config = G1RecorderConfig(poseless_streams=["cmd_vel"])

    pose = await G1Recorder._resolve_pose(recorder, "cmd_vel", object(), 1.0)

    assert pose is None
