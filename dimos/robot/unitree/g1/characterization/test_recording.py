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

"""The G1 reader preserves mem2 timestamps, SI values, and named frames."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.robot.unitree.g1.characterization.recording import (
    filter_pose_outliers,
    read_plant_recording,
    read_recording,
)
from dimos.robot.unitree.g1.frames import (
    pelvis_T_mid360,
    world_T_pelvis_from_mid360_odometry,
)


def _pointlio_odom(ts_s: float) -> Odometry:
    world_T_pelvis = np.eye(4)
    world_T_pelvis[:3, :3] = Rotation.from_euler("z", 0.3).as_matrix()
    world_T_pelvis[:3, 3] = (1.0, 2.0, 0.74)
    world_T_mid360 = world_T_pelvis @ pelvis_T_mid360()
    raw_rotation = world_T_mid360[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    return Odometry(
        ts=ts_s,
        frame_id="odom",
        child_frame_id="mid360_link",
        pose=Pose(
            position=world_T_mid360[:3, 3],
            orientation=Quaternion.from_rotation_matrix(raw_rotation),
        ),
        twist=Twist(linear=[0.2, 0.1, 0.0], angular=[0.0, 0.0, 0.4]),
    )


def _write_fixture(path: Path) -> None:
    store = SqliteStore(path=str(path))
    store.start()
    try:
        store.stream("tele_cmd_vel", Twist).append(
            Twist(linear=[0.3, 0.0, 0.0]), ts=9.99, pose=None
        )
        store.stream("cmd_vel", Twist).append(Twist(linear=[0.3, 0.0, 0.0]), ts=10.0, pose=None)
        store.stream("pointlio_odometry", Odometry).append(
            _pointlio_odom(10.01), ts=10.01, pose=None
        )
        store.stream("motor_command", MotorCommandArray).append(
            MotorCommandArray(
                q=[0.1, 0.2, 0.3],
                dq=[0.3, 0.4, 0.5],
                kp=[1.0, 2.0, 3.0],
                kd=[3.0, 4.0, 5.0],
                tau=[5.0, 6.0, 7.0],
            ),
            ts=10.02,
            pose=None,
        )
        store.stream("motor_states", JointState).append(
            JointState(
                name=["g1/waist_yaw", "g1/waist_roll", "g1/waist_pitch"],
                position=[0.0, 0.0, 0.0],
                velocity=[0.4, 0.5, 0.6],
                effort=[0.6, 0.7, 0.8],
            ),
            ts=10.03,
            pose=None,
        )
        store.stream("imu", Imu).append(Imu(ts=10.04), ts=10.04, pose=None)
    finally:
        store.stop()


def test_read_recording_converts_world_T_mid360_to_world_T_pelvis(tmp_path: Path) -> None:
    path = tmp_path / "fixture.db"
    _write_fixture(path)

    recording = read_recording(path, cache=False)

    np.testing.assert_allclose(recording.command_t_s, [10.0])
    np.testing.assert_allclose(recording.command_body_twist, [[0.3, 0.0, 0.0]])
    np.testing.assert_allclose(recording.teleop_t_s, [9.99])
    np.testing.assert_allclose(recording.world_p_pelvis_m, [[1.0, 2.0, 0.74]], atol=1e-8)
    np.testing.assert_allclose(
        Rotation.from_quat(recording.world_q_pelvis_xyzw).as_euler("xyz"),
        [[0.0, 0.0, 0.3]],
        atol=1e-8,
    )


def test_read_plant_recording_preserves_low_level_units(tmp_path: Path) -> None:
    path = tmp_path / "fixture.db"
    _write_fixture(path)

    recording = read_plant_recording(path, cache=False)

    assert recording.motor_names == ("g1/waist_yaw", "g1/waist_roll", "g1/waist_pitch")
    np.testing.assert_allclose(recording.motor_command_q_rad, [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(recording.motor_command_tau_ff_nm, [[5.0, 6.0, 7.0]])
    np.testing.assert_allclose(recording.motor_state_dq_rad_s, [[0.4, 0.5, 0.6]])
    np.testing.assert_allclose(recording.imu_accel_m_s2, [[0.0, 0.0, 0.0]])


def test_filter_pose_outliers_replaces_only_isolated_teleport() -> None:
    world_p_frame_m = np.column_stack((np.arange(9) * 0.01, np.zeros((9, 2))))
    yaw_rad = np.arange(9) * 0.01
    world_q_frame_xyzw = Rotation.from_euler(
        "xyz", np.column_stack((np.zeros((9, 2)), yaw_rad))
    ).as_quat()
    world_p_frame_m[4] += (1.0, -1.0, 0.5)
    world_q_frame_xyzw[4] = Rotation.from_euler("z", 1.0).as_quat()

    filtered_p_m, filtered_q_xyzw, outlier = filter_pose_outliers(
        world_p_frame_m, world_q_frame_xyzw
    )

    np.testing.assert_array_equal(outlier, np.arange(9) == 4)
    np.testing.assert_allclose(filtered_p_m[:, 0], np.arange(9) * 0.01)
    np.testing.assert_allclose(
        Rotation.from_quat(filtered_q_xyzw).as_euler("zyx")[:, 0],
        yaw_rad,
    )


def test_mid360_to_pelvis_uses_measured_waist_pose() -> None:
    waist_rad = (0.2, -0.1, 0.15)
    world_T_pelvis = np.eye(4)
    world_T_pelvis[:3, :3] = Rotation.from_euler("xyz", (0.05, -0.1, 0.4)).as_matrix()
    world_T_pelvis[:3, 3] = (1.0, -2.0, 0.74)
    world_T_mid360 = world_T_pelvis @ pelvis_T_mid360(*waist_rad)
    raw_rotation = world_T_mid360[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    odometry = Odometry(
        frame_id="odom",
        child_frame_id="mid360_link",
        pose=Pose(
            position=world_T_mid360[:3, 3],
            orientation=Quaternion.from_rotation_matrix(raw_rotation),
        ),
    )

    actual = world_T_pelvis_from_mid360_odometry(odometry, *waist_rad)

    np.testing.assert_allclose(actual, world_T_pelvis, atol=1e-10)
