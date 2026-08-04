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

import math

import pytest

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.navigation.nav_3d.mls_planner.odom_body_frame import OdomBodyFrame


def _level(mount_rotation, orientation, mount_translation=(0.0, 0.0, 0.0)):
    """Run one odometry message through the handler and return the output."""
    module = OdomBodyFrame(
        mount_rotation=list(mount_rotation),
        mount_translation=list(mount_translation),
        body_frame_id="base_link",
    )
    try:
        captured = []
        module.body_odometry.subscribe(captured.append)
        module._on_odometry(
            Odometry(
                ts=1.0,
                frame_id="odom",
                child_frame_id="mid360_link",
                pose=Pose(Vector3(1.0, 2.0, 3.0), orientation),
            )
        )
        return captured[0]
    finally:
        module.stop()


def test_composes_out_the_mount_pitch():
    # A level body reads its own mount tilt as the sensor's world orientation, so
    # composing the mount out returns identity.
    mount = Quaternion.from_euler(Vector3(0.0, 0.3, 0.0))
    out = _level(mount.to_tuple(), mount)
    assert out.orientation.angle_to(Quaternion(0.0, 0.0, 0.0, 1.0)) < 1e-5


def test_preserves_body_yaw_under_mount_tilt():
    # A body yawed by a known angle keeps that yaw after the mount is composed out.
    mount = Quaternion.from_euler(Vector3(0.0, 0.3, 0.0))
    body = Quaternion.from_euler(Vector3(0.0, 0.0, 0.7))
    out = _level(mount.to_tuple(), body * mount)
    assert out.orientation.angle_to(body) < 1e-5


def test_relabels_child_frame_and_a_zero_arm_passes_position_through():
    out = _level([0.0, 0.0, 0.0, 1.0], Quaternion(0.0, 0.0, 0.0, 1.0))
    assert out.child_frame_id == "base_link"
    assert out.position.to_tuple() == (1.0, 2.0, 3.0)


def test_the_lever_arm_moves_the_body_off_the_sensor():
    """LIO reports the SENSOR's position; the body is that less the mount arm.

    Unlevel this and the planner reasons about a footprint 0.30 m ahead of the
    robot, which is 70% of a half-length on the Go2 -- it still reaches goals,
    it just judges every clearance for a body that is not there.
    """
    # facing +x, so the arm comes off unrotated and the arithmetic is readable
    out = _level([0.0, 0.0, 0.0, 1.0], Quaternion(0.0, 0.0, 0.0, 1.0), _LEVER)
    assert out.position.to_tuple() == pytest.approx(
        (1.0 - _LEVER[0], 2.0 - _LEVER[1], 3.0 - _LEVER[2]), abs=1e-12
    )


def test_the_arm_swings_with_the_body_not_the_map():
    """Yawed a quarter turn, the forward arm comes off the body's y, not its x."""
    yaw90 = Quaternion.from_euler(Vector3(0.0, 0.0, math.pi / 2))
    out = _level([0.0, 0.0, 0.0, 1.0], yaw90, (1.0, 0.0, 0.0))
    assert out.position.to_tuple() == pytest.approx((1.0, 2.0 - 1.0, 3.0), abs=1e-12)


# The Go2's real mount, MID360_MOUNT_RPY_DEG = (-60, 0, -90), and a body rolled,
# pitched and yawed so no term of the quaternion product drops out. The rust port
# is pinned to these same numbers in odom_body_frame.rs; keeping the pair in sync
# is what makes its 1e-12 parity assertion mean anything.
_MOUNT = (-0.35355339059327373, 0.3535533905932737, -0.6123724356957945, 0.6123724356957946)
_SENSOR = (-0.5450515607112322, 0.13515397172907628, -0.39632409041884364, 0.7263466221066786)
_LEVELED = (-0.13432939990042636, 0.019615081741061635, 0.3470173839448213, 0.9279815710081614)
# base_link -> mid360_link on the Go2: CAMERA_XYZ + MID360_XYZ from the zenoh
# connection, and _BODY.rotate_vector(_LEVER), both pinned in the rust too.
_LEVER = (0.29515, -0.00003, 0.16297)
_BODY = (-0.1343293999004263, 0.01961508174106149, 0.3470173839448213, 0.9279815710081614)
_OFFSET = (0.21459713303331934, 0.23136344799234826, 0.11870876011049823)


def test_go2_mount_vectors_the_rust_port_is_pinned_to():
    assert Quaternion.from_euler(
        Vector3(*(math.radians(d) for d in (-60.0, 0.0, -90.0)))
    ).to_tuple() == pytest.approx(_MOUNT, abs=1e-12)

    out = _level(_MOUNT, Quaternion(*_SENSOR))
    assert out.orientation.to_tuple() == pytest.approx(_LEVELED, abs=1e-12)

    # the arm is the sum of the two static mount edges, and is rotated by the
    # LEVELLED attitude -- rotating it by the sensor's would tilt it twice
    assert Quaternion(*_BODY).rotate_vector(Vector3(*_LEVER)).to_tuple() == pytest.approx(
        _OFFSET, abs=1e-12
    )
