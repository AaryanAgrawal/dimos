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

"""The G1 nav blueprint publishes a tf tree, and odometry grounds to a level base.

tf gives each frame one parent. G1TfPublisher's mount tree is rooted at
mid360_link so it never writes the frame PointLio owns. The mount chain itself
is covered by test_g1_tf_publisher; here we check the nav consumers' use of it:
GoalRelay-style odometry resolution and ground projection, and the chain's
agreement with the URDF it was copied from.
"""

import numpy as np
import pytest

from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.navigation.tf_pose import OdomBasePose, base_height_above_ground
from dimos.protocol.tf.tf import TF
from dimos.robot.unitree.g1.config import G1
from dimos.robot.unitree.g1.g1_tf_publisher import (
    MID360_HEIGHT,
    NOMINAL_BASE_HEIGHT,
    G1TfPublisher,
    base_to_torso,
    mount_transforms,
    torso_to_mid360,
)

# Rx(pi): the upside-down sensor mount.
_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def test_no_frame_has_two_tf_parents() -> None:
    # The blueprint imports the DDS driver, which needs the optional unitree sdk.
    pytest.importorskip("unitree_sdk2py")
    from dimos.robot.unitree.g1.blueprints.navigation.unitree_g1_nav_3d import unitree_g1_nav_3d

    children: dict[str, set[str]] = {}
    for atom in unitree_g1_nav_3d.blueprints:
        if atom.module is G1TfPublisher:
            children["G1TfPublisher"] = {t.child_frame_id for t in mount_transforms()}
        if atom.module is PointLio:
            sensor_frame = atom.kwargs.get("sensor_frame_id", "mid360_link")
            children["PointLio"] = {sensor_frame}
    assert set(children) == {"G1TfPublisher", "PointLio"}
    claimed: set[str] = set()
    for publisher, frames in children.items():
        clash = claimed & frames
        assert not clash, f"{publisher} also writes {sorted(clash)}"
        claimed |= frames


def test_chain_matches_urdf() -> None:
    """The hardcoded mount chain composes to the URDF rest pose, plus the flip."""
    yourdfpy = pytest.importorskip("yourdfpy")
    urdf = yourdfpy.URDF.load(str(G1.model_path), load_meshes=False)
    urdf.update_cfg(np.zeros(len(urdf.actuated_joint_names)))
    expected = urdf.get_transform("mid360_link", "pelvis") @ _FLIP
    chain = base_to_torso(0.0, 0.0, 0.0) + torso_to_mid360()
    np.testing.assert_allclose(chain.to_matrix(), expected, atol=1e-9)
    mount_z = urdf.get_transform("mid360_link", "pelvis")[2, 3]
    assert MID360_HEIGHT == pytest.approx(NOMINAL_BASE_HEIGHT + mount_z, abs=1e-6)


def test_odometry_resolves_to_level_grounded_base() -> None:
    """The GoalRelay path: sensor odometry -> base pose -> ground projection.

    Feed the attitude Point-LIO reports when the base is level. The resolved
    base must be upright despite the upside-down mount, directly below the
    sensor by the mount height, and ground-project to z = -MID360_HEIGHT.
    """
    buffer = TF()
    for transform in mount_transforms():
        transform.ts = 1.0
        buffer.receive_transform(transform)

    level_attitude = torso_to_mid360().rotation
    odom = Odometry(
        ts=1.0,
        frame_id="odom",
        child_frame_id="mid360_link",
        pose=Pose(0.0, 0.0, 0.0, *level_attitude),
    )

    base_pose = OdomBasePose(buffer, "base_link")
    start = base_pose.resolve(odom)
    assert start is not None

    np.testing.assert_allclose(start.orientation.to_rotation_matrix(), np.eye(3), atol=1e-9)
    assert start.position.z == pytest.approx(NOMINAL_BASE_HEIGHT - MID360_HEIGHT)
    assert np.hypot(start.position.x, start.position.y) < 0.005

    leg = base_pose.sensor_to_base("mid360_link")
    assert leg is not None
    base_height = base_height_above_ground(MID360_HEIGHT, -leg)
    assert base_height == pytest.approx(NOMINAL_BASE_HEIGHT)
    assert start.position.z - base_height == pytest.approx(-MID360_HEIGHT)
