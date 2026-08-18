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

"""The published mount tree composes back to the g1.urdf geometry.

mount_transforms() inverts two edges to root the tree at mid360_link, so the
geometry a consumer reads back is not the geometry written in FRAMES. These
pin the composed result at rest and under waist articulation, which is what
nav actually uses.
"""

import math

from dimos.protocol.tf.tf import MultiTBuffer
from dimos.robot.unitree.g1.g1_tf_publisher import (
    D435_PITCH,
    MID360_PITCH,
    base_to_torso,
    mount_transforms,
)

# base_link -> mid360_link, summed down the rest-pose chain.
MOUNT_X = -0.0039635 + 0.0002835
MOUNT_Z = 0.044 + 0.41618


def _buffer(
    waist_yaw: float = 0.0, waist_roll: float = 0.0, waist_pitch: float = 0.0
) -> MultiTBuffer:
    buffer = MultiTBuffer()
    buffer.receive_transform(*mount_transforms(waist_yaw, waist_roll, waist_pitch))
    return buffer


def test_rest_pose_matches_urdf() -> None:
    """The lidar sits MOUNT_Z above base_link, the offset every ground projection uses."""
    leg = _buffer().get("mid360_link", "base_link")
    assert leg is not None
    base_to_sensor = -leg
    assert abs(base_to_sensor.translation.z - MOUNT_Z) < 1e-6
    assert abs(base_to_sensor.translation.x - MOUNT_X) < 1e-6
    assert abs((-leg).rotation.euler.y - MID360_PITCH) < 1e-6


def test_rest_pose_base_to_torso_matches_urdf_offsets() -> None:
    rest = base_to_torso(0.0, 0.0, 0.0)
    assert abs(rest.translation.x - (-0.0039635)) < 1e-6
    assert abs(rest.translation.z - 0.044) < 1e-6
    assert abs(rest.rotation.euler.x) < 1e-6
    assert abs(rest.rotation.euler.y) < 1e-6
    assert abs(rest.rotation.euler.z) < 1e-6


def test_waist_yaw_rotates_base_link_against_the_torso() -> None:
    """A twisted waist must show up as opposite yaw on base_link, not be baked away."""
    yaw = math.pi / 4
    leg = _buffer(waist_yaw=yaw).get("torso_link", "base_link")
    assert leg is not None
    assert abs(leg.rotation.euler.z - (-yaw)) < 1e-6


def test_waist_pitch_composes_with_the_mid360_mount_pitch() -> None:
    pitch = 0.3
    leg = _buffer(waist_pitch=pitch).get("mid360_link", "base_link")
    assert leg is not None
    assert abs((-leg).rotation.euler.y - (MID360_PITCH + pitch)) < 1e-6


def test_waist_yaw_leaves_mount_height_alone() -> None:
    """The waist yaw axis is vertical, so twisting must not move the lidar height."""
    leg = _buffer(waist_yaw=1.0).get("mid360_link", "base_link")
    assert leg is not None
    assert abs((-leg).translation.z - MOUNT_Z) < 1e-6


def test_d435_hangs_off_base_link() -> None:
    """The tree is rooted at mid360_link, so the camera edge is reachable by composition."""
    camera = _buffer().get("base_link", "d435_link")
    assert camera is not None
    assert abs(camera.translation.x - (-0.0039635 + 0.0576235)) < 1e-6
    assert abs(camera.translation.z - (0.044 + 0.42987)) < 1e-6
    assert abs(camera.rotation.euler.y - D435_PITCH) < 1e-6


def test_pelvis_height_matches_config_note() -> None:
    """mid360 1.2m above ground implies the 0.74m nominal standing pelvis height."""
    leg = _buffer().get("mid360_link", "base_link")
    assert leg is not None
    assert math.isclose(1.2 - (-leg).translation.z, 0.74, abs_tol=0.005)
