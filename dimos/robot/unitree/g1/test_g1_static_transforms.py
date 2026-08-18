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

"""The published mount tree composes back to the g1.urdf rest-pose geometry.

mount_transforms() inverts two of the FRAMES edges to root the tree at
mid360_link, so the geometry a consumer reads back is not the geometry written
in FRAMES. These pin the composed result, which is what nav actually uses.
"""

import math

from dimos.protocol.tf.tf import MultiTBuffer
from dimos.robot.unitree.g1.g1_static_transforms import mount_transforms

# base_link -> mid360_link, summed down the FRAMES chain at rest pose.
MOUNT_X = -0.0039635 + 0.0002835
MOUNT_Z = 0.044 + 0.41618
MID360_PITCH = 0.04014257279586953
D435_PITCH = 0.8307767239493009


def _buffer() -> MultiTBuffer:
    buffer = MultiTBuffer()
    buffer.receive_transform(*mount_transforms())
    return buffer


def test_mount_height_survives_the_inversion() -> None:
    """The lidar sits MOUNT_Z above base_link, the offset every ground projection uses."""
    leg = _buffer().get("mid360_link", "base_link")
    assert leg is not None
    base_to_sensor = -leg
    assert abs(base_to_sensor.translation.z - MOUNT_Z) < 1e-6
    assert abs(base_to_sensor.translation.x - MOUNT_X) < 1e-6


def test_mount_pitch_survives_the_inversion() -> None:
    """A sign flip here tilts every deprojected cloud rather than failing loudly."""
    leg = _buffer().get("mid360_link", "base_link")
    assert leg is not None
    pitch = (-leg).rotation.euler.y
    assert abs(pitch - MID360_PITCH) < 1e-6


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
