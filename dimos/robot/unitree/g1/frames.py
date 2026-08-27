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

"""Measured G1 frame transforms shared by runtime visualization and evaluation."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Mid360Odometry(Protocol):
    x: float
    y: float
    z: float
    orientation: object


_MID360_MOUNT_UNROLL = np.diag([1.0, -1.0, -1.0])


def pelvis_T_mid360(
    waist_yaw_rad: float = 0.0,
    waist_roll_rad: float = 0.0,
    waist_pitch_rad: float = 0.0,
) -> NDArray[np.float64]:
    """Return the URDF-convention pelvis_T_mid360 at the measured waist pose."""
    from dimos.robot.unitree.g1.g1_tf_publisher import base_to_torso, torso_to_mid360

    pelvis_T_physical_mid360 = (
        base_to_torso(waist_yaw_rad, waist_roll_rad, waist_pitch_rad) + torso_to_mid360()
    ).to_matrix()
    pelvis_T_mid360 = pelvis_T_physical_mid360.copy()
    pelvis_T_mid360[:3, :3] = pelvis_T_physical_mid360[:3, :3] @ _MID360_MOUNT_UNROLL
    return np.asarray(pelvis_T_mid360, dtype=np.float64)


def world_T_mid360_from_odometry(odom: Mid360Odometry) -> NDArray[np.float64]:
    """Convert Point-LIO odometry into the URDF-convention world_T_mid360."""
    orientation = odom.orientation
    to_rotation_matrix = getattr(orientation, "to_rotation_matrix", None)
    if not callable(to_rotation_matrix):
        raise TypeError("mid360 odometry orientation needs to_rotation_matrix()")
    world_T_mid360 = np.eye(4, dtype=np.float64)
    world_T_mid360[:3, :3] = to_rotation_matrix() @ _MID360_MOUNT_UNROLL
    world_T_mid360[:3, 3] = (odom.x, odom.y, odom.z)
    return world_T_mid360


def world_T_pelvis_from_mid360_odometry(
    odom: Mid360Odometry,
    waist_yaw_rad: float = 0.0,
    waist_roll_rad: float = 0.0,
    waist_pitch_rad: float = 0.0,
) -> NDArray[np.float64]:
    """Convert Point-LIO world_T_mid360 odometry into world_T_pelvis."""
    return world_T_mid360_from_odometry(odom) @ np.linalg.inv(
        pelvis_T_mid360(waist_yaw_rad, waist_roll_rad, waist_pitch_rad)
    )


def pointlio_ground_z_m(nominal_pelvis_height_m: float) -> float:
    """Return ground z in the Point-LIO boot frame for a nominal stance."""
    return -(float(pelvis_T_mid360()[2, 3]) + nominal_pelvis_height_m)
