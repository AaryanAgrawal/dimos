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

"""The Point-LIO ghost is visual-only and cannot change simulated state."""

import mujoco
import numpy as np

from dimos.robot.unitree.g1.characterization.render import (
    _GHOST_BODY,
    _GHOST_GEOM,
    _add_reference_ghost,
    _camera_azimuth_deg,
    _camera_distance_m,
)


def test_reference_ghost_has_no_collision_or_physics_state() -> None:
    spec = mujoco.MjSpec()
    _add_reference_ghost(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _GHOST_GEOM)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _GHOST_BODY)
    qpos_before = data.qpos.copy()

    data.mocap_pos[int(model.body_mocapid[body_id])] = (1.0, 2.0, 3.0)
    mujoco.mj_forward(model, data)

    assert model.geom_contype[geom_id] == 0
    assert model.geom_conaffinity[geom_id] == 0
    np.testing.assert_array_equal(data.qpos, qpos_before)


def test_camera_distance_keeps_diverged_poses_in_frame() -> None:
    assert _camera_distance_m(np.zeros(3), np.array([8.0, 0.0, 0.0])) == 10.6
    assert _camera_azimuth_deg(np.zeros(3), np.array([8.0, 0.0, 0.0])) == 90.0
