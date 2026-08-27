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

"""G1 plant overrides touch only the measured leg and foot parameters."""

import mujoco
import numpy as np
import pytest

from dimos.robot.unitree.g1.characterization.mujoco_model import (
    G1MujocoPhysics,
    apply_g1_mujoco_physics,
    build_g1_mujoco_spec,
    g1_mujoco_binding,
)
from dimos.robot.unitree.g1.wholebody_connection import G1_JOINT_NAMES


def test_physics_override_changes_only_leg_drives_and_feet() -> None:
    model = build_g1_mujoco_spec().compile()
    names = tuple(G1_JOINT_NAMES)
    binding = g1_mujoco_binding(model, names)
    original_armature = model.dof_armature.copy()
    original_damping = model.dof_damping.copy()
    original_frictionloss = model.dof_frictionloss.copy()
    original_friction = model.geom_friction.copy()
    physics = G1MujocoPhysics(0.03, 0.02, 0.8, 0.7, 0.04)

    apply_g1_mujoco_physics(model, names, physics)

    np.testing.assert_allclose(model.dof_armature[binding.joint_qvel[:12]], 0.03)
    np.testing.assert_allclose(model.dof_damping[binding.joint_qvel[:12]], 0.02)
    np.testing.assert_allclose(model.dof_frictionloss[binding.joint_qvel[:12]], 0.8)
    np.testing.assert_array_equal(
        model.dof_armature[binding.joint_qvel[12:]],
        original_armature[binding.joint_qvel[12:]],
    )
    np.testing.assert_array_equal(
        model.dof_damping[binding.joint_qvel[12:]],
        original_damping[binding.joint_qvel[12:]],
    )
    np.testing.assert_array_equal(
        model.dof_frictionloss[binding.joint_qvel[12:]],
        original_frictionloss[binding.joint_qvel[12:]],
    )
    foot_geoms: list[int] = []
    for side in ("left", "right"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"/{side}_ankle_roll_link")
        foot_geoms.extend(np.flatnonzero(model.geom_bodyid == body_id).tolist())
    np.testing.assert_allclose(model.geom_friction[foot_geoms, 0], 0.7)
    other_geoms = np.setdiff1d(np.arange(model.ngeom), foot_geoms)
    np.testing.assert_array_equal(model.geom_friction[other_geoms], original_friction[other_geoms])


def test_physics_rejects_zero_contact_time() -> None:
    with pytest.raises(ValueError, match="contact time constant must be positive"):
        G1MujocoPhysics(0.03, 0.02, 0.8, 0.7, 0.0)
