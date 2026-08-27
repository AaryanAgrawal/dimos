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

"""MuJoCo implementation of the fixed G1 plant replay plan."""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dimos.robot.unitree.g1.characterization.mujoco_model import (
    G1MujocoBinding,
    G1MujocoPhysics,
    apply_g1_mujoco_physics,
    build_g1_mujoco_spec,
    g1_mujoco_binding,
)
from dimos.robot.unitree.g1.characterization.plant import PlantPrediction, PlantReplayPlan


def _snap(
    data: mujoco.MjData,
    binding: G1MujocoBinding,
    plan: PlantReplayPlan,
    index: int,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    root_q = plan.root_world_q_xyzw[index]
    data.qpos[binding.root_qpos : binding.root_qpos + 3] = plan.root_world_p_m[index]
    data.qpos[binding.root_qpos + 3 : binding.root_qpos + 7] = root_q[[3, 0, 1, 2]]
    data.qpos[binding.joint_qpos] = plan.state_q_rad[index]
    data.qvel[binding.root_qvel : binding.root_qvel + 6] = plan.root_world_velocity[index]
    data.qvel[binding.joint_qvel] = plan.state_dq_rad_s[index]


class G1MujocoBackend:
    """One immutable production model reused across fixed plant replay plans."""

    def __init__(
        self,
        motor_names: tuple[str, ...],
        physics: G1MujocoPhysics | None = None,
    ) -> None:
        self._model = build_g1_mujoco_spec().compile()
        if physics is not None:
            apply_g1_mujoco_physics(self._model, motor_names, physics)
        self._binding = g1_mujoco_binding(self._model, motor_names)

    def model_sha256(self) -> str:
        """Hash the compiled MJB bytes, including the attached meshes."""
        buffer = np.empty(mujoco.mj_sizeModel(self._model), dtype=np.uint8)
        mujoco.mj_saveModel(self._model, buffer=buffer)
        return hashlib.sha256(buffer).hexdigest()

    def rollout(self, plan: PlantReplayPlan) -> PlantPrediction:
        """Execute one precomputed plan without changing the model."""
        model_dt_s = float(self._model.opt.timestep)
        if not np.isclose(model_dt_s, plan.physics_dt_s):
            raise ValueError(
                f"plan/model timestep mismatch: plan={plan.physics_dt_s}s model={model_dt_s}s"
            )
        if plan.command_q_rad.shape[1] != len(self._binding.joint_qpos):
            raise ValueError(
                f"plan/model DOF mismatch: plan={plan.command_q_rad.shape[1]} "
                f"model={len(self._binding.joint_qpos)}"
            )
        data = mujoco.MjData(self._model)
        n_steps, dof = plan.command_q_rad.shape
        q = np.empty((n_steps, dof))
        dq = np.empty_like(q)
        tau = np.empty_like(q)
        root_p = np.empty((n_steps, 3))
        root_q = np.empty((n_steps, 4))
        for index in range(n_steps):
            if plan.reinitialize[index]:
                _snap(data, self._binding, plan, index)
                mujoco.mj_forward(self._model, data)
            current_q = data.qpos[self._binding.joint_qpos]
            current_dq = data.qvel[self._binding.joint_qvel]
            control = (
                plan.command_kp_nm_rad[index] * (plan.command_q_rad[index] - current_q)
                + plan.command_kd_nm_s_rad[index] * (plan.command_dq_rad_s[index] - current_dq)
                + plan.command_tau_ff_nm[index]
            )
            data.ctrl[self._binding.actuators] = control
            mujoco.mj_step(self._model, data)
            q[index] = data.qpos[self._binding.joint_qpos]
            dq[index] = data.qvel[self._binding.joint_qvel]
            tau[index] = data.actuator_force[self._binding.actuators]
            root_p[index] = data.qpos[self._binding.root_qpos : self._binding.root_qpos + 3]
            w, x, y, z = data.qpos[self._binding.root_qpos + 3 : self._binding.root_qpos + 7]
            root_q[index] = (x, y, z, w)
        Rotation.from_quat(root_q)
        return PlantPrediction(q, dq, tau, root_p, root_q)
