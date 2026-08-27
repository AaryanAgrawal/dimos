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

"""Plant replay plans are complete and deterministic before physics runs."""

from dataclasses import replace

import numpy as np

from dimos.control.tasks.g1_groot_wbc_task.g1_groot_wbc_task import G1_GROOT_KD, G1_GROOT_KP
from dimos.robot.unitree.g1.characterization.plant import (
    build_replay_plan,
    groot_command_contract,
    sample_replay_plans,
)
from dimos.robot.unitree.g1.characterization.recording import G1PlantRecording, G1Recording
from dimos.robot.unitree.g1.wholebody_connection import G1_JOINT_NAMES


def _recordings() -> tuple[G1PlantRecording, G1Recording]:
    t_s = 1_700_000_000.0 + np.arange(0.0, 40.0, 0.01)
    dof = 2
    zeros = np.zeros((len(t_s), dof))
    quaternion = np.tile([0.0, 0.0, 0.0, 1.0], (len(t_s), 1))
    plant = G1PlantRecording(
        motor_names=("a", "b"),
        motor_command_t_s=t_s,
        motor_command_q_rad=zeros,
        motor_command_dq_rad_s=zeros,
        motor_command_kp_nm_rad=np.ones_like(zeros),
        motor_command_kd_nm_s_rad=np.ones_like(zeros),
        motor_command_tau_ff_nm=zeros,
        motor_state_t_s=t_s,
        motor_state_q_rad=zeros,
        motor_state_dq_rad_s=zeros,
        motor_state_tau_est_nm=zeros,
        imu_t_s=t_s,
        imu_q_xyzw=quaternion,
        imu_gyro_rad_s=np.zeros((len(t_s), 3)),
        imu_accel_m_s2=np.zeros((len(t_s), 3)),
    )
    high_level = G1Recording(
        command_t_s=t_s,
        command_body_twist=np.zeros((len(t_s), 3)),
        teleop_t_s=t_s,
        teleop_body_twist=np.zeros((len(t_s), 3)),
        pointlio_t_s=t_s,
        world_p_mid360_m=np.tile([0.0, 0.0, 0.0], (len(t_s), 1)),
        world_q_mid360_xyzw=quaternion,
        world_p_pelvis_m=np.tile([0.0, 0.0, -0.4], (len(t_s), 1)),
        world_q_pelvis_xyzw=quaternion,
        pointlio_world_twist=np.zeros((len(t_s), 3)),
    )
    return plant, high_level


def test_plan_schedule_is_seeded_and_fixed() -> None:
    plant, high_level = _recordings()

    first = build_replay_plan(plant, high_level, duration_s=5.0, seed=7)
    repeated = build_replay_plan(plant, high_level, duration_s=5.0, seed=7)
    different = build_replay_plan(plant, high_level, duration_s=5.0, seed=8)

    np.testing.assert_array_equal(first.reinitialize, repeated.reinitialize)
    assert not np.array_equal(first.reinitialize, different.reinitialize)
    np.testing.assert_allclose(first.step_t_s + first.physics_dt_s, first.step_t_s + 0.005)
    assert first.command_q_rad.shape == first.reference_q_rad.shape == (1000, 2)


def test_sampled_windows_cover_the_recording_deterministically() -> None:
    plant, high_level = _recordings()

    plans = sample_replay_plans(plant, high_level, n_segments=4, segment_duration_s=2.0, seed=11)
    repeated = sample_replay_plans(plant, high_level, n_segments=4, segment_duration_s=2.0, seed=11)

    starts = [plan.step_t_s[0] - plant.motor_state_t_s[0] for plan in plans]
    repeated_starts = [plan.step_t_s[0] - plant.motor_state_t_s[0] for plan in repeated]
    np.testing.assert_allclose(starts, repeated_starts)
    assert 0.0 <= starts[0] < 9.5
    assert 28.5 <= starts[-1] <= 38.0


def test_groot_contract_checks_every_command_and_joint_order() -> None:
    plant, _ = _recordings()
    count = len(plant.motor_command_t_s)
    command_q = np.zeros((count, len(G1_JOINT_NAMES)))
    plant = replace(
        plant,
        motor_names=tuple(G1_JOINT_NAMES),
        motor_command_q_rad=command_q,
        motor_command_dq_rad_s=command_q,
        motor_command_kp_nm_rad=np.tile(G1_GROOT_KP, (count, 1)),
        motor_command_kd_nm_s_rad=np.tile(G1_GROOT_KD, (count, 1)),
    )

    contract = groot_command_contract(plant)

    assert contract.status == "pass"
    assert contract.matching_command_fraction == 1.0
    assert contract.joint_order_matches is True
