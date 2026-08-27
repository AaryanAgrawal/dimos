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

"""Rigid world-frame differences must disappear before trajectory scoring."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dimos.robot.unitree.g1.characterization.comparison import (
    G1SimulationRecording,
    compare_trajectories,
)
from dimos.robot.unitree.g1.characterization.recording import G1Recording


def test_comparison_reanchors_world_frames_and_preserves_error_units() -> None:
    t_s = 1_700_000_000.0 + np.arange(0.0, 40.0, 0.05)
    elapsed_s = t_s - t_s[0]
    command = np.column_stack((np.full(len(t_s), 0.2), np.zeros((len(t_s), 2))))
    yaw = 0.02 * elapsed_s
    local_p = np.column_stack((0.2 * elapsed_s, np.zeros(len(t_s)), np.zeros(len(t_s))))
    hardware_yaw0 = 1.1
    hardware_rotation = Rotation.from_euler("z", hardware_yaw0)
    hardware_p = hardware_rotation.apply(local_p) + np.array([4.0, -3.0, 0.74])
    hardware_q = Rotation.from_rotvec(
        np.column_stack((np.zeros((len(t_s), 2)), yaw + hardware_yaw0))
    ).as_quat()
    sim_p = local_p + np.array([-2.0, 5.0, 0.74])
    sim_p[:, 1] += 0.1
    sim_q = Rotation.from_rotvec(np.column_stack((np.zeros((len(t_s), 2)), yaw))).as_quat()
    hardware = G1Recording(
        t_s,
        command,
        t_s,
        command,
        t_s,
        hardware_p,
        hardware_q,
        hardware_p,
        hardware_q,
        command,
    )
    simulation = G1SimulationRecording(
        t_s,
        command,
        t_s,
        sim_p,
        sim_q,
        t_s,
        ("joint",),
        np.zeros((len(t_s), 1)),
    )

    _, result = compare_trajectories(hardware, simulation)

    assert result.command_replay_status == "pass"
    assert result.command_level_sequence_exact is True
    assert result.command_transition_timing_max_abs_error_s == 0.0
    assert result.command_max_abs_error == (0.0, 0.0, 0.0)
    assert result.position_rmse_m == pytest.approx(0.0, abs=1e-10)
    assert result.yaw_rmse_rad == pytest.approx(0.0, abs=1e-10)
