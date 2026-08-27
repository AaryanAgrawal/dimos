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

"""The six-direction reducer recovers a known SIMULATED FOPDT response."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dimos.control.benchmarking.plant import (
    FopdtChannelParams,
    TwistBasePlantParams,
    TwistBasePlantSim,
)
from dimos.robot.unitree.g1.characterization.comparison import G1SimulationRecording
from dimos.robot.unitree.g1.characterization.recording import G1Recording
from dimos.robot.unitree.g1.characterization.response import (
    ResponseSpan,
    _transient_error,
    _VelocityTrack,
    characterize,
    directional_transient_errors,
)

_DT_S = 0.02
_PLANT = TwistBasePlantParams(
    vx=FopdtChannelParams(K=0.8, tau=0.35, L=0.10),
    vy=FopdtChannelParams(K=0.8, tau=0.35, L=0.10),
    wz=FopdtChannelParams(K=0.8, tau=0.35, L=0.10),
)


def _command_schedule() -> np.ndarray:
    commands: list[list[float]] = []
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        commands.extend([[0.0, 0.0, 0.0]] * 50)
        command = [0.0, 0.0, 0.0]
        command[axis] = 0.5 * sign
        commands.extend([command] * 200)
        commands.extend([[0.0, 0.0, 0.0]] * 100)
    return np.asarray(commands)


def _simulated_recording() -> G1Recording:
    command = _command_schedule()
    t_s = 1_700_000_000.0 + np.arange(len(command)) * _DT_S
    plant = TwistBasePlantSim(_PLANT)
    plant.reset(0.0, 0.0, 0.0, _DT_S)
    pose: list[list[float]] = []
    body_twist: list[list[float]] = []
    yaw: list[float] = []
    for vx, vy, wz in command:
        plant.step(float(vx), float(vy), float(wz), _DT_S)
        pose.append([plant.x, plant.y, 0.74])
        body_twist.append([plant.vx, plant.vy, plant.wz])
        yaw.append(plant.yaw)
    yaw_rad = np.unwrap(np.asarray(yaw))
    quaternion = Rotation.from_rotvec(
        np.column_stack((np.zeros_like(yaw_rad), np.zeros_like(yaw_rad), yaw_rad))
    ).as_quat()
    body = np.asarray(body_twist)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    world_twist = np.column_stack(
        (c * body[:, 0] - s * body[:, 1], s * body[:, 0] + c * body[:, 1], body[:, 2])
    )
    position = np.asarray(pose)
    return G1Recording(
        t_s,
        command,
        t_s,
        command,
        t_s,
        position,
        quaternion,
        position,
        quaternion,
        world_twist,
    )


def test_characterize_recovers_known_six_direction_response() -> None:
    result = characterize(_simulated_recording())

    assert result.health.status == "pass"
    assert [direction.n_good_fits for direction in result.directions] == [1] * 6
    for direction in result.directions:
        assert direction.motion_floor_command == pytest.approx(0.5)
        assert direction.K_median == pytest.approx(0.8, abs=0.08)
        assert direction.tau_median_s == pytest.approx(0.35, abs=0.12)
        assert direction.deadtime_median_s == pytest.approx(0.10, abs=0.08)
        assert direction.ceiling_observed is False


def test_transient_error_is_zero_for_the_same_trajectory() -> None:
    recording = _simulated_recording()
    simulation = G1SimulationRecording(
        command_t_s=recording.command_t_s,
        command_body_twist=recording.command_body_twist,
        sim_t_s=recording.pointlio_t_s,
        sim_world_p_pelvis_m=recording.world_p_pelvis_m,
        sim_world_q_pelvis_xyzw=recording.world_q_pelvis_xyzw,
        motor_t_s=recording.pointlio_t_s,
        motor_names=("joint",),
        motor_q_rad=np.zeros((len(recording.pointlio_t_s), 1)),
    )

    errors = directional_transient_errors(
        recording,
        simulation,
        levels_per_direction=1,
    )

    assert [error.nrmse for error in errors] == pytest.approx([0.0] * 6, abs=1e-12)


def test_transient_error_uses_each_actual_command_time() -> None:
    t_s = np.arange(0.0, 4.0, 0.01)
    reference_speed = np.maximum(t_s - 1.0, 0.0)
    predicted_speed = np.maximum(t_s - 1.05, 0.0)
    reference = _VelocityTrack(t_s, np.column_stack((reference_speed, t_s * 0.0, t_s * 0.0)))
    predicted = _VelocityTrack(t_s, np.column_stack((predicted_speed, t_s * 0.0, t_s * 0.0)))
    reference_span = ResponseSpan(0, 1, 0.5, 0.5, 1.0, 3.0)
    predicted_span = ResponseSpan(0, 1, 0.5, 0.5, 1.05, 3.05)

    error = _transient_error(
        "forward",
        "m/s",
        [(reference_span, predicted_span)],
        reference,
        predicted,
        1.5,
        0.1,
    )

    assert error.nrmse == pytest.approx(0.0, abs=1e-12)
