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

"""Tests for the recorded G1 SONIC real-hardware dry run."""

from __future__ import annotations

import numpy as np
import pytest

from dimos.control.tasks.g1_sonic_wbc_task.hardware_dry_run import (
    display_joint_positions,
    load_hardware_dry_run,
)


def test_recording_is_real_hardware_data_with_motor_commands_disabled() -> None:
    recording = load_hardware_dry_run()

    assert recording.source_samples == 4949
    assert len(recording.timestamps_s) == 358
    assert recording.replay_rate_hz == 20.0
    assert recording.source_git_rev == "f8966dddfe69f917e09edf23febe1ceae15a4322"
    assert recording.source_phase_samples == {
        "idle": 100,
        "observe": 38,
        "takeover": 4741,
        "walk": 70,
    }
    assert recording.source_duration_s == pytest.approx(21.443856781)
    assert recording.max_measured_joint_speed_rad_s == pytest.approx(0.101135604)
    assert recording.max_measured_tilt_deg == pytest.approx(0.396384009)
    assert recording.simulated is False
    assert recording.motor_commands_enabled is False


def test_replay_starts_from_measured_g1_pose_then_shows_sonic_targets() -> None:
    recording = load_hardware_dry_run()
    display_rad = display_joint_positions(recording)
    observe = recording.phase == "observe"

    np.testing.assert_allclose(display_rad[observe], recording.measured_position_rad[observe])
    np.testing.assert_allclose(display_rad[~observe], recording.target_position_rad[~observe])
    np.testing.assert_allclose(
        display_rad[0, :6],
        [-0.14378, 0.02274, 0.16964, 0.29699, -0.17502, 0.02436],
        atol=1e-5,
    )


@pytest.mark.mujoco
def test_mujoco_replay_is_kinematic_and_does_not_step_physics() -> None:
    import mujoco

    from dimos.robot.unitree.g1.blueprints.basic.demo_g1_sonic_real_hardware_dry_run import (
        _MODEL_PATH,
        _motor_qpos_addresses,
    )
    from dimos.simulation.engines.mujoco_engine import MujocoEngine
    from dimos.simulation.mujoco.model import get_assets

    recording = load_hardware_dry_run()
    engine = MujocoEngine(config_path=_MODEL_PATH, headless=True, assets=get_assets())
    addresses = _motor_qpos_addresses(engine.model)
    engine.data.qpos[addresses] = display_joint_positions(recording)[-1]
    mujoco.mj_forward(engine.model, engine.data)

    assert engine.data.time == 0.0
    assert len(addresses) == 29
