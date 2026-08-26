# Copyright 2025-2026 Dimensional Inc.
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

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from dimos.control.components import make_humanoid_joints
from dimos.control.coordinator import ControlCoordinator
from dimos.control.task import CoordinatorState, JointStateSnapshot
from dimos.control.tasks.g1_sonic_wbc_task import g1_sonic_wbc_task
from dimos.control.tasks.g1_sonic_wbc_task.g1_sonic_wbc_task import (
    G1SonicWBCTask,
    G1SonicWBCTaskConfig,
)
from dimos.control.tasks.g1_sonic_wbc_task.sonic_pipeline import (
    DDS_TO_ONNX,
    DEFAULT_ANGLES_DDS,
    SonicPipeline,
    _Trajectory,
)
from dimos.hardware.whole_body.spec import IMUState


class _StubPipeline:
    def __init__(self, **_kwargs: Any) -> None:
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def set_velocity(self, _vx: float, _vy: float, _wz: float) -> None:
        pass

    def step(self, **_kwargs: Any) -> np.ndarray:
        self.step_calls += 1
        return DEFAULT_ANGLES_DDS + 0.01

    def snapshot(self) -> dict[str, Any]:
        return {}


def _state_at(t_now: float, positions: np.ndarray) -> CoordinatorState:
    joints = make_humanoid_joints("g1")
    return CoordinatorState(
        joints=JointStateSnapshot(
            joint_positions=dict(zip(joints, positions, strict=True)),
            joint_velocities={name: 0.0 for name in joints},
            joint_efforts={name: 0.0 for name in joints},
            timestamp=t_now,
        ),
        t_now=t_now,
        dt=0.01,
    )


def test_takeover_waits_for_arm_then_blends_live_policy(monkeypatch: Any) -> None:
    monkeypatch.setattr(g1_sonic_wbc_task, "SonicPipeline", _StubPipeline)
    measured = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
    adapter = MagicMock()
    adapter.read_imu.return_value = IMUState(
        quaternion=(1.0, 0.0, 0.0, 0.0),
        gyroscope=(0.0, 0.0, 0.0),
        accelerometer=(0.0, 0.0, -9.81),
        rpy=(0.0, 0.0, 0.0),
    )
    task = G1SonicWBCTask(
        "sonic_wbc",
        G1SonicWBCTaskConfig(
            encoder_onnx="encoder.onnx",
            decoder_onnx="decoder.onnx",
            planner_onnx="planner.onnx",
            joint_names=make_humanoid_joints("g1"),
            zmq_enabled=False,
        ),
        adapter,
    )
    task.start()

    hold = task.compute(_state_at(0.0, measured))
    assert hold is None
    assert task._pipeline.step_calls == 0

    assert task.arm(ramp_seconds=1.0)
    ramp_start = task.compute(_state_at(1.0, measured))
    ramp_mid = task.compute(_state_at(1.5, measured))
    ramp_end = task.compute(_state_at(2.0, measured))
    assert ramp_start is not None and ramp_mid is not None and ramp_end is not None
    np.testing.assert_allclose(ramp_start.positions, measured)
    policy_target = DEFAULT_ANGLES_DDS + 0.01
    np.testing.assert_allclose(ramp_mid.positions, 0.5 * (measured + policy_target), atol=1e-7)
    np.testing.assert_allclose(ramp_end.positions, policy_target, atol=1e-7)
    assert task._pipeline.step_calls == 3

    policy = task.compute(_state_at(2.02, DEFAULT_ANGLES_DDS))
    assert policy is not None
    np.testing.assert_allclose(policy.positions, DEFAULT_ANGLES_DDS + 0.01)
    assert task._pipeline.reset_calls == 2


def test_takeover_blend_starts_when_planner_target_is_ready(monkeypatch: Any) -> None:
    monkeypatch.setattr(g1_sonic_wbc_task, "SonicPipeline", _StubPipeline)
    measured = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
    adapter = MagicMock()
    adapter.read_imu.return_value = IMUState(quaternion=(1.0, 0.0, 0.0, 0.0))
    task = G1SonicWBCTask(
        "sonic_wbc",
        G1SonicWBCTaskConfig(
            encoder_onnx="encoder.onnx",
            decoder_onnx="decoder.onnx",
            planner_onnx="planner.onnx",
            joint_names=make_humanoid_joints("g1"),
            zmq_enabled=False,
        ),
        adapter,
    )
    task.start()
    assert task.arm(ramp_seconds=1.0)
    task._pipeline.step = MagicMock(
        side_effect=[None, DEFAULT_ANGLES_DDS + 0.01, DEFAULT_ANGLES_DDS + 0.01]
    )

    assert task.compute(_state_at(1.0, measured)) is None
    first_target = task.compute(_state_at(3.0, measured))
    mid_target = task.compute(_state_at(3.5, measured))

    assert first_target is not None and mid_target is not None
    np.testing.assert_allclose(first_target.positions, measured)
    np.testing.assert_allclose(
        mid_target.positions,
        0.5 * (measured + DEFAULT_ANGLES_DDS + 0.01),
        atol=1e-7,
    )


def test_dry_run_arm_never_emits_motor_targets(monkeypatch: Any) -> None:
    monkeypatch.setattr(g1_sonic_wbc_task, "SonicPipeline", _StubPipeline)
    adapter = MagicMock()
    adapter.read_imu.return_value = IMUState(
        quaternion=(1.0, 0.0, 0.0, 0.0),
        gyroscope=(0.0, 0.0, 0.0),
        accelerometer=(0.0, 0.0, -9.81),
    )
    task = G1SonicWBCTask(
        "sonic_wbc",
        G1SonicWBCTaskConfig(
            encoder_onnx="encoder.onnx",
            decoder_onnx="decoder.onnx",
            planner_onnx="planner.onnx",
            joint_names=make_humanoid_joints("g1"),
            zmq_enabled=False,
            auto_dry_run=True,
        ),
        adapter,
    )
    task.start()
    assert task.arm(ramp_seconds=1.0)

    assert task.compute(_state_at(0.0, DEFAULT_ANGLES_DDS)) is None
    assert task.compute(_state_at(1.0, DEFAULT_ANGLES_DDS)) is None
    assert task.compute(_state_at(1.02, DEFAULT_ANGLES_DDS)) is None
    assert task._pipeline.step_calls == 3


def test_leaving_armed_dry_run_restarts_live_blend(monkeypatch: Any) -> None:
    monkeypatch.setattr(g1_sonic_wbc_task, "SonicPipeline", _StubPipeline)
    adapter = MagicMock()
    adapter.read_imu.return_value = IMUState(quaternion=(1.0, 0.0, 0.0, 0.0))
    task = G1SonicWBCTask(
        "sonic_wbc",
        G1SonicWBCTaskConfig(
            encoder_onnx="encoder.onnx",
            decoder_onnx="decoder.onnx",
            planner_onnx="planner.onnx",
            joint_names=make_humanoid_joints("g1"),
            zmq_enabled=False,
            auto_dry_run=True,
            default_ramp_seconds=10.0,
        ),
        adapter,
    )
    task.start()
    assert task.arm(ramp_seconds=0.0)
    assert task.compute(_state_at(0.0, DEFAULT_ANGLES_DDS)) is None

    task.set_dry_run(False)
    first_live = task.compute(_state_at(1.0, DEFAULT_ANGLES_DDS))

    assert first_live is not None
    np.testing.assert_allclose(first_live.positions, DEFAULT_ANGLES_DDS)
    assert task.state_snapshot()["arming"] is True


def test_coordinator_estop_makes_sonic_inert_until_cleared(monkeypatch: Any) -> None:
    monkeypatch.setattr(g1_sonic_wbc_task, "SonicPipeline", _StubPipeline)
    adapter = MagicMock()
    adapter.read_imu.return_value = IMUState(quaternion=(1.0, 0.0, 0.0, 0.0))
    task = G1SonicWBCTask(
        "sonic_wbc",
        G1SonicWBCTaskConfig(
            encoder_onnx="encoder.onnx",
            decoder_onnx="decoder.onnx",
            planner_onnx="planner.onnx",
            joint_names=make_humanoid_joints("g1"),
            zmq_enabled=False,
        ),
        adapter,
    )
    task.start()
    coordinator = ControlCoordinator(publish_joint_state=False)
    try:
        assert coordinator.add_task(task)
        assert task.arm(ramp_seconds=0.0)
        assert task.compute(_state_at(0.0, DEFAULT_ANGLES_DDS)) is not None

        coordinator.set_estop(True)

        assert not task.is_active()
        assert task.compute(_state_at(0.02, DEFAULT_ANGLES_DDS)) is None
        assert not task.arm(ramp_seconds=0.0)
        coordinator.set_estop(False)
        assert task.is_active()
        assert task.arm(ramp_seconds=0.0)
    finally:
        coordinator.stop()


def test_planner_initial_context_and_seed_match_nvidia_reference() -> None:
    pipeline = SonicPipeline.__new__(SonicPipeline)
    pipeline._trajectory = None
    pipeline._cur_quat = np.array([0.5, 0.5, 0.5, 0.5])
    pipeline._cur_q_dds = np.arange(29, dtype=np.float32)

    context = pipeline._build_planner_context()
    inputs = pipeline._planner_inputs_dict(0, np.zeros(3), np.array([1.0, 0.0, 0.0]), -1.0, -1.0)

    np.testing.assert_array_equal(context[:, 3:7], np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)))
    np.testing.assert_array_equal(context[:, 7:36], np.tile(pipeline._cur_q_dds, (4, 1)))
    np.testing.assert_array_equal(inputs["random_seed"], np.array([1234]))


def test_auto_locomotion_modes_match_nvidia_speed_ranges() -> None:
    pipeline = SonicPipeline.__new__(SonicPipeline)

    assert pipeline._auto_mode(0.0) == 0
    assert pipeline._auto_mode(0.4) == 1
    assert pipeline._auto_mode(0.8) == 2
    assert pipeline._auto_mode(2.5) == 3


def test_planner_replan_context_interpolates_50hz_motion_at_30hz() -> None:
    pipeline = SonicPipeline.__new__(SonicPipeline)
    pipeline._trajectory = _Trajectory(12)
    pipeline._trajectory.num_frames = 12
    pipeline._traj_frame = 1
    for frame in range(12):
        pipeline._trajectory.root_pos[frame] = [frame, 2.0 * frame, 3.0 * frame]
        pipeline._trajectory.joint_pos[frame] = frame + np.arange(29, dtype=np.float32)

    context = pipeline._build_planner_context()

    sampled_frames = 3.0 + np.arange(4) * (50.0 / 30.0)
    np.testing.assert_allclose(context[:, 0], sampled_frames, atol=1e-6)
    expected_joints = sampled_frames[:, None] + np.arange(29, dtype=np.float32)[DDS_TO_ONNX]
    np.testing.assert_allclose(context[:, 7:36], expected_joints, atol=1e-6)


def test_policy_emits_nothing_until_planner_reference_is_ready() -> None:
    pipeline = SonicPipeline.__new__(SonicPipeline)
    pipeline._step_count = 0
    pipeline._nan_reported = 0
    pipeline._last_targets_dds = np.full(29, 9.0, dtype=np.float32)
    pipeline._check_planner_result = MagicMock()
    pipeline._trajectory = None
    pipeline._mode_queue = []
    pipeline._replan_timer = 0.0
    pipeline._vx = 0.0
    pipeline._vy = 0.0
    pipeline._mode_override = None
    pipeline._use_stream = False
    pipeline._needs_replan = True
    pipeline._submit_planner = MagicMock()

    targets = pipeline.step(
        q_dds=DEFAULT_ANGLES_DDS,
        dq_dds=np.zeros(29, dtype=np.float32),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        gyro_body=np.zeros(3, dtype=np.float32),
        gravity_body=np.array([0.0, 0.0, -1.0]),
    )

    pipeline._submit_planner.assert_called_once_with()
    assert targets is None
