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

"""Faster-than-real-time evaluation through the production GR00T task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.control.task import CoordinatorState, JointStateSnapshot
from dimos.control.tasks.g1_groot_wbc_task.g1_groot_wbc_task import (
    G1_GROOT_KD,
    G1_GROOT_KP,
    G1GrootWBCTask,
    G1GrootWBCTaskConfig,
    g1_joints,
    g1_legs_waist,
)
from dimos.hardware.whole_body.spec import IMUState
from dimos.robot.unitree.g1.characterization.comparison import (
    G1SimulationRecording,
    sample_zoh,
)
from dimos.robot.unitree.g1.characterization.mujoco_model import (
    G1MujocoPhysics,
    apply_g1_mujoco_physics,
    build_g1_mujoco_spec,
    g1_mujoco_binding,
)
from dimos.robot.unitree.g1.wholebody_connection import G1_JOINT_NAMES
from dimos.utils.data import get_data

_POLICY_DT_S = 0.02  # GR00T training and production simulation inference rate is 50 Hz.


@dataclass(frozen=True)
class _Trace:
    t_s: NDArray[np.float64]
    root_p_m: NDArray[np.float64]
    root_q_xyzw: NDArray[np.float64]
    motor_q_rad: NDArray[np.float64]


def _empty_trace(ticks: int, dof: int) -> _Trace:
    return _Trace(
        np.empty(ticks),
        np.empty((ticks, 3)),
        np.empty((ticks, 4)),
        np.empty((ticks, dof)),
    )


class _UnusedAdapter:
    def read_imu(self) -> IMUState:
        return IMUState()


def _sensor_slice(model: mujoco.MjModel, name: str, width: int) -> slice:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"/{name}")
    if sensor_id < 0 or int(model.sensor_dim[sensor_id]) != width:
        raise KeyError(f"MuJoCo sensor {name!r} with width {width} is required")
    start = int(model.sensor_adr[sensor_id])
    return slice(start, start + width)


def _groot_task() -> G1GrootWBCTask:
    model_dir = Path(get_data("groot"))
    config = G1GrootWBCTaskConfig(
        balance_onnx=model_dir / "balance.onnx",
        walk_onnx=model_dir / "walk.onnx",
        joint_names=list(g1_legs_waist),
        all_joint_names=list(g1_joints),
        auto_arm=True,
        decimation=1,
        default_ramp_seconds=0.0,
    )
    task = G1GrootWBCTask("g1_groot_fit", config, _UnusedAdapter())  # type: ignore[arg-type]
    task.start()
    return task


class GrootClosedLoopRunner:
    """Run the unchanged task and ONNX policy directly against one MuJoCo model."""

    def __init__(self, physics: G1MujocoPhysics | None = None) -> None:
        self._names = tuple(G1_JOINT_NAMES)
        self._model = build_g1_mujoco_spec().compile()
        if physics is not None:
            apply_g1_mujoco_physics(self._model, self._names, physics)
        self._binding = g1_mujoco_binding(self._model, self._names)
        self._gyro = _sensor_slice(self._model, "imu-angular-velocity", 3)
        physics_steps = round(_POLICY_DT_S / float(self._model.opt.timestep))
        if not np.isclose(physics_steps * self._model.opt.timestep, _POLICY_DT_S):
            raise ValueError("MuJoCo timestep must divide the 50 Hz GR00T policy period")
        self._physics_steps = physics_steps
        self._kp = np.asarray(G1_GROOT_KP)
        self._kd = np.asarray(G1_GROOT_KD)

    def _state(self, data: mujoco.MjData, t_s: float) -> CoordinatorState:
        binding = self._binding
        q = data.qpos[binding.joint_qpos]
        dq = data.qvel[binding.joint_qvel]
        root_q_wxyz = tuple(data.qpos[binding.root_qpos + 3 : binding.root_qpos + 7])
        imu = IMUState(root_q_wxyz, tuple(data.sensordata[self._gyro]), (0.0, 0.0, 0.0))
        return CoordinatorState(
            joints=JointStateSnapshot(
                dict(zip(self._names, q, strict=True)),
                dict(zip(self._names, dq, strict=True)),
            ),
            imu={"g1": imu},
            t_now=t_s,
            dt=_POLICY_DT_S,
        )

    def _physics_tick(
        self,
        data: mujoco.MjData,
        target_q_rad: NDArray[np.float64],
    ) -> None:
        binding = self._binding
        for _ in range(self._physics_steps):
            q = data.qpos[binding.joint_qpos]
            dq = data.qvel[binding.joint_qvel]
            data.ctrl[binding.actuators] = self._kp * (target_q_rad - q) - self._kd * dq
            mujoco.mj_step(self._model, data)

    @staticmethod
    def _validate_commands(
        command_t_s: NDArray[np.float64],
        command_body_twist: NDArray[np.float64],
        lead_in_s: float,
        tail_s: float,
    ) -> None:
        if lead_in_s < 0.0 or tail_s < 0.0:
            raise ValueError("lead_in_s and tail_s must be non-negative")
        if len(command_t_s) < 2 or command_body_twist.shape != (len(command_t_s), 3):
            raise ValueError("commands need at least two timestamps and shape (N, 3)")
        if np.any(np.diff(command_t_s) < 0.0):
            raise ValueError("command timestamps must be sorted")

    def _control_tick(
        self,
        task: G1GrootWBCTask,
        data: mujoco.MjData,
        target_q_rad: NDArray[np.float64],
        command: NDArray[np.float64],
        relative_s: float,
    ) -> None:
        task.set_velocity_command(*command.tolist(), t_now=relative_s)
        output = task.compute(self._state(data, relative_s))
        if output is not None and output.positions is not None:
            target_q_rad[:15] = output.positions
        self._physics_tick(data, target_q_rad)

    def _record_tick(
        self,
        trace: _Trace,
        tick: int,
        epoch_s: float,
        data: mujoco.MjData,
    ) -> None:
        trace.t_s[tick] = epoch_s + _POLICY_DT_S
        trace.root_p_m[tick] = data.qpos[self._binding.root_qpos : self._binding.root_qpos + 3]
        root_q_wxyz = data.qpos[self._binding.root_qpos + 3 : self._binding.root_qpos + 7]
        trace.root_q_xyzw[tick] = root_q_wxyz[[1, 2, 3, 0]]
        trace.motor_q_rad[tick] = data.qpos[self._binding.joint_qpos]

    def _rollout(
        self,
        command_t_s: NDArray[np.float64],
        command_body_twist: NDArray[np.float64],
        lead_in_s: float,
        tail_s: float,
    ) -> _Trace:
        data = mujoco.MjData(self._model)
        task = _groot_task()
        target_q_rad = np.zeros(len(self._names))
        start_s = float(command_t_s[0])
        duration_s = float(command_t_s[-1] - start_s + lead_in_s + tail_s)
        ticks = int(np.ceil(duration_s / _POLICY_DT_S))
        trace = _empty_trace(ticks, len(self._names))
        for tick in range(ticks):
            relative_s = tick * _POLICY_DT_S - lead_in_s
            epoch_s = start_s + relative_s
            command = (
                np.zeros(3)
                if relative_s < 0.0
                else sample_zoh(command_t_s, command_body_twist, np.asarray([epoch_s]))[0]
            )
            self._control_tick(task, data, target_q_rad, command, relative_s)
            self._record_tick(trace, tick, epoch_s, data)
        return trace

    def run(
        self,
        command_t_s: NDArray[np.float64],
        command_body_twist: NDArray[np.float64],
        *,
        lead_in_s: float = 5.0,
        tail_s: float = 1.0,
    ) -> G1SimulationRecording:
        """Replay one timestamped twist signal without wall-clock sleeps."""
        self._validate_commands(command_t_s, command_body_twist, lead_in_s, tail_s)
        trace = self._rollout(command_t_s, command_body_twist, lead_in_s, tail_s)
        return G1SimulationRecording(
            command_t_s=command_t_s,
            command_body_twist=command_body_twist,
            sim_t_s=trace.t_s,
            sim_world_p_pelvis_m=trace.root_p_m,
            sim_world_q_pelvis_xyzw=trace.root_q_xyzw,
            motor_t_s=trace.t_s,
            motor_names=self._names,
            motor_q_rad=trace.motor_q_rad,
        )
