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

"""Align G1 hardware Point-LIO reference with MuJoCo pelvis ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from dimos.memory.cli.dataset import open_store
from dimos.robot.unitree.g1.characterization.recording import G1Recording, measured_pelvis_pose


@dataclass(frozen=True)
class G1SimulationRecording:
    """Actual GR00T MuJoCo output; poses are world_T_pelvis and units are SI."""

    command_t_s: NDArray[np.float64]
    command_body_twist: NDArray[np.float64]
    sim_t_s: NDArray[np.float64]
    sim_world_p_pelvis_m: NDArray[np.float64]
    sim_world_q_pelvis_xyzw: NDArray[np.float64]
    motor_t_s: NDArray[np.float64]
    motor_names: tuple[str, ...]
    motor_q_rad: NDArray[np.float64]


@dataclass(frozen=True)
class AlignedTrajectories:
    """Command-time tracks re-anchored into each pelvis's initial yaw frame."""

    t_s: NDArray[np.float64]
    command_body_twist: NDArray[np.float64]
    hardware_p_m: NDArray[np.float64]
    hardware_yaw_rad: NDArray[np.float64]
    sim_p_m: NDArray[np.float64]
    sim_yaw_rad: NDArray[np.float64]


@dataclass(frozen=True)
class ComparisonResult:
    """Numerical discrepancy; Point-LIO remains a measured reference."""

    duration_s: float
    n_samples: int
    source_command_events: int
    replay_command_events: int
    command_transitions: int
    command_level_sequence_exact: bool
    command_transition_timing_median_abs_error_s: float
    command_transition_timing_p95_abs_error_s: float
    command_transition_timing_max_abs_error_s: float
    command_rmse: tuple[float, float, float]
    command_max_abs_error: tuple[float, float, float]
    position_rmse_m: float
    position_p90_error_m: float
    final_position_error_m: float
    yaw_rmse_rad: float
    yaw_p90_error_rad: float
    final_yaw_error_rad: float
    simulation_min_pelvis_height_m: float
    simulation_max_abs_roll_rad: float
    simulation_max_abs_pitch_rad: float
    simulation_finite: bool
    command_replay_status: str
    warnings: tuple[str, ...]
    reference: str = "Point-LIO world_T_pelvis; measured reference, not ground truth"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matrix(rows: list[object], fields: tuple[str, ...]) -> NDArray[np.float64]:
    return np.asarray([[float(getattr(row.data, field)) for field in fields] for row in rows])


def read_simulation(path: str | Path) -> G1SimulationRecording:
    """Read actual GR00T replay outputs through the mem2 store seam."""
    store = open_store(Path(path))
    try:
        command_rows = list(store.stream("cmd_vel").order_by("ts"))
        pose_rows = list(store.stream("sim_odom").order_by("ts"))
        motor_rows = list(store.stream("motor_states").order_by("ts"))
        if not command_rows or not pose_rows or not motor_rows:
            raise ValueError("simulation needs non-empty cmd_vel, sim_odom, and motor_states")
        command = np.asarray(
            [[row.data.linear.x, row.data.linear.y, row.data.angular.z] for row in command_rows]
        )
        position = _matrix(pose_rows, ("x", "y", "z"))
        quaternion = np.asarray(
            [
                [
                    row.data.orientation.x,
                    row.data.orientation.y,
                    row.data.orientation.z,
                    row.data.orientation.w,
                ]
                for row in pose_rows
            ]
        )
        motor_names = tuple(motor_rows[0].data.name)
        motor_q = np.asarray([row.data.position for row in motor_rows])
    finally:
        store.stop()
    if any(tuple(row.data.name) != motor_names for row in motor_rows):
        raise ValueError("motor_states joint order changed during simulation")
    return G1SimulationRecording(
        command_t_s=np.asarray([row.ts for row in command_rows]),
        command_body_twist=command,
        sim_t_s=np.asarray([row.ts for row in pose_rows]),
        sim_world_p_pelvis_m=position,
        sim_world_q_pelvis_xyzw=quaternion,
        motor_t_s=np.asarray([row.ts for row in motor_rows]),
        motor_names=motor_names,
        motor_q_rad=motor_q,
    )


def sample_zoh(
    source_t_s: NDArray[np.float64],
    values: NDArray[np.float64],
    query_t_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    index = np.searchsorted(source_t_s, query_t_s, side="right") - 1
    return values[np.clip(index, 0, len(values) - 1)]


def _command_transitions(
    t_s: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    changed = np.r_[True, np.any(values[1:] != values[:-1], axis=1)]
    return t_s[changed] - t_s[0], values[changed]


def _command_replay_health(
    hardware: G1Recording, simulation: G1SimulationRecording
) -> tuple[bool, float, float, float]:
    source_t_s, source_levels = _command_transitions(
        hardware.command_t_s, hardware.command_body_twist
    )
    replay_t_s, replay_levels = _command_transitions(
        simulation.command_t_s, simulation.command_body_twist
    )
    sequence_exact = bool(np.array_equal(replay_levels, source_levels))
    if not sequence_exact:
        return sequence_exact, float("inf"), float("inf"), float("inf")
    timing_error_s = np.abs(replay_t_s - source_t_s)
    return (
        sequence_exact,
        float(np.median(timing_error_s)),
        float(np.quantile(timing_error_s, 0.95)),
        float(np.max(timing_error_s)),
    )


def interpolate_pose(
    source_t_s: NDArray[np.float64],
    position_m: NDArray[np.float64],
    quaternion_xyzw: NDArray[np.float64],
    query_t_s: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    position = np.column_stack(
        [np.interp(query_t_s, source_t_s, position_m[:, axis]) for axis in range(3)]
    )
    rotation = Slerp(source_t_s, Rotation.from_quat(quaternion_xyzw))(query_t_s)
    return position, rotation.as_quat()


def _relative_planar(
    position_m: NDArray[np.float64], quaternion_xyzw: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    yaw_rad = np.unwrap(Rotation.from_quat(quaternion_xyzw).as_euler("xyz")[:, 2])
    initial_yaw = yaw_rad[0]
    c, s = np.cos(initial_yaw), np.sin(initial_yaw)
    delta = position_m - position_m[0]
    local = delta.copy()
    local[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    local[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    return local, yaw_rad - initial_yaw


def align_trajectories(
    hardware: G1Recording,
    simulation: G1SimulationRecording,
    *,
    sample_rate_hz: float = 20.0,
) -> AlignedTrajectories:
    """Align on first replayed command and remove each world's initial SE(2) pose."""
    hardware_t0 = float(hardware.command_t_s[0])
    sim_t0 = float(simulation.command_t_s[0])
    start_s = max(
        0.0,
        float(hardware.pointlio_t_s[0] - hardware_t0),
        float(simulation.sim_t_s[0] - sim_t0),
    )
    end_s = min(
        float(hardware.command_t_s[-1] - hardware_t0),
        float(simulation.command_t_s[-1] - sim_t0),
        float(hardware.pointlio_t_s[-1] - hardware_t0),
        float(simulation.sim_t_s[-1] - sim_t0),
    )
    if end_s <= start_s:
        raise ValueError(
            f"hardware/simulation have no overlapping command time: {start_s=} {end_s=}"
        )
    t_s = np.arange(start_s, end_s, 1.0 / sample_rate_hz)
    hardware_t_s, hardware_p_m, hardware_q_xyzw, _ = measured_pelvis_pose(hardware)
    hardware_p, hardware_q = interpolate_pose(
        hardware_t_s - hardware_t0,
        hardware_p_m,
        hardware_q_xyzw,
        t_s,
    )
    sim_p, sim_q = interpolate_pose(
        simulation.sim_t_s - sim_t0,
        simulation.sim_world_p_pelvis_m,
        simulation.sim_world_q_pelvis_xyzw,
        t_s,
    )
    hardware_local_p, hardware_yaw = _relative_planar(hardware_p, hardware_q)
    sim_local_p, sim_yaw = _relative_planar(sim_p, sim_q)
    command = sample_zoh(
        hardware.command_t_s - hardware_t0,
        hardware.command_body_twist,
        t_s,
    )
    return AlignedTrajectories(
        t_s=t_s,
        command_body_twist=command,
        hardware_p_m=hardware_local_p,
        hardware_yaw_rad=hardware_yaw,
        sim_p_m=sim_local_p,
        sim_yaw_rad=sim_yaw,
    )


def compare_trajectories(
    hardware: G1Recording,
    simulation: G1SimulationRecording,
) -> tuple[AlignedTrajectories, ComparisonResult]:
    """Score root motion after confirming the two runs received the same twist."""
    aligned = align_trajectories(hardware, simulation)
    hardware_t0 = hardware.command_t_s[0]
    sim_t0 = simulation.command_t_s[0]
    hardware_command = sample_zoh(
        hardware.command_t_s - hardware_t0,
        hardware.command_body_twist,
        aligned.t_s,
    )
    sim_command = sample_zoh(
        simulation.command_t_s - sim_t0,
        simulation.command_body_twist,
        aligned.t_s,
    )
    command_error = sim_command - hardware_command
    command_sequence_exact, timing_median_s, timing_p95_s, timing_max_s = _command_replay_health(
        hardware, simulation
    )
    transition_t_s, _ = _command_transitions(hardware.command_t_s, hardware.command_body_twist)
    position_error = np.linalg.norm(aligned.sim_p_m[:, :2] - aligned.hardware_p_m[:, :2], axis=1)
    yaw_error = np.arctan2(
        np.sin(aligned.sim_yaw_rad - aligned.hardware_yaw_rad),
        np.cos(aligned.sim_yaw_rad - aligned.hardware_yaw_rad),
    )
    command_rmse = tuple(np.sqrt(np.mean(command_error**2, axis=0)).tolist())
    command_max = tuple(np.max(np.abs(command_error), axis=0).tolist())
    sim_euler_rad = Rotation.from_quat(simulation.sim_world_q_pelvis_xyzw).as_euler("xyz")
    simulation_finite = bool(
        np.all(np.isfinite(simulation.sim_world_p_pelvis_m))
        and np.all(np.isfinite(simulation.sim_world_q_pelvis_xyzw))
        and np.all(np.isfinite(simulation.motor_q_rad))
    )
    warnings = []
    if not command_sequence_exact:
        warnings.append("simulation did not record the source twist level sequence exactly")
    if timing_p95_s > 0.1:
        warnings.append("source/simulation command timing differs by more than 0.1 s at p95")
    if aligned.t_s[-1] - aligned.t_s[0] < 30.0:
        warnings.append("comparison overlap is shorter than 30 s")
    if not simulation_finite:
        warnings.append("simulation contains non-finite state")
    return aligned, ComparisonResult(
        duration_s=float(aligned.t_s[-1] - aligned.t_s[0]),
        n_samples=len(aligned.t_s),
        source_command_events=len(hardware.command_t_s),
        replay_command_events=len(simulation.command_t_s),
        command_transitions=len(transition_t_s),
        command_level_sequence_exact=command_sequence_exact,
        command_transition_timing_median_abs_error_s=timing_median_s,
        command_transition_timing_p95_abs_error_s=timing_p95_s,
        command_transition_timing_max_abs_error_s=timing_max_s,
        command_rmse=command_rmse,
        command_max_abs_error=command_max,
        position_rmse_m=float(np.sqrt(np.mean(position_error**2))),
        position_p90_error_m=float(np.quantile(position_error, 0.9)),
        final_position_error_m=float(position_error[-1]),
        yaw_rmse_rad=float(np.sqrt(np.mean(yaw_error**2))),
        yaw_p90_error_rad=float(np.quantile(np.abs(yaw_error), 0.9)),
        final_yaw_error_rad=float(abs(yaw_error[-1])),
        simulation_min_pelvis_height_m=float(np.min(simulation.sim_world_p_pelvis_m[:, 2])),
        simulation_max_abs_roll_rad=float(np.max(np.abs(sim_euler_rad[:, 0]))),
        simulation_max_abs_pitch_rad=float(np.max(np.abs(sim_euler_rad[:, 1]))),
        simulation_finite=simulation_finite,
        command_replay_status="pass" if not warnings else "warn",
        warnings=tuple(warnings),
    )
