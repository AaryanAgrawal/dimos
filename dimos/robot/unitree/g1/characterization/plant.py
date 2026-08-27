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

"""Ivan-style fixed replay plans and scores for the G1 low-level plant."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from dimos.robot.unitree.g1.characterization.comparison import (
    interpolate_pose,
    sample_zoh,
)
from dimos.robot.unitree.g1.characterization.recording import (
    G1PlantRecording,
    G1Recording,
    measured_pelvis_pose,
)
from dimos.robot.unitree.g1.characterization.response import (
    ResponseSpan,
    body_velocity,
    response_spans,
)
from dimos.robot.unitree.g1.frames import pointlio_ground_z_m

_NOMINAL_PELVIS_HEIGHT_M = 0.74  # GR00T height command; fixes Point-LIO boot z to floor z=0.
PLANT_CLIP_RANGE_S = (0.05, 0.8)  # Ivan's Go2 range; vary only after loop-2 evidence.


@dataclass(frozen=True)
class PlantHealth:
    """Low-level stream timing and shape checks."""

    motor_command_rate_hz: float
    motor_state_rate_hz: float
    imu_rate_hz: float
    motor_command_max_gap_s: float
    motor_state_max_gap_s: float
    imu_max_gap_s: float
    overlap_s: float
    dof: int
    status: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GrootCommandContract:
    """Evidence that recorded low-level commands match this GR00T task's gains."""

    kp_max_abs_error_nm_rad: float
    kd_max_abs_error_nm_s_rad: float
    dq_max_abs_rad_s: float
    matching_command_fraction: float
    joint_order_matches: bool
    status: str


@dataclass(frozen=True)
class PlantReplayPlan:
    """Complete command, reference, and snap schedule decided before physics."""

    seed: int
    physics_dt_s: float
    step_t_s: NDArray[np.float64]
    reinitialize: NDArray[np.bool_]
    command_q_rad: NDArray[np.float64]
    command_dq_rad_s: NDArray[np.float64]
    command_kp_nm_rad: NDArray[np.float64]
    command_kd_nm_s_rad: NDArray[np.float64]
    command_tau_ff_nm: NDArray[np.float64]
    state_q_rad: NDArray[np.float64]
    state_dq_rad_s: NDArray[np.float64]
    root_world_p_m: NDArray[np.float64]
    root_world_q_xyzw: NDArray[np.float64]
    root_world_velocity: NDArray[np.float64]
    reference_q_rad: NDArray[np.float64]
    reference_dq_rad_s: NDArray[np.float64]
    reference_tau_est_nm: NDArray[np.float64]
    reference_root_world_p_m: NDArray[np.float64]
    reference_root_world_q_xyzw: NDArray[np.float64]


@dataclass(frozen=True)
class PlantPrediction:
    """MuJoCo predictions at each plan step's end."""

    q_rad: NDArray[np.float64]
    dq_rad_s: NDArray[np.float64]
    tau_nm: NDArray[np.float64]
    root_world_p_m: NDArray[np.float64]
    root_world_q_xyzw: NDArray[np.float64]


@dataclass(frozen=True)
class DirectionalPlantReplay:
    """One labelled low-level replay window from a held twist level."""

    direction: str
    command: float
    unit: str
    split: str
    span_start_epoch_s: float
    plan: PlantReplayPlan


@dataclass(frozen=True)
class PlantScore:
    """Dimensioned residual summary against hardware measurements."""

    joint_q_rmse_rad: float
    joint_q_p90_abs_rad: float
    joint_dq_rmse_rad_s: float
    joint_dq_p90_abs_rad_s: float
    joint_tau_rmse_nm: float
    joint_tau_p90_abs_nm: float
    root_position_rmse_m: float
    root_position_p90_m: float
    root_rotation_rmse_rad: float
    root_rotation_p90_rad: float
    n_steps: int


def _rate_and_gap(t_s: NDArray[np.float64]) -> tuple[float, float]:
    positive = np.diff(t_s)
    positive = positive[positive > 0.0]
    if not len(positive):
        return 0.0, math.inf
    return 1.0 / float(np.median(positive)), float(np.max(positive))


def plant_health(recording: G1PlantRecording) -> PlantHealth:
    """Reject low-rate, gapped, or dimensionally inconsistent plant inputs."""
    command_rate, command_gap = _rate_and_gap(recording.motor_command_t_s)
    state_rate, state_gap = _rate_and_gap(recording.motor_state_t_s)
    imu_rate, imu_gap = _rate_and_gap(recording.imu_t_s)
    widths = {
        recording.motor_command_q_rad.shape[1],
        recording.motor_state_q_rad.shape[1],
        len(recording.motor_names),
    }
    warnings = []
    if len(widths) != 1:
        warnings.append(f"motor command/state/name widths disagree: {sorted(widths)}")
    if command_rate < 40.0 or command_gap > 0.1:
        warnings.append("motor_command timing is too sparse for 50 Hz policy replay")
    if state_rate < 100.0 or state_gap > 0.1:
        warnings.append("motor_states timing is too sparse for plant scoring")
    if imu_rate < 100.0 or imu_gap > 0.1:
        warnings.append("IMU timing is too sparse for plant initialization")
    starts = (
        recording.motor_command_t_s[0],
        recording.motor_state_t_s[0],
        recording.imu_t_s[0],
    )
    ends = (
        recording.motor_command_t_s[-1],
        recording.motor_state_t_s[-1],
        recording.imu_t_s[-1],
    )
    overlap_s = max(0.0, min(ends) - max(starts))
    if overlap_s < 30.0:
        warnings.append("low-level stream overlap is shorter than 30 s")
    return PlantHealth(
        motor_command_rate_hz=command_rate,
        motor_state_rate_hz=state_rate,
        imu_rate_hz=imu_rate,
        motor_command_max_gap_s=command_gap,
        motor_state_max_gap_s=state_gap,
        imu_max_gap_s=imu_gap,
        overlap_s=overlap_s,
        dof=recording.motor_state_q_rad.shape[1],
        status="pass" if not warnings else "warn",
        warnings=tuple(warnings),
    )


def groot_command_contract(recording: G1PlantRecording) -> GrootCommandContract:
    """Check recorded gains and desired velocity against the current GR00T contract."""
    from dimos.control.tasks.g1_groot_wbc_task.g1_groot_wbc_task import (
        G1_GROOT_KD,
        G1_GROOT_KP,
    )
    from dimos.robot.unitree.g1.wholebody_connection import G1_JOINT_NAMES

    kp = np.asarray(G1_GROOT_KP)
    kd = np.asarray(G1_GROOT_KD)
    joint_order_matches = recording.motor_names == tuple(G1_JOINT_NAMES)
    if recording.motor_command_q_rad.shape[1] != len(kp):
        return GrootCommandContract(math.inf, math.inf, math.inf, 0.0, joint_order_matches, "fail")
    kp_abs_error = np.abs(recording.motor_command_kp_nm_rad - kp)
    kd_abs_error = np.abs(recording.motor_command_kd_nm_s_rad - kd)
    dq_abs = np.abs(recording.motor_command_dq_rad_s)
    kp_error = float(np.max(np.abs(np.median(recording.motor_command_kp_nm_rad, axis=0) - kp)))
    kd_error = float(np.max(np.abs(np.median(recording.motor_command_kd_nm_s_rad, axis=0) - kd)))
    dq_max = float(np.max(np.abs(recording.motor_command_dq_rad_s)))
    matching = np.all((kp_abs_error < 1e-6) & (kd_abs_error < 1e-6) & (dq_abs < 1e-6), axis=1)
    matching_fraction = float(np.mean(matching))
    status = (
        "pass"
        if joint_order_matches
        and max(kp_error, kd_error, dq_max) < 1e-6
        and matching_fraction >= 0.95
        else "fail"
    )
    return GrootCommandContract(
        kp_error,
        kd_error,
        dq_max,
        matching_fraction,
        joint_order_matches,
        status,
    )


def _linear_sample(
    source_t_s: NDArray[np.float64],
    values: NDArray[np.float64],
    query_t_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.column_stack(
        [np.interp(query_t_s, source_t_s, values[:, axis]) for axis in range(values.shape[1])]
    )


def _reinitialization_mask(
    n_steps: int,
    dt_s: float,
    clip_range_s: tuple[float, float],
    seed: int,
) -> NDArray[np.bool_]:
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_steps, dtype=np.bool_)
    index = 0
    while index < n_steps:
        mask[index] = True
        clip_steps = max(1, round(float(rng.uniform(*clip_range_s)) / dt_s))
        index += clip_steps
    return mask


def _overlap_start(plant: G1PlantRecording, high_level: G1Recording) -> float:
    return max(
        plant.motor_command_t_s[0],
        plant.motor_state_t_s[0],
        plant.imu_t_s[0],
        high_level.pointlio_t_s[0],
    )


def _overlap_end(plant: G1PlantRecording, high_level: G1Recording) -> float:
    return min(
        plant.motor_command_t_s[-1],
        plant.motor_state_t_s[-1],
        plant.imu_t_s[-1],
        high_level.pointlio_t_s[-1],
    )


def _pelvis_world_linear_velocity(
    recording: G1Recording,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t_s, body_twist = body_velocity(recording)
    pose_t_s, position_m, quaternion_xyzw, _ = measured_pelvis_pose(recording)
    yaw_rad = np.interp(
        t_s,
        pose_t_s,
        np.unwrap(Rotation.from_quat(quaternion_xyzw).as_euler("xyz")[:, 2]),
    )
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    world_v_xy_m_s = np.column_stack(
        (
            c * body_twist[:, 0] - s * body_twist[:, 1],
            s * body_twist[:, 0] + c * body_twist[:, 1],
        )
    )
    pelvis_z_m = np.interp(t_s, pose_t_s, position_m[:, 2])
    return t_s, np.column_stack((world_v_xy_m_s, np.gradient(pelvis_z_m, t_s)))


def build_replay_plan(
    plant: G1PlantRecording,
    high_level: G1Recording,
    *,
    start_s: float = 0.0,
    duration_s: float = 30.0,
    physics_dt_s: float = 0.005,
    clip_range_s: tuple[float, float] = PLANT_CLIP_RANGE_S,
    seed: int = 0,
) -> PlantReplayPlan:
    """Freeze one deterministic open-loop schedule before loading MuJoCo."""
    if duration_s <= 0.0 or physics_dt_s <= 0.0:
        raise ValueError("duration_s and physics_dt_s must be positive")
    t0_s = _overlap_start(plant, high_level) + start_s
    t1_s = min(t0_s + duration_s, _overlap_end(plant, high_level))
    if t1_s - t0_s < physics_dt_s:
        raise ValueError(f"recording has no requested plant window: {t0_s=} {t1_s=}")
    step_t_s = np.arange(t0_s, t1_s - physics_dt_s * 0.5, physics_dt_s)
    reference_t_s = step_t_s + physics_dt_s
    state_q = _linear_sample(plant.motor_state_t_s, plant.motor_state_q_rad, step_t_s)
    state_dq = _linear_sample(plant.motor_state_t_s, plant.motor_state_dq_rad_s, step_t_s)
    pointlio_t_s, world_p_pelvis_m, world_q_pelvis_xyzw, _ = measured_pelvis_pose(high_level)
    root_p, root_q = interpolate_pose(
        pointlio_t_s,
        world_p_pelvis_m,
        world_q_pelvis_xyzw,
        step_t_s,
    )
    reference_p, reference_q = interpolate_pose(
        pointlio_t_s,
        world_p_pelvis_m,
        world_q_pelvis_xyzw,
        reference_t_s,
    )
    ground_z_m = pointlio_ground_z_m(_NOMINAL_PELVIS_HEIGHT_M)
    root_p[:, 2] -= ground_z_m
    reference_p[:, 2] -= ground_z_m
    velocity_t_s, pelvis_world_v_m_s = _pelvis_world_linear_velocity(high_level)
    linear_velocity = _linear_sample(velocity_t_s, pelvis_world_v_m_s, step_t_s)
    body_gyro = _linear_sample(plant.imu_t_s, plant.imu_gyro_rad_s, step_t_s)
    world_gyro = Rotation.from_quat(root_q).apply(body_gyro)
    return PlantReplayPlan(
        seed=seed,
        physics_dt_s=physics_dt_s,
        step_t_s=step_t_s,
        reinitialize=_reinitialization_mask(len(step_t_s), physics_dt_s, clip_range_s, seed),
        command_q_rad=sample_zoh(plant.motor_command_t_s, plant.motor_command_q_rad, step_t_s),
        command_dq_rad_s=sample_zoh(
            plant.motor_command_t_s, plant.motor_command_dq_rad_s, step_t_s
        ),
        command_kp_nm_rad=sample_zoh(
            plant.motor_command_t_s, plant.motor_command_kp_nm_rad, step_t_s
        ),
        command_kd_nm_s_rad=sample_zoh(
            plant.motor_command_t_s, plant.motor_command_kd_nm_s_rad, step_t_s
        ),
        command_tau_ff_nm=sample_zoh(
            plant.motor_command_t_s, plant.motor_command_tau_ff_nm, step_t_s
        ),
        state_q_rad=state_q,
        state_dq_rad_s=state_dq,
        root_world_p_m=root_p,
        root_world_q_xyzw=root_q,
        root_world_velocity=np.column_stack((linear_velocity, world_gyro)),
        reference_q_rad=_linear_sample(
            plant.motor_state_t_s, plant.motor_state_q_rad, reference_t_s
        ),
        reference_dq_rad_s=_linear_sample(
            plant.motor_state_t_s, plant.motor_state_dq_rad_s, reference_t_s
        ),
        reference_tau_est_nm=_linear_sample(
            plant.motor_state_t_s, plant.motor_state_tau_est_nm, reference_t_s
        ),
        reference_root_world_p_m=reference_p,
        reference_root_world_q_xyzw=reference_q,
    )


def sample_replay_plans(
    plant: G1PlantRecording,
    high_level: G1Recording,
    *,
    n_segments: int = 8,
    segment_duration_s: float = 8.0,
    seed: int = 0,
) -> tuple[PlantReplayPlan, ...]:
    """Choose seeded, stratified windows across the overlap before any rollout."""
    if n_segments < 1 or segment_duration_s <= 0.0:
        raise ValueError("n_segments and segment_duration_s must be positive")
    available_s = _overlap_end(plant, high_level) - _overlap_start(plant, high_level)
    latest_start_s = available_s - segment_duration_s
    if latest_start_s < 0.0:
        raise ValueError(
            f"recording overlap {available_s:.3f}s is shorter than segment {segment_duration_s:.3f}s"
        )
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, latest_start_s, n_segments + 1)
    starts = [float(rng.uniform(edges[index], edges[index + 1])) for index in range(n_segments)]
    return tuple(
        build_replay_plan(
            plant,
            high_level,
            start_s=start_s,
            duration_s=segment_duration_s,
            seed=seed + index,
        )
        for index, start_s in enumerate(starts)
    )


def _span_label(span: ResponseSpan) -> tuple[str, str]:
    directions = {
        (0, 1): ("forward", "m/s"),
        (0, -1): ("backward", "m/s"),
        (1, 1): ("left", "m/s"),
        (1, -1): ("right", "m/s"),
        (2, 1): ("ccw", "rad/s"),
        (2, -1): ("cw", "rad/s"),
    }
    return directions[(span.axis, span.sign)]


def _spans_by_direction(recording: G1Recording) -> dict[str, list[ResponseSpan]]:
    by_direction: dict[str, dict[float, ResponseSpan]] = {}
    for span in response_spans(recording):
        direction, _ = _span_label(span)
        level = round(abs(span.command), 6)
        previous = by_direction.setdefault(direction, {}).get(level)
        if previous is None or span.end_s - span.start_s > previous.end_s - previous.start_s:
            by_direction[direction][level] = span
    return {
        direction: sorted(spans.values(), key=lambda item: abs(item.command))
        for direction, spans in by_direction.items()
    }


def _directional_replay(
    plant: G1PlantRecording,
    high_level: G1Recording,
    span: ResponseSpan,
    *,
    level_split: str,
    response_window_s: float,
    pre_roll_s: float,
    seed: int,
) -> DirectionalPlantReplay:
    overlap_start_s = _overlap_start(plant, high_level)
    start_epoch_s = max(overlap_start_s, span.start_s - pre_roll_s)
    end_epoch_s = min(span.end_s, span.start_s + response_window_s)
    direction, unit = _span_label(span)
    plan = build_replay_plan(
        plant,
        high_level,
        start_s=start_epoch_s - overlap_start_s,
        duration_s=end_epoch_s - start_epoch_s,
        seed=seed,
    )
    return DirectionalPlantReplay(
        direction,
        abs(span.command),
        unit,
        level_split,
        span.start_s,
        plan,
    )


def directional_replay_plans(
    plant: G1PlantRecording,
    high_level: G1Recording,
    *,
    levels_per_direction: int = 8,
    response_window_s: float = 2.0,
    pre_roll_s: float = 0.25,
    seed: int = 0,
    split: str = "all",
) -> tuple[DirectionalPlantReplay, ...]:
    """Select equal, disjoint command levels for train and validation replay."""
    if levels_per_direction < 2 or response_window_s <= 0.0 or pre_roll_s < 0.0:
        raise ValueError("need at least two levels and non-negative replay durations")
    if split not in {"all", "train", "validation"}:
        raise ValueError(f"split must be all, train, or validation; got {split!r}")
    by_direction = _spans_by_direction(high_level)
    selected: list[DirectionalPlantReplay] = []
    for direction_index, direction in enumerate(
        ("forward", "backward", "left", "right", "ccw", "cw")
    ):
        spans = by_direction.get(direction, [])
        if len(spans) < levels_per_direction:
            raise ValueError(
                f"{direction} has {len(spans)} distinct command levels; need {levels_per_direction}"
            )
        for level_index, span in enumerate(spans[-levels_per_direction:]):
            level_split = "train" if level_index % 2 == 0 else "validation"
            if split != "all" and split != level_split:
                continue
            selected.append(
                _directional_replay(
                    plant,
                    high_level,
                    span,
                    level_split=level_split,
                    response_window_s=response_window_s,
                    pre_roll_s=pre_roll_s,
                    seed=seed + direction_index * levels_per_direction + level_index,
                )
            )
    return tuple(selected)


def score_prediction(plan: PlantReplayPlan, prediction: PlantPrediction) -> PlantScore:
    """Score only signals measured on both sides, in their physical units."""
    q_error = prediction.q_rad - plan.reference_q_rad
    dq_error = prediction.dq_rad_s - plan.reference_dq_rad_s
    tau_error = prediction.tau_nm - plan.reference_tau_est_nm
    position_error = np.linalg.norm(
        prediction.root_world_p_m - plan.reference_root_world_p_m, axis=1
    )
    rotation_error = (
        Rotation.from_quat(prediction.root_world_q_xyzw).inv()
        * Rotation.from_quat(plan.reference_root_world_q_xyzw)
    ).magnitude()
    return PlantScore(
        joint_q_rmse_rad=float(np.sqrt(np.mean(q_error**2))),
        joint_q_p90_abs_rad=float(np.quantile(np.abs(q_error), 0.9)),
        joint_dq_rmse_rad_s=float(np.sqrt(np.mean(dq_error**2))),
        joint_dq_p90_abs_rad_s=float(np.quantile(np.abs(dq_error), 0.9)),
        joint_tau_rmse_nm=float(np.sqrt(np.mean(tau_error**2))),
        joint_tau_p90_abs_nm=float(np.quantile(np.abs(tau_error), 0.9)),
        root_position_rmse_m=float(np.sqrt(np.mean(position_error**2))),
        root_position_p90_m=float(np.quantile(position_error, 0.9)),
        root_rotation_rmse_rad=float(np.sqrt(np.mean(rotation_error**2))),
        root_rotation_p90_rad=float(np.quantile(rotation_error, 0.9)),
        n_steps=len(plan.step_t_s),
    )
