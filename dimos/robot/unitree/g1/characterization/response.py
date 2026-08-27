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

"""Fit the six directional command-to-pelvis responses and their envelope."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from dimos.robot.unitree.g1.characterization.recording import (
    G1Recording,
    filter_pose_outliers,
    measured_pelvis_pose,
    pointlio_pose_sample_mask,
)

if TYPE_CHECKING:
    from dimos.robot.unitree.g1.characterization.comparison import G1SimulationRecording

_AXES = ("vx", "vy", "wz")
_DIRECTIONS = {
    (0, 1): "forward",
    (0, -1): "backward",
    (1, 1): "left",
    (1, -1): "right",
    (2, 1): "ccw",
    (2, -1): "cw",
}
_UNITS = {0: "m/s", 1: "m/s", 2: "rad/s"}
_VELOCITY_SMOOTHING_S = 0.4  # Five fresh samples at the measured 10 Hz LIO update rate.


@dataclass(frozen=True)
class ResponseSpan:
    """One constant, single-axis command interval."""

    axis: int
    sign: int
    command: float
    command_step: float
    start_s: float
    end_s: float


@dataclass(frozen=True)
class StepFit:
    """One measured directional response; all fields use SI units."""

    direction: str
    unit: str
    command: float
    command_step: float
    duration_s: float
    settled_speed: float
    K: float
    tau_s: float
    deadtime_s: float
    rmse: float
    r_squared: float
    response_vx_rms_m_s: float
    response_vy_rms_m_s: float
    response_wz_rms_rad_s: float
    stationary_noise: float
    movement_snr: float
    n_samples: int
    moved: bool
    converged: bool


@dataclass(frozen=True)
class DirectionResult:
    """Median identified region and observed directional envelope."""

    direction: str
    unit: str
    n_steps: int
    n_good_fits: int
    motion_floor_command: float | None
    motion_floor_achieved: float | None
    max_command_tested: float | None
    max_achieved: float | None
    K_median: float | None
    K_p10_p90: tuple[float, float] | None
    tau_median_s: float | None
    tau_p10_p90_s: tuple[float, float] | None
    deadtime_median_s: float | None
    deadtime_p10_p90_s: tuple[float, float] | None
    ceiling_observed: bool


@dataclass(frozen=True)
class DirectionTransientError:
    """Baseline-subtracted transient discrepancy in one command direction."""

    direction: str
    unit: str
    n_levels: int
    n_samples: int
    rmse: float
    nrmse: float
    reference_peak: float
    predicted_peak: float


@dataclass(frozen=True)
class _VelocityTrack:
    t_s: NDArray[np.float64]
    body_twist: NDArray[np.float64]


@dataclass(frozen=True)
class RecordingHealth:
    """Measured trust signals for command/Point-LIO timing and pose quality."""

    command_rate_hz: float
    pointlio_publish_rate_hz: float
    pointlio_pose_update_rate_hz: float
    command_max_gap_s: float
    pointlio_publish_max_gap_s: float
    pointlio_pose_update_max_gap_s: float
    command_pointlio_median_skew_s: float
    command_pointlio_p95_skew_s: float
    teleop_command_matching_fraction: float
    teleop_command_median_skew_s: float
    teleop_command_p95_skew_s: float
    overlap_s: float
    mid360_pose_outlier_fraction: float
    mid360_max_position_step_m: float
    mid360_max_rotation_step_rad: float
    mid360_twist_correlation: tuple[float, float, float]
    mid360_twist_rmse: tuple[float, float, float]
    status: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CharacterizationResult:
    """Complete high-level artifact before planner-specific conservative reduction."""

    health: RecordingHealth
    directions: tuple[DirectionResult, ...]
    steps: tuple[StepFit, ...]
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deduplicate_time(
    t_s: NDArray[np.float64], *values: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    keep = np.r_[True, np.diff(t_s) > 1e-4]
    return (t_s[keep], *(value[keep] for value in values))


def _body_velocity_from_pose(
    t_s: NDArray[np.float64],
    position_m: NDArray[np.float64],
    quaternion: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t_s, position_m, quaternion = _deduplicate_time(t_s, position_m, quaternion)
    if len(t_s) < 3:
        raise ValueError(f"body velocity needs at least 3 distinct poses; got {len(t_s)}")
    yaw_rad = np.unwrap(Rotation.from_quat(quaternion).as_euler("xyz")[:, 2])
    nominal_dt_s = float(np.median(np.diff(t_s)))
    window = max(5, round(_VELOCITY_SMOOTHING_S / nominal_dt_s))
    window += 1 - window % 2
    window = min(window, len(t_s) if len(t_s) % 2 else len(t_s) - 1)
    if window >= 5:
        position_m = savgol_filter(position_m, window, 2, axis=0)
        yaw_rad = savgol_filter(yaw_rad, window, 2)
    world_v_m_s = np.gradient(position_m[:, :2], t_s, axis=0)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    twist = np.column_stack(
        (
            c * world_v_m_s[:, 0] + s * world_v_m_s[:, 1],
            -s * world_v_m_s[:, 0] + c * world_v_m_s[:, 1],
            np.gradient(yaw_rad, t_s),
        )
    )
    return t_s, twist


def body_velocity(recording: G1Recording) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Differentiate world_T_pelvis and express planar twist in pelvis frame."""
    pointlio_t_s, world_p_pelvis_m, world_q_pelvis_xyzw, _ = measured_pelvis_pose(recording)
    return _body_velocity_from_pose(pointlio_t_s, world_p_pelvis_m, world_q_pelvis_xyzw)


def _constant_runs(command: NDArray[np.float64], tolerance: float) -> list[tuple[int, int]]:
    changed = np.any(np.abs(np.diff(command, axis=0)) > tolerance, axis=1)
    boundaries = [0, *(np.flatnonzero(changed) + 1).tolist(), len(command)]
    return list(pairwise(boundaries))


def _response_spans_from_commands(
    t_s: NDArray[np.float64],
    command: NDArray[np.float64],
    *,
    min_duration_s: float = 0.6,
    command_tolerance: float = 1e-3,
) -> list[ResponseSpan]:
    spans: list[ResponseSpan] = []
    previous = np.zeros(3)
    for start, end in _constant_runs(command, command_tolerance):
        level = np.median(command[start:end], axis=0)
        active = np.flatnonzero(np.abs(level) > command_tolerance)
        duration_s = float(t_s[end - 1] - t_s[start])
        if active.size == 1 and duration_s >= min_duration_s:
            axis = int(active[0])
            delta = float(level[axis] - previous[axis])
            sign = 1 if level[axis] > 0 else -1
            if sign * delta > command_tolerance:
                spans.append(
                    ResponseSpan(
                        axis=axis,
                        sign=sign,
                        command=float(level[axis]),
                        command_step=delta,
                        start_s=float(t_s[start]),
                        end_s=float(t_s[end - 1]),
                    )
                )
        previous = level
    return spans


def response_spans(
    recording: G1Recording,
    *,
    min_duration_s: float = 0.6,
    command_tolerance: float = 1e-3,
) -> list[ResponseSpan]:
    """Find held single-axis commands, including upward ramp increments."""
    return _response_spans_from_commands(
        recording.command_t_s,
        recording.command_body_twist,
        min_duration_s=min_duration_s,
        command_tolerance=command_tolerance,
    )


def _fopdt(
    t_s: NDArray[np.float64], K: float, tau_s: float, deadtime_s: float, u: float
) -> NDArray[np.float64]:
    active = np.maximum(t_s - deadtime_s, 0.0)
    return K * u * (1.0 - np.exp(-active / tau_s))


def _fit_curve(
    t_s: NDArray[np.float64], response: NDArray[np.float64], command_step: float
) -> tuple[float, float, float, float, float, bool]:
    tail = response[t_s >= t_s[-1] * 0.7]
    K0 = float(np.clip(np.median(tail) / command_step, 0.01, 4.9))
    noise = max(float(np.std(response[t_s <= min(0.25, t_s[-1])])), 1e-4)
    moving = np.flatnonzero(response > 4.0 * noise)
    L0 = float(np.clip(t_s[moving[0]] if moving.size else 0.05, 0.0, 0.95))

    def model(t: NDArray[np.float64], K: float, tau: float, deadtime: float) -> NDArray[np.float64]:
        return _fopdt(t, K, tau, deadtime, command_step)

    try:
        params, _ = curve_fit(
            model,
            t_s,
            response,
            p0=(K0, 0.3, L0),
            bounds=((0.0, 0.01, 0.0), (5.0, 5.0, 1.0)),
            maxfev=5000,
        )
    except (RuntimeError, ValueError):
        return math.nan, math.nan, math.nan, math.nan, math.nan, False
    predicted = model(t_s, *params)
    residual = response - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    variance = float(np.sum((response - np.mean(response)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / variance if variance > 0.0 else math.nan
    return float(params[0]), float(params[1]), float(params[2]), rmse, r_squared, True


def _step_fit(
    span: ResponseSpan,
    odom_t_s: NDArray[np.float64],
    twist: NDArray[np.float64],
) -> StepFit:
    pre = (odom_t_s >= span.start_s - 0.35) & (odom_t_s < span.start_s)
    fit = (odom_t_s >= span.start_s) & (odom_t_s <= min(span.end_s, span.start_s + 4.0))
    t_s = odom_t_s[fit] - span.start_s
    baseline = np.median(twist[pre], axis=0) if np.any(pre) else np.zeros(3)
    response = span.sign * (twist[fit, span.axis] - baseline[span.axis])
    stationary_noise = max(float(np.std(twist[pre, span.axis])), 1e-4) if np.any(pre) else math.nan
    if len(t_s) < 5:
        return StepFit(
            direction=_DIRECTIONS[(span.axis, span.sign)],
            unit=_UNITS[span.axis],
            command=abs(span.command),
            command_step=abs(span.command_step),
            duration_s=span.end_s - span.start_s,
            settled_speed=math.nan,
            K=math.nan,
            tau_s=math.nan,
            deadtime_s=math.nan,
            rmse=math.nan,
            r_squared=math.nan,
            response_vx_rms_m_s=math.nan,
            response_vy_rms_m_s=math.nan,
            response_wz_rms_rad_s=math.nan,
            stationary_noise=stationary_noise,
            movement_snr=math.nan,
            n_samples=len(t_s),
            moved=False,
            converged=False,
        )
    K, tau_s, deadtime_s, rmse, r_squared, converged = _fit_curve(
        t_s, response, abs(span.command_step)
    )
    tail = t_s >= max(0.0, t_s[-1] * 0.7) if len(t_s) else np.zeros(0, dtype=bool)
    settled = (
        float(np.median(span.sign * twist[fit, span.axis][tail])) if np.any(tail) else math.nan
    )
    response_rms = np.sqrt(np.mean((twist[fit] - baseline) ** 2, axis=0))
    movement_snr = settled / stationary_noise
    moved = bool(np.isfinite(movement_snr) and movement_snr >= 4.0)
    return StepFit(
        direction=_DIRECTIONS[(span.axis, span.sign)],
        unit=_UNITS[span.axis],
        command=abs(span.command),
        command_step=abs(span.command_step),
        duration_s=span.end_s - span.start_s,
        settled_speed=settled,
        K=K,
        tau_s=tau_s,
        deadtime_s=deadtime_s,
        rmse=rmse,
        r_squared=r_squared,
        response_vx_rms_m_s=float(response_rms[0]),
        response_vy_rms_m_s=float(response_rms[1]),
        response_wz_rms_rad_s=float(response_rms[2]),
        stationary_noise=stationary_noise,
        movement_snr=movement_snr,
        n_samples=len(t_s),
        moved=moved,
        converged=converged,
    )


def fit_trajectory_steps(
    command_t_s: NDArray[np.float64],
    command_body_twist: NDArray[np.float64],
    pose_t_s: NDArray[np.float64],
    world_p_pelvis_m: NDArray[np.float64],
    world_q_pelvis_xyzw: NDArray[np.float64],
) -> list[StepFit]:
    """Fit command steps against one measured or simulated pelvis trajectory."""
    odom_t_s, twist = _body_velocity_from_pose(pose_t_s, world_p_pelvis_m, world_q_pelvis_xyzw)
    spans = _response_spans_from_commands(command_t_s, command_body_twist)
    return [_step_fit(span, odom_t_s, twist) for span in spans]


def fit_steps(recording: G1Recording) -> list[StepFit]:
    """Fit every eligible upward command increment in the recording."""
    pose_t_s, position_m, quaternion_xyzw, _ = measured_pelvis_pose(recording)
    return fit_trajectory_steps(
        recording.command_t_s,
        recording.command_body_twist,
        pose_t_s,
        position_m,
        quaternion_xyzw,
    )


def _spread(values: list[float]) -> tuple[float | None, tuple[float, float] | None]:
    if not values:
        return None, None
    data = np.asarray(values)
    return float(np.median(data)), (float(np.quantile(data, 0.1)), float(np.quantile(data, 0.9)))


def _direction_result(direction: str, unit: str, steps: list[StepFit]) -> DirectionResult:
    good = [step for step in steps if step.converged and step.r_squared >= 0.5]
    moved = [step for step in steps if step.moved]
    floor_step = min(moved, key=lambda step: step.command, default=None)
    K, K_range = _spread([step.K for step in good])
    tau, tau_range = _spread([step.tau_s for step in good])
    deadtime, deadtime_range = _spread([step.deadtime_s for step in good])
    command_levels = [step.command for step in steps]
    achieved = [step.settled_speed for step in moved if np.isfinite(step.settled_speed)]
    return DirectionResult(
        direction=direction,
        unit=unit,
        n_steps=len(steps),
        n_good_fits=len(good),
        motion_floor_command=floor_step.command if floor_step is not None else None,
        motion_floor_achieved=floor_step.settled_speed if floor_step is not None else None,
        max_command_tested=max(command_levels, default=None),
        max_achieved=max(achieved, default=None),
        K_median=K,
        K_p10_p90=K_range,
        tau_median_s=tau,
        tau_p10_p90_s=tau_range,
        deadtime_median_s=deadtime,
        deadtime_p10_p90_s=deadtime_range,
        ceiling_observed=False,
    )


def direction_results(steps: list[StepFit]) -> tuple[DirectionResult, ...]:
    """Reduce step fits into the six directional envelopes."""
    return tuple(
        _direction_result(
            direction,
            _UNITS[axis],
            [step for step in steps if step.direction == direction],
        )
        for (axis, _sign), direction in _DIRECTIONS.items()
    )


def _span_response(
    span: ResponseSpan,
    track: _VelocityTrack,
    relative_t_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    baseline = (track.t_s >= span.start_s - 0.35) & (track.t_s < span.start_s)
    if not np.any(baseline):
        raise ValueError(f"no velocity baseline before command at {span.start_s}s")
    baseline_speed = float(np.median(track.body_twist[baseline, span.axis]))
    speed = np.interp(
        span.start_s + relative_t_s,
        track.t_s,
        track.body_twist[:, span.axis],
    )
    return span.sign * (speed - baseline_speed)


def _transient_tracks(
    reference: G1Recording,
    predicted: G1SimulationRecording,
) -> tuple[_VelocityTrack, _VelocityTrack]:
    reference_track = _VelocityTrack(*body_velocity(reference))
    predicted_track = _VelocityTrack(
        *_body_velocity_from_pose(
            predicted.sim_t_s,
            predicted.sim_world_p_pelvis_m,
            predicted.sim_world_q_pelvis_xyzw,
        )
    )
    return reference_track, predicted_track


def _transient_spans(
    command_t_s: NDArray[np.float64],
    command_body_twist: NDArray[np.float64],
    direction: str,
    levels: int,
    split: str,
) -> list[ResponseSpan]:
    distinct: dict[float, ResponseSpan] = {}
    for span in _response_spans_from_commands(command_t_s, command_body_twist):
        if _DIRECTIONS[(span.axis, span.sign)] != direction:
            continue
        level = round(abs(span.command), 6)
        previous = distinct.get(level)
        if previous is None or span.end_s - span.start_s > previous.end_s - previous.start_s:
            distinct[level] = span
    spans = sorted(distinct.values(), key=lambda span: abs(span.command))[-levels:]
    if len(spans) != levels:
        raise ValueError(f"{direction} has {len(spans)} levels; need {levels}")
    if split == "all":
        return spans
    parity = 0 if split == "train" else 1
    return [span for index, span in enumerate(spans) if index % 2 == parity]


def _matched_transient_spans(
    reference: G1Recording,
    predicted: G1SimulationRecording,
    direction: str,
    levels: int,
    split: str,
) -> list[tuple[ResponseSpan, ResponseSpan]]:
    reference_spans = _transient_spans(
        reference.command_t_s,
        reference.command_body_twist,
        direction,
        levels,
        split,
    )
    predicted_spans = _transient_spans(
        predicted.command_t_s,
        predicted.command_body_twist,
        direction,
        levels,
        split,
    )
    predicted_by_level = {round(abs(span.command), 6): span for span in predicted_spans}
    missing = [
        span.command
        for span in reference_spans
        if round(abs(span.command), 6) not in predicted_by_level
    ]
    if missing:
        raise ValueError(f"{direction} replay is missing command levels {missing}")
    return [(span, predicted_by_level[round(abs(span.command), 6)]) for span in reference_spans]


def _transient_error(
    direction: str,
    unit: str,
    spans: list[tuple[ResponseSpan, ResponseSpan]],
    reference: _VelocityTrack,
    predicted: _VelocityTrack,
    response_window_s: float,
    sample_period_s: float,
) -> DirectionTransientError:
    pairs = []
    for reference_span, predicted_span in spans:
        duration_s = min(
            response_window_s,
            reference_span.end_s - reference_span.start_s,
            predicted_span.end_s - predicted_span.start_s,
        )
        relative_t_s = np.arange(sample_period_s, duration_s, sample_period_s)
        pairs.append(
            (
                _span_response(reference_span, reference, relative_t_s),
                _span_response(predicted_span, predicted, relative_t_s),
            )
        )
    reference_values = np.concatenate([pair[0] for pair in pairs])
    predicted_values = np.concatenate([pair[1] for pair in pairs])
    rmse = float(np.sqrt(np.mean((predicted_values - reference_values) ** 2)))
    scale = float(np.max(np.abs(reference_values)))
    if scale <= 0.0:
        raise ValueError(f"{direction} reference transient scale is zero")
    return DirectionTransientError(
        direction,
        unit,
        len(spans),
        len(reference_values),
        rmse,
        rmse / scale,
        scale,
        float(np.max(np.abs(predicted_values))),
    )


def directional_transient_errors(
    reference: G1Recording,
    predicted: G1SimulationRecording,
    *,
    levels_per_direction: int = 8,
    response_window_s: float = 2.0,
    sample_period_s: float = 0.1,
    split: str = "all",
) -> tuple[DirectionTransientError, ...]:
    """Compare Ivan-style baseline-subtracted step responses at fixed levels."""
    if response_window_s <= 0.0 or sample_period_s <= 0.0:
        raise ValueError("response window and sample period must be positive")
    if split not in {"all", "train", "validation"}:
        raise ValueError(f"split must be all, train, or validation; got {split!r}")
    tracks = _transient_tracks(reference, predicted)
    return tuple(
        _transient_error(
            direction,
            _UNITS[axis],
            _matched_transient_spans(
                reference,
                predicted,
                direction,
                levels_per_direction,
                split,
            ),
            *tracks,
            response_window_s,
            sample_period_s,
        )
        for (axis, _sign), direction in _DIRECTIONS.items()
    )


def _rate_and_gap(t_s: NDArray[np.float64]) -> tuple[float, float]:
    dt_s = np.diff(t_s)
    positive = dt_s[dt_s > 0.0]
    if not len(positive):
        return 0.0, math.inf
    return 1.0 / float(np.median(positive)), float(np.max(positive))


def _nearest_skew_s(
    source_t_s: NDArray[np.float64], reference_t_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    indices = np.searchsorted(reference_t_s, source_t_s)
    right = np.clip(indices, 0, len(reference_t_s) - 1)
    left = np.clip(indices - 1, 0, len(reference_t_s) - 1)
    return np.minimum(
        np.abs(source_t_s - reference_t_s[left]), np.abs(source_t_s - reference_t_s[right])
    )


def _nearest_values(
    source_t_s: NDArray[np.float64],
    values: NDArray[np.float64],
    query_t_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    indices = np.searchsorted(source_t_s, query_t_s)
    right = np.clip(indices, 0, len(source_t_s) - 1)
    left = np.clip(indices - 1, 0, len(source_t_s) - 1)
    use_left = np.abs(query_t_s - source_t_s[left]) <= np.abs(query_t_s - source_t_s[right])
    return values[np.where(use_left, left, right)]


def recording_health(recording: G1Recording) -> RecordingHealth:
    """Score timing plus the independent Point-LIO twist/pose-derivative agreement."""
    fresh = pointlio_pose_sample_mask(recording)
    world_p_mid360_m, world_q_mid360_xyzw, pose_outlier = filter_pose_outliers(
        recording.world_p_mid360_m[fresh], recording.world_q_mid360_xyzw[fresh]
    )
    odom_t_s, pose_twist = _body_velocity_from_pose(
        recording.pointlio_t_s[fresh],
        world_p_mid360_m,
        world_q_mid360_xyzw,
    )
    raw = np.column_stack(
        (
            np.interp(odom_t_s, recording.pointlio_t_s, recording.pointlio_world_twist[:, 0]),
            np.interp(odom_t_s, recording.pointlio_t_s, recording.pointlio_world_twist[:, 1]),
            np.interp(odom_t_s, recording.pointlio_t_s, recording.pointlio_world_twist[:, 2]),
        )
    )
    yaw = np.unwrap(Rotation.from_quat(world_q_mid360_xyzw).as_euler("xyz")[:, 2])
    yaw_i = np.interp(odom_t_s, recording.pointlio_t_s[fresh], yaw)
    c, s = np.cos(yaw_i), np.sin(yaw_i)
    raw_body = np.column_stack(
        (c * raw[:, 0] + s * raw[:, 1], -s * raw[:, 0] + c * raw[:, 1], raw[:, 2])
    )
    correlation = tuple(
        float(np.corrcoef(pose_twist[:, i], raw_body[:, i])[0, 1]) for i in range(3)
    )
    rmse = tuple(
        float(np.sqrt(np.mean((pose_twist[:, i] - raw_body[:, i]) ** 2))) for i in range(3)
    )
    command_rate, command_gap = _rate_and_gap(recording.command_t_s)
    pointlio_publish_rate, pointlio_publish_gap = _rate_and_gap(recording.pointlio_t_s)
    pointlio_pose_rate, pointlio_pose_gap = _rate_and_gap(recording.pointlio_t_s[fresh])
    skew_s = _nearest_skew_s(recording.command_t_s, recording.pointlio_t_s)
    median_skew_s = float(np.median(skew_s))
    p95_skew_s = float(np.quantile(skew_s, 0.95))
    teleop_skew_s = _nearest_skew_s(recording.command_t_s, recording.teleop_t_s)
    teleop_at_command = _nearest_values(
        recording.teleop_t_s,
        recording.teleop_body_twist,
        recording.command_t_s,
    )
    teleop_matches = np.all(np.abs(teleop_at_command - recording.command_body_twist) < 1e-6, axis=1)
    teleop_matching_fraction = float(np.mean(teleop_matches))
    teleop_median_skew_s = float(np.median(teleop_skew_s))
    teleop_p95_skew_s = float(np.quantile(teleop_skew_s, 0.95))
    overlap = max(
        0.0,
        min(recording.command_t_s[-1], odom_t_s[-1]) - max(recording.command_t_s[0], odom_t_s[0]),
    )
    position_step_m = np.linalg.norm(np.diff(recording.world_p_mid360_m, axis=0), axis=1)
    rotation_step_rad = (
        Rotation.from_quat(recording.world_q_mid360_xyzw[:-1]).inv()
        * Rotation.from_quat(recording.world_q_mid360_xyzw[1:])
    ).magnitude()
    warnings: list[str] = []
    if pointlio_pose_rate < 15.0 or pointlio_pose_gap > 0.2:
        warnings.append("Point-LIO timing is too sparse for reliable dead-time fitting")
    if p95_skew_s > 0.1:
        warnings.append("command/Point-LIO timestamp skew exceeds 0.1 s at p95")
    if teleop_matching_fraction < 0.99:
        warnings.append("teleop and policy command twists match on fewer than 99% of samples")
    if any(not np.isfinite(value) or value < 0.5 for value in correlation):
        warnings.append("Point-LIO reported twist disagrees with its pose derivative")
    if overlap < 30.0:
        warnings.append("command/Point-LIO overlap is shorter than 30 s")
    return RecordingHealth(
        command_rate_hz=command_rate,
        pointlio_publish_rate_hz=pointlio_publish_rate,
        pointlio_pose_update_rate_hz=pointlio_pose_rate,
        command_max_gap_s=command_gap,
        pointlio_publish_max_gap_s=pointlio_publish_gap,
        pointlio_pose_update_max_gap_s=pointlio_pose_gap,
        command_pointlio_median_skew_s=median_skew_s,
        command_pointlio_p95_skew_s=p95_skew_s,
        teleop_command_matching_fraction=teleop_matching_fraction,
        teleop_command_median_skew_s=teleop_median_skew_s,
        teleop_command_p95_skew_s=teleop_p95_skew_s,
        overlap_s=overlap,
        mid360_pose_outlier_fraction=float(np.mean(pose_outlier)),
        mid360_max_position_step_m=float(np.max(position_step_m)),
        mid360_max_rotation_step_rad=float(np.max(rotation_step_rad)),
        mid360_twist_correlation=correlation,
        mid360_twist_rmse=rmse,
        status="pass" if not warnings else "warn",
        warnings=tuple(warnings),
    )


def characterize(recording: G1Recording) -> CharacterizationResult:
    """Produce the six-direction high-level characterization artifact."""
    steps = fit_steps(recording)
    return CharacterizationResult(
        health=recording_health(recording),
        directions=direction_results(steps),
        steps=tuple(steps),
    )
