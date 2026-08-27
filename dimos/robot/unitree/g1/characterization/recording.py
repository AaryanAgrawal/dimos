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

"""Read G1 mem2 recordings through the store seam into frame-labelled arrays."""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import pairwise
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

from dimos.memory.cli.dataset import open_store
from dimos.memory.store.base import Store
from dimos.robot.unitree.g1.frames import (
    world_T_mid360_from_odometry,
    world_T_pelvis_from_mid360_odometry,
)

_CACHE_TAG = "mem2-v2"
_POSE_OUTLIER_M = 0.05  # Reject isolated 5 cm local-median jumps, not sustained motion.
_ROTATION_OUTLIER_RAD = 0.10  # Same rejection at 5.7 deg for isolated orientation jumps.
_POSE_MEDIAN_SAMPLES = 5  # Smallest useful centered quadratic-free neighborhood.
_POSE_REPEAT_M = 1e-9
_POSE_REPEAT_RAD = 1e-9


@dataclass(frozen=True)
class G1Recording:
    """Synchronized source streams; time is epoch seconds and units are SI."""

    command_t_s: NDArray[np.float64]
    command_body_twist: NDArray[np.float64]
    teleop_t_s: NDArray[np.float64]
    teleop_body_twist: NDArray[np.float64]
    pointlio_t_s: NDArray[np.float64]
    world_p_mid360_m: NDArray[np.float64]
    world_q_mid360_xyzw: NDArray[np.float64]
    world_p_pelvis_m: NDArray[np.float64]
    world_q_pelvis_xyzw: NDArray[np.float64]
    pointlio_world_twist: NDArray[np.float64]


@dataclass(frozen=True)
class G1PlantRecording:
    """Low-level command/state streams used for open-loop plant grounding."""

    motor_names: tuple[str, ...]
    motor_command_t_s: NDArray[np.float64]
    motor_command_q_rad: NDArray[np.float64]
    motor_command_dq_rad_s: NDArray[np.float64]
    motor_command_kp_nm_rad: NDArray[np.float64]
    motor_command_kd_nm_s_rad: NDArray[np.float64]
    motor_command_tau_ff_nm: NDArray[np.float64]
    motor_state_t_s: NDArray[np.float64]
    motor_state_q_rad: NDArray[np.float64]
    motor_state_dq_rad_s: NDArray[np.float64]
    motor_state_tau_est_nm: NDArray[np.float64]
    imu_t_s: NDArray[np.float64]
    imu_q_xyzw: NDArray[np.float64]
    imu_gyro_rad_s: NDArray[np.float64]
    imu_accel_m_s2: NDArray[np.float64]


_Recording = TypeVar("_Recording", G1Recording, G1PlantRecording)


def _cache_path(path: Path, kind: str) -> Path:
    stat = path.stat()
    cache_dir = Path.home() / ".cache" / "dimos_g1_characterization"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{path.stem}.{kind}.{_CACHE_TAG}.{int(stat.st_mtime)}.{stat.st_size}.npz"


def _load_cache(path: Path, cls: type[_Recording]) -> _Recording | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        values = {field.name: data[field.name] for field in fields(cls)}
        if "motor_names" in values:
            values["motor_names"] = tuple(values["motor_names"].tolist())
        return cls(**values)


def _save_cache(path: Path, recording: G1Recording | G1PlantRecording) -> None:
    np.savez_compressed(path, **vars(recording))


def filter_pose_outliers(
    world_p_frame_m: NDArray[np.float64],
    world_q_frame_xyzw: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Replace only isolated pose teleports and return their health mask."""
    median_p_m = median_filter(world_p_frame_m, size=(_POSE_MEDIAN_SAMPLES, 1), mode="nearest")
    euler_rad = np.unwrap(Rotation.from_quat(world_q_frame_xyzw).as_euler("xyz"), axis=0)
    median_euler_rad = median_filter(euler_rad, size=(_POSE_MEDIAN_SAMPLES, 1), mode="nearest")
    rotation_error_rad = (
        Rotation.from_euler("xyz", median_euler_rad).inv() * Rotation.from_euler("xyz", euler_rad)
    ).magnitude()
    outlier = (np.linalg.norm(world_p_frame_m - median_p_m, axis=1) > _POSE_OUTLIER_M) | (
        rotation_error_rad > _ROTATION_OUTLIER_RAD
    )
    filtered_p_m = world_p_frame_m.copy()
    filtered_euler_rad = euler_rad.copy()
    sample = np.arange(len(outlier))
    valid = ~outlier
    if np.any(outlier) and np.sum(valid) >= 2:
        for axis in range(3):
            filtered_p_m[outlier, axis] = np.interp(
                sample[outlier], sample[valid], world_p_frame_m[valid, axis]
            )
            filtered_euler_rad[outlier, axis] = np.interp(
                sample[outlier], sample[valid], euler_rad[valid, axis]
            )
    return filtered_p_m, Rotation.from_euler("xyz", filtered_euler_rad).as_quat(), outlier


def _fill_static_pose_samples(
    t_s: NDArray[np.float64], changed: NDArray[np.bool_], nominal_period_s: float
) -> NDArray[np.bool_]:
    keep = changed.copy()
    anchors = [*np.flatnonzero(changed), len(t_s) - 1]
    for start, end in pairwise(anchors):
        cursor = start
        while t_s[end] - t_s[cursor] > 1.5 * nominal_period_s:  # Bound static gaps to 1.5 cycles.
            cursor = int(np.searchsorted(t_s, t_s[cursor] + nominal_period_s))
            keep[min(cursor, end - 1)] = True
    return keep


def pointlio_pose_sample_mask(recording: G1Recording) -> NDArray[np.bool_]:
    """Keep distinct LIO solutions plus cadence samples through static periods."""
    position_step_m = np.linalg.norm(np.diff(recording.world_p_mid360_m, axis=0), axis=1)
    rotation_step_rad = (
        Rotation.from_quat(recording.world_q_mid360_xyzw[:-1]).inv()
        * Rotation.from_quat(recording.world_q_mid360_xyzw[1:])
    ).magnitude()
    changed = np.r_[
        True, (position_step_m > _POSE_REPEAT_M) | (rotation_step_rad > _POSE_REPEAT_RAD)
    ]
    change_t_s = recording.pointlio_t_s[changed]
    change_dt_s = np.diff(change_t_s)
    publish_dt_s = np.diff(recording.pointlio_t_s)
    periods_s = change_dt_s[change_dt_s > 0.0]
    if not len(periods_s):
        periods_s = publish_dt_s[publish_dt_s > 0.0]
    return _fill_static_pose_samples(recording.pointlio_t_s, changed, float(np.median(periods_s)))


def measured_pelvis_pose(
    recording: G1Recording,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    """Return fresh, outlier-filtered world_T_pelvis measurements and health mask."""
    fresh = pointlio_pose_sample_mask(recording)
    position_m, quaternion_xyzw, outlier = filter_pose_outliers(
        recording.world_p_pelvis_m[fresh], recording.world_q_pelvis_xyzw[fresh]
    )
    return recording.pointlio_t_s[fresh], position_m, quaternion_xyzw, outlier


def _rows(store: Store, name: str) -> list[object]:
    if name not in store.list_streams():
        raise ValueError(f"recording needs stream {name!r}; got {store.list_streams()}")
    return list(store.stream(name).order_by("ts"))


def _read_high_level(path: Path) -> G1Recording:
    store = open_store(path)
    try:
        command_rows = _rows(store, "cmd_vel")
        teleop_rows = _rows(store, "tele_cmd_vel")
        odom_rows = _rows(store, "pointlio_odometry")
        motor_rows = _rows(store, "motor_states")
        if not motor_rows:
            raise ValueError(f"{path}: motor_states must be non-empty for lidar-to-pelvis FK")
        motor_names = tuple(motor_rows[0].data.name)
        if any(tuple(row.data.name) != motor_names for row in motor_rows):
            raise ValueError("motor_states joint order changed during recording")
        required_waist = ("g1/waist_yaw", "g1/waist_roll", "g1/waist_pitch")
        missing = [name for name in required_waist if name not in motor_names]
        if missing:
            raise ValueError(f"motor_states needs waist joints; missing {missing}")
        command_t_s = np.asarray([row.ts for row in command_rows], dtype=np.float64)
        command = np.asarray(
            [[row.data.linear.x, row.data.linear.y, row.data.angular.z] for row in command_rows],
            dtype=np.float64,
        )
        teleop_t_s = np.asarray([row.ts for row in teleop_rows], dtype=np.float64)
        teleop = np.asarray(
            [[row.data.linear.x, row.data.linear.y, row.data.angular.z] for row in teleop_rows],
            dtype=np.float64,
        )
        pointlio_t_s = np.asarray([row.ts for row in odom_rows], dtype=np.float64)
        motor_t_s = np.asarray([row.ts for row in motor_rows], dtype=np.float64)
        motor_q_rad = _rectangular(motor_rows, "position", "motor_states")
        waist_rad = np.column_stack(
            [
                np.interp(pointlio_t_s, motor_t_s, motor_q_rad[:, motor_names.index(name)])
                for name in required_waist
            ]
        )
        world_T_mid360 = np.asarray([world_T_mid360_from_odometry(row.data) for row in odom_rows])
        world_T_pelvis = np.asarray(
            [
                world_T_pelvis_from_mid360_odometry(row.data, *waist)
                for row, waist in zip(odom_rows, waist_rad, strict=True)
            ]
        )
        pointlio_twist = np.asarray(
            [[row.data.vx, row.data.vy, row.data.wz] for row in odom_rows],
            dtype=np.float64,
        )
    finally:
        store.stop()
    if not len(command_t_s) or not len(teleop_t_s) or not len(pointlio_t_s):
        raise ValueError(f"{path}: cmd_vel, tele_cmd_vel, and pointlio_odometry must be non-empty")
    return G1Recording(
        command_t_s=command_t_s,
        command_body_twist=command,
        teleop_t_s=teleop_t_s,
        teleop_body_twist=teleop,
        pointlio_t_s=pointlio_t_s,
        world_p_mid360_m=world_T_mid360[:, :3, 3],
        world_q_mid360_xyzw=Rotation.from_matrix(world_T_mid360[:, :3, :3]).as_quat(),
        world_p_pelvis_m=world_T_pelvis[:, :3, 3],
        world_q_pelvis_xyzw=Rotation.from_matrix(world_T_pelvis[:, :3, :3]).as_quat(),
        pointlio_world_twist=pointlio_twist,
    )


def _rectangular(rows: list[object], attr: str, label: str) -> NDArray[np.float64]:
    values = [list(getattr(row.data, attr)) for row in rows]
    widths = {len(value) for value in values}
    if len(widths) != 1:
        raise ValueError(f"{label}.{attr} has changing widths: {sorted(widths)}")
    return np.asarray(values, dtype=np.float64)


def _read_plant(path: Path) -> G1PlantRecording:
    store = open_store(path)
    try:
        command = _rows(store, "motor_command")
        state = _rows(store, "motor_states")
        imu = _rows(store, "imu")
        empty = [
            name
            for name, rows in (("motor_command", command), ("motor_states", state), ("imu", imu))
            if not rows
        ]
        if empty:
            raise ValueError(f"{path}: required streams are empty: {', '.join(empty)}")
        motor_names = tuple(state[0].data.name)
        if not motor_names:
            raise ValueError(f"{path}: motor_states has no named joints")
        if any(tuple(row.data.name) != motor_names for row in state):
            raise ValueError("motor_states joint order changed during recording")
        result = G1PlantRecording(
            motor_names=motor_names,
            motor_command_t_s=np.asarray([row.ts for row in command]),
            motor_command_q_rad=_rectangular(command, "q", "motor_command"),
            motor_command_dq_rad_s=_rectangular(command, "dq", "motor_command"),
            motor_command_kp_nm_rad=_rectangular(command, "kp", "motor_command"),
            motor_command_kd_nm_s_rad=_rectangular(command, "kd", "motor_command"),
            motor_command_tau_ff_nm=_rectangular(command, "tau", "motor_command"),
            motor_state_t_s=np.asarray([row.ts for row in state]),
            motor_state_q_rad=_rectangular(state, "position", "motor_states"),
            motor_state_dq_rad_s=_rectangular(state, "velocity", "motor_states"),
            motor_state_tau_est_nm=_rectangular(state, "effort", "motor_states"),
            imu_t_s=np.asarray([row.ts for row in imu]),
            imu_q_xyzw=np.asarray(
                [
                    [
                        row.data.orientation.x,
                        row.data.orientation.y,
                        row.data.orientation.z,
                        row.data.orientation.w,
                    ]
                    for row in imu
                ]
            ),
            imu_gyro_rad_s=np.asarray(
                [
                    [
                        row.data.angular_velocity.x,
                        row.data.angular_velocity.y,
                        row.data.angular_velocity.z,
                    ]
                    for row in imu
                ]
            ),
            imu_accel_m_s2=np.asarray(
                [
                    [
                        row.data.linear_acceleration.x,
                        row.data.linear_acceleration.y,
                        row.data.linear_acceleration.z,
                    ]
                    for row in imu
                ]
            ),
        )
    finally:
        store.stop()
    return result


def read_recording(path: str | Path, *, cache: bool = True) -> G1Recording:
    """Read the high-level command/Point-LIO contract, cached by file identity."""
    recording_path = Path(path)
    cache_path = _cache_path(recording_path, "response")
    cached = _load_cache(cache_path, G1Recording) if cache else None
    if cached is not None:
        return cached
    recording = _read_high_level(recording_path)
    if cache:
        _save_cache(cache_path, recording)
    return recording


def read_plant_recording(path: str | Path, *, cache: bool = True) -> G1PlantRecording:
    """Read low-level commands, motor state, and IMU for open-loop replay."""
    recording_path = Path(path)
    cache_path = _cache_path(recording_path, "plant")
    cached = _load_cache(cache_path, G1PlantRecording) if cache else None
    if cached is not None:
        return cached
    recording = _read_plant(recording_path)
    if cache:
        _save_cache(cache_path, recording)
    return recording
