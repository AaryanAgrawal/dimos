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

"""Render GR00T MuJoCo ground truth beside a visual-only Point-LIO ghost."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from dimos.robot.unitree.g1.characterization.comparison import (
    G1SimulationRecording,
    align_trajectories,
    interpolate_pose,
)
from dimos.robot.unitree.g1.characterization.mujoco_model import (
    build_g1_mujoco_spec,
    g1_mujoco_binding,
)
from dimos.robot.unitree.g1.characterization.recording import G1Recording, measured_pelvis_pose

_GHOST_BODY = "pointlio_pelvis_reference"
_GHOST_GEOM = "pointlio_pelvis_reference_geom"


@dataclass(frozen=True)
class VideoResult:
    """Verified video properties."""

    path: Path
    frame_count: int
    fps: float
    playback_speed: float
    source_duration_s: float


def _add_reference_ghost(spec: mujoco.MjSpec) -> None:
    """Add Ivan-style visual pose marker with no collision or mass."""
    body = spec.worldbody.add_body(name=_GHOST_BODY, mocap=True)
    geom = body.add_geom(name=_GHOST_GEOM)
    geom.type = mujoco.mjtGeom.mjGEOM_BOX  # type: ignore[attr-defined]
    geom.size = [0.20, 0.12, 0.12]  # m; large enough to remain visible at the maximum separation.
    geom.rgba = [0.1, 1.0, 0.2, 0.55]
    geom.contype = 0
    geom.conaffinity = 0


def build_render_model() -> mujoco.MjModel:
    """Compile the same empty scene and robot MJCF used by the replay blueprint."""
    spec = build_g1_mujoco_spec()
    _add_reference_ghost(spec)
    return spec.compile()


def _interpolate_motor_q(
    simulation: G1SimulationRecording,
    query_t_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    relative_t_s = simulation.motor_t_s - simulation.command_t_s[0]
    return np.column_stack(
        [
            np.interp(query_t_s, relative_t_s, simulation.motor_q_rad[:, joint])
            for joint in range(simulation.motor_q_rad.shape[1])
        ]
    )


def _ghost_in_sim_world(
    hardware_p_m: NDArray[np.float64],
    hardware_q_xyzw: NDArray[np.float64],
    sim_p_m: NDArray[np.float64],
    sim_q_xyzw: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    hardware_rotation = Rotation.from_quat(hardware_q_xyzw)
    sim_rotation = Rotation.from_quat(sim_q_xyzw)
    world_rotation = sim_rotation[0] * hardware_rotation[0].inv()
    position = sim_p_m[0] + world_rotation.apply(hardware_p_m - hardware_p_m[0])
    rotation = world_rotation * hardware_rotation
    return position, rotation.as_quat()


def _path_pixels(
    hardware_xy_m: NDArray[np.float64],
    sim_xy_m: NDArray[np.float64],
    *,
    width: int,
    height: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    both = np.vstack((hardware_xy_m, sim_xy_m))
    lower = both.min(axis=0)
    span = np.maximum(both.max(axis=0) - lower, 0.2)
    scale = min((width - 30) / span[0], (height - 50) / span[1])

    def convert(points: NDArray[np.float64]) -> NDArray[np.int32]:
        pixels = (points - lower) * scale + np.array([15.0, 35.0])
        pixels[:, 1] = height - pixels[:, 1]
        return pixels.astype(np.int32)

    return convert(hardware_xy_m), convert(sim_xy_m)


def _draw_overlay(
    frame_bgr: NDArray[np.uint8],
    t_s: float,
    command: NDArray[np.float64],
    hardware_path: NDArray[np.int32],
    sim_path: NDArray[np.int32],
    path_index: int,
) -> None:
    panel_w, panel_h = 340, 250
    panel = np.full((panel_h, panel_w, 3), 25, dtype=np.uint8)
    cv2.polylines(panel, [hardware_path], False, (80, 230, 80), 1, cv2.LINE_AA)
    cv2.polylines(panel, [sim_path], False, (255, 220, 80), 1, cv2.LINE_AA)
    cv2.polylines(panel, [hardware_path[: path_index + 1]], False, (30, 255, 30), 3, cv2.LINE_AA)
    cv2.polylines(panel, [sim_path[: path_index + 1]], False, (255, 160, 20), 3, cv2.LINE_AA)
    cv2.putText(panel, "top-down pelvis path", (12, 22), 0, 0.55, (230, 230, 230), 1)
    cv2.putText(panel, "Point-LIO", (12, 45), 0, 0.48, (30, 255, 30), 1)
    cv2.putText(panel, "SIM ground truth", (165, 45), 0, 0.48, (255, 160, 20), 1)
    frame_bgr[15 : 15 + panel_h, 15 : 15 + panel_w] = panel
    text = f"command time {t_s:6.1f} s   vx {command[0]:+.2f} m/s   vy {command[1]:+.2f} m/s   wz {command[2]:+.2f} rad/s"
    cv2.putText(frame_bgr, text, (20, frame_bgr.shape[0] - 24), 0, 0.65, (255, 255, 255), 2)


def _set_model_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_qpos: int,
    joint_addresses: NDArray[np.int64],
    sim_p_m: NDArray[np.float64],
    sim_q_xyzw: NDArray[np.float64],
    motor_q_rad: NDArray[np.float64],
    ghost_p_m: NDArray[np.float64],
    ghost_q_xyzw: NDArray[np.float64],
) -> None:
    data.qpos[root_qpos : root_qpos + 3] = sim_p_m
    data.qpos[root_qpos + 3 : root_qpos + 7] = sim_q_xyzw[[3, 0, 1, 2]]
    data.qpos[joint_addresses] = motor_q_rad
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _GHOST_BODY)
    mocap_id = int(model.body_mocapid[body_id])
    data.mocap_pos[mocap_id] = ghost_p_m
    data.mocap_quat[mocap_id] = ghost_q_xyzw[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)


def _camera_distance_m(sim_p_m: NDArray[np.float64], ghost_p_m: NDArray[np.float64]) -> float:
    """Keep both poses inside the default 45 deg camera field with body margin."""
    separation_m = float(np.linalg.norm(sim_p_m[:2] - ghost_p_m[:2]))
    return max(3.5, 1.0 + 1.2 * separation_m)


def _camera_azimuth_deg(sim_p_m: NDArray[np.float64], ghost_p_m: NDArray[np.float64]) -> float:
    """View the final separation side-on so neither trajectory hides the other."""
    delta_m = ghost_p_m[:2] - sim_p_m[:2]
    return float(np.degrees(np.arctan2(delta_m[1], delta_m[0])) + 90.0)


def render_comparison(
    hardware: G1Recording,
    simulation: G1SimulationRecording,
    out_path: Path,
    *,
    fps: float = 30.0,
    playback_speed: float = 6.0,
    width: int = 1280,
    height: int = 720,
) -> VideoResult:
    """Render the actual simulated state plus re-anchored Point-LIO pose marker."""
    aligned = align_trajectories(hardware, simulation)
    query_t_s = np.arange(aligned.t_s[0], aligned.t_s[-1], playback_speed / fps)
    hardware_t_s, hardware_position_m, hardware_quaternion_xyzw, _ = measured_pelvis_pose(hardware)
    hardware_p, hardware_q = interpolate_pose(
        hardware_t_s - hardware.command_t_s[0],
        hardware_position_m,
        hardware_quaternion_xyzw,
        query_t_s,
    )
    sim_p, sim_q = interpolate_pose(
        simulation.sim_t_s - simulation.command_t_s[0],
        simulation.sim_world_p_pelvis_m,
        simulation.sim_world_q_pelvis_xyzw,
        query_t_s,
    )
    ghost_p, ghost_q = _ghost_in_sim_world(hardware_p, hardware_q, sim_p, sim_q)
    motor_q = _interpolate_motor_q(simulation, query_t_s)
    command_index = np.searchsorted(aligned.t_s, query_t_s, side="right") - 1
    command_index = np.clip(command_index, 0, len(aligned.t_s) - 1)
    command = aligned.command_body_twist[command_index]
    hardware_path, sim_path = _path_pixels(
        aligned.hardware_p_m[:, :2], aligned.sim_p_m[:, :2], width=340, height=250
    )

    model = build_render_model()
    camera_distance_m = max(
        _camera_distance_m(sim, ghost) for sim, ghost in zip(sim_p, ghost_p, strict=True)
    )
    model.stat.extent = max(
        model.stat.extent, camera_distance_m
    )  # MuJoCo scales fog and clipping by extent.
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    binding = g1_mujoco_binding(model, simulation.motor_names)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE  # type: ignore[attr-defined]
    camera.distance = 3.5
    camera.azimuth = _camera_azimuth_deg(sim_p[-1], ghost_p[-1])
    camera.elevation = -18.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"failed to open video writer for {out_path}")
    try:
        for index, t_s in enumerate(query_t_s):
            _set_model_pose(
                model,
                data,
                binding.root_qpos,
                binding.joint_qpos,
                sim_p[index],
                sim_q[index],
                motor_q[index],
                ghost_p[index],
                ghost_q[index],
            )
            camera.lookat[:] = (sim_p[index] + ghost_p[index]) * 0.5
            camera.lookat[2] = 0.8
            camera.distance = _camera_distance_m(sim_p[index], ghost_p[index])
            renderer.update_scene(data, camera=camera)
            frame_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            path_index = int(command_index[index])
            _draw_overlay(
                frame_bgr,
                float(t_s),
                command[index],
                hardware_path,
                sim_path,
                path_index,
            )
            writer.write(frame_bgr)
    finally:
        writer.release()
        renderer.close()
    capture = cv2.VideoCapture(str(out_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        measured_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count != len(query_t_s) or abs(measured_fps - fps) > 0.1:
        raise RuntimeError(
            f"video verification failed: got frames={frame_count} fps={measured_fps}, "
            f"want frames={len(query_t_s)} fps={fps}"
        )
    return VideoResult(
        path=out_path,
        frame_count=frame_count,
        fps=measured_fps,
        playback_speed=playback_speed,
        source_duration_s=float(query_t_s[-1] - query_t_s[0]),
    )
