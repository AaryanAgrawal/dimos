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

"""SONIC (GEAR-SONIC) inference pipeline, simulator-agnostic.

Planner (10 Hz, background thread) -> Encoder (50 Hz) -> Decoder (50 Hz)
producing 29 joint position targets. Ported from the Matrix project's
parity-verified reimplementation of NVIDIA's C++ reference
(GR00T-WholeBodyControl/gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp);
all observation layouts, joint orderings, gains, and the encoder-injection
rule match that reference. See sonic-notebook/DECISIONS.md D3: upper-body
targets enter ONLY through the encoder observation - never override the
decoder's output.

This module has no DimOS or simulator dependencies: callers feed joint
state (DDS/MuJoCo order), an IMU quaternion (w,x,y,z), and body-frame
angular velocity; ``step()`` returns 29 position targets in DDS order.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore[import-untyped]

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# ---------------------------------------------------------------------------
# Motor constants (policy_parameters.hpp)
# ---------------------------------------------------------------------------

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2 * math.pi
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

EFFORT_5020 = 25.0
EFFORT_7520_14 = 88.0
EFFORT_7520_22 = 139.0
EFFORT_4010 = 5.0

# PD gains in DDS/MuJoCo joint order, matching the C++ kps/kds arrays
# exactly - including the x2 on ankles and waist roll/pitch. The policy
# was trained against these; the blueprint must pass them as wb_config.
_KP_LEG = [
    STIFFNESS_7520_22,
    STIFFNESS_7520_22,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    2.0 * STIFFNESS_5020,
    2.0 * STIFFNESS_5020,
]
_KD_LEG = [
    DAMPING_7520_22,
    DAMPING_7520_22,
    DAMPING_7520_14,
    DAMPING_7520_22,
    2.0 * DAMPING_5020,
    2.0 * DAMPING_5020,
]
_KP_ARM = [
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_4010,
    STIFFNESS_4010,
]
_KD_ARM = [
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_4010,
    DAMPING_4010,
]
SONIC_KP: list[float] = [
    *_KP_LEG,
    *_KP_LEG,
    STIFFNESS_7520_14,
    2.0 * STIFFNESS_5020,
    2.0 * STIFFNESS_5020,  # waist
    *_KP_ARM,
    *_KP_ARM,
]
SONIC_KD: list[float] = [
    *_KD_LEG,
    *_KD_LEG,
    DAMPING_7520_14,
    2.0 * DAMPING_5020,
    2.0 * DAMPING_5020,  # waist
    *_KD_ARM,
    *_KD_ARM,
]

# ---------------------------------------------------------------------------
# Joint orderings. "DDS order" here equals the MuJoCo order used across
# DimOS G1 code (legs L/R, waist, arms L/R). "ONNX order" is SONIC's
# interleaved left/right BFS training order.
# ---------------------------------------------------------------------------

NUM_JOINTS = 29
HISTORY_LEN = 10

# ONNX index -> DDS index (isaaclab_to_mujoco in the C++)
ONNX_TO_DDS = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.intp,
)
# DDS index -> ONNX index (mujoco_to_isaaclab in the C++)
DDS_TO_ONNX = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.intp,
)

DEFAULT_ANGLES_DDS = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

_SCALE_LEG = [
    0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_7520_14 / STIFFNESS_7520_14,
    0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_5020 / STIFFNESS_5020,
]
_SCALE_ARM = [
    0.25 * EFFORT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_4010 / STIFFNESS_4010,
    0.25 * EFFORT_4010 / STIFFNESS_4010,
]
ACTION_SCALE_DDS = np.array(
    [
        *_SCALE_LEG,
        *_SCALE_LEG,
        0.25 * EFFORT_7520_14 / STIFFNESS_7520_14,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        *_SCALE_ARM,
        *_SCALE_ARM,
    ],
    dtype=np.float32,
)

DEFAULT_ANGLES_ONNX = DEFAULT_ANGLES_DDS[ONNX_TO_DDS]
ACTION_SCALE_ONNX = ACTION_SCALE_DDS[ONNX_TO_DDS]

# 17 upper-body joints (waist + arms) in ONNX-order indices, matching the
# C++ upper_body_joint_isaaclab_order_in_isaaclab_index.
UPPER_BODY_ONNX_INDICES = np.array(
    [2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    dtype=np.intp,
)

# ---------------------------------------------------------------------------
# Observation dimensions (observation_config.yaml; offsets verified against
# the C++ observation registry)
# ---------------------------------------------------------------------------

ENCODER_OBS_DIM = 1762
ENCODER_TOKEN_DIM = 64
DECODER_OBS_DIM = 994

DEFAULT_HEIGHT = 0.788740
POLICY_DT = 0.02
REPLAN_INTERVAL_DEFAULT = 1.0
REPLAN_INTERVAL_RUNNING = 0.1
BLEND_FRAMES = 8
LOOK_AHEAD_FRAMES = 2

_IDENTITY_6D = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

# LocomotionMode (localmotion_kplanner.hpp) - the full 27.
LOCOMOTION_MODES: dict[str, int] = {
    "IDLE": 0,
    "SLOW_WALK": 1,
    "WALK": 2,
    "RUN": 3,
    "IDEL_SQUAT": 4,
    "IDEL_KNEEL_TWO_LEGS": 5,
    "IDEL_KNEEL": 6,
    "IDEL_LYING_FACE_DOWN": 7,
    "CRAWLING": 8,
    "IDEL_BOXING": 9,
    "WALK_BOXING": 10,
    "LEFT_PUNCH": 11,
    "RIGHT_PUNCH": 12,
    "RANDOM_PUNCH": 13,
    "ELBOW_CRAWLING": 14,
    "LEFT_HOOK": 15,
    "RIGHT_HOOK": 16,
    "FORWARD_JUMP": 17,
    "STEALTH_WALK": 18,
    "INJURED_WALK": 19,
    "LEDGE_WALKING": 20,
    "OBJECT_CARRYING": 21,
    "STEALTH_WALK_2": 22,
    "HAPPY_DANCE_WALK": 23,
    "ZOMBIE_WALK": 24,
    "GUN_WALK": 25,
    "SCARE_WALK": 26,
}
STATIC_MODES = {0, 4, 5, 6, 7, 9}


# ---------------------------------------------------------------------------
# Quaternion helpers ([w, x, y, z] convention throughout)
# ---------------------------------------------------------------------------


def _quat_conjugate(q: NDArray) -> NDArray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply(q1: NDArray, q2: NDArray) -> NDArray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_rotmat(q: NDArray) -> NDArray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n > 1e-10:
        w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_6d(rot: NDArray) -> NDArray:
    return np.array(
        [rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]],
        dtype=np.float32,
    )


def _quat_lerp(q0: NDArray, q1: NDArray, t: float) -> NDArray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1.0 - t) * q0 + t * q1
    n = np.linalg.norm(q)
    return (q / n if n > 1e-10 else q0).astype(np.float32)


def _yaw_from_quat(q: NDArray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _calc_heading_quat(q: NDArray) -> NDArray:
    half = _yaw_from_quat(q) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _calc_heading_quat_inv(q: NDArray) -> NDArray:
    half = -_yaw_from_quat(q) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


class _Trajectory:
    """50 Hz reference motion (joint data stored in ONNX order)."""

    __slots__ = ("joint_pos", "joint_vel", "num_frames", "root_pos", "root_quat")

    def __init__(self, max_frames: int) -> None:
        self.joint_pos = np.zeros((max_frames, NUM_JOINTS), dtype=np.float32)
        self.joint_vel = np.zeros((max_frames, NUM_JOINTS), dtype=np.float32)
        self.root_pos = np.zeros((max_frames, 3), dtype=np.float32)
        self.root_quat = np.zeros((max_frames, 4), dtype=np.float32)
        self.root_quat[:, 0] = 1.0
        self.num_frames = 0


class SonicPipeline:
    """Planner -> encoder -> decoder pipeline over ONNX Runtime.

    Callers drive it at 50 Hz via :meth:`step`. The planner runs on a
    single background worker so ``step()`` never blocks on the 774 MB
    planner model.
    """

    def __init__(
        self,
        encoder_path: str | Path,
        decoder_path: str | Path,
        planner_path: str | Path,
        providers: list[str] | None = None,
    ) -> None:
        cpu = ["CPUExecutionProvider"]
        fast = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._encoder = ort.InferenceSession(str(encoder_path), providers=fast)
        self._decoder = ort.InferenceSession(str(decoder_path), providers=fast)
        try:
            self._planner = ort.InferenceSession(str(planner_path), providers=fast)
        except Exception:
            self._planner = ort.InferenceSession(str(planner_path), providers=cpu)
        self._encoder_input = self._encoder.get_inputs()[0].name
        self._decoder_input = self._decoder.get_inputs()[0].name
        logger.info(
            "SonicPipeline models loaded",
            encoder_providers=self._encoder.get_providers(),
            planner_providers=self._planner.get_providers(),
        )

        self._standing_token = self._build_standing_token()

        self._his_ang_vel = np.zeros((HISTORY_LEN, 3), dtype=np.float32)
        self._his_joint_pos = np.zeros((HISTORY_LEN, NUM_JOINTS), dtype=np.float32)
        self._his_joint_vel = np.zeros((HISTORY_LEN, NUM_JOINTS), dtype=np.float32)
        self._his_action = np.zeros((HISTORY_LEN, NUM_JOINTS), dtype=np.float32)
        self._his_gravity = np.zeros((HISTORY_LEN, 3), dtype=np.float32)
        self._history_ptr = 0
        self._last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._obs_buffer = np.zeros(DECODER_OBS_DIM, dtype=np.float32)

        self._trajectory: _Trajectory | None = None
        self._traj_frame = 0
        self._heading_delta_quat = np.array([1, 0, 0, 0], dtype=np.float64)
        self._heading_initialized = False

        self._planner_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sonic-planner"
        )
        self._planner_future: Future | None = None
        self._replan_timer = 0.0
        self._needs_replan = True
        self._step_count = 0

        # Commands
        self._vx = 0.0
        self._vy = 0.0
        self._yaw_rate = 0.0
        self._height_cmd = -1.0  # -1 = mode default
        self._mode_override: int | None = None
        self._upper_targets_dds = DEFAULT_ANGLES_DDS[15:].copy()

        # Latest robot state fed by step() (for planner input building)
        self._cur_quat = np.array([1, 0, 0, 0], dtype=np.float64)
        self._cur_q_dds = DEFAULT_ANGLES_DDS.copy()
        self._nan_reported = 0
        self._last_targets_dds = DEFAULT_ANGLES_DDS.copy()

    # -- commands ---------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        if abs(vx - self._vx) > 0.05 or abs(vy - self._vy) > 0.05 or abs(wz - self._yaw_rate) > 0.1:
            self._needs_replan = True
        self._vx, self._vy, self._yaw_rate = vx, vy, wz

    def set_mode(self, mode: int | str | None) -> int | None:
        """Force a LocomotionMode (int or name); None returns to speed-auto."""
        if isinstance(mode, str):
            mode = LOCOMOTION_MODES[mode.upper()]
        if mode is not None and not 0 <= int(mode) <= 26:
            raise ValueError(f"locomotion mode out of range: {mode}")
        if mode != self._mode_override:
            self._needs_replan = True
        self._mode_override = None if mode is None else int(mode)
        return self._mode_override

    def set_base_height(self, height: float) -> None:
        if abs(height - self._height_cmd) > 0.01:
            self._needs_replan = True
        self._height_cmd = float(height)

    def set_upper_body(self, targets_dds_14: NDArray) -> None:
        self._upper_targets_dds = np.asarray(targets_dds_14, dtype=np.float32).flatten()[:14]

    def reset(self) -> None:
        self._his_ang_vel[:] = 0.0
        self._his_joint_pos[:] = 0.0
        self._his_joint_vel[:] = 0.0
        self._his_action[:] = 0.0
        self._his_gravity[:] = 0.0
        self._history_ptr = 0
        self._last_action[:] = 0.0
        self._obs_buffer[:] = 0.0
        self._trajectory = None
        self._traj_frame = 0
        self._replan_timer = 0.0
        self._step_count = 0
        self._needs_replan = True
        self._heading_delta_quat = np.array([1, 0, 0, 0], dtype=np.float64)
        self._heading_initialized = False
        if self._planner_future is not None and not self._planner_future.done():
            self._planner_future.cancel()
        self._planner_future = None
        self._upper_targets_dds = DEFAULT_ANGLES_DDS[15:].copy()
        self._mode_override = None

    # -- encoder ----------------------------------------------------------

    def _build_standing_token(self) -> NDArray:
        enc_obs = np.zeros(ENCODER_OBS_DIM, dtype=np.float32)
        for i in range(10):
            enc_obs[4 + i * NUM_JOINTS : 4 + (i + 1) * NUM_JOINTS] = DEFAULT_ANGLES_ONNX
            enc_obs[601 + i * 6 : 601 + (i + 1) * 6] = _IDENTITY_6D
        out = self._encoder.run(None, {self._encoder_input: enc_obs.reshape(1, -1)})
        return out[0].squeeze().astype(np.float32)

    def _has_upper_body_targets(self) -> bool:
        return not np.allclose(self._upper_targets_dds, DEFAULT_ANGLES_DDS[15:], atol=1e-6)

    def _upper_body_17_onnx(self) -> NDArray:
        full = DEFAULT_ANGLES_ONNX.copy()
        for dds_i in range(15, 29):
            full[DDS_TO_ONNX[dds_i]] = self._upper_targets_dds[dds_i - 15]
        return full[UPPER_BODY_ONNX_INDICES]

    def _inject_upper_body(self, enc_obs: NDArray) -> None:
        """Encoder-observation injection (D3): positions replaced, velocities
        zeroed for the 17 upper-body joints across all 10 frames."""
        upper_vals = self._upper_body_17_onnx()
        for i in range(10):
            pos = 4 + i * NUM_JOINTS
            vel = 294 + i * NUM_JOINTS
            for k, idx in enumerate(UPPER_BODY_ONNX_INDICES):
                enc_obs[pos + idx] = upper_vals[k]
                enc_obs[vel + idx] = 0.0

    def _build_encoder_obs(self, base_quat: NDArray) -> NDArray:
        enc_obs = np.zeros(ENCODER_OBS_DIM, dtype=np.float32)
        traj = self._trajectory
        assert traj is not None
        f_curr = min(self._traj_frame, traj.num_frames - 1)

        for i in range(10):
            f = min(f_curr + i * 5, traj.num_frames - 1)
            enc_obs[4 + i * NUM_JOINTS : 4 + (i + 1) * NUM_JOINTS] = traj.joint_pos[f]
            enc_obs[294 + i * NUM_JOINTS : 294 + (i + 1) * NUM_JOINTS] = traj.joint_vel[f]

        if self._has_upper_body_targets():
            self._inject_upper_body(enc_obs)

        q_base_inv = _quat_conjugate(base_quat)
        for i in range(10):
            f = min(f_curr + i * 5, traj.num_frames - 1)
            q_aligned = _quat_multiply(
                self._heading_delta_quat, traj.root_quat[f].astype(np.float64)
            )
            q_rel = _quat_multiply(q_base_inv, q_aligned)
            enc_obs[601 + i * 6 : 601 + (i + 1) * 6] = _rotmat_to_6d(_quat_to_rotmat(q_rel))
        return enc_obs

    # -- planner ----------------------------------------------------------

    def _auto_mode(self, speed: float) -> int:
        if speed < 0.05:
            return 0
        if speed < 0.4:
            return 1
        if speed < 1.2:
            return 2
        return 3

    def _build_planner_context(self) -> NDArray:
        context = np.zeros((4, 36), dtype=np.float32)
        if self._trajectory is not None and self._trajectory.num_frames > 4:
            traj = self._trajectory
            start = min(self._traj_frame + LOOK_AHEAD_FRAMES, traj.num_frames - 1)
            for n in range(4):
                f = min(round(start + n * (50.0 / 30.0)), traj.num_frames - 1)
                context[n, 0:3] = traj.root_pos[f]
                context[n, 3:7] = traj.root_quat[f]
                context[n, 7:36] = traj.joint_pos[f][DDS_TO_ONNX]
        else:
            root_pos = np.array([0.0, 0.0, DEFAULT_HEIGHT], dtype=np.float32)
            for n in range(4):
                context[n, 0:3] = root_pos
                context[n, 3:7] = self._cur_quat
                context[n, 7:36] = self._cur_q_dds
        return context

    def _build_planner_inputs(self) -> dict:
        speed = math.hypot(self._vx, self._vy)
        yaw = _yaw_from_quat(self._cur_quat)
        cos_h, sin_h = math.cos(yaw), math.sin(yaw)
        world_vx = self._vx * cos_h - self._vy * sin_h
        world_vy = self._vx * sin_h + self._vy * cos_h

        mode = self._mode_override if self._mode_override is not None else self._auto_mode(speed)

        if speed > 0.05 and mode not in STATIC_MODES:
            move_dir = np.array([world_vx / speed, world_vy / speed, 0.0], dtype=np.float32)
        else:
            move_dir = np.zeros(3, dtype=np.float32)

        target_yaw = yaw + self._yaw_rate * 1.0
        face_dir = np.array([math.cos(target_yaw), math.sin(target_yaw), 0.0], dtype=np.float32)

        if mode == 1:
            target_vel = max(0.2, min(speed, 0.8))
        elif mode == 3:
            target_vel = max(1.5, min(speed, 3.0))
        else:
            target_vel = -1.0

        return {
            "context_mujoco_qpos": self._build_planner_context().reshape(1, 4, 36),
            "target_vel": np.array([target_vel], dtype=np.float32),
            "mode": np.array([mode], dtype=np.int64),
            "movement_direction": move_dir.reshape(1, 3),
            "facing_direction": face_dir.reshape(1, 3),
            "random_seed": np.array([42], dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": np.array(
                [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64
            ),
            "height": np.array([self._height_cmd], dtype=np.float32),
        }

    def _submit_planner(self) -> None:
        if self._planner_future is not None and not self._planner_future.done():
            return
        try:
            inputs = self._build_planner_inputs()
        except Exception as exc:
            logger.warning("SonicPipeline planner input build failed", error=repr(exc))
            return
        self._planner_future = self._planner_executor.submit(self._planner.run, None, inputs)

    def _check_planner_result(self) -> None:
        if self._planner_future is None or not self._planner_future.done():
            return
        try:
            self._apply_planner_result(self._planner_future.result())
        except Exception as exc:
            logger.warning("SonicPipeline planner inference failed", error=repr(exc))
        self._planner_future = None

    def _apply_planner_result(self, result: list) -> None:
        qpos_30hz = result[0].squeeze()
        num_frames = int(result[1].item())
        if num_frames < 2:
            return
        if self._nan_check("planner_qpos", qpos_30hz[:num_frames]):
            return
        new_traj = self._resample_to_50hz(qpos_30hz, num_frames)

        if self._trajectory is not None and self._trajectory.num_frames > 0:
            old, old_f = self._trajectory, self._traj_frame
            blend = min(BLEND_FRAMES, new_traj.num_frames)
            for f in range(blend):
                of = min(old_f + f, old.num_frames - 1)
                w_new = (f + 1) / (blend + 1)
                w_old = 1.0 - w_new
                new_traj.joint_pos[f] = w_old * old.joint_pos[of] + w_new * new_traj.joint_pos[f]
                new_traj.root_pos[f] = w_old * old.root_pos[of] + w_new * new_traj.root_pos[f]
                new_traj.root_quat[f] = _quat_lerp(old.root_quat[of], new_traj.root_quat[f], w_new)
            for f in range(min(blend, new_traj.num_frames - 1)):
                new_traj.joint_vel[f] = (new_traj.joint_pos[f + 1] - new_traj.joint_pos[f]) * 50.0

        if not self._heading_initialized and new_traj.num_frames > 0:
            init_heading = _calc_heading_quat(self._cur_quat)
            init_ref_inv = _calc_heading_quat_inv(new_traj.root_quat[0])
            self._heading_delta_quat = _quat_multiply(init_heading, init_ref_inv)
            self._heading_initialized = True

        self._trajectory = new_traj
        self._traj_frame = 0

    def _resample_to_50hz(self, qpos_30hz: NDArray, n30: int) -> _Trajectory:
        n50 = max(2, int(n30 / 30.0 * 50.0))
        traj = _Trajectory(n50)
        for f in range(n50):
            f30 = f / 50.0 * 30.0
            f0 = min(int(f30), n30 - 1)
            f1 = min(f0 + 1, n30 - 1)
            alpha = (f30 - f0) if f0 < n30 - 1 else 0.0
            traj.root_pos[f] = (1 - alpha) * qpos_30hz[f0, 0:3] + alpha * qpos_30hz[f1, 0:3]
            traj.root_quat[f] = _quat_lerp(qpos_30hz[f0, 3:7], qpos_30hz[f1, 3:7], alpha)
            raw = (1 - alpha) * qpos_30hz[f0, 7:36] + alpha * qpos_30hz[f1, 7:36]
            traj.joint_pos[f] = raw[ONNX_TO_DDS]
        for f in range(n50 - 1):
            traj.joint_vel[f] = (traj.joint_pos[f + 1] - traj.joint_pos[f]) * 50.0
        if n50 > 1:
            traj.joint_vel[-1] = traj.joint_vel[-2]
        traj.num_frames = n50
        return traj

    # -- step -------------------------------------------------------------

    def _nan_check(self, name: str, arr: NDArray) -> bool:
        if np.isnan(arr).any() or np.isinf(arr).any():
            if self._nan_reported < 10:
                logger.warning(
                    "SonicPipeline non-finite tensor",
                    tensor=name,
                    step=self._step_count,
                    sample=np.asarray(arr).ravel()[:8].tolist(),
                )
                self._nan_reported += 1
            return True
        return False

    def step(
        self,
        q_dds: NDArray,
        dq_dds: NDArray,
        base_quat_wxyz: NDArray,
        gyro_body: NDArray,
        gravity_body: NDArray,
    ) -> NDArray:
        """One 50 Hz policy step. Returns 29 position targets, DDS order."""
        self._step_count += 1

        # Input sentries: a non-finite or degenerate input poisons the
        # heading math and the planner. Hold the previous targets instead.
        bad = (
            self._nan_check("q_dds", np.asarray(q_dds))
            or self._nan_check("dq_dds", np.asarray(dq_dds))
            or self._nan_check("base_quat", np.asarray(base_quat_wxyz))
            or self._nan_check("gyro", np.asarray(gyro_body))
            or self._nan_check("gravity", np.asarray(gravity_body))
        )
        qn = float(np.linalg.norm(np.asarray(base_quat_wxyz, dtype=np.float64)))
        if qn < 0.5:
            if self._nan_reported < 10:
                logger.warning(
                    "SonicPipeline degenerate base quaternion",
                    norm=qn,
                    step=self._step_count,
                )
                self._nan_reported += 1
            bad = True
        if bad:
            return self._last_targets_dds.copy()
        self._cur_quat = np.asarray(base_quat_wxyz, dtype=np.float64)
        self._cur_q_dds = np.asarray(q_dds, dtype=np.float32)

        self._check_planner_result()

        self._replan_timer += POLICY_DT
        speed = math.hypot(self._vx, self._vy)
        mode = self._mode_override if self._mode_override is not None else self._auto_mode(speed)
        moving = speed > 0.05 or (self._mode_override is not None and mode not in STATIC_MODES)
        interval = REPLAN_INTERVAL_RUNNING if speed >= 1.2 else REPLAN_INTERVAL_DEFAULT
        traj_low = (
            self._trajectory is not None
            and self._traj_frame > self._trajectory.num_frames - 20
            and moving
        )
        # A forced non-static mode needs planner output even at zero twist.
        mode_needs_traj = (
            self._mode_override is not None
            and mode not in STATIC_MODES
            and self._replan_timer >= interval
        )
        if (
            self._needs_replan
            or (self._replan_timer >= interval and moving)
            or traj_low
            or mode_needs_traj
        ):
            self._submit_planner()
            self._replan_timer = 0.0
            self._needs_replan = False

        # Encoder token
        if self._trajectory is not None and self._trajectory.num_frames > 0:
            token = self._run_encoder(self._build_encoder_obs(self._cur_quat))
        elif self._has_upper_body_targets():
            enc_obs = np.zeros(ENCODER_OBS_DIM, dtype=np.float32)
            for i in range(10):
                enc_obs[4 + i * NUM_JOINTS : 4 + (i + 1) * NUM_JOINTS] = DEFAULT_ANGLES_ONNX
                enc_obs[601 + i * 6 : 601 + (i + 1) * 6] = _IDENTITY_6D
            self._inject_upper_body(enc_obs)
            token = self._run_encoder(enc_obs)
        else:
            token = self._standing_token

        # Proprio history (ONNX order)
        q_onnx = self._cur_q_dds[ONNX_TO_DDS]
        dq_onnx = np.asarray(dq_dds, dtype=np.float32)[ONNX_TO_DDS]
        ptr = self._history_ptr
        self._his_ang_vel[ptr] = np.asarray(gyro_body, dtype=np.float32)
        self._his_joint_pos[ptr] = q_onnx - DEFAULT_ANGLES_ONNX
        self._his_joint_vel[ptr] = dq_onnx
        self._his_action[ptr] = self._last_action
        self._his_gravity[ptr] = np.asarray(gravity_body, dtype=np.float32)
        self._history_ptr = (ptr + 1) % HISTORY_LEN

        obs = self._obs_buffer
        obs[0:ENCODER_TOKEN_DIM] = token
        order = np.array(
            [(self._history_ptr + j) % HISTORY_LEN for j in range(HISTORY_LEN)],
            dtype=np.intp,
        )
        obs[64:94] = self._his_ang_vel[order].ravel()
        obs[94:384] = self._his_joint_pos[order].ravel()
        obs[384:674] = self._his_joint_vel[order].ravel()
        obs[674:964] = self._his_action[order].ravel()
        obs[964:994] = self._his_gravity[order].ravel()

        self._nan_check("token", token)
        self._nan_check("decoder_obs", obs)
        out = self._decoder.run(None, {self._decoder_input: obs.reshape(1, -1)})
        actions = out[0].squeeze()[:NUM_JOINTS].astype(np.float32)
        if self._nan_check("actions", actions):
            return self._last_targets_dds.copy()
        self._last_action = actions.copy()

        # All 29 decoder actions applied directly - no post-decoder override
        # (D3; matches C++ CreatePolicyCommand).
        targets_onnx = DEFAULT_ANGLES_ONNX + actions * ACTION_SCALE_ONNX
        self._last_targets_dds = targets_onnx[DDS_TO_ONNX].copy()

        if self._trajectory is not None:
            self._traj_frame = min(self._traj_frame + 1, self._trajectory.num_frames - 1)

        return targets_onnx[DDS_TO_ONNX]

    def _run_encoder(self, enc_obs: NDArray) -> NDArray:
        out = self._encoder.run(None, {self._encoder_input: enc_obs.reshape(1, -1)})
        return out[0].squeeze().astype(np.float32)

    # -- telemetry --------------------------------------------------------

    def snapshot(self) -> dict:
        speed = math.hypot(self._vx, self._vy)
        mode = self._mode_override if self._mode_override is not None else self._auto_mode(speed)
        return {
            "mode": mode,
            "mode_override": self._mode_override,
            "speed": speed,
            "trajectory": self._trajectory is not None,
            "traj_frame": self._traj_frame,
            "traj_frames_total": (self._trajectory.num_frames if self._trajectory else 0),
            "action_norm": float(np.linalg.norm(self._last_action)),
            "upper_body_active": self._has_upper_body_targets(),
        }
