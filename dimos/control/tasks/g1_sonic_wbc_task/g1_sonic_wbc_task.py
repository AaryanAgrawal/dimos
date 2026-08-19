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

"""SONIC whole-body-control task for the Unitree G1 humanoid.

Runs the GEAR-SONIC planner+encoder+decoder pipeline inside the
coordinator tick loop. Unlike the GR00T decoupled task, SONIC is a
UNIFIED policy: it claims all 29 joints at WBC priority. Precise arm
servoing is not this task's job (upper-body targets are encoder hints,
per sonic-notebook DECISIONS.md D3) - pair with the decoupled task and
hot-swap when manipulation accuracy matters.

Locomotion modes (the 27 GEAR modes: squat, kneel, crawl, boxing, dances,
carrying, jump...) are RPC-reachable via coordinator.task_invoke:

    task_invoke("sonic_wbc", "set_locomotion_mode", {"mode": "HAPPY_DANCE_WALK"})
    task_invoke("sonic_wbc", "set_locomotion_mode", {"mode": None})  # speed-auto
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from dimos.control.hardware_interface import ConnectedWholeBody
from dimos.control.task import (
    BaseControlTask,
    ControlMode,
    CoordinatorState,
    JointCommandOutput,
    ResourceClaim,
)
from dimos.control.tasks.g1_sonic_wbc_task.sonic_pipeline import (
    DEFAULT_ANGLES_DDS,
    LOCOMOTION_MODES,
    NUM_JOINTS,
    SonicPipeline,
)
from dimos.protocol.service.spec import BaseConfig
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.hardware.whole_body.spec import WholeBodyAdapter
    from dimos.msgs.geometry_msgs.Twist import Twist

logger = setup_logger()


@dataclass
class G1SonicWBCTaskConfig:
    """Configuration for the SONIC WBC task.

    joint_names must be all 29 G1 coordinator joint names in DDS order -
    SONIC is a unified whole-body policy and claims every joint.
    """

    encoder_onnx: str | Path
    decoder_onnx: str | Path
    planner_onnx: str | Path
    joint_names: list[str]
    priority: int = 50
    decimation: int = 1
    timeout: float = 1.0
    auto_arm: bool = False
    auto_dry_run: bool = False
    default_ramp_seconds: float = 10.0


class G1SonicWBCTask(BaseControlTask):
    """GEAR-SONIC unified 29-DOF whole-body policy as a coordinator task.

    State machine, caches, and safety semantics mirror G1GrootWBCTask:
    active-but-unarmed echoes measured positions (pure damping), arm()
    ramps to the SONIC default pose, dry-run computes without emitting.
    """

    def __init__(
        self,
        name: str,
        config: G1SonicWBCTaskConfig,
        adapter: WholeBodyAdapter,
    ) -> None:
        if len(config.joint_names) != NUM_JOINTS:
            raise ValueError(
                f"G1SonicWBCTask '{name}' requires exactly {NUM_JOINTS} joint "
                f"names (unified whole-body policy), got {len(config.joint_names)}"
            )
        if config.decimation < 1:
            raise ValueError(f"G1SonicWBCTask '{name}' requires decimation >= 1")

        self._name = name
        self._config = config
        self._adapter = adapter
        self._joint_names_list = list(config.joint_names)
        self._joint_names_set = frozenset(config.joint_names)

        self._pipeline = SonicPipeline(
            encoder_path=config.encoder_onnx,
            decoder_path=config.decoder_onnx,
            planner_path=config.planner_onnx,
        )

        self._default_29 = DEFAULT_ANGLES_DDS.copy()

        self._tick_count = 0
        self._last_targets: list[float] | None = None

        # Last-known-good caches; same missing-joint policy as the GR00T
        # task - never substitute 0.0, a zero pose reads as "legs straight"
        # and provokes a snap-back.
        self._cached_q_29 = self._default_29.copy()
        self._cached_dq_29 = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._state_seen = False

        self._active = False
        self._armed = False
        self._arming = False
        self._arm_pending = False
        self._dry_run = bool(config.auto_dry_run)
        self._arming_duration = 0.0
        self._arming_start_t = 0.0
        self._ramp_start: NDArray[np.float32] | None = None
        self._last_dry_run_log_t = 0.0
        self._last_diag_log_t = 0.0

        self._cmd_lock = threading.Lock()
        self._cmd = np.zeros(3, dtype=np.float32)
        self._last_cmd_time = 0.0

    # -- ControlTask protocol ----------------------------------------------

    def claim(self) -> ResourceClaim:
        return ResourceClaim(
            joints=self._joint_names_set,
            priority=self._config.priority,
            mode=ControlMode.SERVO_POSITION,
        )

    def is_active(self) -> bool:
        return self._active

    def _refresh_state_caches(self, state: CoordinatorState) -> bool:
        all_present = True
        for i, jname in enumerate(self._joint_names_list):
            pos = state.joints.get_position(jname)
            vel = state.joints.get_velocity(jname)
            if pos is None:
                all_present = False
            else:
                self._cached_q_29[i] = pos
            if vel is None:
                all_present = False
            else:
                self._cached_dq_29[i] = vel
        if all_present:
            self._state_seen = True
        return all_present

    def compute(self, state: CoordinatorState) -> JointCommandOutput | None:
        if not self._active:
            return None

        fresh = self._refresh_state_caches(state)
        if not self._state_seen and not fresh:
            return None

        current_29 = self._cached_q_29.copy()

        if self._arm_pending:
            self._ramp_start = current_29.copy()
            self._arming_start_t = state.t_now
            if self._arming_duration > 0.0:
                self._arming = True
                self._armed = False
                logger.info(
                    "G1SonicWBCTask arming: ramp to SONIC default pose",
                    task=self._name,
                    ramp_seconds=self._arming_duration,
                )
            else:
                self._arming = False
                self._armed = True
                self._reset_policy_state()
                logger.info("G1SonicWBCTask armed (no ramp)", task=self._name)
            self._arm_pending = False

        if not self._armed and not self._arming:
            self._last_targets = current_29.tolist()
            return JointCommandOutput(
                joint_names=self._joint_names_list,
                positions=self._last_targets,
                mode=ControlMode.SERVO_POSITION,
            )

        if self._arming:
            assert self._ramp_start is not None
            elapsed = state.t_now - self._arming_start_t
            alpha = (
                1.0 if self._arming_duration <= 0.0 else min(1.0, elapsed / self._arming_duration)
            )
            target = self._ramp_start + alpha * (self._default_29 - self._ramp_start)
            self._last_targets = target.tolist()
            if alpha >= 1.0:
                self._arming = False
                self._armed = True
                self._reset_policy_state()
                logger.info(
                    "G1SonicWBCTask ramp complete - policy armed",
                    task=self._name,
                    mode="dry-run" if self._dry_run else "live",
                )
            return JointCommandOutput(
                joint_names=self._joint_names_list,
                positions=self._last_targets,
                mode=ControlMode.SERVO_POSITION,
            )

        # Armed: run the pipeline at the decimated rate.
        self._tick_count += 1
        if self._tick_count % self._config.decimation != 0:
            if self._dry_run or self._last_targets is None:
                return None
            return JointCommandOutput(
                joint_names=self._joint_names_list,
                positions=self._last_targets,
                mode=ControlMode.SERVO_POSITION,
            )

        q_29 = self._cached_q_29.copy()
        dq_29 = self._cached_dq_29.copy()

        if state.imu:
            imu = next(iter(state.imu.values()))
        else:
            imu = self._adapter.read_imu()
        gyro = np.asarray(imu.gyroscope, dtype=np.float32)
        quat = np.asarray(imu.quaternion, dtype=np.float64)
        gravity = self._projected_gravity(imu.quaternion)

        with self._cmd_lock:
            if (
                self._config.timeout > 0.0
                and self._last_cmd_time > 0.0
                and (state.t_now - self._last_cmd_time) > self._config.timeout
            ):
                cmd = np.zeros(3, dtype=np.float32)
            else:
                cmd = self._cmd.copy()
        self._pipeline.set_velocity(float(cmd[0]), float(cmd[1]), float(cmd[2]))

        targets_29 = self._pipeline.step(
            q_dds=q_29,
            dq_dds=dq_29,
            base_quat_wxyz=quat,
            gyro_body=gyro,
            gravity_body=gravity,
        )
        self._last_targets = targets_29.tolist()

        if (state.t_now - self._last_diag_log_t) >= 5.0:
            logger.info("G1SonicWBCTask", task=self._name, **self._pipeline.snapshot())
            self._last_diag_log_t = state.t_now

        if self._dry_run:
            if (state.t_now - self._last_dry_run_log_t) >= 1.0:
                max_delta = float(np.max(np.abs(targets_29 - current_29)))
                logger.info(
                    "G1SonicWBCTask DRY-RUN",
                    task=self._name,
                    max_dq_rad=max_delta,
                )
                self._last_dry_run_log_t = state.t_now
            return None

        return JointCommandOutput(
            joint_names=self._joint_names_list,
            positions=self._last_targets,
            mode=ControlMode.SERVO_POSITION,
        )

    def on_preempted(self, by_task: str, joints: frozenset[str]) -> None:
        if joints & self._joint_names_set:
            logger.warning(
                "G1SonicWBCTask preempted",
                task=self._name,
                by_task=by_task,
                joints=joints,
            )

    # -- command inputs ------------------------------------------------------

    def set_velocity_command(self, vx: float, vy: float, yaw_rate: float, t_now: float) -> None:
        with self._cmd_lock:
            self._cmd[:] = [vx, vy, yaw_rate]
            self._last_cmd_time = t_now

    def on_twist_command(self, msg: Twist, t_now: float) -> None:
        self.set_velocity_command(
            float(msg.linear.x), float(msg.linear.y), float(msg.angular.z), t_now
        )

    def set_locomotion_mode(self, mode: int | str | None) -> dict[str, Any]:
        """Force one of the 27 GEAR locomotion modes; None = speed-auto."""
        applied = self._pipeline.set_mode(mode)
        logger.info(
            "G1SonicWBCTask locomotion mode",
            task=self._name,
            requested=mode,
            applied=applied,
        )
        return {"mode_override": applied}

    def list_locomotion_modes(self) -> dict[str, int]:
        return dict(LOCOMOTION_MODES)

    def set_base_height(self, height: float) -> None:
        self._pipeline.set_base_height(float(height))

    def set_upper_body(self, positions: list[float]) -> bool:
        """14 arm-joint encoder hints, DDS order (indices 15-28)."""
        if len(positions) != 14:
            raise ValueError(f"set_upper_body expects 14 values, got {len(positions)}")
        self._pipeline.set_upper_body(np.asarray(positions, dtype=np.float32))
        return True

    def clear_upper_body(self) -> None:
        self._pipeline.set_upper_body(DEFAULT_ANGLES_DDS[15:].copy())

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._active = True
        self._armed = False
        self._arming = False
        self._arm_pending = False
        self._dry_run = bool(self._config.auto_dry_run)
        self._last_targets = None
        self._reset_policy_state()
        with self._cmd_lock:
            self._cmd[:] = 0.0
            self._last_cmd_time = 0.0
        logger.info(
            "G1SonicWBCTask started",
            task=self._name,
            armed=False,
            dry_run=self._dry_run,
        )
        if self._config.auto_arm:
            self.arm(self._config.default_ramp_seconds)

    def stop(self) -> None:
        self._active = False
        self._armed = False
        self._arming = False
        self._arm_pending = False
        self._last_targets = None
        logger.info("G1SonicWBCTask stopped", task=self._name)

    def arm(self, ramp_seconds: float | None = None) -> bool:
        if not self._active:
            logger.warning("G1SonicWBCTask arm() before start(); ignoring", task=self._name)
            return False
        if self._armed or self._arming or self._arm_pending:
            return False
        ramp = ramp_seconds if ramp_seconds is not None else self._config.default_ramp_seconds
        self._arming_duration = max(0.0, float(ramp))
        self._arm_pending = True
        logger.info(
            "G1SonicWBCTask arm requested",
            task=self._name,
            ramp_seconds=self._arming_duration,
        )
        return True

    def disarm(self) -> bool:
        if not self._armed and not self._arming and not self._arm_pending:
            return False
        self._armed = False
        self._arming = False
        self._arm_pending = False
        self._ramp_start = None
        self._reset_policy_state()
        logger.info("G1SonicWBCTask disarmed (holding current pose)", task=self._name)
        return True

    def reset_runtime_state(self, reactivate: bool | None = None) -> bool:
        was_armed = self._armed or self._arming or self._arm_pending
        should_reactivate = was_armed if reactivate is None else bool(reactivate)

        self._armed = False
        self._arming = False
        self._arm_pending = False
        self._ramp_start = None
        self._arming_start_t = 0.0
        self._last_targets = None
        self._state_seen = False
        self._cached_q_29[:] = self._default_29
        self._cached_dq_29[:] = 0.0
        self._reset_policy_state()
        with self._cmd_lock:
            self._cmd[:] = 0.0
            self._last_cmd_time = 0.0

        if self._active and should_reactivate:
            self._arming_duration = 0.0
            self._arm_pending = True

        logger.info(
            "G1SonicWBCTask runtime state reset",
            task=self._name,
            reactivate=should_reactivate,
        )
        return True

    def set_dry_run(self, enabled: bool) -> None:
        new_val = bool(enabled)
        if new_val == self._dry_run:
            return
        self._dry_run = new_val
        self._last_dry_run_log_t = 0.0
        logger.info("G1SonicWBCTask dry_run changed", task=self._name, dry_run=new_val)

    def state_snapshot(self) -> dict[str, Any]:
        snap = {
            "active": self._active,
            "armed": self._armed,
            "arming": self._arming,
            "arm_pending": self._arm_pending,
            "dry_run": self._dry_run,
            "arming_duration": self._arming_duration,
        }
        snap.update(self._pipeline.snapshot())
        return snap

    # -- internal ------------------------------------------------------------

    def _reset_policy_state(self) -> None:
        self._pipeline.reset()
        self._tick_count = 0

    @staticmethod
    def _projected_gravity(quaternion: tuple[float, ...]) -> NDArray[np.float32]:
        w, x, y, z = quaternion
        gx = 2.0 * (-x * z + w * y)
        gy = 2.0 * (-y * z - w * x)
        gz = -(w * w - x * x - y * y + z * z)
        return np.array([gx, gy, gz], dtype=np.float32)


class G1SonicWBCTaskParams(BaseConfig):
    encoder_onnx: str | Path
    decoder_onnx: str | Path
    planner_onnx: str | Path
    hardware_id: str
    auto_arm: bool = False
    auto_dry_run: bool = False
    default_ramp_seconds: float = 10.0
    decimation: int | None = None


def create_task(cfg: Any, hardware: Any) -> G1SonicWBCTask:
    params = G1SonicWBCTaskParams.model_validate(cfg.params)
    hw = hardware.get(params.hardware_id) if hardware else None
    if hw is None:
        raise ValueError(
            f"G1SonicWBCTask {cfg.name!r} references unknown hardware "
            f"{params.hardware_id!r}. Declare the hardware before the task "
            f"in the blueprint config."
        )
    if not isinstance(hw, ConnectedWholeBody):
        raise TypeError(
            f"G1SonicWBCTask {cfg.name!r} requires a WHOLE_BODY hardware "
            f"component for {params.hardware_id!r}, got {type(hw).__name__}."
        )

    kwargs: dict[str, Any] = dict(
        encoder_onnx=params.encoder_onnx,
        decoder_onnx=params.decoder_onnx,
        planner_onnx=params.planner_onnx,
        joint_names=cfg.joint_names,
        priority=cfg.priority,
        auto_arm=params.auto_arm,
        auto_dry_run=params.auto_dry_run,
        default_ramp_seconds=params.default_ramp_seconds,
    )
    if params.decimation is not None:
        kwargs["decimation"] = params.decimation
    return G1SonicWBCTask(
        cfg.name,
        G1SonicWBCTaskConfig(**kwargs),
        adapter=hw.adapter,
    )
