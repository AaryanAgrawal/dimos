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

"""Unitree G1 SONIC (GEAR-SONIC) whole-body-control blueprint.

Unified 29-DOF policy: planner + encoder + decoder. All 27 GEAR locomotion
modes are reachable at runtime through the coordinator RPC surface:

    coordinator.task_invoke("sonic_wbc", "set_locomotion_mode",
                            {"mode": "HAPPY_DANCE_WALK"})

Usage:
    dimos --simulation mujoco run unitree-g1-sonic-wbc    # sim
    dimos run unitree-g1-sonic-wbc                        # real hardware

Real hardware note: SONIC uses armature-derived PD gains (SONIC_KP/KD),
NOT the GR00T gain table. Never run this blueprint while the C++
g1_deploy_onnx_ref binary owns rt/lowcmd.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from dimos.control.components import HardwareComponent, HardwareType
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.control.tasks.g1_groot_wbc_task.g1_groot_wbc_task import g1_joints
from dimos.control.tasks.g1_sonic_wbc_task.sonic_pipeline import SONIC_KD, SONIC_KP
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.core.stream import Out
from dimos.core.transport import LCMTransport
from dimos.hardware.whole_body.spec import WholeBodyConfig
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.utils.data import LfsPath
from dimos.visualization.vis_module import vis_module

# SONIC model files. Default to the local GR00T-WholeBodyControl checkout;
# override with SONIC_MODEL_DIR / SONIC_PLANNER_PATH for other machines.
_SONIC_RELEASE_DIR = Path(
    os.environ.get(
        "SONIC_MODEL_DIR",
        Path.home() / "Desktop/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release",
    )
)
_SONIC_PLANNER_PATH = Path(
    os.environ.get(
        "SONIC_PLANNER_PATH",
        Path.home()
        / "Desktop/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
    )
)

_MJCF_PATH = LfsPath("mujoco_sim/g1_gear_wbc.xml")
_G1_NUM_MOTORS = len(g1_joints)
_cmd_vel_topic = "/cmd_vel" if global_config.simulation else "/g1/cmd_vel"

_adapter_address: str | Path

if global_config.simulation and global_config.simulation != "mujoco":
    raise ValueError("unitree-g1-sonic-wbc only supports --simulation mujoco")

if global_config.simulation == "mujoco":
    from dimos.simulation.engines.mujoco_sim_module import MujocoSimModule
    from dimos.simulation.engines.robot_sim_binding import (
        RobotSimSpec,
        mjcf_joint_names_from_hardware,
    )

    _g1_sim_joints = tuple(g1_joints)
    _g1_sim_spec = RobotSimSpec(
        robot_id="g1",
        hardware_joints=_g1_sim_joints,
        root_body_names=("pelvis",),
        root_joint_names=("floating_base_joint",),
        require_floating_base=True,
        model_joint_names=mjcf_joint_names_from_hardware(_g1_sim_joints),
        imu_gyro_names=(
            "imu-pelvis-angular-velocity",
            "imu-torso-angular-velocity",
            "imu-angular-velocity",
            "gyro_pelvis",
            "imu_gyro",
        ),
        imu_accel_names=(
            "imu-pelvis-linear-acceleration",
            "imu-torso-linear-acceleration",
            "imu-linear-acceleration",
            "accelerometer_pelvis",
            "imu_accel",
        ),
        require_imu=True,
    )

    _backend = MujocoSimModule.blueprint(
        address=_MJCF_PATH,
        headless=True,
        dof=_G1_NUM_MOTORS,
        inject_legacy_assets=True,
        robot_sim_spec=_g1_sim_spec,
    )
    _adapter_type = "sim_mujoco_g1"
    _adapter_address = _MJCF_PATH
    _tick_rate = 50.0
    _auto_arm = True
    _auto_dry_run = False
    _default_ramp_seconds = 0.0
    _decimation = 1
    _n_workers = 2
else:
    from dimos.robot.unitree.g1.wholebody_connection import G1WholeBodyConnection

    _backend = G1WholeBodyConnection.blueprint(release_sport_mode=True)
    _adapter_type = "transport_lcm"
    _adapter_address = ""
    _tick_rate = 100.0
    _auto_arm = False
    _auto_dry_run = True
    _default_ramp_seconds = 10.0
    _decimation = 2  # 100 Hz tick / 2 = 50 Hz policy
    _n_workers = 10


class _G1SonicCoordinator(ControlCoordinator):
    g1_joints: Out[JointState]


_coordinator = _G1SonicCoordinator.blueprint(
    instance_name="ControlCoordinator",
    publish_robot_joint_states=True,
    tick_rate=_tick_rate,
    hardware=[
        HardwareComponent(
            hardware_id="g1",
            hardware_type=HardwareType.WHOLE_BODY,
            joints=g1_joints,
            adapter_type=_adapter_type,
            address=_adapter_address,
            wb_config=WholeBodyConfig(kp=tuple(SONIC_KP), kd=tuple(SONIC_KD)),
        ),
    ],
    tasks=[
        TaskConfig(
            name="sonic_wbc",
            type="g1_sonic_wbc",
            joint_names=g1_joints,
            priority=50,
            auto_start=True,
            params={
                "encoder_onnx": str(_SONIC_RELEASE_DIR / "model_encoder.onnx"),
                "decoder_onnx": str(_SONIC_RELEASE_DIR / "model_decoder.onnx"),
                "planner_onnx": str(_SONIC_PLANNER_PATH),
                "hardware_id": "g1",
                "auto_arm": _auto_arm,
                "auto_dry_run": _auto_dry_run,
                "default_ramp_seconds": _default_ramp_seconds,
                "decimation": _decimation,
            },
        ),
    ],
).transports(
    {
        ("joint_command", JointState): LCMTransport("/g1/joint_command", JointState),
        ("g1_joints", JointState): LCMTransport("/g1/joints", JointState),
        ("cmd_vel", Twist): LCMTransport(_cmd_vel_topic, Twist),
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("imu", Imu): LCMTransport("/g1/imu", Imu),
        ("motor_command", MotorCommandArray): LCMTransport("/g1/motor_command", MotorCommandArray),
    }
)

_remappings = [(_G1SonicCoordinator, "twist_command", "cmd_vel")]

unitree_g1_sonic_wbc = (
    autoconnect(_backend, _coordinator, vis_module(viewer_backend=global_config.viewer))
    .remappings(cast("Any", _remappings))
    .global_config(robot_model="unitree_g1", n_workers=_n_workers)
)
