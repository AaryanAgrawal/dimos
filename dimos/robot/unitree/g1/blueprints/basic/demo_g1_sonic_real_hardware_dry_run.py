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

"""Replay G1 SONIC real-hardware dry-run targets in MuJoCo without physics."""

from __future__ import annotations

import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from dimos.control.tasks.g1_groot_wbc_task.g1_groot_wbc_task import g1_joints
from dimos.control.tasks.g1_sonic_wbc_task.hardware_dry_run import (
    RECORDING_FILENAME,
    G1SonicHardwareDryRun,
    display_joint_positions,
    load_hardware_dry_run,
)
from dimos.simulation.engines.mujoco_engine import MujocoEngine
from dimos.simulation.engines.robot_sim_binding import mjcf_joint_names_from_hardware
from dimos.simulation.mujoco.model import get_assets
from dimos.utils.data import LfsPath

_MODEL_PATH = LfsPath("mujoco_sim/g1_gear_wbc.xml")
_PHASE_LABEL = {
    "observe": "UNITREE RUN MODE — MEASURED POSE",
    "takeover": "SONIC TAKEOVER — PROPOSED, NOT COMMANDED",
    "idle": "SONIC IDLE — PROPOSED, NOT COMMANDED",
    "walk": "SONIC WALK 0.20 m/s — PROPOSED, NOT COMMANDED",
}


def _motor_qpos_addresses(model: mujoco.MjModel) -> list[int]:
    names = mjcf_joint_names_from_hardware(tuple(g1_joints))
    addresses: list[int] = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing G1 joint {name!r}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def _set_viewer_text(viewer: Any, phase: str, elapsed_s: float) -> None:
    viewer.set_texts(
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "REAL G1 DATA DRY RUN\nKINEMATIC REPLAY — NO PHYSICS / NO MOTOR COMMANDS",
            f"{_PHASE_LABEL[phase]}\nTIME {elapsed_s:05.2f} s",
        )
    )


def _play_once(
    viewer: Any,
    engine: MujocoEngine,
    recording: G1SonicHardwareDryRun,
    positions_rad: NDArray[np.float64],
    qpos_addresses: list[int],
) -> bool:
    started_at = time.monotonic()
    initial_timestamp_s = float(recording.timestamps_s[0])
    for index, timestamp_s in enumerate(recording.timestamps_s):
        if not viewer.is_running():
            return False
        elapsed_s = float(timestamp_s) - initial_timestamp_s
        while time.monotonic() - started_at < elapsed_s:
            time.sleep(0.001)
        phase = str(recording.phase[index])
        _set_viewer_text(viewer, phase, elapsed_s)
        with viewer.lock():
            engine.data.qpos[qpos_addresses] = positions_rad[index]
            engine.data.qvel[:] = 0.0
            mujoco.mj_forward(engine.model, engine.data)
        viewer.sync()
    return True


def main() -> None:
    """Open the fixed-base kinematic replay and loop until its viewer closes."""
    recording = load_hardware_dry_run()
    positions_rad = display_joint_positions(recording)
    engine = MujocoEngine(
        config_path=_MODEL_PATH,
        headless=True,
        assets=get_assets(),
    )
    qpos_addresses = _motor_qpos_addresses(engine.model)
    print(
        f"Playing {RECORDING_FILENAME}: real G1 data, motor commands disabled, "
        "MuJoCo physics disabled"
    )
    with mujoco.viewer.launch_passive(engine.model, engine.data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.55]
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -12.0
        while viewer.is_running() and _play_once(
            viewer, engine, recording, positions_rad, qpos_addresses
        ):
            time.sleep(1.0)


if __name__ == "__main__":
    main()
