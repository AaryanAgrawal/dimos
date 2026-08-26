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

"""Load the recorded G1 SONIC real-hardware dry run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dimos.control.tasks.g1_sonic_wbc_task.sonic_pipeline import NUM_JOINTS

RECORDING_FILENAME = "g1_sonic_real_hardware_dry_run_20260825.npz"
RECORDING_PATH = Path(__file__).parent / "test_data" / RECORDING_FILENAME
_PHASES = frozenset({"observe", "takeover", "idle", "walk"})


@dataclass(frozen=True)
class G1SonicHardwareDryRun:
    timestamps_s: NDArray[np.float64]
    phase: NDArray[Any]
    measured_position_rad: NDArray[np.float64]
    target_position_rad: NDArray[np.float64]
    source_samples: int
    replay_rate_hz: float
    source_git_rev: str
    source_phase_samples: dict[str, int]
    source_duration_s: float
    max_measured_joint_speed_rad_s: float
    max_measured_tilt_deg: float
    simulated: bool
    motor_commands_enabled: bool

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_s[-1] - self.timestamps_s[0])


def _scalar(data: Any, key: str) -> Any:
    return data[key].item()


def _validate(recording: G1SonicHardwareDryRun) -> None:
    samples = len(recording.timestamps_s)
    expected_joint_shape = (samples, NUM_JOINTS)
    if recording.measured_position_rad.shape != expected_joint_shape:
        raise ValueError(
            f"measured positions must have shape {expected_joint_shape}, "
            f"got {recording.measured_position_rad.shape}"
        )
    if recording.target_position_rad.shape != expected_joint_shape:
        raise ValueError(
            f"targets must have shape {expected_joint_shape}, "
            f"got {recording.target_position_rad.shape}"
        )
    if len(recording.phase) != samples or np.any(np.diff(recording.timestamps_s) <= 0.0):
        raise ValueError("recording phases and strictly increasing timestamps must align")
    if set(recording.phase.tolist()) != _PHASES:
        raise ValueError(f"recording phases must be {sorted(_PHASES)}")
    if recording.simulated or recording.motor_commands_enabled:
        raise ValueError("recording must be real hardware data captured without motor commands")


def load_hardware_dry_run(path: Path = RECORDING_PATH) -> G1SonicHardwareDryRun:
    """Load and validate the recorded real-hardware dry run."""
    with np.load(path, allow_pickle=False) as data:
        names = data["source_phase_names"].tolist()
        counts = data["source_phase_samples"].tolist()
        recording = G1SonicHardwareDryRun(
            timestamps_s=np.asarray(data["timestamps_s"], dtype=np.float64),
            phase=np.asarray(data["phase"]),
            measured_position_rad=np.asarray(data["joint_position_rad"], dtype=np.float64),
            target_position_rad=np.asarray(data["position_target_rad"], dtype=np.float64),
            source_samples=int(_scalar(data, "source_samples")),
            replay_rate_hz=float(_scalar(data, "replay_rate_hz")),
            source_git_rev=str(_scalar(data, "source_git_rev")),
            source_phase_samples=dict(zip(names, counts, strict=True)),
            source_duration_s=float(_scalar(data, "source_duration_s")),
            max_measured_joint_speed_rad_s=float(_scalar(data, "max_measured_joint_speed_rad_s")),
            max_measured_tilt_deg=float(_scalar(data, "max_measured_tilt_deg")),
            simulated=bool(_scalar(data, "simulated")),
            motor_commands_enabled=bool(_scalar(data, "motor_commands_enabled")),
        )
    _validate(recording)
    return recording


def display_joint_positions(recording: G1SonicHardwareDryRun) -> NDArray[np.float64]:
    """Show measured pose before takeover and proposed SONIC targets afterward."""
    positions = recording.target_position_rad.copy()
    observe = recording.phase == "observe"
    positions[observe] = recording.measured_position_rad[observe]
    return positions
