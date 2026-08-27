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

"""Generate reproducible G1 command-response JSON and figures from mem2."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from dimos.robot.unitree.g1.characterization.comparison import (
    AlignedTrajectories,
    compare_trajectories,
    read_simulation,
)
from dimos.robot.unitree.g1.characterization.mujoco_model import (
    G1_BASELINE_MUJOCO_PHYSICS,
    G1MujocoPhysics,
)
from dimos.robot.unitree.g1.characterization.plant import (
    PLANT_CLIP_RANGE_S,
    PlantScore,
    groot_command_contract,
    plant_health,
    sample_replay_plans,
    score_prediction,
)
from dimos.robot.unitree.g1.characterization.plant_mujoco import G1MujocoBackend
from dimos.robot.unitree.g1.characterization.recording import (
    G1Recording,
    measured_pelvis_pose,
    read_plant_recording,
    read_recording,
)
from dimos.robot.unitree.g1.characterization.render import render_comparison
from dimos.robot.unitree.g1.characterization.response import (
    CharacterizationResult,
    StepFit,
    body_velocity,
    characterize,
    direction_results,
    directional_transient_errors,
    fit_trajectory_steps,
)
from dimos.utils.data import get_data


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n")


def _plot_timeseries(recording: G1Recording, out_path: Path) -> None:
    command_t_s = recording.command_t_s - recording.command_t_s[0]
    odom_t_s, twist = body_velocity(recording)
    odom_t_s = odom_t_s - recording.command_t_s[0]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    labels = (("vx", "m/s"), ("vy", "m/s"), ("wz", "rad/s"))
    for axis, (name, unit) in enumerate(labels):
        axes[axis].step(
            command_t_s, recording.command_body_twist[:, axis], where="post", label="command"
        )
        axes[axis].plot(odom_t_s, twist[:, axis], lw=1.0, label="Point-LIO pose derivative")
        axes[axis].set_ylabel(f"{name} ({unit})")
        axes[axis].grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("recording time (s)")
    fig.suptitle("G1 GR00T hardware command response — Point-LIO reference")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_envelope(result: CharacterizationResult, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, direction in zip(axes.flat, result.directions, strict=True):
        steps = [step for step in result.steps if step.direction == direction.direction]
        ax.scatter([step.command for step in steps], [step.settled_speed for step in steps])
        ax.set_title(direction.direction)
        ax.set_xlabel(f"command ({direction.unit})")
        ax.set_ylabel(f"achieved ({direction.unit})")
        ax.grid(alpha=0.25)
    fig.suptitle("G1 GR00T directional envelope — ceiling is unverified without saturation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_envelope_comparison(
    hardware: CharacterizationResult,
    simulated_steps: list[StepFit],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, direction in zip(axes.flat, hardware.directions, strict=True):
        real = [step for step in hardware.steps if step.direction == direction.direction]
        simulated = [step for step in simulated_steps if step.direction == direction.direction]
        ax.scatter(
            [step.command for step in real],
            [step.settled_speed for step in real],
            label="hardware Point-LIO",
        )
        ax.scatter(
            [step.command for step in simulated],
            [step.settled_speed for step in simulated],
            marker="x",
            label="SIMULATED ground truth",
        )
        ax.set_title(direction.direction)
        ax.set_xlabel(f"command ({direction.unit})")
        ax.set_ylabel(f"achieved ({direction.unit})")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("G1 GR00T directional response — hardware vs actual MuJoCo replay")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_trajectory(recording: G1Recording, out_path: Path) -> None:
    _, world_p_pelvis_m, world_q_pelvis_xyzw, _ = measured_pelvis_pose(recording)
    yaw = Rotation.from_quat(world_q_pelvis_xyzw).as_euler("xyz")[:, 2]
    x = world_p_pelvis_m[:, 0]
    y = world_p_pelvis_m[:, 1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y, lw=1.5, label="Point-LIO world_T_pelvis")
    stride = max(1, len(x) // 60)
    ax.quiver(x[::stride], y[::stride], np.cos(yaw[::stride]), np.sin(yaw[::stride]), scale=25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("G1 hardware trajectory — outlier-filtered Point-LIO reference")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_comparison(aligned: AlignedTrajectories, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(
        aligned.hardware_p_m[:, 0],
        aligned.hardware_p_m[:, 1],
        label="hardware Point-LIO reference",
    )
    axes[0].plot(aligned.sim_p_m[:, 0], aligned.sim_p_m[:, 1], label="SIMULATED ground truth")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("initial-pelvis x (m)")
    axes[0].set_ylabel("initial-pelvis y (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    position_error_m = np.linalg.norm(aligned.sim_p_m[:, :2] - aligned.hardware_p_m[:, :2], axis=1)
    yaw_error_rad = np.arctan2(
        np.sin(aligned.sim_yaw_rad - aligned.hardware_yaw_rad),
        np.cos(aligned.sim_yaw_rad - aligned.hardware_yaw_rad),
    )
    axes[1].plot(aligned.t_s, position_error_m, label="planar position error (m)")
    axes[1].plot(aligned.t_s, np.abs(yaw_error_rad), label="absolute yaw error (rad)")
    axes[1].set_xlabel("command time (s)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("G1 hardware reference vs actual GR00T MuJoCo replay")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_plant_scores(scores: list[PlantScore], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    values = (
        ("joint q RMSE (rad)", [score.joint_q_rmse_rad for score in scores]),
        ("joint dq RMSE (rad/s)", [score.joint_dq_rmse_rad_s for score in scores]),
        ("joint torque RMSE (N m)", [score.joint_tau_rmse_nm for score in scores]),
        ("root position RMSE (m)", [score.root_position_rmse_m for score in scores]),
        ("root rotation RMSE (rad)", [score.root_rotation_rmse_rad for score in scores]),
    )
    for axis, (label, score_values) in zip(axes.flatten()[:-1], values, strict=True):
        axis.bar(np.arange(len(scores)), score_values)
        axis.set_xlabel("seeded segment")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle("SIMULATED baseline G1 plant vs hardware clips")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _groot_model_hashes() -> dict[str, str]:
    model_dir = Path(get_data("groot"))
    return {name: _sha256(model_dir / name) for name in ("balance.onnx", "walk.onnx")}


def _plant_report(
    db_path: Path,
    recording: G1Recording,
    out_dir: Path,
    *,
    git_sha: str,
    git_dirty: bool,
    recording_sha256: str,
    command: str | None,
    n_segments: int,
    segment_duration_s: float,
    physics: G1MujocoPhysics | None,
) -> None:
    plant = read_plant_recording(db_path)
    health = plant_health(plant)
    contract = groot_command_contract(plant)
    if contract.status != "pass":
        raise ValueError(f"recorded motor commands do not match GR00T: {asdict(contract)}")
    plans = sample_replay_plans(
        plant,
        recording,
        n_segments=n_segments,
        segment_duration_s=segment_duration_s,
        seed=0,
    )
    backend = G1MujocoBackend(plant.motor_names, physics)
    predictions = [backend.rollout(plan) for plan in plans]
    scores = [
        score_prediction(plan, prediction)
        for plan, prediction in zip(plans, predictions, strict=True)
    ]
    artifact = {
        "schema_version": 1,
        "provenance": {
            "hardware_recording_sha256": recording_sha256,
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "command": command,
            "seed": 0,
            "groot_model_sha256": _groot_model_hashes(),
            "mujoco_model_sha256": backend.model_sha256(),
            "simulation_label": "SIMULATED open-loop MuJoCo plant",
        },
        "method": {
            "n_segments": n_segments,
            "segment_duration_s": segment_duration_s,
            "physics_dt_s": plans[0].physics_dt_s,
            "reinitialization_clip_range_s": PLANT_CLIP_RANGE_S,
            "physics_override": asdict(physics) if physics is not None else None,
        },
        "health": asdict(health),
        "groot_command_contract": asdict(contract),
        "plans": [
            {
                "seed": plan.seed,
                "start_epoch_s": float(plan.step_t_s[0]),
                "duration_s": float(plan.step_t_s[-1] - plan.step_t_s[0]),
                "reinitializations": int(np.sum(plan.reinitialize)),
            }
            for plan in plans
        ],
        "scores": [asdict(score) for score in scores],
    }
    _write_json(out_dir / "plant_grounding.json", artifact)
    _plot_plant_scores(scores, out_dir / "plant_grounding.png")


def write_report(
    db_path: Path,
    out_dir: Path,
    *,
    simulation_db: Path | None = None,
    command: str | None = None,
    render_video: bool = False,
    video_playback_speed: float = 6.0,
    plant_segments: int = 8,
    plant_segment_duration_s: float = 8.0,
    plant_physics: G1MujocoPhysics | None = None,
) -> CharacterizationResult:
    """Analyze one real recording and write its evidence bundle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    recording = read_recording(db_path)
    result = characterize(recording)
    artifact = {
        "schema_version": 1,
        "provenance": {
            "recording": str(db_path.resolve()),
            "recording_size_bytes": db_path.stat().st_size,
            "recording_sha256": _sha256(db_path),
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
            "seed": result.seed,
            "command": command,
            "reference": "Point-LIO world_T_pelvis; measured reference, not ground truth",
        },
        "result": asdict(result),
    }
    _write_json(out_dir / "g1_groot_characterization.json", artifact)
    _plot_timeseries(recording, out_dir / "command_response.png")
    _plot_envelope(result, out_dir / "directional_envelope.png")
    _plot_trajectory(recording, out_dir / "hardware_trajectory.png")
    _plant_report(
        db_path,
        recording,
        out_dir,
        git_sha=str(artifact["provenance"]["git_sha"]),
        git_dirty=bool(artifact["provenance"]["git_dirty"]),
        recording_sha256=str(artifact["provenance"]["recording_sha256"]),
        command=command,
        n_segments=plant_segments,
        segment_duration_s=plant_segment_duration_s,
        physics=plant_physics,
    )
    if simulation_db is not None:
        simulation = read_simulation(simulation_db)
        aligned, comparison = compare_trajectories(recording, simulation)
        simulation_steps = fit_trajectory_steps(
            simulation.command_t_s,
            simulation.command_body_twist,
            simulation.sim_t_s,
            simulation.sim_world_p_pelvis_m,
            simulation.sim_world_q_pelvis_xyzw,
        )
        simulation_directions = direction_results(simulation_steps)
        transient_errors = directional_transient_errors(recording, simulation)
        comparison_artifact = {
            "schema_version": 1,
            "provenance": {
                "hardware_recording_sha256": artifact["provenance"]["recording_sha256"],
                "simulation_recording": str(simulation_db.resolve()),
                "simulation_recording_sha256": _sha256(simulation_db),
                "git_sha": artifact["provenance"]["git_sha"],
                "git_dirty": artifact["provenance"]["git_dirty"],
                "command": command,
                "groot_model_sha256": _groot_model_hashes(),
                "simulation_label": "SIMULATED MuJoCo root pose ground truth",
            },
            "trajectory_comparison": comparison.to_dict(),
            "simulated_directional_response": {
                "directions": [asdict(direction) for direction in simulation_directions],
                "steps": [asdict(step) for step in simulation_steps],
            },
            "baseline_subtracted_transient_error": {
                "directions": [asdict(error) for error in transient_errors],
                "mean_nrmse": float(np.mean([error.nrmse for error in transient_errors])),
                "worst_nrmse": float(np.max([error.nrmse for error in transient_errors])),
            },
        }
        _plot_comparison(aligned, out_dir / "hardware_vs_sim.png")
        _plot_envelope_comparison(
            result, simulation_steps, out_dir / "hardware_vs_sim_envelope.png"
        )
        if render_video:
            video = render_comparison(
                recording,
                simulation,
                out_dir / "hardware_vs_sim.mp4",
                playback_speed=video_playback_speed,
            )
            comparison_artifact["video"] = {
                **asdict(video),
                "path": str(video.path.resolve()),
            }
        _write_json(out_dir / "hardware_vs_sim.json", comparison_artifact)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-speed", type=float, default=6.0)
    parser.add_argument("--plant-segments", type=int, default=8)
    parser.add_argument("--plant-segment-duration", type=float, default=8.0)
    parser.add_argument("--baseline-plant", action="store_true")
    args = parser.parse_args()
    if args.video and args.simulation is None:
        parser.error("--video needs --simulation")
    command = " ".join(shlex.quote(part) for part in sys.argv)
    result = write_report(
        args.recording,
        args.out,
        simulation_db=args.simulation,
        command=command,
        render_video=args.video,
        video_playback_speed=args.video_speed,
        plant_segments=args.plant_segments,
        plant_segment_duration_s=args.plant_segment_duration,
        plant_physics=G1_BASELINE_MUJOCO_PHYSICS if args.baseline_plant else None,
    )
    print(json.dumps(asdict(result.health), indent=2))


if __name__ == "__main__":
    main()
