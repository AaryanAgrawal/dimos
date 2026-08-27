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

"""Fit measured G1 actuator physics, then gate it through the real GR00T task."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

from dimos.robot.unitree.g1.characterization.closed_loop import GrootClosedLoopRunner
from dimos.robot.unitree.g1.characterization.comparison import G1SimulationRecording
from dimos.robot.unitree.g1.characterization.mujoco_model import (
    G1_BASELINE_MUJOCO_PHYSICS,
    G1MujocoPhysics,
    apply_g1_mujoco_physics,
    build_g1_mujoco_spec,
)
from dimos.robot.unitree.g1.characterization.plant import (
    DirectionalPlantReplay,
    PlantScore,
    directional_replay_plans,
    score_prediction,
)
from dimos.robot.unitree.g1.characterization.plant_mujoco import G1MujocoBackend
from dimos.robot.unitree.g1.characterization.recording import (
    G1PlantRecording,
    G1Recording,
    read_plant_recording,
    read_recording,
)
from dimos.robot.unitree.g1.characterization.response import directional_transient_errors

# Bounds enclose stock plus both exploratory loops; contact stayed at the production optimum.
_BOUNDS = ((0.005, 0.08), (0.0001, 0.08), (0.1, 3.5))
_FOOT_SLIDE_FRICTION = 1.0
_FOOT_CONTACT_TIME_CONSTANT_S = 0.02
_TARGET_MEAN_NRMSE = 0.15
_TARGET_WORST_NRMSE = 0.20
_PLANT_CHANNELS = ("joint_q", "joint_dq", "joint_tau", "root_p", "root_R")


@dataclass(frozen=True)
class NormalizedPlantResidual:
    """Candidate residual divided by stock, balanced across channels and directions."""

    mean: float
    by_channel: dict[str, float]
    by_direction: dict[str, float]


@dataclass(frozen=True)
class FitEvaluation:
    """Open-loop and fast production-task screen for one physical candidate."""

    physics: G1MujocoPhysics
    train_normalized_residual: NormalizedPlantResidual
    validation_normalized_residual: NormalizedPlantResidual
    transient_nrmse_by_direction: dict[str, float]
    transient_mean_nrmse: float
    transient_worst_nrmse: float
    min_pelvis_height_m: float
    max_abs_roll_rad: float
    max_abs_pitch_rad: float
    screen_passed: bool


@dataclass(frozen=True)
class _FitReplays:
    train: tuple[DirectionalPlantReplay, ...]
    validation: tuple[DirectionalPlantReplay, ...]
    stock_train: NDArray[np.float64]
    stock_validation: NDArray[np.float64]


def _physics(values: NDArray[np.float64]) -> G1MujocoPhysics:
    return G1MujocoPhysics(
        leg_armature_kg_m2=float(values[0]),
        leg_damping_nm_s_rad=float(values[1]),
        leg_frictionloss_nm=float(values[2]),
        foot_slide_friction=_FOOT_SLIDE_FRICTION,
        foot_contact_time_constant_s=_FOOT_CONTACT_TIME_CONSTANT_S,
    )


def _score_vector(score: PlantScore) -> NDArray[np.float64]:
    return np.asarray(
        [
            score.joint_q_rmse_rad,
            score.joint_dq_rmse_rad_s,
            score.joint_tau_rmse_nm,
            score.root_position_rmse_m,
            score.root_rotation_rmse_rad,
        ]
    )


def _plant_scores(
    backend: G1MujocoBackend,
    replays: tuple[DirectionalPlantReplay, ...],
) -> NDArray[np.float64]:
    return np.asarray(
        [_score_vector(score_prediction(item.plan, backend.rollout(item.plan))) for item in replays]
    )


def _normalized_residual(
    backend: G1MujocoBackend,
    replays: tuple[DirectionalPlantReplay, ...],
    stock_scores: NDArray[np.float64],
) -> NormalizedPlantResidual:
    ratios = _plant_scores(backend, replays) / stock_scores
    directions = tuple(dict.fromkeys(item.direction for item in replays))
    return NormalizedPlantResidual(
        float(np.mean(ratios)),
        dict(zip(_PLANT_CHANNELS, np.mean(ratios, axis=0).tolist(), strict=True)),
        {
            direction: float(np.mean(ratios[[item.direction == direction for item in replays]]))
            for direction in directions
        },
    )


def _search_open_loop(
    plant: G1PlantRecording,
    train: tuple[DirectionalPlantReplay, ...],
    stock_train: NDArray[np.float64],
    *,
    seed: int,
    maxiter: int,
    popsize: int,
) -> NDArray[np.float64]:
    def objective(values: NDArray[np.float64]) -> float:
        backend = G1MujocoBackend(plant.motor_names, _physics(values))
        return _normalized_residual(backend, train, stock_train).mean

    # Seeded bounded global search: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    result = differential_evolution(
        objective,
        _BOUNDS,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=False,
        updating="immediate",
    )
    return np.asarray(result.x)


def _stability(simulation: G1SimulationRecording) -> tuple[float, float, float]:
    from scipy.spatial.transform import Rotation

    position_m = simulation.sim_world_p_pelvis_m
    euler_rad = Rotation.from_quat(simulation.sim_world_q_pelvis_xyzw).as_euler("xyz")
    return (
        float(np.min(position_m[:, 2])),
        float(np.max(np.abs(euler_rad[:, 0]))),
        float(np.max(np.abs(euler_rad[:, 1]))),
    )


def _accepted(
    train_residual: NormalizedPlantResidual,
    validation_residual: NormalizedPlantResidual,
    transient_nrmse: NDArray[np.float64],
    stability: tuple[float, float, float],
) -> bool:
    min_height_m, max_roll_rad, max_pitch_rad = stability
    return bool(
        train_residual.mean < 1.0
        and validation_residual.mean < 1.0
        and float(np.mean(transient_nrmse)) < _TARGET_MEAN_NRMSE
        and float(np.max(transient_nrmse)) < _TARGET_WORST_NRMSE
        and min_height_m > 0.65
        and max(max_roll_rad, max_pitch_rad) < 0.5
    )


def _evaluate(
    physics: G1MujocoPhysics,
    plant: G1PlantRecording,
    recording: G1Recording,
    train: tuple[DirectionalPlantReplay, ...],
    validation: tuple[DirectionalPlantReplay, ...],
    stock_train: NDArray[np.float64],
    stock_validation: NDArray[np.float64],
) -> FitEvaluation:
    backend = G1MujocoBackend(plant.motor_names, physics)
    train_residual = _normalized_residual(backend, train, stock_train)
    validation_residual = _normalized_residual(backend, validation, stock_validation)
    simulation = GrootClosedLoopRunner(physics).run(
        recording.command_t_s, recording.command_body_twist
    )
    errors = directional_transient_errors(recording, simulation)
    values = np.asarray([error.nrmse for error in errors])
    stability = _stability(simulation)
    return FitEvaluation(
        physics,
        train_residual,
        validation_residual,
        {error.direction: error.nrmse for error in errors},
        float(np.mean(values)),
        float(np.max(values)),
        *stability,
        _accepted(train_residual, validation_residual, values, stability),
    )


def _refine_friction(
    values: NDArray[np.float64],
    count: int,
) -> list[G1MujocoPhysics]:
    lower = max(_BOUNDS[2][0], float(values[2]) - 0.75)
    upper = min(_BOUNDS[2][1], float(values[2]) + 0.75)
    return [
        _physics(np.asarray([values[0], values[1], friction]))
        for friction in np.linspace(lower, upper, count)
    ]


def _best(evaluations: list[FitEvaluation]) -> FitEvaluation:
    passed = [evaluation for evaluation in evaluations if evaluation.screen_passed]
    candidates = passed or evaluations
    return min(
        candidates,
        key=lambda evaluation: (
            evaluation.validation_normalized_residual.mean,
            evaluation.train_normalized_residual.mean,
            evaluation.transient_worst_nrmse,
            evaluation.transient_mean_nrmse,
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())


def _save_model(path: Path, motor_names: tuple[str, ...], physics: G1MujocoPhysics) -> str:
    model = build_g1_mujoco_spec().compile()
    apply_g1_mujoco_physics(model, motor_names, physics)
    mujoco.mj_saveModel(model, str(path))
    return _sha256(path)


def _fit_provenance(recording_path: Path, seed: int) -> dict[str, object]:
    return {
        "recording": str(recording_path.resolve()),
        "recording_sha256": _sha256(recording_path),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "seed": seed,
        "command": " ".join(sys.argv),
    }


def _fit_method(mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "train_validation_split": "alternating levels from each direction's highest eight",
        "open_loop_channels": _PLANT_CHANNELS,
        "bounds": {
            "leg_armature_kg_m2": _BOUNDS[0],
            "leg_damping_nm_s_rad": _BOUNDS[1],
            "leg_frictionloss_nm": _BOUNDS[2],
        },
        "acceptance": {
            "heldout_normalized_residual_lt": 1.0,
            "transient_mean_nrmse_lt": _TARGET_MEAN_NRMSE,
            "transient_worst_nrmse_lt": _TARGET_WORST_NRMSE,
        },
        "promotion_gate": "full actual DimOS blueprint replay with the same response thresholds",
    }


def _write_artifact(
    path: Path,
    recording_path: Path,
    seed: int,
    evaluations: list[FitEvaluation],
    best: FitEvaluation,
    model_path: Path,
    model_sha256: str,
    mode: str,
) -> None:
    artifact = {
        "schema_version": 1,
        "provenance": _fit_provenance(recording_path, seed),
        "method": _fit_method(mode),
        "evaluations": [asdict(evaluation) for evaluation in evaluations],
        "best": asdict(best),
        "mujoco_model": {"path": str(model_path), "sha256": model_sha256},
    }
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def _fit_replays(
    plant: G1PlantRecording,
    recording: G1Recording,
    seed: int,
) -> _FitReplays:
    train = directional_replay_plans(plant, recording, seed=seed, split="train")
    validation = directional_replay_plans(plant, recording, seed=seed, split="validation")
    stock = G1MujocoBackend(plant.motor_names, G1_BASELINE_MUJOCO_PHYSICS)
    return _FitReplays(
        train,
        validation,
        _plant_scores(stock, train),
        _plant_scores(stock, validation),
    )


def _fit_evaluations(
    physics_candidates: list[G1MujocoPhysics],
    plant: G1PlantRecording,
    recording: G1Recording,
    replays: _FitReplays,
) -> list[FitEvaluation]:
    return [
        _evaluate(
            physics,
            plant,
            recording,
            replays.train,
            replays.validation,
            replays.stock_train,
            replays.stock_validation,
        )
        for physics in physics_candidates
    ]


def _physics_candidates(
    args: argparse.Namespace,
    plant: G1PlantRecording,
    replays: _FitReplays,
) -> tuple[str, list[G1MujocoPhysics]]:
    if args.candidate is not None:
        return "candidate_validation", [_physics(np.asarray(args.candidate))]
    values = _search_open_loop(
        plant,
        replays.train,
        replays.stock_train,
        seed=args.seed,
        maxiter=args.maxiter,
        popsize=args.popsize,
    )
    return "seeded_search", _refine_friction(values, args.friction_refinements)


def run_fit(args: argparse.Namespace) -> FitEvaluation:
    recording = read_recording(args.recording)
    plant = read_plant_recording(args.recording)
    replays = _fit_replays(plant, recording, args.seed)
    mode, physics_candidates = _physics_candidates(args, plant, replays)
    evaluations = _fit_evaluations(physics_candidates, plant, recording, replays)
    best = _best(evaluations)
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "g1_groot_tuned.mjb"
    model_sha256 = _save_model(model_path, plant.motor_names, best.physics)
    _write_artifact(
        args.out / "plant_fit.json",
        args.recording,
        args.seed,
        evaluations,
        best,
        model_path,
        model_sha256,
        mode,
    )
    return best


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=4)
    parser.add_argument("--popsize", type=int, default=4)
    parser.add_argument("--friction-refinements", type=int, default=7)
    parser.add_argument(
        "--candidate",
        type=float,
        nargs=3,
        metavar=("ARMATURE_KG_M2", "DAMPING_NM_S_RAD", "FRICTIONLOSS_NM"),
    )
    return parser.parse_args()


def main() -> None:
    best = run_fit(_parse_args())
    print(json.dumps(asdict(best), indent=2))


if __name__ == "__main__":
    main()
