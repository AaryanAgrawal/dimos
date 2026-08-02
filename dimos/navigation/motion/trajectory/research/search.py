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

"""Search leg-joint physics for the values that make sim behave like hardware.

    python -m dimos.navigation.motion.trajectory.research.search DATASET POLICY

Uses Optuna's CMA-ES sampler. The objective is continuous, low-dimensional and
noisy, which is what CMA-ES is built for; TPE is the better default when
parameters are categorical or conditional, and neither applies here.

The noise floor is measured once and reused for every trial. It costs four
extra rollouts, and holding it fixed is what makes losses comparable between
trials -- recomputing it per trial would let a trial win by getting noisier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dimos.navigation.motion.trajectory.research.evaluate import evaluate, measure_noise

# name -> (low, high, log). Menagerie defaults are armature 0.01, damping 2.0,
# frictionloss 0.2; the ranges bracket them by about a decade each way.
SPACE: dict[str, tuple[float, float, bool]] = {
    "armature": (0.001, 0.2, True),
    "damping": (0.2, 10.0, True),
    "frictionloss": (0.0, 2.0, False),
    # Not a physics property: how long the command takes to reach the policy on
    # hardware. Fitted rather than assumed because the recording cannot separate
    # genuine transport latency from a stamping offset -- see FINDINGS.
    "command_delay": (0.0, 0.8, False),
    # Competing explanation for the same lag: a heavier / more rotationally
    # sluggish trunk. Inertia delays and smooths the response; transport delay
    # only shifts it. Letting both into one space lets the data choose.
    "trunk_mass_scale": (0.6, 2.0, False),
    "trunk_inertia_scale": (0.4, 4.0, True),
}

# Statistics grouped into a few objectives. Seven separate objectives would make
# almost every trial non-dominated -- NSGA-II degrades badly past three or four
# -- and these three are the ones that actually trade against each other: the
# physics-only search bought gait accuracy with friction, the delay search bought
# rotation accuracy and gave gait back.
OBJECTIVES: dict[str, tuple[str, ...]] = {
    "gait": ("gait_hz", "height_std"),
    "translation": ("speed", "speed_gain", "speed_lag"),
    "rotation": ("yaw_rate_gain", "yaw_lag"),
}


def _objective_values(snr: dict[str, float], groups: dict[str, tuple[str, ...]]) -> list[float]:
    """Root-mean-square SNR within each group, so groups of different size compare."""
    return [
        float(np.sqrt(np.mean([snr[k] ** 2 for k in keys if k in snr]))) for keys in groups.values()
    ]


def run(
    dataset: str | Path,
    policy_bin: str | Path,
    *,
    trials: int = 100,
    start: float = 6.0,
    seconds: float = 20.0,
    seeds: int = 4,
    space: dict[str, tuple[float, float, bool]] | None = None,
    storage: str | None = None,
    study_name: str = "go2-physics",
) -> dict[str, Any]:
    """Minimize the noise-weighted sim-vs-real loss over physics parameters."""
    import optuna

    space = space or SPACE
    noise = measure_noise(dataset, policy_bin, start=start, seconds=seconds, seeds=seeds)

    baseline = evaluate(dataset, policy_bin, start=start, seconds=seconds, noise=noise)

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: trial.suggest_float(name, lo, hi, log=log)
            for name, (lo, hi, log) in space.items()
        }
        delay = params.pop("command_delay", 0.0)
        report = evaluate(
            dataset,
            policy_bin,
            start=start,
            seconds=seconds,
            physics=params,
            command_delay=delay,
            noise=noise,
        )
        for key, value in report.snr().items():
            trial.set_user_attr(key, value)
        return report.loss()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.CmaEsSampler(seed=0),
        storage=storage,
        study_name=study_name,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    best_params = dict(study.best_params)
    best_delay = best_params.pop("command_delay", 0.0)
    best = evaluate(
        dataset,
        policy_bin,
        start=start,
        seconds=seconds,
        physics=best_params,
        command_delay=best_delay,
        noise=noise,
    )
    return {
        "baseline_loss": baseline.loss(),
        "best_loss": study.best_value,
        "best_params": study.best_params,
        "baseline_table": baseline.table(),
        "best_table": best.table(),
    }


def run_multi(
    dataset: str | Path,
    policy_bin: str | Path,
    *,
    trials: int = 200,
    start: float = 6.0,
    seconds: float = 15.0,
    seeds: int = 4,
    space: dict[str, tuple[float, float, bool]] | None = None,
    groups: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Multi-objective search; returns the Pareto front rather than one winner.

    A single scalar loss hides the trade: gait accuracy and rotation accuracy
    pull the parameters in different directions, and averaging them just picks
    an arbitrary point on that curve. NSGA-II keeps the whole curve.
    """
    import optuna

    space = space or SPACE
    groups = groups or OBJECTIVES
    noise = measure_noise(dataset, policy_bin, start=start, seconds=seconds, seeds=seeds)

    def objective(trial: optuna.Trial) -> tuple[float, ...]:
        params = {
            name: trial.suggest_float(name, lo, hi, log=log)
            for name, (lo, hi, log) in space.items()
        }
        delay = params.pop("command_delay", 0.0)
        report = evaluate(
            dataset,
            policy_bin,
            start=start,
            seconds=seconds,
            physics=params,
            command_delay=delay,
            noise=noise,
        )
        snr = report.snr()
        for key, value in snr.items():
            trial.set_user_attr(key, value)
        return tuple(_objective_values(snr, groups))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        directions=["minimize"] * len(groups),
        sampler=optuna.samplers.NSGAIISampler(seed=0),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    front = sorted(
        (
            {
                "objectives": dict(zip(groups, t.values, strict=True)),
                "params": t.params,
                "snr": {k: v for k, v in t.user_attrs.items()},
            }
            for t in study.best_trials
        ),
        key=lambda row: sum(row["objectives"].values()),
    )
    return {"groups": list(groups), "front": front, "n_trials": trials}


def format_front(result: dict[str, Any], limit: int = 12) -> str:
    groups = result["groups"]
    head = "  ".join(f"{g:>11}" for g in groups)
    lines = [
        f"Pareto front ({len(result['front'])} of {result['n_trials']} trials)",
        f"{head}   parameters",
    ]
    for row in result["front"][:limit]:
        vals = "  ".join(f"{row['objectives'][g]:11.2f}" for g in groups)
        params = " ".join(f"{k}={v:.4g}" for k, v in sorted(row["params"].items()))
        lines.append(f"{vals}   {params}")
    if len(result["front"]) > limit:
        lines.append(f"... {len(result['front']) - limit} more")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(prog="trajectory.research.search")
    ap.add_argument("dataset")
    ap.add_argument("policy")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--start", type=float, default=6.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--storage", default=None, help="e.g. sqlite:///search.db to resume")
    ap.add_argument("--json", default="", help="write the result here")
    ap.add_argument(
        "--multi",
        action="store_true",
        help="multi-objective (NSGA-II); print the Pareto front instead of one winner",
    )
    args = ap.parse_args()

    if args.multi:
        result = run_multi(
            args.dataset,
            args.policy,
            trials=args.trials,
            start=args.start,
            seconds=args.seconds,
        )
        print(format_front(result))
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2))
        return

    result = run(
        args.dataset,
        args.policy,
        trials=args.trials,
        start=args.start,
        seconds=args.seconds,
        storage=args.storage,
    )
    print("\n=== baseline ===")
    print(result["baseline_table"])
    print("\n=== best ===")
    print(result["best_table"])
    print(f"\nloss {result['baseline_loss']:.2f} -> {result['best_loss']:.2f}")
    print("params:", json.dumps(result["best_params"], indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
