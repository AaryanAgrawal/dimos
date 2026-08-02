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

from dimos.navigation.motion.trajectory.research.evaluate import evaluate, measure_noise

# name -> (low, high, log). Menagerie defaults are armature 0.01, damping 2.0,
# frictionloss 0.2; the ranges bracket them by about a decade each way.
SPACE: dict[str, tuple[float, float, bool]] = {
    "armature": (0.001, 0.2, True),
    "damping": (0.2, 10.0, True),
    "frictionloss": (0.0, 2.0, False),
}


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
        physics = {
            name: trial.suggest_float(name, lo, hi, log=log)
            for name, (lo, hi, log) in space.items()
        }
        report = evaluate(
            dataset, policy_bin, start=start, seconds=seconds, physics=physics, noise=noise
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

    best = evaluate(
        dataset,
        policy_bin,
        start=start,
        seconds=seconds,
        physics=study.best_params,
        noise=noise,
    )
    return {
        "baseline_loss": baseline.loss(),
        "best_loss": study.best_value,
        "best_params": study.best_params,
        "baseline_table": baseline.table(),
        "best_table": best.table(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="trajectory.research.search")
    ap.add_argument("dataset")
    ap.add_argument("policy")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--start", type=float, default=6.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--storage", default=None, help="e.g. sqlite:///search.db to resume")
    ap.add_argument("--json", default="", help="write the result here")
    args = ap.parse_args()

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
