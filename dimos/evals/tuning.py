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

"""House conventions for optuna studies. There is no framework here on purpose.

An eval declares its knobs by calling ``trial.suggest_*`` inside its own
objective - that *is* the search space, and it is what lets a knob exist
only in some branches. This module adds the two things that should not be
re-decided per eval: where the study is stored, and that the sampler is
seeded so a rerun repeats.

    from dimos.evals.tuning import study

    def objective(trial):
        cfg = Thing(rate=trial.suggest_float("rate", 0.1, 10, log=True))
        return accept_rate(cfg), median_error(cfg)

    s = study("my-eval", ["maximize", "minimize"])
    s.optimize(objective, n_trials=100)
    for t in s.best_trials:  # the Pareto front; there is no single best
        print(t.values, t.params)

Studies are keyed by name and resume: the same name against the same
storage continues where it left off, so a run can be stopped and picked up.
Browse one with ``uvx optuna-dashboard sqlite:///optuna.db``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

STORAGE = "sqlite:///optuna.db"


def study(
    name: str,
    directions: list[str],
    *,
    names: list[str] | None = None,
    seed: int = 0,
    storage: str = STORAGE,
) -> optuna.Study:
    """A named, resumable study. ``directions`` is one entry per objective value.

    ``["minimize"]`` for a single score; ``["maximize", "minimize"]`` for a
    two-value objective, whose result is a Pareto front in ``best_trials``
    rather than one winner.

    ``names`` labels the objectives, so the dashboard and ``best_trials``
    read as ``good``/``seconds`` instead of an unlabelled list of floats.
    Pass one name per direction.
    """
    import optuna

    s = optuna.create_study(
        study_name=name,
        directions=directions,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=seed),
        load_if_exists=True,
    )
    if names:
        s.set_metric_names(names)
    return s
