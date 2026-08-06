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

"""The gold oracle must survive its own judge.

Regression for the densify_states truncation bug (fixed at d7c1b7c88):
the reference planner published yaw so coarsely that the judge's station
scoring swapped the body box for its circumscribing rotation cylinder and
vetoed the reference's own path on gen028. The invariant whose absence
produced that: for every scenario, `--planner gold` is never vetoed and
scores ~1.0 against itself.

Seeds: 28 is the historical regression world; 0 and 30 sample the mixed
embodiment roster.

Since planner/revision.md the two sides no longer measure the same box: the
search plans with the swept box the edge's own heading needs, while the judge
still sweeps the all-gait UNION, which is up to (union - narrowest row) wider.
So the oracle's own path can read as touching the judge's box without the robot
touching anything, and the invariant has to say so in bounded terms rather than
pretend the mismatch is not there. Closing it is the revision's own acceptance
item — the judge gains a per-mode envelope-violation metric — and until it
lands, `envelope_slack` is exactly how far the two shapes are allowed to
disagree.
"""

from __future__ import annotations

import pytest

from dimos.navigation.motion.embodiment import EMBODIMENTS, Embodiment
from dimos.navigation.motion.geometry import AvoidanceConfig
from dimos.navigation.motion.planner.referee.score import score_world
from dimos.navigation.motion.planner.referee.sim import Verdict, judge
from dimos.navigation.motion.scenarios import SCENARIOS, Scenario, generate

GEN_SEEDS = [0, 28, 30]
WORLD_IDS = [sc.name for sc in SCENARIOS] + [f"gen{s:03d}" for s in GEN_SEEDS]


def envelope_slack(emb: Embodiment) -> float:
    """How far the judge's union box can stick out past a row the body walks in.

    Zero for an embodiment with no measured rows: there, both sides sweep the
    same box and any contact the judge reports is a real one.
    """
    if not emb.envelope:
        return 0.0
    return max(
        (emb.width - min(r[2] for r in emb.envelope)) / 2.0,
        (emb.length - min(r[1] for r in emb.envelope)) / 2.0,
    )


@pytest.fixture(scope="module")
def verdicts() -> dict[str, tuple[Scenario, Verdict]]:
    cfg = AvoidanceConfig()
    roster = list(EMBODIMENTS.values())
    worlds = list(SCENARIOS) + [generate(s, None, roster[s % len(roster)]) for s in GEN_SEEDS]
    return {sc.name: (sc, judge(sc, cfg, planner="gold")) for sc in worlds}


@pytest.mark.parametrize("name", WORLD_IDS)
def test_gold_survives_its_own_judge(
    verdicts: dict[str, tuple[Scenario, Verdict]], name: str
) -> None:
    sc, v = verdicts[name]
    w = score_world(v)
    slack = envelope_slack(sc.emb)
    assert v.min_truth > -slack - 1e-6, (
        f"gold's path is {-v.min_truth:.4f} m into truth on {name}, past the "
        f"{slack:.4f} m the union may disagree with its own rows by"
    )
    if not slack:
        assert w["dq"] is False, f"gold DQ'd on {name}"
    if sc.expect != "refuse":
        assert not v.veto, f"gold vetoed its own path on {name} (min_scored {v.min_scored:.3f})"
    if v.gold is not None and not w["dq"]:
        # Gold vs itself: only densify-vs-raw resampling separates them.
        assert w["gold"] > 0.9, f"gold scored {w['gold']} against itself on {name}"
