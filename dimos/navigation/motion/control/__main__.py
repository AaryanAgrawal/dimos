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

"""Run and score closed-loop episodes.

python -m dimos.navigation.motion.control --ls                 # scenario names
python -m dimos.navigation.motion.control --view -s corridor   # watch one
python -m dimos.navigation.motion.control --view --gen 8 -s gen003   # a generated one
python -m dimos.navigation.motion.control --score              # curated battery
python -m dimos.navigation.motion.control --score --gen 8      # + generated
"""

import argparse
import json

import numpy as np

from dimos.navigation.motion.control.controller import load as load_controller
from dimos.navigation.motion.control.episode import (
    DomainRandomization,
    EpisodeConfig,
    run_episode,
)
from dimos.navigation.motion.control.judge import print_row, score_episode, summarize
from dimos.navigation.motion.planner.autoresearch.scenarios import SCENARIOS, generated
from dimos.navigation.motion.simulation.policy import FreePolicy
from dimos.utils.data import get_data

DEFAULT_POLICY = "ml-trajectory-research/freewalk_mcf.bin"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--scenario", help="run a single scenario by name")
    ap.add_argument("--ls", action="store_true", help="list scenario names and exit")
    ap.add_argument("--gen", type=int, default=0, help="add N generated worlds")
    ap.add_argument("--view", action="store_true", help="live MuJoCo viewer")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer speed factor")
    ap.add_argument("--score", action="store_true", help="judge each episode + summary")
    ap.add_argument("--json", action="store_true", help="summary as JSON only")
    ap.add_argument("--replan-hz", type=float, default=0.0, help="0 = plan once")
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="FREE policy blob")
    ap.add_argument("--planner", default="target", help="referee planner registry name")
    ap.add_argument(
        "--controller", default="pursuit", help="controller registry name or module:factory"
    )
    ap.add_argument("--blind", action="store_true", help="drop the clearance annotation")
    ap.add_argument("--dr", action="store_true", help="randomize mechanisms per episode")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for --dr")
    args = ap.parse_args()

    scenarios = list(SCENARIOS)
    if args.gen:
        scenarios += generated(args.gen)
    if args.ls:
        for sc in scenarios:
            emb = f" [{sc.emb.tag}]" if sc.emb.tag != "go2" else ""
            print(f"{sc.name:<20s} {sc.expect:<7s}{emb} {sc.note}")
        return
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            ap.error(f"no scenario named {args.scenario!r}")

    policy = FreePolicy.load(get_data(args.policy))
    cfg = EpisodeConfig(
        replan_hz=args.replan_hz, planner=args.planner, annotate_clearance=not args.blind
    )
    dr = DomainRandomization() if args.dr else None
    make_controller = load_controller(args.controller)
    rng = np.random.default_rng(args.seed)

    rows = []
    for sc in scenarios:
        result = run_episode(
            sc, make_controller(), policy, cfg, view=args.view, speed=args.speed, dr=dr, rng=rng
        )
        row = score_episode(result)
        rows.append(row)
        if not args.json:
            print_row(row, sc)
    summary = summarize(rows)
    print(json.dumps({"summary": summary} if args.json else summary))


if __name__ == "__main__":
    main()
