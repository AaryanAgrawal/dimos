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

"""Replay a recorded Go2 mcap into flat-ground MuJoCo and score joint tracking.

python -m dimos.navigation.motion.trajectory.research <file>.mcap

Recordings and the networks that produced them ship as the
``ml-trajectory-research`` LFS archive; resolve one with
``get_data("ml-trajectory-research/unitree_himloco01.mcap")``.
"""

from __future__ import annotations

import argparse

import numpy as np

from dimos.navigation.motion.trajectory.research import model as go2_model, replay as replay_mod


def main() -> None:
    ap = argparse.ArgumentParser(prog="trajectory.research")
    ap.add_argument("dataset", help="recorded .mcap")
    ap.add_argument("--limit", type=int, default=5000, help="lowcmd samples to replay")
    ap.add_argument("--no-seed", action="store_true", help="start from keyframe, not lowstate")
    args = ap.parse_args()

    commands = replay_mod.read_commands(args.dataset, limit=args.limit)
    states = replay_mod.read_states(args.dataset, limit=args.limit)
    if not commands:
        raise SystemExit("no lowcmd in dataset")

    print(f"lowcmd  {len(commands)} samples, {commands[-1].ts - commands[0].ts:.2f}s")
    print(f"lowstate {len(states)} samples")

    rollout = replay_mod.replay(
        commands, init_state=None if args.no_seed else (states[0] if states else None)
    )
    model, _ = go2_model.load()

    print(f"\nbase drift: {np.linalg.norm(rollout.base_pos[-1] - rollout.base_pos[0]):.3f} m")
    print(f"base z: {rollout.base_pos[0, 2]:.3f} -> {rollout.base_pos[-1, 2]:.3f} m")

    if states:
        e = replay_mod.tracking_error(rollout, states, model)
        print(
            f"\njoint RMS overall: {e['rms_overall']:.4f} rad   max |err|: {e['max_abs']:.4f} rad"
        )
        for name, v in zip(go2_model.MUJOCO_ACTUATOR_NAMES, e["rms_per_joint"], strict=False):
            print(f"  {name:10s} {v:.4f}")


if __name__ == "__main__":
    main()
