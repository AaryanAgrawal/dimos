# Tools

Closed-loop episodes: referee world -> evolved planner -> controller -> matched sim.
Run from the repo root. Blue floor boxes in the viewer = the plan's expected body poses.

```bash
# list scenario names / watch one (curated or generated)
python -m dimos.navigation.motion.control --ls
python -m dimos.navigation.motion.control --view -s corridor
python -m dimos.navigation.motion.control --view --gen 8 -s gen003

# slow motion, receding horizon
python -m dimos.navigation.motion.control --view -s zigzag_room --speed 0.5 --replan-hz 5

# score the curated 16 (per-world rows + summary JSON)
python -m dimos.navigation.motion.control --score

# + N generated worlds / machine-readable
python -m dimos.navigation.motion.control --score --gen 8
python -m dimos.navigation.motion.control --score --json

# other planner, policy, or controller (candidates load as module:factory)
python -m dimos.navigation.motion.control --score --planner target-py
python -m dimos.navigation.motion.control --score --controller my.candidate:make
python -m dimos.navigation.motion.control --view -s slalom --policy ml-trajectory-research/freewalk_mcf.bin

# domain randomization (per-episode mechanism draws) and the blind A/B
python -m dimos.navigation.motion.control --score --dr --seed 3
python -m dimos.navigation.motion.control --score --blind

# tests and types
python -m pytest dimos/navigation/motion/control -q
python -m mypy dimos/navigation/motion/control
```

Score per world = `gate * (100*arrived + 10*precision + 1*(pace+composure)/2)`,
max 111 (referee scale). Gate 0 on wall contact or fall; refusal counts as
arrival on `expect="refuse"` worlds. Precision is judged in clearance space:
the share of ticks the body spent under the embodiment's 0.05 m floor against
truth — deviation with room around it is free, the same deviation beside a
wall is the violation. Under replanning, cross-track is scored against the
plan that was ACTIVE at each tick (a replan never amnesties past drift) and
`churn` measures how far the follower forced the deterministic planner to
re-route -- plan flip-flop reads as churn, not as clean tracking. Columns:
min body clearance, below-floor fraction, cross-track p95 (diagnostic),
churn, tilt p99.
