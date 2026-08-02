# Tools

Closed-loop episodes: referee world -> evolved planner -> controller -> matched sim.
Run from the repo root. Blue floor boxes in the viewer = the plan's expected body poses.

```bash
# watch one scenario (names from planner/autoresearch/scenarios.py)
python -m dimos.navigation.motion.control --view -s corridor

# slow motion, receding horizon
python -m dimos.navigation.motion.control --view -s zigzag_room --speed 0.5 --replan-hz 5

# score the curated 16 (per-world rows + summary JSON)
python -m dimos.navigation.motion.control --score

# + N generated worlds / machine-readable
python -m dimos.navigation.motion.control --score --gen 8
python -m dimos.navigation.motion.control --score --json

# other planner or policy
python -m dimos.navigation.motion.control --score --planner target-py
python -m dimos.navigation.motion.control --view -s slalom --policy ml-trajectory-research/freewalk_mcf.bin

# tests and types
python -m pytest dimos/navigation/motion/control -q
python -m mypy dimos/navigation/motion/control
```

Score per world = `gate * (100*progress + 10*tracking + 1*composure)`, max 111
(referee scale). Gate 0 on wall contact or fall; refusal counts as arrival on
`expect="refuse"` worlds. Row columns: cross-track p95 vs the plan, tilt p99,
slew-saturation fraction.
