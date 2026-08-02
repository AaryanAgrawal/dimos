# ml-trajectory-control research

Self-contained: needs only this project's venv. No `~/coding/go2` checkout.

Data is the `ml-trajectory-research` LFS archive —
`get_data("ml-trajectory-research/unitree_himloco01.mcap")`, or the paths below
once extracted under `data/`.

**Tests**

```bash
python -m pytest dimos/navigation/motion/trajectory/research -q
python -m mypy dimos/navigation/motion/trajectory/research
```

**Watch the policy walk** (MuJoCo window; recorded commands drive it)

```bash
python -m dimos.navigation.motion.trajectory.research data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --view
```

Add `--speed 0.5` for slow motion, `--seconds 20` to cut it short, and
`--start 6` to skip the stand-up at the beginning of a run — the simulator
always begins standing, so t=0 is not a fair comparison.

**With the recorded pose as a ghost box**

```bash
python -m dimos.navigation.motion.trajectory.research data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --view --ghost
```

The green box is the recorded `base` pose, anchored so t=0 sits at the robot's
start pose.

`--mount-yaw 94` is the robot's forward heading within the tracker's xy plane,
fitted from the data. `--tracker-z 0.207` is how far the tracker sits from
`base` along the tracker's z — positive because the tracker is mounted
inverted. That one is a guess; tune it until the box sits on the body.

**Headless, with numbers**

```bash
python -m dimos.navigation.motion.trajectory.research data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --seconds 8
```

**Constant command instead of a recording**

```python
import numpy as np
from dimos.navigation.motion.trajectory.research.policy import FreePolicy
from dimos.navigation.motion.trajectory.research.walk import walk

policy = FreePolicy.load("data/ml-trajectory-research/freewalk_mcf.bin")
walk(policy, command=np.array([0.5, 0.0, 0.0]), seconds=4, view=True)
```

**Score sim against the recording** (~1.6 s)

```bash
python -m dimos.navigation.motion.trajectory.research data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --eval
```

Add `--physics armature=0.03,damping=2.0` to try parameter values. Each row is
a statistic, scaled by its own noise floor — SNR under ~1 means sim and hardware
agree to within what chaos already does to a rollout.

**Search physics for the best match** (~1.6 s per trial)

```bash
python -m dimos.navigation.motion.trajectory.research.search data/ml-trajectory-research/unitree_himloco01.mcap data/ml-trajectory-research/freewalk_mcf.bin --trials 100
```

Optuna + CMA-ES over leg-joint physics, an explicit command delay, trunk
mass/inertia and foot friction. `--storage sqlite:///search.db` to resume,
`--json out.json` to save the result. `--multi` switches to NSGA-II and prints
the Pareto front over the gait/translation/rotation objective groups.

**Replay `lowcmd` instead of the policy** — drop `--policy`. The robot collapses;
that is the data, not a bug. See `FINDINGS.md`.

## Files

| | |
|---|---|
| `policy.py` | "FREE" v1 blob reader + HIM forward (numpy; matches MNN to 1.1e-4) |
| `walk.py` | policy → MuJoCo, driven by a constant or a recording's `control_log` |
| `model.py` | menagerie go2 scene (with or without ghost body), motor permutation |
| `vive.py` | recorded tracker pose → base track, anchored at a chosen time |
| `metrics.py` | filtering + chaos-tolerant statistics for comparing runs |
| `evaluate.py` | one call: run, summarize both sides, weight by noise floor |
| `search.py` | Optuna/CMA-ES over leg-joint physics |
| `replay.py` | the `lowcmd` replay path |
| `FINDINGS.md` | what the recordings do and don't contain |

## Where this is going

`FINDINGS.md` is the state of the sim-to-real work: what is calibrated, what
the two recordings can and cannot answer, the modelling gaps found so far,
and the proposed next steps — the sim oscillating roughly twice as fast and
twice as hard as the real robot is the dominant open gap.
