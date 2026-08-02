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

Add `--speed 0.5` for slow motion, `--seconds 20` to cut it short.

**With the recorded pose as a ghost box**

```bash
python -m dimos.navigation.motion.trajectory.research data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --view --ghost
```

The green box is recorded `base_link`, anchored so t=0 sits at the robot's start
pose. `--tracker-z -0.15` sets how far below the tracker base_link is assumed to
be — a guess; tune it until the box sits on the body.

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

**Replay `lowcmd` instead of the policy** — drop `--policy`. The robot collapses;
that is the data, not a bug. See `FINDINGS.md`.

## Files

| | |
|---|---|
| `policy.py` | "FREE" v1 blob reader + HIM forward (numpy; matches MNN to 1.1e-4) |
| `walk.py` | policy → MuJoCo, driven by a constant or a recording's `control_log` |
| `model.py` | menagerie go2 scene (with or without ghost body), motor permutation |
| `vive.py` | recorded tracker pose → base_link track, anchored at t=0 |
| `replay.py` | the `lowcmd` replay path |
| `FINDINGS.md` | what the recordings do and don't contain |

## Not done yet

A trajectory *error number*. The ghost shows the two side by side, but scoring
needs the body→tracker offset pinned down — `--tracker-z` is a guess, and the
in-plane offset is not modelled at all. `v11_final.bin` also has no runner: it
is obs/frame **46** (gait height as channel 45), so `walk.py` needs that extra
input threaded through before it will load.
