# ml-trajectory-control research

Self-contained: needs only this worktree's venv. No `~/coding/go2` checkout.

Data is the `ml-trajectory-research` LFS archive —
`get_data("ml-trajectory-research/unitree_himloco01.mcap")`, or the paths below
once extracted under `data/`.

```bash
cd ~/coding/dimos-ml-trajectory-control
D=data/ml-trajectory-research
```

**Tests**

```bash
direnv exec . python -m pytest dimos/navigation/motion/trajectory/research -q
direnv exec . python -m mypy dimos/navigation/motion/trajectory/research
```

**Watch the policy walk** (MuJoCo window; recorded commands drive it)

```bash
direnv exec . python -m dimos.navigation.motion.trajectory.research \
  $D/unitree_himloco01.mcap --policy $D/freewalk_mcf.bin --view
```

Add `--speed 0.5` for slow motion, `--seconds 20` to cut it short.

**Headless, with numbers**

```bash
direnv exec . python -m dimos.navigation.motion.trajectory.research \
  $D/unitree_himloco01.mcap --policy $D/freewalk_mcf.bin --seconds 8
```

**Constant command instead of a recording**

```python
from dimos.navigation.motion.trajectory.research.policy import FreePolicy
from dimos.navigation.motion.trajectory.research.walk import walk
import numpy as np
walk(FreePolicy.load(f"{D}/freewalk_mcf.bin"), command=np.array([0.5, 0, 0]), seconds=4, view=True)
```

**Replay `lowcmd` instead of the policy** — drop `--policy`. The robot collapses;
that is the data, not a bug. See `FINDINGS.md`.

## Files

| | |
|---|---|
| `policy.py` | "FREE" v1 blob reader + HIM forward (numpy; matches MNN to 1.1e-4) |
| `walk.py` | policy → MuJoCo, driven by a constant or a recording's `control_log` |
| `model.py` | menagerie go2 scene, Unitree↔MuJoCo motor permutation |
| `replay.py` | the `lowcmd` replay path |
| `FINDINGS.md` | what the recordings do and don't contain |

## Not done yet

Comparing simulated base trajectory against `vive_pose`. Needs the rigid
body→tracker offset first (measured or fitted) — without it any error number is
meaningless. `v11_final.bin` also has no runner: it is obs/frame **46**
(gait height as channel 45), so `walk.py` needs that extra input threaded
through before it will load.
