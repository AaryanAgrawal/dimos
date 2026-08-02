# Tools

Every entry point in this package, copy-paste ready. Run from the repo root.
Two recording/net pairs ship in the `ml-trajectory-research` LFS archive:

| recording | net | |
|---|---|---|
| `unitree_himloco01.mcap` | `freewalk_mcf.bin` | plain walking, 45-obs |
| `unitree_v11_gait_height01.mcap` | `v11_final.bin` | 46-obs, commandable body height (crouches to 0.10 m around t=32) |

**Watch the fitted sim next to reality** — MuJoCo window, recorded commands
drive the policy, green ghost box is the recorded pose:

```bash
python -m dimos.navigation.motion.simulation data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --view --ghost --fitted --start 6
```

`--start 6` skips the stand-up (the sim always begins standing). `--speed 0.5`
slow motion, `--seconds 20` cut short. Without `--fitted` you watch stock
menagerie physics — visibly wobblier than the real robot.

**Score a configuration** — per-statistic sim vs real table, noise-floor
units; SNR under ~1 means matched to within what chaos already does:

```bash
python -m dimos.navigation.motion.simulation data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --eval --fitted
```

**Try your own physics** — override any key on top of (or instead of) the
preset:

```bash
python -m dimos.navigation.motion.simulation data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --eval --fitted --physics damping=0.5,leg_mass_scale=1.2 --command-delay 0.02 --actuator-tau 0.01
```

**Refit** — CMA-ES over the 11-parameter space against one recording:

```bash
python -m dimos.navigation.motion.simulation.search data/ml-trajectory-research/unitree_himloco01.mcap data/ml-trajectory-research/freewalk_mcf.bin --trials 100
```

`--multi` switches to NSGA-II and prints the Pareto front over the
gait/translation/rotation/legs groups. `--storage sqlite:///search.db`
resumes, `--json out.json` saves.

**Joint refit** — score every trial on several recordings at once (this is
how `FITTED_*` was produced; single-recording fits absorb that run's style):

```bash
python -m dimos.navigation.motion.simulation.search data/ml-trajectory-research/unitree_himloco01.mcap data/ml-trajectory-research/freewalk_mcf.bin --also data/ml-trajectory-research/unitree_v11_gait_height01.mcap data/ml-trajectory-research/v11_final.bin --seed-fitted --trials 300
```

`--seed-fitted` enqueues the current preset as trial 0, so the search has to
beat it, not rediscover it.

**Replay the recorded lowcmd** — drop `--policy`. The robot collapses in
seconds; that is what open-loop replay of a closed loop's commands does, and
it is why everything above compares command-to-command instead:

```bash
python -m dimos.navigation.motion.simulation data/ml-trajectory-research/unitree_himloco01.mcap --view
```

**Drive it by hand** — constant command instead of a recording:

```python
import numpy as np
from dimos.navigation.motion.simulation.policy import FreePolicy
from dimos.navigation.motion.simulation.walk import walk

policy = FreePolicy.load("data/ml-trajectory-research/freewalk_mcf.bin")
walk(policy, command=np.array([0.5, 0.0, 0.0]), seconds=4, view=True)
```

**Tests and types**:

```bash
python -m pytest dimos/navigation/motion/simulation -q
python -m mypy dimos/navigation/motion/simulation
```

Tracker knobs, rarely needed: `--mount-yaw 94` (robot forward within the
tracker xy plane, fitted) and `--tracker-z 0.207` (tracker offset from
`base`; a guess — tune until the ghost sits on the body).
