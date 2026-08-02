# Go2 sim-to-real: where we are

Goal: make MuJoCo behave like the real Go2 well enough to train against.

**Status.** The loop is closed and measurable. The same HIMLoco policy that ran
on the robot runs in sim, driven by the recorded commands, scored against
VR-tracker ground truth. Two modelling gaps have been found, fixed and
validated; one remains.

Best configuration, in noise-floor units (1.0 = as close as two identical
simulators land from each other):

    gait 0.67    translation 2.20    rotation 2.06

against an untouched baseline where the turn lag alone was **10**.

```
armature 0.0201   damping 0.5648   frictionloss 1.6276
command_delay 0.3172   actuator_tau 0.0448
trunk_mass_scale 1.4063   trunk_inertia_scale 3.399
```

---

## 1. What the recordings contain

`data/ml-trajectory-research/` — two runs with the networks that produced them.

| stream | rate | use |
|---|---|---|
| `control_log` | 48 Hz | the policy's input: `{"action":"walk","vx","vy","vyaw"}` |
| `vive_pose` | 253 Hz | ground-truth body pose (JSON: `p`, `q`, `t_host`) |
| `policy_state` | once | which policy — check it matches the `.bin` |

**There is no joint-space data.** Not commanded (`lowcmd.q` identically zero),
not measured (`lowstate.q` calf angles are in their mechanical range in 0.0% of
rows), not velocity or torque (identically zero). Comparison happens at
body-pose level and cannot be moved to joint level without a new capture.

This also means the simulator's initial pose cannot be restored from a
recording — it always starts standing, so `--start 6` is needed to skip the
robot getting to its feet.

## 2. Calibration that is settled

**Target frame.** The official URDF root link is `base` (there is no
`base_link`), collision box `0.3762 x 0.0935 x 0.114` at origin, i.e. the trunk's
geometric centre, level with the hips. menagerie inherits it one-to-one, so
MuJoCo `qpos[0:3]` *is* that frame — nothing to map.

**Tracker mount**, fitted sim-free from the recording:

* quaternion is **wxyz**, frame is **z-up**
* tracker is mounted **inverted** — `R[2,2] = -0.997`
* robot forward is **+94°** in the tracker's xy plane, i.e. along tracker +y.
  Two independent fits agree (93.6° circular mean; 94.0° by maximizing
  cos(velocity, command)). This was the "mirrored" look.

**Not settled: the tracker's translation.** `--tracker-z 0.207` is a guess and
the in-plane offset is unmodelled. It cannot be recovered from these recordings
— the Vive origin is the room calibration, not the floor.

## 3. Why the judge is distributional

The gait is chaotic. A 3° initial-pose perturbation grows to 136 cm of position
error in 12 s, **non-monotonically** — a 17° perturbation ends up closer. By a
10 s horizon the sim-vs-real gap is *smaller* than the gap between two identical
simulators. Trajectory error carries no information about physics.

So `metrics.py` compares distributions — speed, gains, response lags, body-bob
amplitude and frequency — and `evaluate.py` divides each by its own noise floor,
measured from perturbed rollouts. Nothing is fitted against a statistic whose
sim-real difference does not exceed its own noise.

## 4. Mechanisms found so far

**Command delay ≈ 0.45 s.** On hardware the operator's command crosses a network
and the robot's filtering before the policy sees it; in sim it lands on the same
tick. Cross-correlation measures 0.46–0.50 s directly (sharp peak, 0.88 against
0.66 at zero lag) and the search independently recovered 0.451 s.

*Not inertia* — the alternative hypothesis. Tripling trunk rotational inertia
moves the lag not at all (0.06 → 0.06 against 0.46 real). The policy is a 50 Hz
closed loop: it compensates for a heavier body on the first tick, but cannot
compensate for a command it has not received.

**Actuator lag ≈ 10–45 ms.** A MuJoCo motor delivers requested torque on the
same step; a real BLDC through a gearbox does not. Adding it collapsed the
Pareto front *toward the origin* rather than sliding it along:

| | without | with |
|---|---|---|
| min distance to origin | 3.82 | **2.87** |
| best worst-objective | 2.71 | **2.20** |
| best gait | 1.05 | **0.67** |

The decisive evidence: `actuator_tau` is searched from 0, so an ideal actuator
is free — and **not one of twenty Pareto-optimal trials chose zero** (median
0.022 s). A spurious knob would leave some optima at the identity value.

## 5. How to run it

```bash
# score one configuration (~1.6 s)
python -m dimos.navigation.motion.trajectory.research data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --eval

# watch it, with the recorded pose as a ghost box
python -m dimos.navigation.motion.trajectory.research data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --view --ghost

# multi-objective search, Pareto front
python -m dimos.navigation.motion.trajectory.research.search data/ml-trajectory-research/unitree_himloco01.mcap data/ml-trajectory-research/freewalk_mcf.bin --multi --trials 150
```

---

# Next steps

## A. Foot contact — the standing hypothesis

Rotation is the worst remaining objective (2.06) and is the one thing neither
mechanism so far touches: both model *when* torque arrives, neither models what
the foot does once it lands.

The menagerie Go2 sets **`condim="1"`** on its geoms — frictionless point
contacts that generate **no tangential force at all**. A quadruped turns by
shearing its feet against the ground. This is the obvious suspect.

Try, in order: `condim=3` with the existing `friction=0.6`, then friction as a
search parameter, then `solref`/`solimp` contact stiffness and damping. Judge
by the same test: does the front collapse toward the origin, and does the search
decline to return the parameter to its default?

*Cost:* small — `condim` and `friction` are `model.geom_condim` /
`model.geom_friction` patches in the same `_physics` context manager.

## B. Pin the tracker translation

`--tracker-z 0.207` is a guess, the in-plane offset is unmodelled, and
`height_mean` is excluded from scoring entirely because of it. Two routes:

1. **Measure it** with a ruler against the trunk centre. Ten minutes, ends the
   ambiguity.
2. **Fit it** from the `v = ω × r` lever-arm signature — sim-free, but weakly
   conditioned here (body tilt is only 3.7° mean). A recording with deliberate
   pitching would fix that.

Route 1 is better value. It also unlocks `height_mean` as an eighth statistic.

## C. Validate on the second recording

Everything so far is fitted and scored on `unitree_himloco01`. The v11 run is
untouched and would be a genuine held-out check — but it needs the 46-channel
observation threading first (gait height as channel 45), which `walk.py` does
not do.

Fitting and validating on one recording is the weakest part of the current
result. This is how to fix it.

## D. Capture data that answers what these cannot

Worth doing on the next robot session:

* **A run with deliberate pitch and roll** — makes the lever arm observable and
  strengthens every rotational statistic.
* **A run with low-level control actually enabled**, if the hardware supports
  it, so `lowcmd`/`lowstate` carry real joint data. That would let the
  comparison move from body pose to joint space, which is a much tighter test.
* **Longer straight-line segments** — the current runs stay inside a 1.5 m box,
  so translation statistics rest on very little travel.

## E. Housekeeping

* `speed_lag` regressed (0.7 → 3.0) when command delay was introduced and has
  not been revisited.
* The search fits one recording's mount and delay as global constants; if a
  second recording disagrees, they are per-run properties and the harness needs
  to say so.

---

# Traps worth not repeating

Each of these produced a plausible, finite, wrong number.

* **Filter by time, not samples.** Differentiating a 253 Hz recording and a
  50 Hz rollout with the same 25-*sample* window smooths them by 0.1 s and 0.5 s.
  Reported a Go2 walking at 3.87 m/s.
* **Gains are regression slopes, not means of ratios.** Ratios explode near zero
  command and cancel across sign flips — reported the simulator turning
  backwards.
* **Fit response lag before fitting a gain.** A command alternating faster than
  the response lag reads as no response at all.
* **Autocorrelation, not FFT, for gait frequency.** An FFT peak flips between
  fundamental and harmonic across window lengths (1.5 Hz vs 3.3 Hz on the same
  rollout). This caused a retraction that then had to be un-retracted.
* **Never fit the mount against the simulator.** The rollout diverges within
  seconds; a sim-vs-ghost yaw score is nearly flat and its argmin is ~180° wrong.
  The mount is a property of the recording.
* **Check that a patch applied.** A silently failed edit left `--start` shifting
  the ghost but not the commands, so the simulator ran the first seconds of a
  run against a ghost six seconds later — for several rounds of "results".
* **`|q| < 4 rad` is not a validity check.** Against real joint limits, 0.0% of
  recorded calf angles are physical, not the 75% that filter suggested.
