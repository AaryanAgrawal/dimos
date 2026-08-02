# Go2 sim-to-real: where we are

Goal: make MuJoCo behave like the real Go2 well enough to train against.

**Status: matched, on the fitted recording.** The same HIMLoco policy that ran
on the robot runs in sim, driven by the recorded commands, scored against
VR-tracker ground truth — and after modelling three mechanisms and fixing two
judge artifacts, the multi-objective search collapses to a single point with
**every statistic at or below its noise floor**:

    gait 0.23    translation 0.85    rotation 0.16

(noise-floor units: 1.0 = as close as two identical simulators land from each
other under a 3° initial-pose perturbation, measured at default physics). In
absolute terms: speed 0.404 vs 0.427 m/s, yaw gain 0.699 vs 0.678, yaw lag
0.170 vs 0.170 s, pitch bob 0.021 vs 0.020 rad, gait 1.72 vs 1.67 Hz.

```
armature 0.0140   damping 0.2381   frictionloss 0.7372
foot_friction 0.5692   foot_friction_torsional 0.0031
trunk_mass_scale 1.326   trunk_inertia_scale 1.487
command_delay 0.0231   actuator_tau 0.0289
```

These are *physical*: armature near the 0.01 spec, trunk 33% heavier (it
carries a tracker and mounts), inertia ×1.5 rather than the ×3.4 an earlier
contaminated fit produced, and a 23 ms genuine transport delay.

The honest caveat: fitted and validated on **one recording**. See Next steps.

---

## 1. What the recordings contain

`data/ml-trajectory-research/` — two runs with the networks that produced them.

| stream | rate | use |
|---|---|---|
| `control_log` | 48 Hz | the operator's command: `{"action":"walk","vx","vy","vyaw"}` |
| `vive_pose` | 253 Hz | ground-truth body pose (JSON: `p`, `q`, `t_host`) |
| `policy_state` | once | which policy — check it matches the `.bin` |

**There is no joint-space data.** Not commanded (`lowcmd.q` identically zero),
not measured (`lowstate.q` calf angles are in their mechanical range in 0.0% of
rows), not velocity or torque (identically zero). Comparison happens at
body-pose level and cannot be moved to joint level without a new capture.

This also means the simulator's initial pose cannot be restored from a
recording — it always starts standing, so `--start 6` is needed to skip the
robot getting to its feet.

**Both streams share one epoch: the first walk command.** `t_host` and
`log_time` are the same clock (2.5 ms apart, no drift), but the vive stream
starts recording before the operator presses walk — 0.31 s earlier on
himloco01, 4.4 s on v11. Zeroing each stream at its own first message paired
every real pose with a command from its future, and a search then "fitted"
0.317 s of command delay, within 4 ms of the bookkeeping offset.

## 2. Calibration that is settled

**Target frame.** The official URDF root link is `base` (there is no
`base_link`), origin at the trunk's geometric centre, level with the hips.
menagerie inherits it one-to-one, so MuJoCo `qpos[0:3]` *is* that frame.

**Tracker mount**, fitted sim-free from the recording:

* quaternion is **wxyz**, frame is **z-up**
* tracker is mounted **inverted** — `R[2,2] = -0.997`
* robot forward is **+94°** in the tracker's xy plane — i.e. the mount is ~4°
  skewed from square, which is why this is fitted, not assumed. Two
  independent fits agree (93.6° circular mean; 94.0° by maximizing
  cos(velocity, command)).

**Not settled: the tracker's translation.** `--tracker-z 0.207` is a guess and
the in-plane offset is unmodelled. Its influence is now *contained* rather
than fixed: height is compared in sensor space and the orientation statistics
are immune to it (§3). A ruler measurement would end it.

## 3. How the judge works

**Distributional, because the gait is chaotic.** A 3° initial-pose
perturbation grows to 136 cm of position error in 12 s, non-monotonically; by
a 10 s horizon the sim-vs-real gap is smaller than the gap between two
identical simulators. Trajectory error carries no information about physics,
so `metrics.py` compares distributions — speed, gains, response lags, bob
amplitude and frequency, pitch/roll oscillation — and `evaluate.py` divides
each difference by its own noise floor from perturbed rollouts.

**Height in sensor space.** Inverting the guessed tracker offset on the real
data injected the guess into the ground truth: 11.4 mm of the "real" z std was
lever-arm swing against 5.6 mm of actual tracker bob. Instead the real side
keeps the raw tracker height and the sim mounts a *virtual tracker* with the
same guess — the guess distorts both sides identically and cancels.

**Orientation statistics.** `pitch_std` / `roll_std` (detrended, so a constant
mount or room-calibration tilt drops out) carry the gait's body oscillation
and owe nothing to the tracker translation. They are what pinned the sim
oscillating ~2× too fast and ~2.5× too hard before the physics fit — visible
by eye as the exaggerated leg lifts, and confirmed independently on z, pitch
and roll.

## 4. The three mechanisms

**Command slew — known constants, not a knob.** The robot rate-limits operator
commands per-axis before the policy sees them (go2web `policy.rs
ramp_velocity`): max 0.05 / 0.04 / 0.10 (vx/vy/vyaw) per 20 ms tick. The
recorded `control_log` carries the operator *target*; the policy on hardware
only ever saw the ramp. A yaw reversal ramps for 0.4 s where a speed nudge
ramps for a tenth of that — which is why the real yaw answers ~0.1 s later
than the real speed does, an axis-dependent lag no uniform delay could fit
(the pre-slew search traded translation 0.79 against rotation 3.03 and could
not have both). With the slew alone, default physics puts sim yaw_lag at
0.170 vs real 0.170.

**Actuator lag ≈ 20–30 ms.** A MuJoCo motor delivers requested torque on the
same step; a real BLDC through a gearbox does not. First-order lag on the
torque; every Pareto-optimal trial keeps it (searched from 0, so an ideal
actuator is free — none chose it).

**Command delay ≈ 23 ms.** What genuinely remains of transport latency once
the epoch artifact is gone.

**Corrected en route:** the menagerie feet are *not* frictionless — the
`condim="1"` default class only governs the calf capsules; the foot geoms
override with `priority="1" condim="6"` and a full friction cone. The open
question was the friction *values*; the search settled on much lower torsional
friction (0.003 vs the shipped 0.02).

## 5. How to run it

```bash
# score one configuration (~1.6 s)
python -m dimos.navigation.motion.trajectory.research data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --eval

# watch it, with the recorded pose as a ghost box
python -m dimos.navigation.motion.trajectory.research data/ml-trajectory-research/unitree_himloco01.mcap --policy data/ml-trajectory-research/freewalk_mcf.bin --view --ghost

# multi-objective search, Pareto front
python -m dimos.navigation.motion.trajectory.research.search data/ml-trajectory-research/unitree_himloco01.mcap data/ml-trajectory-research/freewalk_mcf.bin --multi --trials 300
```

---

# Next steps

## A. Validate on the held-out recording

Everything above is fitted and scored on `unitree_himloco01`. The v11 run is
untouched and is now a *meaningful* held-out check — the per-recording epoch
is handled, and slew/actuator/delay are supposed to be properties of the
platform, not the run. It needs the 46-channel observation threading first
(gait height as channel 45), which `walk.py` does not do.

If the fitted physics transfers, the sim is ready to train against. If it
does not, the failure pattern says which parameter was soaking up run-specific
error. This is the single highest-value next step.

## B. Measure the tracker translation

A ruler against the trunk centre ends the `--tracker-z 0.207` guess, unlocks
`height_mean` as a tenth statistic, and removes the last systematic the
sensor-space trick only cancels to first order. Ten minutes on the robot.

## C. Capture data that answers what these cannot

* **Low-level control enabled**, if the hardware supports it, so
  `lowcmd`/`lowstate` carry real joint data. That upgrades the whole method
  from distributional matching to short-horizon prediction error
  (re-initialize sim from real state, score 0.5 s predictions) — far better
  conditioned, chaos-free, and it can localize error to individual joints.
* **Deliberate pitch and roll**, to make the tracker lever arm observable.
* **Longer straight lines** — the current runs live in a 1.5 m box.

## D. Housekeeping

* Three parameters sit at or near search bounds (damping at the 0.238 floor,
  torsional friction near 0.002, actuator_tau near 0.029 of 0.05). With every
  objective sub-noise the basin is flat, so their exact values are weakly
  identified — widen the ranges before quoting them as measurements.
* The noise floor shrinks ~10–100× at the fitted physics (the matched sim is
  much less statistically chaotic than the default one), so a standalone
  `--eval` at the best config reports inflated/infinite SNR against its own
  floor. Compare absolute sim/real columns there, or reuse a default-physics
  floor. Four-seed peak-to-peak is also a fragile spread estimator; more seeds
  would firm it up.
* `speed_lag` is the weakest surviving statistic (sim 0.12 vs real 0.08 s,
  the one residual above 1 after the collapse in absolute terms).

## E. Then: train against it

The project goal. With A green, the matched sim is the environment; the
fitted config is the domain-randomization centre, and the noise-floor
machinery doubles as a regression test that future sim changes stay matched.

---

# Traps worth not repeating

Each of these produced a plausible, finite, wrong number.

* **Put both streams on one epoch.** Zeroing each stream at its own first
  message turned "how long the tracker ran before the operator pressed walk"
  into 0.31 s of phantom command delay — which a search then confidently
  fitted, within 4 ms. Per-recording, and 4.4 s on the other run.
* **Never invert a guessed extrinsic onto the ground truth.** The lever-arm
  correction made the "real" height mostly an artifact of the guessed offset.
  Simulate the sensor instead of correcting the measurement.
* **A mechanism beats a knob.** The axis-dependent lag was unfittable by any
  of nine parameters, and was ten lines of the robot's own command shaping
  with published constants. When a search trades two objectives it should be
  able to satisfy, go read the executor.
* **Check what a fitted value is absorbing.** trunk_inertia ×3.4 and 8× spec
  frictionloss were the search compensating for the two artifacts above; they
  relaxed to ×1.5 and 3.7× once the artifacts died. Implausible fitted values
  are structural-error alarms, not measurements.
* **Filter by time, not samples; regression slopes, not means of ratios;
  autocorrelation, not FFT, for gait frequency; fit the lag before the gain.**
  Estimator bugs that each reported confident nonsense (a 3.9 m/s Go2, a
  backwards-turning simulator, a retraction that had to be un-retracted).
* **Never fit the mount against the simulator** — rollouts diverge in seconds,
  so the sim-vs-ghost score is flat and its argmin ~180° wrong. The mount is a
  property of the recording.
* **Check that a patch applied, and don't append a mutable array** — one
  silent edit failure and one aliased `vel_cmd` buffer each invalidated a
  round of numbers; both now have regression tests.
* **`|q| < 4 rad` is not a validity check.** Against real joint limits, 0.0%
  of recorded calf angles are physical.
