# Go2 → MuJoCo replay: what works, and what the 2026-08-02 recordings can't do

## The sim half works

`dimos/navigation/motion/trajectory/research/` — standalone, does not use
`unitree/mujoco_connection` or any of the existing sim stack.

- `model.py` — loads menagerie `unitree_go2/scene.xml` (flat ground, checkered
  plane), builds the Unitree↔MuJoCo motor permutation **by name**.
- `replay.py` — reads `lowcmd`/`lowstate` from an mcap, re-runs the on-board
  control law `tau = kp·(q_des−q) + kd·(dq_des−dq) + tau_ff` against the
  *simulated* joint state, steps MuJoCo, scores against recorded `lowstate`.
- `__main__.py` — `python -m dimos.navigation.motion.trajectory.research <mcap>`

Fit is good: model timestep is **0.002 s = 500 Hz**, exactly the `lowcmd` and
`lowstate` rate, so it's one command per sim step with no resampling. The 12
actuators are `<motor>` (torque) with `gear=1`, so `data.ctrl` *is* N·m,
clamped at ±23.7 (hip/thigh) and ±45.43 (calf).

## The data half is blocked

**`rt/lowcmd` in these recordings is not motor commands.** Decoded (dimos'
decoder is correct — I verified it against a hand-written CDR parse, they
agree byte for byte):

| field                  | what's in it                                                            |
|------------------------|-------------------------------------------------------------------------|
| `q` (target position)  | **identically 0** in 86% of samples; the rest reach 16000 rad           |
| `dq` (target velocity) | 0                                                                       |
| `kd`                   | **identically 0**, all 12 joints, all 99 s                              |
| `mode`                 | random bytes — 182, 225, 190, 48… (valid modes are 0x00/0x01/0x0A)      |
| `kp`                   | structured: `1.8/0.8/1.2` repeating per leg — a per-joint-type constant |
| `tau`                  | structured: `3.5/3.0/2.2` repeating per leg                             |

`kp` and `tau` repeat with period 3 (hip/thigh/calf), so those two fields carry
*something* — but with `q_des = 0` and `kd = 0` there is no trajectory to
replay. No PD controller runs with `kd = 0` on all twelve joints for 99
seconds.

The robot was driven by the **sport-mode API** — `rt/api/sport/request`, 1978
messages at 21 Hz. That controller is Unitree's, closed, and not in the
recording. `policy/lowcmd` (the channel that *would* hold a custom policy's
output) is **empty**.

**`lowstate` is also mostly unusable:**

| field     | status                                                                                                                                              |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `dq`      | **identically zero**, 100% of rows, both files                                                                                                      |
| `tau_est` | **identically zero**, 100% of rows, both files                                                                                                      |
| `q`       | populated, but **~24% of rows are out of physical range** (\|q\| > 4 rad; observed −19.06 … +21.49) — 26,064 bad values in file 1, 18,929 in file 2 |

This matches the known Go2 **Air** limitation: no leg-state telemetry. It is
worse than "empty" — it is partly populated with garbage, so anything that
doesn't range-check will silently train on noise.

Running the replay anyway produces exactly what you'd expect from commanding
zero: the robot collapses. Base z 0.272 → 0.081 m, 2.19 rad RMS joint error.
That number is not a sim-tuning problem and should not be tuned against.

## What these recordings *do* contain, cleanly

- **`vive/pose`** — JSON, clean, 190–251 Hz:
  `{"p":[-0.049,0.074,-0.005],"q":[0.034,0.999,-0.030,-0.006],"t_host":…,"ts":…}`
  Full 6-DoF ground truth. This is the good stuff.
- `vive/imu` (385–500 Hz), `vive/log`
- `odom` 153 Hz, `sportmodestate` 300 Hz, `api_sport_request` 21 Hz
- `color_image_h264` 14 Hz

Note `vive/*` is `[raw bytes]` to dimos — no codec registered — but it's plain
JSON, so decoding is a one-liner, not a codec-authoring job.

## Three ways forward

1. **Re-record with low-level control.** Put the robot in low-level mode so
   `rt/lowcmd` carries real `q_des`/`kd` and `lowstate` reports real `dq`/`tau`.
   This is the only path that makes the original plan — replay motor commands,
   tune sim to match Vive — work as stated. Worth checking whether the Air can
   do this at all before assuming it's a recording-config fix.
2. **Change the input layer**: drive sim from `api_sport_request` (21 Hz
   high-level velocity commands) and score base trajectory against `vive/pose`.
   This sidesteps the motor data entirely, and is arguably the more useful
   model — learning the sport controller's body response is closer to
   "trajectory control" than joint replay is.
3. **Kinematic replay** of `lowstate.q` with the ~24% bad rows filtered. Cheap,
   but you'd be fitting a sim to a signal that is a quarter corrupt and has no
   velocity or torque to constrain it. I would not build on this.

My recommendation is **(2)**, with **(1)** if you can get the robot into
low-level mode — they're complementary, not competing.

---

# Third recording: `20260802-213016.mcap` (16:32)

**Motor commands: still absent. Velocity commands: now present and good.**

The robot was on a **learned policy**, not sport mode — `policy/state` reports
three modes across the run: `freewalk` → `fps` → `v11`. `api_sport_request`
collapsed to 15 messages (was 1978), and `policy_lowcmd` went from empty to
4804 @ 48 Hz. That's the low-level switch from option (1) actually happening.

But `policy/lowcmd` still carries nothing per-joint. Decoded with dimos' own
`GO2_CODECS['rt/lowcmd']` — the validated CDR codec, not a hand-parse:

| field | value |
|---|---|
| `q_des`, `dq_des`, `kd` | identically 0 |
| `kp` | 0.5–1.0 |
| `tau_ff` | 0–40, quantized to 1.6 |
| `mode` | all 256 byte values |

**All twelve motor slots hold identical values in 100% of rows.** Twelve
identical entries cannot describe a walking quadruped, and a `mode` byte
uniform over 0–255 is not an enum. Whatever this channel is, it is not a
per-joint command. `rt/lowcmd` and `lowstate` are unchanged from the first two
files — `dq` and `tau_est` still identically zero, `q` still ~25% out of
physical range.

## What is genuinely new and usable

**`control_log`, 4339 msgs @ 45 Hz** — the actual policy input, as JSON:

```json
{"action":"walk","hold":false,"type":"policy","vx":-0.0121,"vy":0.0,"vyaw":0.0}
```

| action | count |
|---|---|
| `walk` | 3608 |
| `pitch` | 641 |
| `gait_height` | 79 |
| `fps_engage` / `v11_engage` | 1 each |

Command ranges: `vx` ±0.600 m/s (60% nonzero), `vy` ±0.400 (22%), `vyaw` ±1.710
rad/s (47%). Dense, well-exercised in all three axes.

**`vive_pose`, 24921 @ 251 Hz**, full run coverage, no early dropout. Motion is
confined to roughly a 1.5 × 1.3 m box with 0.25 m net displacement — walking
mostly in place rather than traversing.

## What this changes

Option (2) is no longer a fallback, it's the supported path: **45 Hz body
velocity commands in, 251 Hz Vive pose out.** This file is the right data for
it; the first two are not (15 sport requests total).

Option (1) is still blocked. Switching the robot to policy control did *not*
surface per-joint commands — so if you want motor-level replay, the missing
piece is whatever writes `policy/lowcmd`, not the robot's control mode.

**One confound to handle:** three policies (`freewalk`, `fps`, `v11`) run inside
this single 99 s recording. Segment by `policy/state` before fitting anything,
or you'll be tuning one sim against three different controllers.

---

# `unitree_himloco01.mcap` — the loop closes

Single policy: `policy/state` = `{"mode":"freewalk"}`, one entry, 55.5 s.
`control_log` 2596 @ 48 Hz, `vive_pose` 14024 @ 253 Hz, both covering the run.

**Motor commands: absent again, and now obviously so.** Across `policy/lowcmd`
*and* `rt/lowcmd`, decoded with dimos' own codec: `q_des` and `kd` identically
zero, `kp` constant **1.0**, `tau_ff` constant **40.0** — every sample, every
joint. Not noise, not misalignment: template defaults. Checked four ways now
(dimos codec, hand CDR parse, both channels, three files). The publisher never
fills `motor_cmd`. If motor-level replay matters, fix that publisher in
`~/coding/go2`; nothing in the recordings will change it.

## But that no longer blocks the project

`~/coding/go2/models/index.md:12` — **`free_walk` reconstructed = HIMLoco**, the
same policy that produced this recording, and it already runs in MuJoCo on the
*same* menagerie scene this package uses.

Verified working:

```
cd ~/coding/go2 && ./.venv/bin/python models/mcf/mcf_walk.py --vx 0.5 --seconds 3
# final height=0.291  x_travel=+1.168m  vx_mean(2nd half)=+0.530  UPRIGHT (grav_z=-0.99)
```

Commanded 0.5, achieved 0.530. Walks, stays upright.

Its interface is exactly our recorded signal — `--vx --vy --vyaw`, and obs
`[cmd(3), ang_vel(3), grav(3), dof_pos(12), dof_vel(12), prev_action(12)]` ×6
frames. So the sim-to-real comparison needs **no motor data at all**:

| | source |
|---|---|
| policy | `models/mcf/mcf_walk.py` (HIMLoco free_walk, MNN) |
| input | `control_log` vx/vy/vyaw @ 48 Hz |
| ground truth | `vive_pose` @ 253 Hz |
| sim | menagerie `unitree_go2/scene.xml`, flat |

The only work left is to drive the policy from the recorded command *series*
instead of constants, then compare base trajectory to Vive.

## Two things to decide before that comparison means anything

1. **Frame alignment.** `vive_pose` is in the tracker's own frame, with the
   tracker mounted somewhere on the body. Comparing to MuJoCo's base pose needs
   a rigid body→tracker offset — either measured, or fitted (Procrustes/hand-eye
   on the first seconds). Not knowing it makes any RMS number meaningless.
2. **Environments.** `mcf_walk.py` needs `MNN`, which only exists in
   `~/coding/go2/.venv`; this package runs in the dimos worktree venv. Either
   vendor the policy runner in, or shell out and exchange npz.

---

# `unitree_v11_gait_height01.mcap` — v11-final

`policy/state` = `{"mode":"v11"}`, single policy. `control_log`: 2071 `walk`,
19 `gait_height`. Gait height was **swept 0.100 → 0.369 m** (nominal 0.31,
slider range 0–0.4), so this run exercises the extra input deliberately.
Note gait-height messages key it as **`gh`**, not `gait_height`:
`{"action":"gait_height","gh":0.1948,"type":"policy"}`.

## The network

**`~/coding/go2web/policy/assets/v11_final.bin`** — identified from
`go2web/web/src/Policy.svelte:139`, which names it exactly:

> `id: "v11_final"`, `name: "v11-final (gait height)"` — *"v11 gait-height HIM
> walker, run_flat2 final (step 1500) — drive vx/vy/vyaw + the gait-h slider
> (0–0.4 m, nom 0.31)."*

Siblings are earlier checkpoints: `v11_600/800/1000/1200.bin` (all 978,784 B,
2026-07-03).

From `go2web/policy/src/policies/unitree/himloco.rs`:

- **obs_per_frame is 46, not 45** — `gait_height` in raw metres is appended as
  the **last channel, index 45**; further extra channels are zero (`:240`).
  So v11 is *not* drop-in compatible with the free_walk obs layout.
- **PD gains are kp=40, kd=1**, baked into the .bin (`:31`). Worth stating
  plainly: that is the real controller, and it is further confirmation that the
  `kp=1 / kd=0` in every `lowcmd` field is template filler, not gains.
- Speed-banded experts inside one .bin: `0_1` / `1_5` / `only_rotate`, selected
  by the commanded velocity (`:33`, `:130`).

## Runner gap — this one has no sim path yet

| policy | Python/MNN runner | verified in MuJoCo |
|---|---|---|
| freewalk (`himloco01`) | `~/coding/go2/models/mcf/mcf_walk.py` | **yes** — walks, vx 0.53 for 0.5 cmd |
| **v11-final** | **none** | no |

There is no `.mnn` for v11 (only `free_walk/*.mnn` exist) and no MuJoCo harness
anywhere in `go2web` — that crate is the on-robot Rust runtime. To run v11 in
sim, either:

1. write a Python loader for the "FREE" v1 format — the layout is fully
   documented in `~/coding/go2/models/mcf/export_freewalk_bin.py:17`
   (`"FREE", u32 ver=1, u32 hist, u32 obs_per_frame, u32 act_dim, u32 enc_vel,
   u32 enc_lat`, then branch weights), or
2. drive the existing Rust policy crate from a MuJoCo harness.

(1) is small and keeps everything in one process. Worth noting the same loader
would also read `freewalk_mcf.bin`, which would remove the MNN dependency and
the two-venv split that `mcf_walk.py` currently forces.


---

# Correction: `lowstate.q` is not joint angles either

An earlier pass here said ~75% of `lowstate.q` rows were usable, based on a
`|q| < 4 rad` filter. That filter was far too generous. Checked against the
actual mechanical limits from the MJCF:

    FL/FR/RL/RR_calf   [-2.72, -0.84]   -- always negative, by construction
    hip                [-1.05, +1.05]
    thigh              [-1.57, +3.49]  (front) / [-0.52, +4.54] (rear)

Recorded calf angles are inside their range in **0.0%** of rows on himloco01
and **0.1%** on v11. Mean calf is +0.007 rad -- a sign that is mechanically
impossible -- and the values span +/-14 rad.

So there is no joint-space signal in these recordings at all: not commanded
(`lowcmd.q` identically zero), not measured (`lowstate.q` not angles),
not velocity or torque (identically zero). Comparison stays at body-pose level.

## What the simulator starts from, and why the first seconds do not compare

`walk()` resets to the menagerie `home` keyframe (base at z = 0.27, standing)
and then overwrites the twelve leg joints with the policy's own
`default_pose = [0.1, 0.8, -1.5, ...]`. It always begins standing, and nothing
in the recording can change that -- restoring the robot's true initial pose
would need joint angles, which do not exist here.

On himloco01 the tracker sits at 0.166 m for the first ~5 s and then holds
0.229-0.245 m for the remaining 50 s: the robot is standing up. Comparing from
t=0 therefore lines an already-standing simulator against a robot still getting
to its feet. Use `--start 6` to anchor past it.

(The v11 run sits ~6 cm lower throughout, mean tracker z 0.180 vs 0.241 for the
same mount -- consistent with its deliberate gait-height sweep down to 0.10.)

---

# Trajectory error is the wrong objective; use distributions

## The gait is chaotic

Perturbing only the initial joint angles and replaying the same commands:

| perturbation | @1 s | @3 s | @12 s |
|---|---|---|---|
| sigma = 0.05 rad (~3 deg) | 2.3 cm | 6.6 cm | **136 cm** |
| sigma = 0.30 rad | 9.5 cm | 23.0 cm | **58 cm** |

Not monotonic -- the *smaller* perturbation ends up further away. Long-horizon
position is uncorrelated with the initial error, so knowing the robot's true
starting joint angles (which these recordings do not contain anyway) would buy
only the first few hundred milliseconds.

## Which means trajectory matching has almost no signal

Sim-vs-real windowed displacement error, against a chaos floor measured as two
simulators 3 degrees apart:

| window | sim vs real | chaos floor | ratio |
|---|---|---|---|
| 0.5 s | 24.0 cm | 15.3 cm | 1.57 |
| 2 s | 77.8 cm | 51.7 cm | 1.50 |
| 10 s | 145 cm | 165 cm | **0.88** |

At a 10 s horizon the sim-real gap is *smaller* than the noise floor. Fitting
physics parameters to trajectory error cannot work.

## Distributional statistics do survive

`metrics.py`. Over 40 s of himloco01 from `--start 6`, with the chaos spread
taken across four perturbed rollouts:

| statistic | sim | real | diff | chaos | SNR |
|---|---|---|---|---|---|
| speed (m/s) | 0.410 | 0.389 | 0.021 | 0.022 | 1.0 |
| speed_gain | 0.780 | 0.759 | 0.022 | 0.048 | 0.5 |
| height_std | 0.036 | 0.024 | 0.012 | 0.004 | **2.7** |
| gait_hz | 0.575 | 0.675 | 0.100 | 0.150 | 0.7 |
| yaw_rate_gain | -0.084 | 0.528 | 0.612 | 0.029 | 21.1 (unreliable) |
| height_mean | 0.292 | 0.319 | 0.027 | 0.003 | 8.7 (not comparable) |

Read this carefully:

* **speed and speed_gain agree** between sim and hardware to within the chaos
  noise. That is a genuine result -- the policy's translational response
  transfers.
* **height_std is the one usable discriminator so far**: the simulated body bobs
  50% more than the real one (0.036 vs 0.024 m), three times the noise.
* **height_mean is not comparable.** The recorded value is fixed by the unknown
  tracker offset and the anchor height, both chosen by hand.
* **yaw_rate_gain is not trustworthy yet.** A clean constant command gives the
  simulator +0.63, correct sign in both directions, but against the recorded
  schedule it reads -0.08. The recorded turn commands alternate faster than the
  0.4 s filter and the policy's own lag, so an instantaneous regression is the
  wrong estimator. Needs lag compensation or cross-correlation.
* **gait_hz** at 0.58-0.68 Hz is too low for a trotting Go2 (expect ~2 Hz); the
  FFT is locking onto drift rather than the gait. The band or the detrending
  needs work.

## Two estimator bugs found and fixed here

1. **Sample-window filtering.** Differentiating a 253 Hz recording and a 50 Hz
   rollout with the same 25-*sample* window applies 0.1 s of smoothing to one
   and 0.5 s to the other. With Vive dt jitter as large as the interval itself
   (mean 3.96 ms, std 3.88 ms) this reported the robot walking at **3.87 m/s**.
   Everything now resamples to a uniform grid and smooths by a window in
   seconds.
2. **Mean of ratios.** `mean(achieved / commanded)` explodes near zero command
   and cancels across sign flips. Replaced by a least-squares slope through the
   origin.

Also: `vive.read_vive_pose` now takes its clock from the payload's `t_host`
rather than mcap `log_time` -- monotonic, and its worst gap is 15 ms against
92 ms. The payload's own `ts` goes backwards at 91 points and is unusable raw.

---

# Correction: `--start` was not reaching the commands

Every sim-vs-real number above this line, from any run using `--start`, was
wrong. A patch adding the offset to `cmd_at()` silently failed to apply while
the same offset *did* land on the ghost lookup, so the simulator was driven by
the first seconds of a run and scored against a ghost six seconds later. It
looked entirely plausible -- the robot walked, the numbers were finite -- which
is why it survived several rounds. `test_start_offset_reaches_the_command_schedule`
now pins it.

The tell, once looked at: the commands the simulator actually applied had
`vx = 0` where the schedule at that time said `vx = 0.422`.

## Results with commands correctly aligned

40 s of himloco01 from `--start 6`, chaos spread over four perturbed rollouts:

| statistic | sim | real | diff | chaos | SNR |
|---|---|---|---|---|---|
| **gait_hz** | 3.301 | 1.750 | 1.550 | 0.450 | **3.4** |
| **yaw_lag** (s) | 0.070 | 0.490 | 0.420 | 0.030 | **14.0** |
| **yaw_rate_gain** | 0.555 | 0.697 | 0.142 | 0.021 | **6.7** |
| **height_std** | 0.032 | 0.024 | 0.008 | 0.003 | **2.8** |
| speed (m/s) | 0.441 | 0.389 | 0.052 | 0.028 | 1.9 |
| speed_gain | 0.887 | 0.839 | 0.048 | 0.042 | 1.1 |
| speed_lag (s) | 0.290 | 0.390 | 0.100 | 0.030 | 3.3 |
| height_mean | 0.296 | 0.319 | -- | -- | not comparable |

A coherent picture, and it matches watching the viewer: **the simulated gait is
about twice as fast as the real one** (3.3 vs 1.75 Hz), bobs 33% higher, and
responds to a turn command seven times more quickly (0.07 s vs 0.49 s) while
turning less per unit command. The simulated robot is skittering where the real
one strides.

Translation still transfers well -- speed and speed_gain agree to within about
twice the noise.

These four are the physics-fitting targets: **gait_hz, yaw_lag, yaw_rate_gain,
height_std**.

## Estimator fixes in this pass

* **gait_hz** now high-passes by subtracting a 1 s moving average and windows
  with a Hann taper before the FFT, and the search band starts at 1.0 Hz. It
  was locking onto the robot's slow drift around the room and reporting
  0.58 Hz. There is no ground-truth gait rate to check against -- HIMLoco
  free_walk has no clocked gait; the only explicit rate in the fleet is 1.5 Hz
  on an experimental trot-clock policy (`go2web policies/experimental/jun05.rs`)
  -- so this is a plausibility check, not a calibration.
* **Command gains** now search for the policy-to-body lag by cross-correlation
  and regress at that lag, reporting it alongside the gain. The lag turns out
  to be a discriminator in its own right, and the strongest one found so far.
* 50 Hz is the *control* rate (`himloco.rs:185`, "Stateful 50 Hz controller"),
  not a gait frequency -- it confirms `walk.CONTROL_DT = 0.02`.

---

# Retraction: "the simulated gait is twice as fast" was one bad window

`gait_hz` is not stable enough to have said that. Same configuration, same
recording, varying only the window length:

| window | sim gait_hz | real gait_hz |
|---|---|---|
| 15 s | 1.54 | 1.13 |
| 25 s | 1.52 | 1.68 |
| **40 s** | **3.30** | 1.75 |
| 45 s | 1.51 | 1.18 |

The 3.30 is an outlier -- almost certainly a harmonic winning the FFT on that
one window -- and it is the number the previous section built its headline on.
Read across windows, sim sits near 1.5 Hz and the real robot wanders between
1.1 and 1.75. They are not obviously different, and **gait_hz must not be used
as a fitting target until it is estimated properly** (autocorrelation of
vertical velocity, or a median across sub-windows, rather than a single FFT
peak).

What survives as stable and discriminating: **height_std, yaw_rate_gain,
yaw_lag**, with speed and speed_gain agreeing between sim and hardware.

# First physics sweep

Leg-joint parameters, against the real targets (gait_hz 1.75, height_std 0.024,
speed 0.389), 25 s window:

| armature | gait_hz | height_std | speed |
|---|---|---|---|
| **0.01** (menagerie default) | 1.52 | 0.033 | 0.478 |
| 0.03 | 1.28 | 0.035 | 0.483 |
| 0.06 | 1.00 | 0.037 | 0.503 |
| 0.10 | 1.08 | 0.047 | 0.484 |
| 0.20 | 1.08 | 0.062 | **0.106** |

| damping | gait_hz | height_std | speed |
|---|---|---|---|
| **2.0** (default) | 1.52 | 0.033 | 0.478 |
| 4.0 | 1.20 | 0.079 | 0.212 |
| 8.0 | 1.04 | 0.077 | 0.125 |

Two things to take from this:

* **No single parameter closes the gap.** Raising armature pulls `gait_hz` down
  but pushes `height_std` *away* from the real 0.024, and past 0.1 it collapses
  forward speed. Damping degrades speed hard for little gain. The objective is
  genuinely multi-dimensional, which is the honest case for a search rather
  than hand-tuning.
* **The menagerie defaults are already the best row here** on gait_hz and speed.
  Whatever makes the simulated robot look wrong in the viewer is not a simple
  under-damping of the leg joints.

Worth searching next: `armature`, `damping`, `frictionloss`, geom `friction`
(currently 0.6), contact `solref`/`solimp`, and trunk mass/inertia. But not
before the judge is trustworthy -- a search against a statistic that swings
2x on window length will happily fit the noise.
