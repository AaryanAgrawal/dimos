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
