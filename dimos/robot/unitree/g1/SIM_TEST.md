# G1 protective stops — test in simulation

What this branch ships:

```
 sources (many, dumb)          the latch (one)                  safe stop (per device)
 hosted-teleop button          ControlCoordinator               G1WholeBodyConnection
 any module with an EStop ref  estop() / clear_estop()          SimMujocoG1WholeBodyAdapter
                               set_estop(bool) kept              fall or flail -> one damping frame
                                                                 (kp=0, tau=0, kd kept), drop
                                                                 every later command, latch,
                                                                 then estop() up to the coordinator
```

The checks are two pure functions in `protective.py`, run on every state sample by both the hardware
connection and the MuJoCo adapter:

| check | trips when | default | why |
|---|---|---|---|
| fall | pelvis tilt off gravity > `max_tilt_deg` | 45 deg | standing reads ~2 deg; a fall passes 45 on the way down |
| flail | ≥ `flail_joint_count` joints past `flail_joint_speed_rad_s` at once | 3 joints, 5 rad/s | SIMULATED GR00T walk: one knee/ankle reaches 7.5 rad/s, never two joints past 5 at once; lifted 9–11 joints, fallen 13–17 (`context/flail-sim.png` in aaryan-dimensional) |

Per-embodiment defaults live on `G1WholeBodyConnectionConfig`. The MuJoCo adapter uses the same
constants. Only `stop()` / a restart clears the latch; `clear_estop()` on the coordinator never
resumes anything.

Measured on this Mac (SIMULATED, MuJoCo 3.10.0, real GR00T ONNX through the real `MujocoSimModule`
+ `ControlCoordinator`, git 431f889ca): the flail check fires 0.0–0.24 s after the feet leave the
floor and 0.26–0.30 s after a push starts — 0.38–0.45 s before tilt reaches 45 deg. A lifted G1 never
tilts past 39 deg, so the tilt check alone never catches a lift.

## Setup (Linux)

```bash
git fetch fork aaryan/estop && git checkout aaryan/estop
uv sync --group tests
```

MuJoCo 3.10.0 and onnxruntime come with the `tests` group. `get_data` pulls the LFS bundles
(`groot`, `g1_urdf`, `mujoco_sim`) on first use. The registered sim blueprint could not start on
`main` — `SimMujocoG1WholeBodyAdapter` had no `get_limits()` and the coordinator's Protocol check
refused it — fixed on this branch; confirm the blueprint itself comes up:

```bash
dimos --simulation mujoco run unitree-g1-groot-wbc
```

## The rig

`flail_experiment.py` runs the real GR00T policy in headless MuJoCo with a mocap hook at the torso.
One scenario per process (they share an SHM key — never two at once). Each run prints
`SIMULATED MuJoCo ... stop=True|False git=...` and saves every physics step (q, dq, tau, IMU,
height, contacts) to an `.npz`.

```bash
cd dimos && uv run --group tests python dimos/robot/unitree/g1/flail_experiment.py --scenario <name> --seed 0 --out /tmp/<name>.npz
```

`--no-stop` disables the protective check for the BEFORE case (that is `main`'s behaviour: nothing
stops the policy).

## Test 1 — fall

```bash
uv run --group tests python dimos/robot/unitree/g1/flail_experiment.py --scenario fall --seed 0 --out /tmp/fall.npz
```

A 300 N lateral push at the pelvis until the robot tips. Expect in the log:

```
E-STOP reason='flailing, N joints past 5 rad/s'
```

0.26–0.30 s after the push (the flail fires before the tilt: the policy goes haywire while tipping).
After it, the robot is limp — the printed `max|dq|` should be far below the `--no-stop` run of the
same seed (which reached 63–79 rad/s after impact). Seeds 1 and 2 push in different directions.

## Test 2 — lift

```bash
uv run --group tests python dimos/robot/unitree/g1/flail_experiment.py --scenario lift --seed 0 --out /tmp/lift.npz
uv run --group tests python dimos/robot/unitree/g1/flail_experiment.py --scenario lift_free --seed 0 --out /tmp/lift_free.npz
```

The hook raises the torso 0.5 m in 0.5 s and holds (`lift` welds it rigid, `lift_free` pins it so it
swings). Expect the flail trip within 0.25 s of the feet leaving the floor. `--lift-seconds 2.5` is a
gentle hoist; it trips the same way, so the trip is the policy air-stepping, not the yank.

## Test 3 — no false trip

```bash
uv run --group tests python dimos/robot/unitree/g1/flail_experiment.py --scenario stand --seed 0 --out /tmp/stand.npz
uv run --group tests python dimos/robot/unitree/g1/flail_experiment.py --scenario walk --seed 0 --out /tmp/walk.npz
```

5 s standing, 5 s walking at 0.5 m/s. Expect no `E-STOP` line. Measured here: walking peaks at
7.5 rad/s on one joint, never two joints past 5 rad/s at once.

## Test 4 — hardware walk baseline

The thresholds are from sim; sim standing is quieter than the real robot (0.02 vs 0.77 rad/s peak).
The 317 s hardware recording of the G1 walking six directions under GR00T
(`aaryan/g1-groot-characterization`) is the check that they do not false-trip on hardware:

```bash
uv run python -c "from dimos.utils.data import get_data; get_data('g1_groot_characterization_2026-08-27.db')"
uv run --group tests python -m dimos.robot.unitree.g1.protective_replay data/g1_groot_characterization_2026-08-27.db
```

It replays every motor sample through the shipped `stop_reason`. Expect `no trip`, and read off
`most joints past 5 rad/s at once` (must stay under 3, with room) and `max tilt`. If it trips,
raise `flail_joint_count` or `flail_joint_speed_rad_s` in `protective.py` from what it prints.
Not run on this Mac: the LFS object is not in the fork's store.

The G1 fall recording (176,426 samples at 756 Hz, two deliberate falls, PR #3691) runs through the
same command. It is not on this Mac; it was last in `/tmp/g1check/` on the G1. Expect the first trip
near t = 118 s.

## Test 5 — before / after clips for the PR

Same seed, same push: `--no-stop` then default. The rig records no video yet; the shortest path is a
`mujoco.Renderer` in `_Scenario.after`, one frame every 1/30 s, `MUJOCO_GL=egl`, burn
`SIMULATED MuJoCo 3.10.0 BEFORE|AFTER` into each frame, GIF into `aaryan-dimensional/context/` next
to the existing `estop_sim_*.gif`.

## On hardware

Not run. Hardware rung: `dimos run unitree-g1-groot-wbc`, lift the robot off the ground with the
harness — the log must show the `E-STOP` line and the joints must go soft (damping only) within a
quarter second; set it down, `dimos stop`, restart to clear.

## State and what is left

Done on this branch: the structure above, tests (`control/test_estop.py`,
`g1/test_wholebody_estop.py`, `simulation/adapters/whole_body/test_g1.py`), `get_limits` on the sim
adapter, `tick_loop.py` back to `main`, `origin/main` merged in.

Left:

- Run Tests 1–4 on Linux; the hardware walk replay decides whether the defaults ship as they are.
- Before/after clips (Test 5) and a PR body rewrite — the body still describes the Aug 26 design
  (coordinator-side haywire check, `set_estop` from the connection).
- Sam's point on dimos #3621: if the publish loop dies, `_on_motor_command` keeps forwarding. The
  fix is a freshness gate on the last LowState in `_on_motor_command`; not built.
- The control coordinator refactor (PR #3409, unstarted) replaces the connection's `estop()` call-up
  with `ConnectionStatus` and owns the ESTOPPED state. Nothing else here changes.
- The `EStop` Spec protocol now declares `estop()` / `clear_estop()`; hosted teleop still calls
  `set_estop(bool)` directly and is untouched.
