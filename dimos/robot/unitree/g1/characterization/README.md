# G1 GR00T characterization

This package applies Ivan's Go2 process to G1 without changing the GR00T policy or its commands:

```text
hardware mem2
├── cmd_vel + Point-LIO world_T_pelvis
│   └── six-direction command response and observed envelope
└── motor_command + motor_states + IMU
    └── fixed q/dq/torque/root clips
         ├── four command levels per direction -> fit
         └── four disjoint levels per direction -> held-out check
              └── candidate leg physics
                   ├── unchanged GR00T task screen
                   └── actual unitree-g1-groot-replay promotion gate
```

The low-level fit minimizes the candidate-to-baseline residual ratio across joint position,
joint velocity, estimated motor torque, pelvis position, and pelvis rotation. All six directions
have equal numbers of clips. The policy is only a constraint: a candidate is retained when the
held-out plant residual improves, mean response NRMSE is below 15%, every direction is below 20%,
and the full replay remains upright.

For direction `d`, the response metric is

```text
delta_v_d(t) = sign_d * (v_d(t0 + t) - median(v_d before t0))
NRMSE_d = RMS(delta_v_sim - delta_v_hardware) / max(abs(delta_v_hardware))
```

Each trace uses its own sent-command timestamp `t0`, so replay transport delay is not mistaken for
plant delay. Point-LIO is a measured reference, not ground truth; only the MuJoCo root pose is
SIMULATED ground truth.

The identified settings apply only to the 12 leg joints. Waist and arm physics, foot friction,
contact time, GR00T ONNX weights, external twists, gains, and control rates remain unchanged.

| setting | baseline | identified | unit |
|---|---:|---:|---|
| leg armature | 0.010000000 | 0.013836205 | kg m^2 |
| leg damping | 0.001000000 | 0.000563844 | N m s/rad |
| leg friction loss | 0.100000000 | 3.250000000 | N m |

## Reproduce

Materialize the LFS recording, then rerun the direction-balanced candidate check:

```bash
uv run python -c "from dimos.utils.data import get_data; get_data('g1_groot_characterization_2026-08-27.db')"
.venv/bin/python -m dimos.robot.unitree.g1.characterization.plant_fit \
  data/g1_groot_characterization_2026-08-27.db --out=/tmp/g1-groot-fit \
  --seed=0 --candidate 0.013836205 0.000563844 3.25
```

Replay the exact 317.2 s hardware command trace through the production blueprint:

```bash
timeout -s INT 370 .venv/bin/dimos --simulation mujoco run unitree-g1-groot-replay \
  --recording=data/g1_groot_characterization_2026-08-27.db \
  --lead-in-s=5 --db-path=/tmp/g1-groot-tuned.db \
  --enable-pointcloud=false --enable-mujoco-lidar=false --record-tf=false
```

Generate the result bundle and 6x comparison video:

```bash
.venv/bin/python -m dimos.robot.unitree.g1.characterization.report \
  data/g1_groot_characterization_2026-08-27.db \
  --simulation=/tmp/g1-groot-tuned.db \
  --out=dimos/robot/unitree/g1/characterization/results/2026-08-27/tuned \
  --video --video-speed=6
```

Use `--baseline-plant` when regenerating the raw stock report after the production MJCF has been
tuned. JSON artifacts include recording and model hashes, git revision and dirty state, seed,
exact command, frames, units, timing health, and stability health.
