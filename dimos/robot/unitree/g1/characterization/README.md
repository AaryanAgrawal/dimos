# G1 GR00T characterization

This package keeps Ivan's two questions separate:

```text
hardware mem2
├── tele_cmd_vel -> cmd_vel timing/value check
├── fresh Point-LIO world_T_mid360 + measured waist FK -> world_T_pelvis
│   └── directional gain, delay, time constant, motion floor, observed envelope
└── motor_command + motor_states + IMU
    └── seeded open-loop clips against the production G1 MuJoCo model

same cmd_vel ──► actual unitree-g1-groot-replay blueprint
             └── SIMULATED pelvis ground truth vs Point-LIO measured reference
```

The plant is never fit through the policy. A `PlantReplayPlan` freezes every command,
reference sample, and reinitialization before MuJoCo loads. The closed-loop replay uses the actual
GR00T task and ONNX models; the green pelvis ghost is visual-only and has no collision or physics
state.

Point-LIO published at 29.8 Hz but updated its pose at 10.0 Hz in this run, so repeated poses are not
differentiated. The directional maximum is the largest command tested, not a claimed policy
ceiling. Motion floor is the first response above four times the same run's stationary Point-LIO
noise.

The replay comparison checks the piecewise-constant twist level sequence consumed by GR00T. Raw
duplicate event counts may differ without changing that signal; transition timing and ZOH error are
reported separately. No planner limits are emitted until a second run observes saturation and
improves lateral/yaw fit health.

## Run

Pull the LFS recording, then replay its exact twist stream through GR00T:

```bash
uv run python -c "from dimos.utils.data import get_data; get_data('g1_groot_characterization_2026-08-27.db')"
timeout -s INT 360 .venv/bin/dimos --simulation mujoco run unitree-g1-groot-replay \
  --recording=data/g1_groot_characterization_2026-08-27.db \
  --lead-in-s=5 --db-path=/tmp/g1-groot-replay.db \
  --enable-pointcloud=false --enable-mujoco-lidar=false --record-tf=false
```

Generate the evidence bundle and a 6x Go2-style comparison video:

```bash
.venv/bin/python -m dimos.robot.unitree.g1.characterization.report \
  data/g1_groot_characterization_2026-08-27.db \
  --simulation=/tmp/g1-groot-replay.db \
  --out=dimos/robot/unitree/g1/characterization/results/2026-08-27 \
  --video --video-speed=6
```

Every JSON artifact records the source SHA-256, git revision, seed, exact command, stream health,
frames, units, and GR00T model hashes. Point-LIO is always labelled as a measured reference; only
the MuJoCo root pose is SIMULATED ground truth.
