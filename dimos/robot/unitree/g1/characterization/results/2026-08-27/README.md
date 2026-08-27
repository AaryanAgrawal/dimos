# G1 GR00T characterization - 2026-08-27

This is the first raw baseline: recorded hardware is the Point-LIO measured reference, and MuJoCo
root pose is SIMULATED ground truth. The replay used the exact recorded twist levels and current
`balance.onnx`/`walk.onnx`; no ONNX weights or command gains were changed.

## Raw result

| direction | max command tested | hardware achieved | SIMULATED achieved | max error | response NRMSE |
|---|---:|---:|---:|---:|---:|
| forward | 0.864 m/s | 0.806 m/s | 0.781 m/s | -3.1% | 16.1% |
| backward | 0.405 m/s | 0.640 m/s | 0.470 m/s | -26.5% | 18.6% |
| left | 0.387 m/s | 0.263 m/s | 0.304 m/s | +15.5% | 29.9% |
| right | 0.495 m/s | 0.343 m/s | 0.493 m/s | +44.0% | 35.3% |
| CCW | 1.746 rad/s | 1.281 rad/s | 1.523 rad/s | +18.9% | 16.4% |
| CW | 1.089 rad/s | 0.722 rad/s | 1.004 rad/s | +39.0% | 26.0% |

Response NRMSE compares settled speed at every tested level at or above the hardware motion floor,
normalized by that direction's maximum measured speed. The full 317.2 s path ends 7.61 m and 1.92
rad apart, but that accumulated drift is secondary to the reset-per-command response on a long run.

The replay was stable: minimum SIMULATED pelvis height was 0.725 m, maximum absolute roll was 0.132
rad, and maximum absolute pitch was 0.197 rad. All 133 command transitions had the exact level
sequence; transition timing p95 error was 0.051 s.

Point-LIO published at 29.8 Hz but produced fresh poses at 10.0 Hz with a 0.233 s maximum gap, so
dead-time fits are not reliable. Its pose derivative agrees with reported linear twist, but reported
yaw rate has the opposite convention; the response analysis therefore uses the pose derivative.

## Plant grounding

Eight seeded 8.0 s open-loop clips replay the recorded GR00T q/dq/kp/kd/torque command against the
production model. Median SIMULATED errors are 0.030 rad joint position, 0.433 rad/s joint velocity,
3.43 N m motor torque, 0.055 m pelvis position, and 0.073 rad pelvis rotation.

A five-knob fit varied leg armature, damping, friction loss, foot friction, and contact time. It
improved several channels but worsened the worst directional response. It is rejected; stock
MuJoCo remains canonical.

## What the next hardware run must add

1. Publish fresh Point-LIO pose at least 20 Hz with sensor timestamps; keep every current stream.
2. For each single axis, hold zero for 5 s, command 25%, 50%, 75%, and 100% of the maximum tested
   above for 5 s each, and return to zero for 5 s between levels. Repeat every direction twice.
3. With a safety operator, try 110% only when the previous level settled safely. Stop increasing
   when achieved-speed gain drops by 15%, balance degrades, or the operator stops the run.
4. Record a second independent run for held-out validation. Do not tune on and report the same run.

No planner ceiling is emitted from this run because no saturation was observed. The values above
are maximum commands tested, not policy limits.

## Artifacts

- `raw/hardware_vs_sim_envelope.png`: the local response comparison that decides closeness.
- `raw/hardware_vs_sim.mp4`: 6x measured-reference versus SIMULATED replay.
- `raw/plant_grounding.json`: seeded low-level q/dq/torque/root scores and hashes.
- `raw/hardware_vs_sim.json`: exact command-replay, stability, response, and trajectory metrics.
- `raw/g1_groot_characterization.json`: hardware health, steps, motion floors, and envelopes.
