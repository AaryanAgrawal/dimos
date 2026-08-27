# G1 GR00T characterization - 2026-08-27

The identified production plant passes the full actual DimOS replay. The Point-LIO hardware pose
is a measured reference; the MuJoCo root is SIMULATED ground truth. GR00T ONNX weights and the 133
recorded command transitions are unchanged.

## Six-direction response

NRMSE compares the baseline-subtracted first 2.0 s after the eight highest command levels in each
direction, using each run's actual command timestamp.

| direction | stock SIMULATED | tuned SIMULATED | change |
|---|---:|---:|---:|
| forward | 12.26% | 7.66% | -37.5% |
| backward | 14.51% | 15.04% | +3.6% |
| left | 21.26% | 18.58% | -12.6% |
| right | 22.95% | 14.91% | -35.0% |
| CCW | 18.41% | 14.61% | -20.6% |
| CW | 22.40% | 14.73% | -34.2% |
| **mean** | **18.63%** | **14.25%** | **-23.5%** |

Backward response worsens by 0.53 percentage points but remains below the 20% direction gate. The
worst tuned direction is left at 18.58%.

## Plant and trajectory checks

Alternating levels from each direction's highest eight give 24 fit clips and 24 disjoint held-out
clips. The held-out normalized residual is 0.877, a 12.34% improvement over stock. Every measured
channel improves: joint position 13.38%, joint velocity 18.29%, estimated torque 11.81%, pelvis
position 6.38%, and pelvis rotation 11.85%.

| full 317.2 s trajectory metric | stock SIMULATED | tuned SIMULATED | change |
|---|---:|---:|---:|
| position RMSE | 5.212 m | 4.337 m | -16.8% |
| position p90 | 7.793 m | 6.467 m | -17.0% |
| final position separation | 7.607 m | 6.931 m | -8.9% |
| yaw RMSE | 0.792 rad | 0.418 rad | -47.2% |
| yaw p90 | 1.410 rad | 0.660 rad | -53.2% |
| final yaw separation | 1.920 rad | 0.522 rad | -72.8% |

The tuned replay is stable: minimum SIMULATED pelvis height is 0.716 m, maximum absolute roll is
0.123 rad, and maximum absolute pitch is 0.167 rad. Command levels are exact and transition timing
p95 error is 0.052 s. Long-horizon separation is reported as a sanity check, not used as the fit
objective, because small velocity and yaw errors accumulate over this interleaved 317.2 s trace.

## Evidence

- `tuned/plant_fit.json`: settings, direction-balanced train/held-out residuals, policy screen, hashes, and exact command.
- `tuned/hardware_vs_sim.json`: full production-blueprint response, trajectory, command, and stability metrics.
- `tuned/hardware_vs_sim.mp4`: 6x Point-LIO measured reference beside SIMULATED MuJoCo ground truth.
- `tuned/hardware_vs_sim.png`: full planar trajectory and error trace.
- `tuned/hardware_vs_sim_envelope.png`: measured versus SIMULATED settled directional response.
- `tuned/plant_grounding.json`: low-level q/dq/torque/root replay scores.
- `raw/`: the same evidence for the stock baseline.

This is one hardware run with command levels held out inside that run, not an independent hardware
validation set. Point-LIO pose updates are 10.0 Hz with a 0.233 s maximum gap, so dead-time fits are
not reliable. A second recording with fresh pose timestamps at 20 Hz or faster is still required
before treating the observed maximum commands as planner limits.
