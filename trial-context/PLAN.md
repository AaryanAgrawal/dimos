# Plan: fiducial relocalization + held-out benchmark

Living plan. Update after every step. Source of truth for this effort.

## Goal
A fiducial marker prior inside the relocalization module that measurably improves accuracy where lidar is weak, and a held-out benchmark (--eval) that proves it honestly on same-scene pairs, replay or live.

## Where we are (2026-07-20)
Prior built, judged, fused, no regression. Ambiguity gate landed in the detector. Held-out benchmark works: survey2 relocalizes against survey1's map, the prior proposes in 24/28 cycles but wins 0 (loses on wall fitness because marker poses still carry flip + calibration error). The gate cut survey orientation error (tag 2 88->22, tag 7 54->6) but tag 6 stays 48 (not a simple flip; calibration/geometry). No plumbing gap left; the gap is marker-pose accuracy.

## Architecture (the right shape)
- **The blueprint is the eval.** dimos swaps the connection (replay vs webrtc vs sim) by config; modules are identical. So one blueprint runs replay and live; --eval only adds a collector. Replay and recording are inverses over the same db (record = live + Recorder writing; replay = ReplayConnection reading).
- **--eval is a collector module**, not a post-hoc harness, so it works live and replay in-process. It subscribes to the published world->map TF, odom, and the reloc health, computes per-source stats at shutdown, prints the table, and writes the trajectory plot + json. On replay it also loads PGO truth for accuracy.
- **Held-out is the rule.** Premap from run A, replay run B of the same scene. Never same-recording (that is memorization). The same-scene test IS the relocalization succeeding.
- **Everything tunable is config**, ready for DIM-944 sweeps.

## Same-scene pairs (from benchmark_setup.yaml + disk)
- sf_office survey1 (premap + gated marker map) -> survey2 (replay). Primary, 2D, go2, fiducial. Confirmed.
- mid360_gir_park1 -> park1_2. Same GIR park, mid360, outdoor. Secondary; park1_2 needs a marker survey; mid360 tilt caveat.
- go2_hongkong_office, hk_building_all_around: tag-free -> lidar-only held-out (no prior).

## Build (make + test everything)
Track A - dimos module consolidation:
- [x] Ambiguity gate in the detector (marker_detect + marker_transformer + module config, gate on for the stream module).
- [x] Proposal census log (relocalize candidates: ransac=N fiducial=M).
- [ ] Hyperparameters -> config, same shape as fitness_threshold: min_local_points, gravity_tilt_max_deg, reloc_interval_s, and the aggregation gates. Thread gravity_tilt into relocalize()/refine_candidates.
- [ ] Marker loader reads JSON ({meta, markers:{id:{translation,rotation}}}); convert the sf marker maps to JSON.
- [ ] Health log: source first.
- [ ] Blueprint rename -> unitree-go2-relocalization-fiducial.

Track B - the eval:
- [ ] RelocEval collector module: subscribes world->map TF + odom + health; at shutdown prints the per-source table (won, %traj, med_err, med_fit, success) + proposed-vs-won; writes json + trajectory plot (colored by winning prior + markers).
- [ ] --eval flag on `dimos run` that attaches it (replay and live).
- [ ] Held-out accuracy: Umeyama-align survey1-map to survey2-truth so per-source med_err is real, not "-".
- [ ] score_replay 0-accept guard.

Track C - docs:
- [ ] relocalization.md: promote held-out to a rule; update the log example with source=; fix the use_fiducial_prior row + add marker_map_file, ambiguity_ratio_min, and the tunables; pointer to --eval + the benchmark doc.
- [ ] New relocalization_benchmark.md: held-out method, --eval + its output, metrics, tunables (DIM-944), fiducial caveats (flip, calibration DIM-1308), the survey1/survey2 worked example.

Test:
- Unit tests for config/loader/gate. Lint + types. Run the held-out benchmark (sf_office survey1->survey2) end to end; read the census + source table + plot. Then park1->park1_2.

## The --eval output (agreed shape)
Per-source table: source | won | %traj | med_err | med_fit | success. Plus proposed-vs-won, coverage, first-fix, held-out note. Live: drop med_err/success, add drift/no-jump. Trajectory png: robot path in map frame colored by winning prior + markers.

## Open items / tickets
- Marker-pose accuracy is the one gap to a real benefit: stricter ambiguity ratio, close frontal views, per-unit calibration (DIM-1308). Tag 6 specifically.
- DIM-1281 fiducial prior. DIM-944 tuning (hyperparameters lead in; no autotune built). DIM-1308 Go2 calibration.

## References
- jnav aggregation (Jeff): dimos/navigation/jnav/utils/apriltags.py.
- IPPE planar ambiguity: Collins & Bartoli 2014; Schweighofer & Pinz. AprilTag: Olson 2011. Markley 2007.
- benchmark_setup.yaml (recording manifest), BENCHMARK_METHOD.md.
