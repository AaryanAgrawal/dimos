# Fiducial relocalization — findings

Raw record of everything found and built. The PR selects and orders from this. Each finding
marks its evidence and status. All code is local, nothing pushed.

## Summary
A fiducial marker prior was added inside the relocalization module (Jeff's jnav aggregation, judged
by the shared wall-fitness referee). It does not win yet. The gap is not plumbing — the prior
detects, gates, fuses, and proposes almost every cycle — it is marker-pose accuracy. Single planar
tags mirror-flip on the Go2's wide fisheye, and the camera is on a static, not-per-unit calibration.
A held-out benchmark (survey1 map -> survey2 drive) and an `--eval` collector were built to measure
this honestly, which dimos did not have before.

## What was built (landed, tested)
- Ambiguity-ratio gate in the detector (`marker_detect` -> `marker_transformer` -> stream module
  config, on for the reloc detector, off for other callers). 44 fiducial tests pass.
- Proposal census log (`relocalize candidates: ransac=N fiducial=M`) and a per-candidate judge log
  (`judge: ransac=0.72 fiducial=0.41 (* winner)`), fires when a prior competes.
- Hyperparameters -> Config: `min_local_points`, `reloc_interval_s`, `gravity_tilt_max_deg`
  (threaded through relocalize/refine_candidates), plus `ambiguity_ratio_min`, `marker_map_file`.
- JSON marker loader; health log source-first; blueprint renamed to
  `unitree-go2-relocalization-fiducial`.
- `RelocEval` collector module + `--eval` flag + held-out `eval.py` + Umeyama truth-alignment +
  top-down trajectory plot (colored by winning prior + markers). score_replay 0-accept guard.
- Docs: `relocalization.md` updates + new `relocalization_benchmark.md`.
- 57 relocalization/perception tests pass; ruff + mypy clean.

## Findings (evidence)

1. **The prior works end to end; it loses on accuracy, not plumbing.** Held-out survey2 vs survey1
   map: prior proposed in 24/28 cycles (1-4 candidates), won 0. Every fix `source=ransac`. STATUS: solid.

2. **Why ransac wins, quantified (judge log).** Fiducial reaches the top-10 at wall fitness
   ~0.29-0.59, consistently ~0.05-0.15 below ransac's best (~0.35-0.64). Closest: fiducial=0.556 vs
   ransac=0.607. A near-miss, not garbage. STATUS: solid, measured.

3. **The marker error is a mirror flip (IPPE planar ambiguity).** Per-tag orientation is bimodal,
   two clusters ~90-180 deg apart, both reprojecting <1.2 px. Deviation-from-medoid histogram:
   tag 2 = 4 near / 5 flipped; tag 6 = 2 near / 11 flipped; tag 7 tighter. Absolute reproj cannot
   separate them (reproj<=1 px still spans 179 deg). STATUS: solid, measured. Refs: Collins &
   Bartoli 2014; Schweighofer & Pinz.

4. **The ambiguity-ratio gate fixes it partially.** Re-surveying survey1 with the gate cut
   orientation spread: tag 2 88.5 -> 22.5 deg (19 -> 9 sightings), tag 7 53.9 -> 5.8, tag 5 23 -> 21.
   Tag 6 stayed 46 -> 48 (gate dropped 1 of 113) -> tag 6 is not a simple flip (calibration or a hard
   viewing geometry). STATUS: solid.

5. **The medoid/Huber aggregation works on clean-majority input, fails on flip-majority.** Unit test
   (2 deg synthetic noise) fused < median single; real data tags 2/7 cleaned; tag 6 (11/14 flipped)
   -> medoid followed the flip (94 deg after Huber). Majority-robust, not magic. STATUS: solid.

6. **Relocalization is non-deterministic.** Same premap, same recording, 3 runs: 22 / 20 / 18
   accepts, different poses (first fix 0.753/(9.06,7.10) vs 0.765/(9.79,3.22), same n_pts). RANSAC is
   unseeded in the live pipeline. Implication: a single benchmark run is not reproducible; the eval
   must peg seeds or average N runs. Validates DIM-944. STATUS: solid, measured.

7. **Camera calibration is a suspect for the residual (tag 6).** One static intrinsic
   (`front_camera_720.yaml`, 92 deg wide fisheye, small distortion coeffs) for every Go2, not
   per-unit. Under wrong intrinsics solvePnP fits a wrong pose that still reprojects <1.2 px, so
   low reproj masks it. STATUS: plausible, not isolated (needs a checkerboard). DIM-1308.

8. **No tool exists to detect degraded/wrong intrinsics.** `cameracalibrate` makes a calibration,
   `apriltag` prints a board, `fixture_verification` checks board images — none measures live
   calibration health. And reproj error alone cannot detect it (the flip reprojects well); the tool
   must check pose consistency. STATUS: gap. Added to DIM-1308.

9. **dimos had no held-out relocalization accuracy eval.** Unit test = synthetic same-scene; replay
   test = same-recording premap and only checks a fix publishes. The survey1/survey2 held-out test is
   the honest one and did not exist. STATUS: solid.

10. **The `--eval` architecture.** dimos swaps the connection by config, so one blueprint runs replay
    and live; `--eval` only adds a collector. Held-out is the rule (premap from A, replay B of the
    same scene). Recording gives accuracy (PGO truth), live gives a health report. STATUS: designed +
    built.

11. **The sf_office lidar baseline is confidently-wrong.** 8/47 within 1 m, median 0.41 m at fitness
    ~0.6-0.8 (part of the 0.41 m is the premap-vs-truth PGO offset). So the scene wants a second
    signal. STATUS: solid, with the truth-floor caveat.

## Corrections made (honesty log)
- "Lidar nails sf_office" was wrong — it is 8/47 within 1 m.
- "46% aggregation, unit-tested" was wrong — the 46% (3.47 -> 1.87 m) is `aggregation_fused_vs_single`
  on village3 tag-10, measured on the ambiguity-gated set (the gate that was off live). The unit test
  only asserts fused < median-single on synthetic noise.
- Earlier in the effort: the mid360 "gravity-gate bug" was a harness artifact (a body-frame submap
  re-anchor the pipeline never does), not a production bug.

## Held-out benchmark (the method + the result)
- Pair: survey1 (premap + gated marker map) -> survey2 (replay). Same scene confirmed (survey2
  relocalized: 18-22 accepts across runs). Never same-recording.
- Result today: prior proposes, wins 0, near-miss on fitness. med_err held out until survey2's
  markers are surveyed (to anchor the Umeyama map_A -> map_B alignment).
- Other same-scene pairs on disk: park1 -> park1_2 (mid360, outdoor, 2nd fiducial pair, park1_2 needs
  a survey); HK office / building (tag-free -> lidar-only).

## Open / next
- Close the last ~0.1 fitness on marker accuracy: stricter ambiguity ratio (test 3-4 on tag 6),
  per-unit calibration (DIM-1308), close frontal views, or a multi-tag board.
- Survey survey2's markers so held-out med_err is a real number.
- Determinism: peg RANSAC seeds or average N runs in the eval.
- Native `--eval` CLI wiring (collector module is built; flag is best-effort).

## Tickets
- DIM-1281 fiducial prior. DIM-944 tuning system (hyperparameters lead in; no autotune built).
  DIM-1308 Go2 per-unit calibration + degraded-intrinsics tool.

## Data / artifacts
- USB: `go2_recordings/2026-07-18_sf_office_survey1` and `..._survey2` (dbs + premap + marker maps +
  PGO graph + sightings + rrd + intrinsics + summary.md).
- Trajectory plot: `trial/harness/out/eval/survey2_heldout.trajectory.png`.
- Gated marker map: `trial/harness/out/premaps/sf_office_20260718_survey1/` (`.marker_map.json` + `.yaml`).

## References
- jnav aggregation (Jeff): `dimos/navigation/jnav/utils/apriltags.py`.
- IPPE: Collins & Bartoli 2014. Planar pose ambiguity: Schweighofer & Pinz. AprilTag: Olson 2011.
  Markley quaternion mean 2007. Umeyama 1991.
