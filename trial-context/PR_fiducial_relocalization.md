## Contribution path
- [x] Linked issue or discussion: DIM-920

## Problem
- Relocalizing onto a premap is lidar RANSAC alone.
- A sparse or repetitive room locks onto a confident wrong pose: fitness high, pose meters off, nothing catches it.
- RANSAC also cannot fire until the submap clears its point floor, so early acquisition has no answer at all.
- A surveyed AprilTag is a second signal, but one glimpse at range throws meters, so sightings need fusing first.

## What exists
- One prior: FPFH+RANSAC against the premap, one accept bar, publishing `world -> map` on `/tf`.
- `dimos map global --markers` detects tags and overlays them in rerun. It does not aggregate them, and writes no marker map.

## Result

The fiducial fix landed at 0.867 in 0.1 s; RANSAC took 7.3 s to reach 0.755. Higher fitness, ~70x faster, and it fired 40 seconds before RANSAC could, while the submap was still under the point floor.

## Solution
- Fiducial relocalization prior: a tag's sightings fused into one candidate, proposed on the burst edge.
- Premap creation: PGO-corrected tag aggregation writes `map_T_marker` per id alongside the cloud.
- Per-prior acceptance gating: shared geometric gates, then each prior clears its own bar.

```
SURVEY                                dimos map global <rec>.db --export --markers --marker-size
──────────────────────────────────────────────────────────────────────────────────────────────
 recorded images
      │
      ▼
 DetectMarkers ─► ambiguity gate      solvePnPGeneric returns BOTH IPPE poses; keep the winner
      │                               only if the mirror reprojects >= 2x worse (else it flips ~180°)
      ▼
 per-glimpse gates                    range · view angle · reproj px · tag px
      │                               conditions, not residuals: a uniformly bad set agrees with itself
      ▼
 PGO-correct into the map frame       one static frame, so ALL sightings of a tag fuse together
      │
      ▼
 robust_cluster_pose                  medoid seed      a REAL sighting, never between two modes
      │                               Huber IRLS       translation, w = δ/|r| past δ
      │                               Markley eigen    rotation; ±q is one rotation, so a plain mean cancels
      ▼
 min_observations                     under 3 in-gate -> dropped; Huber has no redundancy to outvote a flip
      │
      ▼
 <rec>.marker_map.json                map_T_marker per id · n_detections · marker_length_m
      │
══════╪═══════════════════════════════════════════════════════════════════════════════════════
LIVE  │
      ├──────────────► marker_length_m ─► the detector solves at the size the survey used
      │                                    (one source of truth; config is the fallback)
      ▼
 camera ─► MarkerDetectionStreamModule
             DetectMarkers ─► same ambiguity gate ─► same per-glimpse gates
             AggregateTagBursts    sliding window: the LIO world frame DRIFTS, so an old
                    │              sighting is a different frame. Must forget.
                    ▼ aggregated_detections        one robust pose per (marker, visit)
             FiducialPrior.observe()
                    map_T_marker @ inv(world_T_marker)
                    = map_T_⟨marker⟩ · ⟨marker⟩_T_world = map_T_world   ─► _pending (under a lock)
                    │
 lidar ─► VoxelGridMapper ─► local_map ─► RelocalizationModule
                    │                          │   priors constructed only when enabled
                    │                 ┌────────┴────────────┐
                    │            RansacPrior           FiducialPrior
                    │            every 2.0 s           on the burst edge
                    │            n_pts >= 50_000       no candidate needs lidar
                    │            FPFH + RANSAC         drains _pending
                    │                 └────────┬────────────┘
                    │                          ▼   ONE prior per fire, never pooled
                    └───────────────►  refine_candidates()
                                         gravity tilt ≤ 10°  ─► NoUprightCandidateError
                                         wall floor  ≥ 100   ─► InsufficientWallEvidenceError
                                         wall rerank ─► wall ICP ─► full-cloud ICP
                                              │ (T, fitness)
                                              ▼
                                       per-prior fitness bar     -o priors.<key>.fitness_threshold
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
                 inv(map_T_world) ─► /tf              accepted_fixes ─► rerun  (--eval only)
                                                      one sphere per fix, coloured by prior
```

- The judge solves `map_T_world`; `/tf` carries its inverse, `world_T_map`.
- A prior returns bare 4x4s and no confidence: ranking is the judge's job, and zero candidates is a valid answer.
- The accept line gains `source=`. `--eval` raises skipped and rejected to warning and prints a per-prior table at stop.

## Breaking Changes

None

---

## How to Test

**Hardware (Go2).** Run on the robot; see Result.

Take any premap recording and run it through once. One pass writes both the premap and the marker map:
```bash
dimos map global <rec>.db --export --markers --marker-size 0.10
#   -> <rec>.pc2.lcm  +  <rec>.marker_map.json
```
- `--export` implies `--pgo`.
- `--marker-size` is the printed tag edge in meters, default `0.10`. It sets metric scale at PnP, so a wrong value makes every marker pose wrong by a constant factor with no error.
- Pass it once, here. The survey stamps it into the marker map; give the same map to the detector below and it reads the size back, so the two cannot solve at different scales.

Then drive, three legs, same premap:
```bash
R="--robot-ip <ip> --unitree-aes-128-key <key>"
MAP=<space>.marker_map.json
M="-o relocalizationmodule.map_file=<space> -o relocalizationmodule.priors.fiducial.marker_map_file=$MAP -o markerdetectionstreammodule.marker_map_file=$MAP"

dimos $R run unitree-go2-relocalization $M                                                        # both priors
dimos $R run unitree-go2-relocalization $M -o relocalizationmodule.priors.ransac.enabled=false    # fiducial only
dimos $R run unitree-go2-relocalization $M -o relocalizationmodule.priors.fiducial.enabled=false  # lidar only
```

Read the first line before moving the robot:
```
relocalize priors   live=['ransac', 'fiducial']   inert=[]
```
- `fiducial` under `inert` means the marker map did not load.
- Then watch for a `relocalize:` line with `source=fiducial`.
- A fresh drive against a premap built from a different recording is the clean test.
- The fiducial leg publishes only while a mapped tag is in frame, so gaps are expected.

Replay, no robot:
```bash
dimos --replay --replay-db=<rec> run unitree-go2-relocalization --eval \
  -o relocalizationmodule.map_file=<space> \
  -o relocalizationmodule.priors.fiducial.marker_map_file=<space>.marker_map.json
```

- Every prior field takes `-o relocalizationmodule.priors.<key>.<field>`: `enabled`, `fitness_threshold`, `interval_s`, `min_local_points`, `marker_map_file`.
- Naming one prior leaves the other at its default.

```bash
uv run pytest dimos/perception/fiducial dimos/robot/test_all_blueprints_generation.py   # 34 passed
uv run ruff format --check dimos/ && uv run ruff check dimos/
uv run mypy dimos/mapping/relocalization/ dimos/perception/fiducial/
```
No test file is added. The one existing test edit is the detector's port assertion, which gains `aggregated_detections`.

## Improvements
- Fuse fixes so they converge instead of replace. The anchor is the last accepted fix today.
- Consume the tag covariance we already publish and drop.
- Bound candidate plausibility before ICP so the pool still competes: Mahalanobis against that covariance, scaled by odometry travelled. Nothing bounds a fix today.
- Corroborate a fix before publishing, at least for the first anchor.
- Accept on more than an inlier ratio: overlap cannot separate the right pose from one a period down a repeating corridor.
- Report the fitness of the pose we publish, not the stage before.
- Read the survey's marker size off the detector's config so the two cannot drift.
- Tune the ICP judge across several recordings.

## Next features
- Self-degrade: publish confidence, withdraw a stale anchor (AMCL). Brings a state machine.
- Keep top-N candidates across fires; the margin is the health signal (RTAB-Map).
- Reloc as a pose-graph edge under a robust loss (Cartographer).
- Derive the judge's density-dependent constants from the incoming cloud instead of one lidar's numbers.

## AI assistance
Claude Code with Claude Opus 4.8.

## Checklist
- [x] This PR is scoped to one issue or clearly stated problem.
- [x] I ran the relevant checks (`uv run pytest`, pre-commit) for the files I changed.
- [ ] I have reviewed and understood every line in this PR.
- [x] I disclosed AI assistance above.
- [ ] I have read and approved the [CLA](https://github.com/dimensionalOS/dimos/blob/main/CLA.md).
