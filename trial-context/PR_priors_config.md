# feat(relocalization): fiducial prior + per-prior config for RelocalizationModule

## Contribution path

- [x] Linked issue / discussion: DIM-920 (relocalization)

## Problem

A robot waking up in a prior map where lidar geometry is ambiguous — a corridor, a bare room, a symmetric hall — relocalizes into the wrong place on wall fitness alone, and every nav goal after it inherits the error. RANSAC is the only source, it cannot answer until the live map is dense enough to search, and an AprilTag surveyed into that map has no way in.

## What this adds

Three things, and nothing else is load-bearing:

1. **The fiducial relocalization prior** — a surveyed AprilTag becomes an absolute fix that competes in the same judge as RANSAC.
2. **Markers** — detect tags, robustly fuse their sightings, and survey them into a marker map.
3. **Eval logging** — `--eval` logs a per-source table so you can see which prior won, live or under replay.

---

## 1 · The fiducial relocalization prior

`RelocalizationModule` used to have one hardcoded source — RANSAC. This generalizes "a relocalization source" into a **prior**: a thing that *proposes* candidate poses on its own trigger. Every prior's candidates go through the one shared judge; the winner is published with its `source`.

```
             lidar                                    camera
               │                                        │
       VoxelGridMapper                             DetectMarkers
       carve 0.05 m                          admit? ambiguity · reproj · size · range · angle
               │                                        │ admitted
               ▼                                        ▼
       global_map (world)                          TagAggregator  (3 in 5 s · Huber + Markley)
               │                                        │ aggregated_detections
       RansacPrior                                  FiducialPrior
       polled: time ≥ 2 s AND n_pts ≥ 50k          event: fires on the burst edge
               │ 34 candidates                          │ 1 candidate per pending tag
               │                                        │ map_T_marker @ inv(world_T_marker)
               └───────────►  refine_candidates  ◄──────┘
                    walls only · ≥100 pts · tilt ≤10° · ICP · max fitness
                              │
                    ACCEPT      per-prior fitness_threshold (0.60)
                              │
                    JUMP GUARD  ≤5 m/s · ≤45°/s · tracking only, first fix exempt
                              │
                    publish world_T_map on /tf
```

**The design is a Protocol + a discriminated union — the Strategy pattern, made declarative.** A prior is a *candidate source* (structural interface); a blueprint *declares* which priors run.

```
  RelocPrior (Protocol)  ── structural typing: a class conforms by SHAPE, not by inheriting ──
     name : str
     propose(global_map, local_map) -> list[Candidate]
        ▲                         ▲
        │ RansacPrior             │ FiducialPrior          ← implicit subtypes; neither says (RelocPrior)
        │  FPFH+RANSAC search     │  observe() a sighting → compose → has_pending → propose()

  PriorConfig  = Annotated[ RansacPriorConfig | FiducialPriorConfig, Field(discriminator="type") ]
                            └──────────── a discriminated (tagged) union ───────────┘
         type="ransac" │                    │ type="fiducial"
                       ▼                    ▼
        RansacPriorConfig            FiducialPriorConfig      ── both extend ──►  PriorConfigBase
          interval_s · min_points      marker_map_file                             enabled=True
                                       marker_length_m · aggregation                fitness_threshold=0.6
```

**The two triggers look like what they are** — RANSAC is polled on each cloud with a time+geometry gate; the fiducial is edge-triggered by a completed tag burst:

```
  module event handlers                          the prior's own state
  ─────────────────────                          ─────────────────────
  _on_local_map(cloud):                          RansacPrior  — a candidate source
     if now-last ≥ interval and n_pts ≥ min:       (the module owns the timer + points gate)
        fire ransac
     if fiducial.has_pending: fire fiducial       FiducialPrior — the burst is the trigger
        (cold-start: a pending fix, no cloud yet)    observe() fills _pending
  _on_aggregated_detections(burst):                  has_pending reports readiness
     fiducial.observe(each tag)                       propose() drains it (consume-on-use)
     if cloud cached and has_pending: fire
```

**The fiducial fix is one frame composition** — a surveyed tag pose against a live-detected one, judged like any other candidate:

```
  map_T_marker    from the surveyed marker map   ─┐
                                                  ├─►  map_T_world = map_T_marker @ inv(world_T_marker)
  world_T_marker  Huber-fused live sightings     ─┘         (then straight into refine_candidates)
```

Per prior, its own trigger and accept bar:

| prior | trigger | `fitness_threshold` | why that bar |
|---|---|---|---|
| ransac | cloud, time ≥ 2 s AND `min_local_points=50000` | 0.60 | a geometric search has only the walls it landed on |
| fiducial | a completed tag burst (3 sightings / 5 s) | 0.60 | a decoded id names the tag, it does not show the composed pose fits the walls |

`enabled` toggles a prior in or out; a fiducial-only or lidar-only blueprint runs the same path.

---

## 2 · Markers

Detect tags, gate each glimpse, and fuse a marker's sightings into one robust pose via `robust_cluster_pose` (Huber-IRLS translation + Markley quaternion mean). One fusion, two windows — because time means different things offline and online:

```
    OFFLINE  (dimos map … --markers)             ONLINE  (live reloc)
    all sightings, whole recording               sliding window: last 5 s, ≥3 sightings
    one static PGO-corrected frame               live LIO frame drifts → must forget
              └──────────► robust_cluster_pose (Huber + Markley) ◄──────────┘
    → one map_T_marker per id                    → world_T_marker to the FiducialPrior
       written to <premap>.marker_map.json
```

The offline survey is a single pass: PGO-corrected `world_T_tag` per sighting, grouped by id, fused, and written alongside the premap as the `marker_map.json` the fiducial prior loads. The live path streams `aggregated_detections` — one aggregated pose per completed burst — additively beside the detector's existing per-frame `detections`.

---

## 3 · Eval logging

`RelocEval` (`--eval`) listens on the real streams — no ground truth, live and under `--replay` alike — and logs a per-source table so you can see which prior is winning:

```
/tf   (world_T_map fixes) ─┐
/odom (robot path) ────────┤ RelocEval  ──►  per-source table (logged at exit + on Ctrl+C)
run-log verbose trace ─────┘                 source · prop · acc · rej · false · %traj · med_fit
```

The run log is the only place each accept's winning `source` + fitness live (the `/tf` carries neither), so `--eval` turns on the module's verbose trace and joins accepts to sources by translation.

---

## Breaking Changes

---
- `fitness_threshold` and `min_local_points` moved onto the prior entries. The old module-level keys raise, naming the new home.
- RANSAC's accept bar is 0.60, was 0.45. Shipped presets are not byte-identical.
- `priors` is required on `RelocalizationModule.Config`.
- `unitree-go2-relocalization` is replaced by three blueprints named by their priors: `-lidar`, `-lidar-fiducial`, `-fiducial`.
---

## Core changes — why

- **`perception/fiducial/apriltag_aggregation.py`** (new) — per-tag robust fusion: Huber-IRLS translation + Markley quaternion mean, each cited inline; a streaming aggregator for the live path.
- **`perception/fiducial/{marker_transformer,marker_detection_stream_module,marker_pose}.py`** — additive: a new `aggregated_detections` stream + burst aggregator beside the existing detector. Upstream `detections` untouched.
- **`mapping/relocalization/{module,priors,relocalize}.py`** — the prior pool, per-prior triggers/thresholds, jump guard, and the shared judge (`refine_candidates` unchanged from arkluc's #2160).
- **`mapping/utils/cli/map.py`** — `--markers` also fuses the sightings and writes `<premap>.marker_map.json` alongside the export.
- **`robot/cli/dimos.py`** — `-o` tolerates pydantic `missing`, so a partial overlay may omit a required field like `relocalizationmodule.priors`. Unknown keys and type errors still raise.
- **`blueprints/smart/unitree_go2.py`** + regenerated `all_blueprints.py` — the three presets. No `core/` or `transport/` change.

## How to Test

**Hardware.** Run on a Go2 against an sf office premap with surveyed tags, started cold in the mapped room:

```
uv run dimos --robot-ip <robot ip> run unitree-go2-relocalization-lidar-fiducial --eval \
  -o relocalizationmodule.map_file=<premap>.pgo_markers.pc2.lcm \
  -o relocalizationmodule.marker_map_file=<premap>.marker_map.json
```

Watch `relocalize accepted` for the winning `source=`; Ctrl+C logs the per-source table.

**Replay**, both priors, no robot:

```
uv run dimos --replay --replay-db=hk_village3 run unitree-go2-relocalization-lidar-fiducial --eval \
  -o relocalizationmodule.map_file=data/replay_gate/hk_village3.pc2.lcm \
  -o relocalizationmodule.marker_map_file=data/replay_gate/hk_village3.marker_map.json
```

Test to read: `test_relocalize.py::test_fiducial_composes_map_T_world_then_consumes_it_once`. Existing suites unchanged.

## Follow-up

To investigate before this is trusted beyond flat, single-floor sites:

- **3D nav.** The judge gates candidates to within `gravity_tilt_max_deg` (10°) of upright and scores wall-only fitness, so `world_T_map` is a gravity-aligned planar correction. How it composes with a 3D nav stack — ramps, stairs, multi-floor — is untested.
- **Robot on a ramp.** On an incline the body tilts with the ground: the 10° gravity gate may reject a valid fix, and wall-only fitness assumes vertical walls.
- **Fiducial without a lidar scan.** The fiducial prior still judges against a cached `global_map`. A decoded tag is an absolute fix on its own; scoring it against the stored premap submap near the tag instead would drop the lidar dependency at acquisition.

## AI assistance

Claude Code (Opus 4.8) — design, implementation, tests; all changes reviewed.

## Checklist

- [ ] This PR is scoped to one clearly stated problem.
- [ ] I ran the relevant checks (`uv run pytest`, pre-commit) for the files I changed.
- [ ] I have reviewed and understood every line in this PR.
- [ ] I disclosed AI assistance above.
- [ ] I have read and approved the [CLA](https://github.com/dimensionalOS/dimos/blob/main/CLA.md).

---

## Announcement (Discord / PR comment)

Relocalization takes priors now, and one of them is an AprilTag.

A robot waking in a prior map waits on RANSAC, which cannot answer until the live map is dense enough to search, and in a corridor or a bare room it can answer confidently and wrongly. A surveyed tag is an absolute fix. Both propose into the same judge, which refines against the premap walls and publishes the winner with its source. Triggers are per prior: the tag fires on a sighting burst, RANSAC on its 2 s timer. Both clear the same 0.60 wall-fitness bar.

```
dimos run unitree-go2-relocalization-lidar-fiducial --eval \
  -o relocalizationmodule.map_file=<premap>.pc2.lcm \
  -o relocalizationmodule.marker_map_file=<premap>.marker_map.json
```

`--eval` logs the per-source table.
