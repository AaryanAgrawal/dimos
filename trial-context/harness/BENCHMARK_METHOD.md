# Benchmark method — grade the REAL pipeline, never a re-implementation

**The one rule.** The relocalization benchmark grades what the shipped dimos pipeline
publishes under `dimos --replay run <blueprint>`. The harness may **orchestrate** (pick
recordings, hold PGO/premap truth, score, analyze) and may **call real dimos for the
legitimate steps** (premap = `dimos map global --pgo --export`; truth = the loop-closure
`PoseGraph.correction_at`). It must **never re-implement a data-path step** — submap
building, candidate generation, gravity/wall handling, pose composition, or frame
conventions — and then grade a fix it computed itself. The instant a harness rebuilds a
data-path step its own way, it becomes a *different program* that drifts from production
and measures itself, not dimos.

**Why this rule exists (the concrete scar).** The old sections harness (`prep.py` +
`run_bench.py`) re-anchored each query submap to the robot's **body frame**
(`world_pts @ inv(P)`, `prep.py::build_sections`) — a step dimos never does. Production
builds the submap in the LIO's gravity-aligned **world** frame via the real
`VoxelGridMapper` and feeds it to `relocalize()` untouched (`module.py::_try_relocalize`,
`voxels.py::VoxelMapTransformer` — one persistent grid, `add_frame` per obs, cumulative,
no window/reset). On the tilted mid360 rig the body-frame re-anchor manufactured a ~52°
tilt and a phantom "gravity-gate bug" **that does not exist in production**. A full night
went into fixing a bug in the test scaffolding. Testing a re-implementation tests the
re-implementation.

## Canonical path (dimos-native)

```
# 1. premap — REAL dimos (reuse an existing exported .pc2.lcm when present):
uv run --project /home/dimos/aaryan-dimensional/dimos \
    dimos map global <rec> --pgo --export --no-gui --device CPU:0

# 2. driver — runs the shipped blueprint, LISTENS only (no data-path step):
uv run --project /home/dimos/aaryan-dimensional/dimos \
    python trial/harness/replay_bench.py <rec> --premap /abs/<rec>.pc2.lcm
#   wraps: dimos --replay --replay-db=<rec> run unitree-go2-fiducial-relocalization \
#          -o relocalizationmodule.map_file=<abs .pc2.lcm> [-o ...]

# 3. scorer — grades the PUBLISHED fixes vs PGO silver truth:
uv run --project /home/dimos/aaryan-dimensional/dimos \
    python trial/harness/score_replay.py <rec>
```

- **`replay_bench.py`** launches the real `unitree-go2-fiducial-relocalization` blueprint,
  captures the `world→map` transform the `RelocalizationModule` actually publishes to the
  TF tree (LCM `/tf`, full rotation) joined to its `relocalize:` accept log lines, and
  recovers recording-ts from the `/odom` clock. It re-implements **no** dimos step.
- **`score_replay.py`** grades `est = inv(world_map_fix)` against `correction_at(t)` (real
  dimos PGO), `success = err_t < 1 m AND err_r < 15°`. Truth is **PGO silver** and says so.
- **LCM bus is EXCLUSIVE**: one replay at a time (a second Coordinator crashes the first).
  The driver refuses to launch while any `dimos --replay` is live. Check `pgrep -af
  'dimos --replay'` first.

## What the harness may legitimately do (not a re-implementation)

- **Premap** via `dimos map global --pgo --export` — real dimos, not a hand-rolled voxel
  accumulate.
- **Truth** via `PoseGraph.correction_at(t)` from the recording's own PGO graph — real
  dimos loop-closure. The replay's `world` frame == the recording's `world_raw` (odom
  origin), so a published `world→map` fix is directly comparable. This is the **reference**,
  external to the system under test — reusing it is correct, not a re-implementation.
- **Orchestrate + score**: pick recordings, hold premap + PGO truth, join logs,
  compute error, stratify, plot.

## Audit — the superseded re-implementations (SILVER numbers archived, path retired)

| Data-path step | Old harness (SUPERSEDED) | Real dimos | Verdict |
|---|---|---|---|
| Submap **frame** | `prep.py::build_sections` re-anchor to gravity-aligned body frame (`world_pts @ inv(Pg)`, translation+yaw) | `module.py::_try_relocalize` feeds the **world-frame** `global_map` to `relocalize()` untouched | **DIVERGENT** — source of the phantom gravity-gate bug; retire |
| Submap **extent** | `prep.py::build_sections` bounded trailing window (≤200 scans → 50k voxels) | `voxels.py::VoxelMapTransformer` **cumulative** grid (no window/reset) | **DIVERGENT** — the bounded window masks the cumulative-warp late divergence the real pipeline shows |
| **Premap** build | `prep.py::build_premap` (dedup 0.3 m + `correction_at` + carve-OFF VoxelGrid) | `dimos map global --pgo --export` | **DIVERGENT** — re-implements pass 2; use the real `--export` |
| **Candidate** gen + judge | `run_bench.py::_eval_one` calls `relocalize()` / `generate_ransac_candidates` + manual pool + `refine_candidates` (no `sources=`) on the body-frame submap | `relocalize()` / `relocalize_with_priors()` inside the module | **DIVERGENT** — grades a self-computed fix; omits the per-source gate |
| **Gravity** handling | `run_bench.py --map-up-from-premap` (`estimate_map_up` + `map_up=` threading) | `relocalize.py::_gravity_tilt_deg` world-z gate; **no `map_up` param exists in any clone** | **DIVERGENT** — references an API absent tree-wide; chases the phantom |
| Lidar **pose** assoc. | `prep.py::reposed_lidar_obs` (nearest-ts odom join) | real LIO poses the clouds in replay | prep-path workaround; replay uses the real LIO |
| **PGO / truth** | `prep.py::build_graph` → `PoseGraph.correction_at` | real dimos loop-closure PGO | **FAITHFUL** — real dimos, truth only; keep |
| Whole **scoring** path | `replay_bench.py` + `score_replay.py` | runs the real blueprint, listens to `/tf` + log | **FAITHFUL** — canonical; re-implements nothing |

`prep.py` and `run_bench.py` are kept (not deleted) — the SILVER numbers they produced are
archived in `WORKSPACE.md` and `out/` — but the sections path is **superseded by
`replay_bench.py` + `score_replay.py`** for any new grading. Do not build new benchmark
numbers on the direct-`relocalize()` path.

## The test before you add a harness step

If the step touches a cloud, a pose, a candidate, a gravity axis, or a frame on the way to
the number you report — **can dimos do it instead?** If yes, call dimos. If you truly can't
test something through real dimos, say so plainly; do not substitute a look-alike.
