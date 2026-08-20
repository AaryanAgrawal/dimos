# Offline relocalization confidence benchmark (the sections harness)

Measures, on real recorded drives (replay, deterministic), whether the
confidence a relocalization method publishes actually predicts correctness —
and what each prior buys. Phase 3 of the trial plan; the successor to the cut
real-life benchmark. Imports dimos as a library from the sibling checkout;
never lives inside dimos.

## CANONICAL PATH — grade the real pipeline (read `BENCHMARK_METHOD.md` first)

New grading goes through the **dimos-native replay path**: run the shipped blueprint
under `dimos --replay` and score what the `RelocalizationModule` actually publishes —
the harness re-implements **no** data-path step. See `BENCHMARK_METHOD.md` for the full
rule + audit.

```
uv run --project /home/dimos/aaryan-dimensional/dimos dimos map global <rec> --pgo --export --no-gui  # premap (real dimos)
uv run --project /home/dimos/aaryan-dimensional/dimos python trial/harness/replay_bench.py <rec> --premap /abs/<rec>.pc2.lcm  # driver: runs blueprint, listens
uv run --project /home/dimos/aaryan-dimensional/dimos python trial/harness/score_replay.py <rec>  # scorer: published fixes vs PGO truth
```

> **SUPERSEDED — the sections harness below (`prep.py` sections + `run_bench.py`'s direct
> `relocalize()`/`refine_candidates()` call).** `prep.py::build_sections` re-anchors the
> submap to the robot **body frame** (`world_pts @ inv(Pg)`) and `prep.py::build_premap`
> re-implements `dimos map global --pgo` — steps production never does (it feeds the
> **world-frame** cumulative `VoxelGridMapper.global_map` to `relocalize()` untouched).
> `run_bench.py` then grades a fix **it computed itself**, not one the module published;
> its `--map-up-from-premap` path even calls a `map_up`/`estimate_map_up` API absent from
> every dimos clone. That body-frame re-anchor manufactured the **phantom mid360
> gravity-gate bug** — the reason this path is retired. These two files are **kept, not
> deleted** (their SILVER numbers are archived in `WORKSPACE.md`); use them only to read
> the archived results, never to produce new benchmark numbers. Canonical replacement:
> `replay_bench.py` + `score_replay.py`.

## One command each (from `dimos/`)

```bash
uv run python ../trial/harness/prep.py hk_village3 --n-queries 120   # sections + premap + PGO truth
uv run python ../trial/harness/markers.py hk_village3               # scatter + marker map + fixes
uv run python ../trial/harness/run_bench.py hk_village3 --config ransac
uv run python ../trial/harness/run_bench.py hk_village3 --config ransac+fiducial \
    --fiducial-fixes ../trial/harness/out/markers/hk_village3.fixes.json
uv run python ../trial/harness/analyze.py hk_village3.ransac hk_village3.ransac_fiducial
# uv run python ../trial/harness/china_markers.py    # EXCLUDED from evidence (Aaryan Jul 17: don't use china_office)
uv run pytest ../trial/harness/tests/                               # confidence math unit tests
```

Figures land in `trial/results/figures/` (tracked); raw results in
`trial/harness/out/` (untracked).

## Design (each choice is load-bearing)

- **#2137's determinism recipe, defects fixed**: `OMP_NUM_THREADS=1` before
  open3d import, per-frame seeds = `frame_idx`, sorted order, fork workers —
  but **no frame is ever excluded** and **the denominator is all sections**
  (crashes / no-candidates count as failures). leshy's harness excluded 5
  frames (3 of them "not disambiguable without a pose prior" — exactly the
  frames a fiducial prior must be scored on) and silently shrank the
  denominator on timeout.
- **Sections mimic the live stack**: trailing-window `VoxelGrid` accumulation
  (carve ON, 0.05 m — the live mapper), gate state recorded at the live
  module's 50k-point threshold; premap rebuilt like `dimos map global --pgo`
  (carve OFF, 0.3 m frame dedup). Sections are re-anchored to the query
  frame's body frame — a kidnapped robot doesn't know the recording's world
  frame.
- **Truth is SILVER and says so**: `T_true = correction_at(ts) ∘
  world_raw_T_body`, from a PoseGraph pickled at prep time and reused by
  every downstream stage (PGO wobbles ~6 cm run-to-run; two runs = two
  frames). The marker revisit test (below) qualifies the truth per recording.
- **The confidence questions are first-class** (`confidence.py`, pure numpy,
  unit-tested): risk–coverage (what does each accept threshold really buy),
  AUROC (does fitness rank correctness), reliability/ECE (is 0.7 "70% sure").
- **Marker revisit test = the PGO qualifier** (leshy's verification: observe
  a marker, drive a loop, observe again — locations must match), stratified
  by time gap because aggregate scatter is a composition trap: short-gap
  pairs have no drift to correct and drown the loop-return pairs the test is
  about. Measured consequences so far: village3 — PGO 3.3× better at 60 s+
  loop returns, 3× WORSE at 30–60 s mid-segment (interpolated corrections
  bend the trajectory between anchors). (A china_office measurement existed
  but is EXCLUDED per Aaryan Jul 17 — don't cite it; the production-lane
  version should come from the purpose-built mid360 walk recording instead.)
  "PGO = silver truth" is a per-recording claim, never a constant.
- **Fiducial fixes are honest about circularity**: the marker map derives
  from the same PoseGraph as the truth (exactly what a deployment survey
  would produce, but truth-correlated); candidates are built from real
  detections bridged over raw odometry with age recorded, and the judge
  still ranks them on geometry alone. The cross-run test (premap from one
  recording, queries from another) is the escape from this correlation —
  future work, data permitting.

## Results (hk_village3, 120 sections, replay, PGO-silver truth ±decimeter floor on long windows)

All numbers independently re-derived by an adversarial verification pass
(bit-identical reruns; every rate re-counted; mechanisms ablation-tested).

| config | success | risk @0.45 gate | risk @0.60 | risk ≤2% gate exists? | median dt |
|---|---|---|---|---|---|
| ransac (today) | 77.5% (93/120) | 22.5% (27/120 accepted wrong) | 22.7% | **no** (best 1/45 @0.959) | 9.7 s |
| ransac+lastpose | 77.5% (93/120) | seed entrenches busts on sparse submaps (4 failures won by the seed) | — | — | 7.2 s |
| ransac+fiducial | 95.8% (115/120) | 4.2% (5/120) | 4.2% | yes: thr 0.911, coverage 0.875 (2/105) | 11.0 s |
| fiducial+judge | 95.0% (114/120, 3 no-marker = failures) | 2.6% (3/117) | 2.6% | ~ (thr 0.82, coverage 0.88) | **0.4 s** |

**Read via the gate split (the decisive cut, adversarially established):**
gate-reached sections (>=50k pts, n=50) — ransac already 50/50; the entire
fiducial gain is on gate-missed sections (61.4% -> 92.9%), i.e. attempts the
live stack refuses today (MIN_LOCAL_POINTS skip). The honest fiducial
headline is **coverage extension into the early/power-on regime + ~25x
compute reduction when tags are visible** — not an accuracy fix of live
behavior. Circularity caveat: the marker map shares the truth's PoseGraph
(90/117 sections had a bar-passing candidate pre-judge, median candidate
error 0.46 m); decorrelation tests (temporal-split map, different-recording
map) are the named next step. Scenario label: tracking-recovery /
power-on-near-tag (candidate ages <=23.5 s) — not kidnap, which severs the
odometry bridge the candidates ride.

**The confidence finding is circularity-proof and is the headline:** fitness
does not gate safely on sparse submaps — 27 confident busts at 0.81–0.96
(failure rotations median 85°, large-rotation wrong-basin class), the run's
highest-fitness answer (0.995) wrong by 2.9 m/157°, no safe threshold
exists, the 0.45-vs-0.6 gate debate is moot (both ~22.5% risk), and even in
the fiducial arm fitness stayed confident-wrong ~2–4% (a 0.995-fitness
2.9 m bust beat a 0.147 m fiducial candidate). Submap size belongs in the
confidence reading. Fragility note: at N=120, threshold conclusions swing on
single samples — quote raw counts.
