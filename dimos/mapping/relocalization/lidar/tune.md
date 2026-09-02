# Tuning lidar relocalization

`relocalize()` places a small live cloud into a prior map. `tune.py` measures
whether it finds the right place, how tightly, and how fast, and `tune`
searches a rig's `RelocalizeConfig` for a better tradeoff between those.

    uv run python -m dimos.mapping.relocalization.lidar.tune run --samples 12
    uv run python -m dimos.mapping.relocalization.lidar.tune run --samples 12 --view
    uv run --group dev python -m dimos.mapping.relocalization.lidar.tune tune --trials 200
    uv run --group dev python -m dimos.mapping.relocalization.lidar.tune verify --study <name>

### Setup, once

`tune` and `verify` need optuna, and everything here needs the rust
raycaster. **Order matters** - install optuna first, build the extension
second:

    uv sync --group dev
    uv run maturin develop --uv -m dimos/mapping/ray_tracing/rust/py/Cargo.toml

The sync prunes maturin-built extensions as extraneous, so doing it the
other way round leaves you without `dimos_voxel_ray_tracing`, which every
command here needs to accumulate a local map. Once the dev group is in the
venv there is nothing left to sync, and both `uv run` and `uv run --group
dev` leave the extension alone.

If a run does die with "dimos_voxel_ray_tracing is not built", something
re-synced: rerun the maturin line.

## What the eval needs

A recording and a premap **in one coordinate frame**. The usual way to get
that is a single recording with its premap assembled from a different
stretch of it — `dimos map global <recording> --export` writes the
`.pc2.lcm`. Ground truth is then the identity, and the error is how far the
recovered transform sits from it, so nothing has to be hand-labelled.

Register the pair in `DATASETS`:

```python
"go2-sf-area1": Dataset(
    recording="recording_go2_mid360_2026-05-29_4-45pm-PST_corrected.db",
    premap="mid360_go2_sf_area1.pc2.lcm",
    window=(0.0, 400.0),   # premap was built from the scans after 400 s
),
```

Both names go through `get_data`. Or pass an unregistered pair inline:

    ... run --dataset my-walk --recording walk.db --premap walk.pc2.lcm --from 0 --to 400

`window` (or `--from/--to`) is the stretch probes may be drawn from. Set it
when you know which part of the recording built the premap; leave it off and
the whole recording is searched.

## Overlap is not optional

A loop walk revisits little of its route, so a premap built from one stretch
covers only part of the rest. A probe standing where the premap has no points
has **no right answer** — no configuration can place it — and counting it as
a failure caps the eval below anything the search can reach. Ask this eval
for a 60% score and it will hand you 60% forever.

So probes are selected by coverage, not by even spacing: `probe_starts` walks
a grid of candidates, measures each one's overlap with the premap
(`coverage`, the share of the local cloud within `COVERAGE_TOL_M` of a premap
point *at the truth pose*), and keeps an even spread of those above
`MIN_COVERAGE`. If none qualify it raises rather than scoring nothing.

`window` narrows where it looks; coverage decides what it uses.

## Reading a probe line

```
start=  63s  cover 100.0%  MISS  off 88.319 m  (65.416 m, 105.00 deg, tilt 55.22)
             fit 10.700 vs truth 0.100  fitness 0.000  0.7s
```

- **cover** — how much of this cloud the premap contains. Below `MIN_COVERAGE`
  the probe is never used.
- **off** — displacement: the median distance the recovered transform moved
  the cloud's own points. This is the honest error. Translation and rotation
  do not combine into a distance — a small angle about a far-off center moves
  a cloud further than a large one about its middle — and displacement is what
  the robot actually feels. The parenthesised translation/rotation/tilt are
  the decomposition, useful for diagnosis, not for ranking.
- **tilt** — the part of the rotation that takes gravity off vertical. Both
  clouds come from a lidar-inertial odometry, so a correct fix has none, and
  tilt in a result is *provably* error rather than a near miss. It is
  reported as a diagnostic, not enforced: constraining the hypothesis to
  yaw was tried and measured as a wash once its pivot bug was fixed, so the
  code went rather than sit unmeasured and on by default.
- **fit … vs truth** — median point-to-premap distance where the aligner put
  the cloud, against where ground truth says it belongs. A miss that fits
  *better* than truth (`TRUTH?`) means the recording's own poses drifted and
  the eval is wrong, not the aligner. A miss that fits worse is a real miss.
- **fitness** — ICP's self-report. Diagnostic only: it never decides
  correctness, because tuning against a score the aligner computes about
  itself is circular.

## Tuning a new rig

`RelocalizeConfig`'s required fields are scales - voxel sizes, neighbourhood radii,
correspondence distances - so they belong to a sensor and an environment,
not to relocalization in general. `PRESETS` names them after the rig they
were measured on. A mid360 walking an outdoor block is the only entry so
far; do not nudge it to suit a new sensor, add one.

The procedure, end to end:

1. **Get a recording and a premap that share a frame.** The cheap way is one
   recording with its premap built from a different stretch of it:
   `dimos map global <recording> --seek <t> --export`. Ground truth is then
   the identity and nothing needs labelling.
2. **Register the pair** in `DATASETS`, with a `window` covering the stretch
   the premap does *not* come from. Include some ground the premap never
   saw - that is the only place a false fix can be caught, and it is the
   failure that matters.
3. **Check the split is real** before trusting anything:
   `... run --samples 12`. The `cover` column should be near 1.0 for probes
   inside the map and near 0.0 outside, with nothing in between. A premap
   whose coverage is smeared means the window is wrong.
4. **Search**: `uv run --group dev python -m ... tune tune --trials 200
   --samples 30`. Wider is better than longer - the probe count sets the
   resolution of the hit rate, and 200 trials against 8 probes mostly finds
   lucky draws.
5. **Verify, always**: `... verify --study <name> --top 8 --repeats 10`.
   A study's front is single draws of a stochastic pipeline. In the run this
   preset came from, twelve trials tied at "100%", none of them actually
   reached it on repeats, and two lost forty points on the holdout.
6. **Add the preset** in `relocalize.py`, named for the rig, with a comment
   saying which study and trial it came from.

Two failure modes worth knowing, because both bit this preset:

- **A knob that looks essential can be an artifact of the others.** An
  ablation that changes one field while the rest sit at old values measures
  the interaction, not the field. Two defaults were set that way and later
  removed.
- **A tuned threshold does not transfer between call patterns.** The cutoff
  fitted for a single shot at two frames was too strict for a retry loop,
  and threw away fixes that were centimetres from correct.

## The objective

`tune` optimises four values at once — `(good ↑, bad ↓, error ↓, seconds ↓)`:

- **good** — share of probes placed within `TOLERANCE_M` / `TOLERANCE_DEG`,
  *and* accepted by the fitness cutoff.
- **bad** — accepted but wrong. Worse than no fix at all: the module publishes
  it as a TF and everything downstream believes it.
- **error** — median displacement of the good fixes. Without it a config
  landing at 2 cm and one at 45 cm score identically under a 50 cm tolerance.
- **seconds** — median per probe. It has to be in the objective rather than
  merely recorded: `ransac_iters` is the speed knob and more iterations never
  hurt quality, so a time-blind search converges on the slowest config there
  is.

There is no single winner. `study.best_trials` is a Pareto front — every
config nothing else beats on all four — printed correctness-first with speed
as the tiebreak. Expect degenerate corners: a cutoff near 1.0 accepts nothing,
so it scores 0% good and 0% bad and is technically non-dominated. Ignore them.

`fitness_threshold` is tuned alongside the aligner because it is what turns a
fitness number into a decision. Note the catch: `icp_dist_factor` and
`voxel_fine` change what fitness *means*, so cutoffs are not comparable across
trials with different values of those.

## Knobs

Every field of `RelocalizeConfig` was a literal in the aligner's body. The
scales are required, so there is no `RelocalizeConfig()` to fall into by
accident; `MID360` is `align_fast` as measured on an outdoor mid360 walk, and
a different sensor or a room-sized map gets its own instance next to it. The
handful of defaulted fields are search budgets and caps, not scales.
`objective()` declares which are searched and over what range; add one by
calling `trial.suggest_*` there, and bump `SPACE`.

## Studies

Results live in `optuna.db` at the repo root, keyed
`<dataset>-v<SPACE>-n<frames>s<samples>`. Same name resumes, which is why the
name carries the dataset, the probe setup, and the space version — widening
the space or changing the score bumps `SPACE` rather than mixing incomparable
trials into one front.

    uvx optuna-dashboard sqlite:///optuna.db

The dashboard's hyperparameter-importance view is the fastest way to find out
which knobs did nothing, so the next study can drop them.

## Noise

RANSAC is deliberately unseeded — the variance is real and the robot will
face it, so hiding it behind a seed would flatter the numbers. The cost is a
noisy objective: with 5 probes the rates quantise to 20% steps and a trial can
beat its neighbour by luck. Raise `--samples` before raising `--trials`.
