# Tuning lidar relocalization

`relocalize()` places a small live cloud into a prior map. `eval.py` measures
whether it finds the right place, how tightly, and how fast, and `tune`
searches `RelocalizeConfig` for a better tradeoff between those.

    uv run python -m dimos.mapping.relocalization.lidar.eval run --frames 10 --samples 5
    uv run python -m dimos.mapping.relocalization.lidar.eval run --frames 10 --view
    uv run python -m dimos.mapping.relocalization.lidar.eval tune --trials 100

`run` needs nothing extra. `tune` needs optuna, which lives in the `dev`
group - and `default-groups = ["tests"]`, so a plain `uv sync` does not
install it:

    uv sync --group dev

On a machine where the maturin extensions are already built, prefer
`VIRTUAL_ENV=$PWD/.venv uv pip install "optuna>=4.9.0"`, since `uv sync`
prunes `dimos_voxel_ray_tracing` and `accumulate` needs it.

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
  clouds come from a gravity-aligned odometry, so a correct fix has none.
  Tilt in a result is *provably* error, which is what `gravity_aligned` in
  `RelocalizeConfig` exploits by flattening the RANSAC hypothesis to yaw
  before ICP refines it. On by default, since both maps normally come from
  lidar-inertial odometry; `--no-gravity` turns it off for a premap whose
  frame you do not know to be level.
- **fit … vs truth** — median point-to-premap distance where the aligner put
  the cloud, against where ground truth says it belongs. A miss that fits
  *better* than truth (`TRUTH?`) means the recording's own poses drifted and
  the eval is wrong, not the aligner. A miss that fits worse is a real miss.
- **fitness** — ICP's self-report. Diagnostic only: it never decides
  correctness, because tuning against a score the aligner computes about
  itself is circular.

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

Every field of `RelocalizeConfig` was a literal in `relocalize()`. The
defaults are `align_fast` as measured, at the scale of an outdoor mid360
walk — a different sensor or a room-sized map wants its own instance.
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
