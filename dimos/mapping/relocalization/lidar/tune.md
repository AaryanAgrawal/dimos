# Tuning lidar relocalization

`relocalize()` places a small live cloud into a prior map. `tune.py` measures
whether it finds the right place, how tightly, and how fast. `tune` searches a
rig's `RelocalizeConfig` for a better tradeoff between those.

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

The sync deletes maturin-built extensions. Doing it the other way round
leaves you without `dimos_voxel_ray_tracing`, which every command here needs.
Once the dev group is installed there is nothing left to sync, and both
`uv run` and `uv run --group dev` leave the extension alone.

If a run dies with "dimos_voxel_ray_tracing is not built", something
re-synced. Rerun the maturin line.

## What the eval needs

A recording and a premap **in one coordinate frame**.

The usual way to get that is a single recording split into two.
One part is used for pulling lidar frames, the other part is used to assemble a premap.

This is done in order to not use lidar frames used for a map in relocalization
(match will be too perfect).

Given they share the ground truth coordinate frame, we can measure error by how far the
relocalized frame sits from ground truth coordinates, so nothing has to be hand-labelled.

The shipped pair was built like this. Both files are in `data/`, so
this is only for reproducing them or doing the same to a new recording:

    dimos map global recording_go2_mid360_2026-05-29_4-45pm-PST_corrected \
      --lidar fastlio_lidar --voxel 0.005 --seek 400 --export --no-gui

`--seek 400` does the split: the premap is the walk after 400 s, and probes
come from before it. `--export` implies `--pgo` and writes
`./<dataset>.pc2.lcm`. The walk is a loop, so the two halves still cover some
of the same ground. Without that they would never match.

Register the pair in `DATASETS`:

```python
"go2-sf-area1": Dataset(
    recording="recording_go2_mid360_2026-05-29_4-45pm-PST_corrected.db",
    premap="recording_go2_mid360_2026-05-29_4-45pm-PST_corrected.pc2.lcm",
    window=(0.0, 400.0),   # premap was built from the scans after 400 s
),
```

Both names go through `get_data`. Or pass an unregistered pair inline:

    ... run --dataset my-walk --recording walk.db --premap walk.pc2.lcm --from 0 --to 400

`window` (or `--from/--to`) is the stretch probes may be drawn from. Set it
when you know which part of the recording built the premap. Leave it off and
the whole recording is searched.

## Overlap is not optional

A loop walk revisits little of its route, so a premap built from one stretch
covers only part of the rest. A probe standing where the premap has no points
has **no right answer**. No config can place it, and counting it as a failure
puts a ceiling on the score that the search can never get past. Ask this eval
for a 60% score and it will hand you 60% forever.

So probes are picked by coverage, not by even spacing. `probe_starts` walks a
grid of candidates and measures each one's overlap with the premap
(`coverage`: the share of the local cloud within `COVERAGE_TOL_M` of a premap
point, at the truth pose). It keeps an even spread of the ones above
`MIN_COVERAGE`. If none qualify it raises instead of scoring nothing.

`window` narrows where it looks. Coverage decides what it uses.

## Reading a probe line

```
start=  63s  cover 100.0%  MISS  off 88.319 m  (65.416 m, 105.00 deg, tilt 55.22)
             fit 10.700 vs truth 0.100  fitness 0.000  0.7s
```

- **cover** how much of this cloud the premap contains. Below `MIN_COVERAGE`
  the probe is never used.
- **off** displacement: the median distance the recovered transform moved the
  cloud's own points. This is the honest error. Translation and rotation do
  not add up into one distance, because a small angle far from the cloud
  moves it more than a big angle through its middle. Displacement is what the
  robot actually feels. The numbers in brackets are the breakdown, useful for
  diagnosis, not for ranking.
- **tilt** the part of the rotation that takes gravity off vertical. Both
  clouds come from a lidar-inertial odometry, so a correct fix has no tilt.
  Tilt in a result is error, not a near miss. It is only reported, not
  enforced: constraining the search to yaw was tried and measured as a wash
  once its pivot bug was fixed, so the code went rather than sit unmeasured
  and on by default.
- **fit ... vs truth** median distance from the cloud to the premap, where
  the aligner put it, against where ground truth says it belongs. A miss that
  fits *better* than truth (`TRUTH?`) means the recording's own poses
  drifted, so the eval is wrong and not the aligner. A miss that fits worse
  is a real miss.
- **fitness** ICP's own score. Diagnostic only. It never decides
  correctness, because tuning against a score the aligner computes about
  itself is circular.

## Tuning a new rig

`RelocalizeConfig`'s required fields are scales: voxel sizes, neighbourhood
radii, correspondence distances. They belong to a sensor and an environment,
not to relocalization in general. `PRESETS` names them after the rig they
were measured on. A mid360 walking an outdoor block is the only entry so far.
Do not nudge it to suit a new sensor, add one.

The procedure:

1. **Get a recording and a premap that share a frame.** The cheap way is one
   recording with its premap built from a different stretch of it:
   `dimos map global <recording> --seek <t> --export`. Ground truth is then
   the identity and nothing needs labelling.
2. **Register the pair** in `DATASETS`, with a `window` covering the stretch
   the premap does *not* come from. Include some ground the premap never saw.
   That is the only place a false fix can be caught, and a false fix is the
   failure that matters.
3. **Check the split is real** before trusting anything:
   `... run --samples 12`. The `cover` column should be near 1.0 for probes
   inside the map and near 0.0 outside, with nothing in between. If coverage
   is smeared, the window is wrong.
4. **Search**: `uv run --group dev python -m ... tune tune --trials 200
   --samples 30`. Wider beats longer. The probe count sets the resolution of
   the hit rate, and 200 trials against 8 probes mostly finds lucky draws.
5. **Verify, always**: `... verify --study <name> --top 8 --repeats 10`.
   A study's front is single draws of a random pipeline. In the run this
   preset came from, twelve trials tied at "100%", none of them actually
   reached it on repeats, and two lost forty points on the holdout.
6. **Add the preset** in `relocalize.py`, named for the rig, with a comment
   saying which study and trial it came from.

Two failure modes worth knowing, because both bit this preset:

- **A knob that looks essential can be an artifact of the others.** An
  ablation that changes one field while the rest sit at old values measures
  the interaction, not the field. Two defaults were set that way and later
  removed.
- **A tuned threshold does not carry over to a different call pattern.** The
  cutoff fitted for a single shot at two frames was too strict for a retry
  loop, and threw away fixes that were centimetres from correct.

## The objective

`tune` optimises four values at once: `(good up, bad down, error down,
seconds down)`.

- **good** share of probes placed within `TOLERANCE_M` / `TOLERANCE_DEG`,
  *and* accepted by the fitness cutoff.
- **bad** accepted but wrong. Worse than no fix at all: the module publishes
  it as a TF and everything downstream believes it.
- **error** median displacement of the good fixes. Without it a config
  landing at 2 cm and one at 45 cm score the same under a 50 cm tolerance.
- **seconds** median per probe. It has to be in the objective and not just
  recorded. `ransac_iters` is the speed knob and more iterations never hurt
  quality, so a search that ignores time ends up on the slowest config there
  is.

There is no single winner. `study.best_trials` is a Pareto front: every config
that nothing else beats on all four. It prints correctness first, speed as
the tiebreak. Expect useless corners, like a cutoff near 1.0 that accepts
nothing, scores 0% good and 0% bad, and technically nothing beats it. Ignore
those.

`fitness_threshold` is tuned alongside the aligner because it is what turns a
fitness number into a decision. One catch: `icp_dist_factor` and `voxel_fine`
change what fitness *means*, so cutoffs cannot be compared across trials with
different values of those.

## Knobs

Every field of `RelocalizeConfig` was a literal in the aligner's body. The
scales are required, so there is no `RelocalizeConfig()` to fall into by
accident. `MID360` is `align_fast` as measured on an outdoor mid360 walk, and
a different sensor or a room-sized map gets its own instance next to it. The
few defaulted fields are search budgets and caps, not scales. `objective()`
declares which fields are searched and over what range. Add one by calling
`trial.suggest_*` there, and bump `SPACE`.

## Studies

Results live in `optuna.db` at the repo root, keyed
`<dataset>-v<SPACE>-n<frames>s<samples>`. The same name resumes, which is why
the name carries the dataset, the probe setup, and the space version.
Widening the space or changing the score bumps `SPACE` instead of mixing
trials that cannot be compared into one front.

    uvx optuna-dashboard sqlite:///optuna.db

The dashboard's hyperparameter-importance view is the fastest way to find out
which knobs did nothing, so the next study can drop them.

## Noise

RANSAC is left unseeded on purpose. The variance is real and the robot will
face it, so hiding it behind a seed would only make the numbers look better
than they are. The cost is a noisy objective: with 5 probes the rates jump in
20% steps and a trial can beat its neighbour by luck. Raise `--samples`
before raising `--trials`.
