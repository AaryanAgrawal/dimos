# PLAN — configuring relocalization priors the dimensional way

How to configure relocalization priors and their hyperparameters. Grounds a follow-up PR (not the
current fiducial PR), sequenced with the Phase-4 fusion arbiter and DIM-944 autotune.

## Problem

Prior wiring is a flat bag on one `Config(ModuleConfig)`
(`dimos/mapping/relocalization/module.py:56-97`).

- Priors toggle by loose booleans on the module — `use_last_pose_seed` (`module.py:77`),
  `use_fiducial_prior` (`module.py:81`) — assembled ad-hoc in `_try_relocalize`
  (`module.py:264-276`).
- RANSAC is special-cased always-on: `[RansacPrior(), *extra_priors]` (`module.py:274`), no toggle,
  no config. Inconsistent with the "every prior is a proposer into the same judge" architecture
  (`priors.py:80-95`).
- Each prior's params are scattered as sibling module fields, indistinguishable from the judge's
  own gates. FiducialPrior's `marker_map_file` / `marker_length_m` / `ambiguity_ratio_min` /
  `camera_info` (`module.py:86-97`) sit next to `fitness_threshold` (`module.py:61`) and
  `reloc_interval_s` (`module.py:68`) — you read `priors.py:181-191` to learn a param exists, then
  hunt `module.py` for where it is surfaced.
- The flat model already leaks: `aruco_dictionary` (`module.py:88`) is declared but never threaded
  into `FiducialPrior`; `age_max_s` and `AggregationConfig` (`priors.py:188-189`) are real prior
  params with no config surface at all — so the age cutoff the fusion arbiter's age-decay reads is
  un-tunable today.
- No per-prior namespace: two priors can't share a param name (each wanting its own `age_max_s`)
  without prefix-mangling.
- Adding a prior = module surgery: a new `use_X` bool, N new flat fields, a branch in
  `_try_relocalize`, and a `_start_X_prior` builder.

## Proposal — a typed list of prior configs

One field on `RelocalizationModule.Config`:

```python
priors: list[PriorConfig] = Field(default_factory=default_priors)
```

`PriorConfig` is a pydantic discriminated union keyed by `type`, each variant carrying its own
params plus a shared base for the fields the arbiter reads. This is the in-tree manipulation
pattern (`dimos/manipulation/planning/kinematics/config.py:26-57`,
`Annotated[A | B | C, Field(discriminator="backend")]`, consumed as a `ModuleConfig` field at
`dimos/manipulation/manipulation_module.py:119`); the typed-list half also ships
(`manipulation_module.py:112` `robots: list[RobotModelConfig]`).

```python
class PriorConfigBase(BaseConfig):
    enabled: bool = True
    tier: int = 0          # source-trust rank the Phase-4 arbiter reads (priors.py:69-73)
    weight: float = 1.0    # covariance/priority weight when >1 source agrees
    max_age_s: float = 120.0  # fix older than this is dropped; arbiter decays within it

class RansacPriorConfig(PriorConfigBase):
    type: Literal["ransac"] = "ransac"

class LastPosePriorConfig(PriorConfigBase):
    type: Literal["last_pose"] = "last_pose"

class FiducialPriorConfig(PriorConfigBase):
    type: Literal["fiducial"] = "fiducial"
    marker_map_file: str | None = None
    marker_length_m: float = 0.10          # tag edge, meters
    ambiguity_ratio_min: float = 2.0       # IPPE flip must reproject >=Nx worse to keep
    aruco_dictionary: str = "DICT_APRILTAG_36h11"
    camera_info: CameraInfo | None = None

PriorConfig = Annotated[
    RansacPriorConfig | LastPosePriorConfig | FiducialPriorConfig,
    Field(discriminator="type"),
]
```

RANSAC becomes a first-class toggleable entry; the fiducial params are grouped under the fiducial
entry, un-scattered. `-o` and `--config` reach every field via the dotted path already advertised
(`dimos/robot/cli/dimos.py:381-382`, tested for the union shape at
`test_manipulation_unit.py:289-303`). YAML:

```yaml
priors:
  - type: ransac                 # was always-on; now an explicit entry
  - type: fiducial
    marker_map_file: go2_office_markers
    marker_length_m: 0.10
    ambiguity_ratio_min: 2.0
    tier: 2                       # arbiter trusts a marker fix over a geometric guess
    max_age_s: 120.0
  - type: last_pose
    enabled: false
```

Each prior class grows a `from_config(cfg) -> RelocPrior` (or a small registry `build(cfg)`), so
`_try_relocalize` becomes: build enabled entries, call `propose`, judge. Adding a prior = one
variant class + one `propose()` (`priors.py:80-95`) — no new module field, no branch.

## Why it is the dimensional way

- pydantic `ModuleConfig`/`BaseConfig` with per-field unit/why comments is the house convention
  (`dimos/core/module.py:106-117`; pgo exemplar `dimos/mapping/loop_closure/pgo.py:97-106`).
- The discriminated-union-on-a-ModuleConfig-field and the `list[SubConfig]` both already ship in
  manipulation — precedent, not invention.
- `-o` overridable and blueprint-presettable (`Module.blueprint(field=value)`,
  `module.py:429-434`): a "lite" preset can pin the prior set.
- It is exactly the shape DIM-944 sweeps ("chainable toolset, tunable hyperparams per step",
  WORKSPACE §4): every knob is an addressable, sweepable field grouped under its prior — including
  the currently-unreachable `max_age_s` and the aggregation sub-gates.
- It is exactly what the Phase-4 fusion arbiter reads (WORKSPACE §4): per-prior `tier`, `weight`,
  `max_age_s`/age-decay all live on the entry the arbiter consumes, instead of a tier-less
  `Candidate` (`priors.py:61-77`). RANSAC's `always | on_demand | never` policy lands as the
  `ransac` entry's `enabled` plus a policy field when the arbiter needs it.

## Migration

- Clean cut preferred: replace the four booleans/param clusters with `priors: list[PriorConfig]`.
  If a deprecation shim is wanted, keep `use_fiducial_prior` etc. as validators that append the
  matching entry and warn — one release, then delete.
- Blueprint presets that pin `use_fiducial_prior=True` / `marker_*` move to a `priors=[...]` list;
  audit reloc blueprints before the cut.
- Fold in the two known leaks while restructuring: deliver `aruco_dictionary` to the prior (dead
  today, `module.py:88`), surface `max_age_s`/aggregation (absent today, `priors.py:188-189`).
- Sequencing: land when the fusion arbiter or a 3rd prior arrives, not inside the current fiducial
  PR. The arbiter is the first consumer of `tier`/`weight`/age-decay.

## Honest tradeoff

Strictly, two priors and two booleans don't need this. The union earns its keep at the 3rd prior or
the fusion arbiter — whichever lands first. Until then the flat booleans are fine; don't refactor
ahead of a consumer.

## Open questions (Aaryan)

1. Clean cut or one-release deprecation shim for the booleans?
2. `tier`/`weight`/`max_age_s` on `PriorConfigBase` now (inert until Phase-4), or added with the
   arbiter PR so the config doesn't advertise knobs nothing reads yet?
3. Registry (`type` -> builder) vs. each variant owning `from_config` — which matches how you want
   third-party/experimental priors added?
4. Does the judge's own surface (`fitness_threshold`, `min_local_points`, `gravity_tilt_max_deg`)
   stay flat on the module, or move into a sibling `judge:` block for symmetry? (Fusion moves
   `min_local_points` per-source per WORKSPACE §4 — argues for grouping.)
