# Quiz bank — Dimensional trial

Running bank of questions on the fiducial-relocalization work, every answer grounded in a `file:line`.
Aaryan answers out loud, then opens the fold; the agent grades and logs the score at the bottom.

Paths are relative to the `dimos` package root (`dimos/dimos/...` on disk). Line numbers are against
`feat/relocalization-fiducial-prior` at `142e51b93`.

---

## 1. Blueprints & config

**Q1.** What does `autoconnect(...)` actually do — and what does it *not* do?

<details><summary>answer</summary>

Merges several `Blueprint`s into one: concatenates the atoms, de-dupes by instance name in reverse so a
later blueprint overrides an earlier one, and unions the transport / global-config / remapping maps.
It wires nothing. Wiring happens later in the coordinator, which groups every declared stream by
`(name, type)` and hands all of them the same transport.
`core/coordination/blueprints.py:388-412`, `:415-423`; `core/coordination/module_coordinator.py:298-320`

</details>

**Q2.** How does a Module declare its ports, and how does the blueprint find them?

<details><summary>answer</summary>

The Module class annotates them: `global_map: In[PointCloud2]`, `loaded_map: Out[PointCloud2]`.
`BlueprintAtom.create` resolves the annotations over the whole MRO and records each `In`/`Out` as a
`StreamRef(name, type, direction)`. `In`/`Out` are `Stream` subclasses holding a transport, not values.
`mapping/relocalization/module.py:96-99`; `core/coordination/blueprints.py:119-127`; `core/stream.py:103`, `:149`, `:223`

</details>

**Q3.** What is a Blueprint, as opposed to a Module?

<details><summary>answer</summary>

A frozen dataclass describing what to run — module classes plus their kwargs — with no instances in it.
`Module.blueprint` is just `partial(Blueprint.create, cls)`. The coordinator deploys the atoms into
real Module objects in worker processes at `build()`.
`core/coordination/blueprints.py:184-216`; `core/module.py:430-434`; `core/coordination/module_coordinator.py:333-356`

</details>

**Q4.** Three places a config value can come from. Name them in precedence order.

<details><summary>answer</summary>

1) the pydantic field default in the module's `Config`; 2) the blueprint preset — the kwargs passed to
`X.blueprint(...)`, stored as `BlueprintAtom.kwargs`; 3) the config file / env / `-o` overlay, merged
over the preset at deploy by `_merge_config_kwargs` (deep merge, override wins).
`mapping/relocalization/module.py:72-91`; `core/coordination/blueprints.py:98-154`;
`core/coordination/worker_manager_python.py:34-41`

</details>

**Q5.** How does `-o markerdetectionstreammodule.aggregation.max_reproj_px=1.5` reach a nested pydantic field?

<details><summary>answer</summary>

`load_config_args` splits the key on `.`, walks/creates nested dicts with `setdefault`, and drops the
string value at the leaf (`/` in a namespaced instance name is escaped to `_`). The resulting overlay is
validated early against `blueprint.config()` — `missing` errors are tolerated because the overlay is
partial, but unknown keys and type errors raise. Pydantic coerces the string at model build.
`robot/cli/dimos.py:271-306`

</details>

**Q6.** Why is that early validation deliberately allowed to swallow `missing` errors?

<details><summary>answer</summary>

Because the overlay is a PARTIAL layer over the blueprint preset that supplies the required fields.
The live case is in the code comment: `MarkerDetectionStreamModuleConfig.marker_length_m` is
`Field(..., gt=0.0)` — required, with the blueprint preset carrying it — so
`-o markerdetectionstreammodule.aggregation.min_observations=3` would be rejected for a field the
operator never touched. Only typos and bad types should fail here.
`robot/cli/dimos.py:296-306`; `perception/fiducial/marker_detection_stream_module.py:60-62`

</details>

**Q7.** What does `--disable MarkerDetectionStreamModule` do?

<details><summary>answer</summary>

The CLI resolves each name to its module class and calls `blueprint.disabled_modules(*classes)`;
`active_blueprints` then filters those atoms out, so the module is never deployed or wired. Anything
holding a ref to it gets a `DisabledModuleProxy` whose every method is a logged no-op.
`robot/cli/dimos.py:316`, `:368-372`; `core/coordination/blueprints.py:218-219`, `:376-381`, `:44-62`

</details>

**Q8.** We shipped three relocalization blueprints, then cut to one. What is the argument, and what does the surviving one declare?

<details><summary>answer</summary>

The three differed only in which priors were configured — config, not topology. The survivor,
`unitree_go2_relocalization`, passes NO kwargs to `RelocalizationModule.blueprint()`: `Config.priors`
already defaults to both entries on, so the blueprint declares nothing about priors at all. The CLI
carries the switches (lidar-only = `--disable MarkerDetectionStreamModule`). Three registry names for
one config difference is three things to keep in step.
`robot/unitree/go2/blueprints/smart/unitree_go2.py:97-105`; `mapping/relocalization/module.py:90-91`;
removal in dimos commit `adf6b8f0b`

</details>

**Q9.** What is `.transports({...})` for, and why does our relocalization blueprint not have one?

<details><summary>answer</summary>

It pins an explicit transport (e.g. a fixed LCM topic) for a `(stream name, type)` key instead of letting
the coordinator pick one. Only needed when a consumer sits outside the blueprint — a cpp/rust or external
module. Both ends of `aggregated_detections` are Python modules in the same blueprint, so autoconnect
wires it; the pin would be dead weight.
`core/coordination/blueprints.py:251-254`; `perception/fiducial/marker_detection_stream_module.py:84-86`;
surviving example `robot/unitree/go2/blueprints/smart/unitree_go2.py:86-93`

</details>

**Q10.** What does `.global_config(n_workers=12, robot_model=...)` set, and why 12 here?

<details><summary>answer</summary>

It stores overrides on the Blueprint; `ModuleCoordinator.build` applies them to the process-global
`GlobalConfig` before deploying, and `WorkerManagerPython` spawns `n_workers` worker processes.
The go2 stack raises the count as modules are added: 10 base → 11 with markers → 12 with relocalization.
`core/coordination/blueprints.py:256-260`; `core/coordination/module_coordinator.py:333-340`;
`core/coordination/worker_manager_python.py:50`, `:59-64`; `robot/unitree/go2/blueprints/smart/unitree_go2.py:49,94,105`

</details>

**Q11.** How is `all_blueprints.py` generated, and what broke when the three blueprints became one?

<details><summary>answer</summary>

A test AST-scans every `dimos/**.py` for top-level assignments that are an `autoconnect(...)` call or end
in a blueprint builder method, and writes the kebab-case name → `module:attr` registry. Locally the test
rewrites the file and fails if the result is uncommitted; in CI it only diffs and fails.
Renaming three blueprints to one left three stale keys — regenerated in dimos commit `490248123`.
`robot/test_all_blueprints_generation.py:38-45`, `:49-88`, `:254-287`; `robot/all_blueprints.py:132`

</details>

**Q12.** An operator wants to run RANSAC only. Write the `-o` that turns the fiducial prior off.

<details><summary>answer</summary>

Trick question: there isn't one. `-o` cannot reach `priors` at all.

    -o relocalizationmodule.priors='[{"type":"ransac"}]'   # str where a list is required
    -o relocalizationmodule.priors.0.enabled=false         # builds {"0": {...}}, a dict

Both die in the CLI's early validation with pydantic `list_type` ("Input should be a valid list") —
`load_config_args` only ever produces strings at the leaf and nested dicts above them, and neither is a
list. Checked by running both through `load_config_args(unitree_go2_relocalization.config(), ...)`.
The working toggle is `--disable MarkerDetectionStreamModule`. `Config.marker_map_file` exists as a
top-level field for exactly this reason — a dotted `-o` cannot index into the priors list.
`robot/cli/dimos.py:286-294`, `:296-306`; `mapping/relocalization/module.py:76-77`, `:90-91`

</details>

**Q13.** Two modules both declare a port named `aggregated_detections`. What makes them the same wire — and what happens if you rename one side?

<details><summary>answer</summary>

`_connect_streams` keys every declared stream by `(remapped_name, type)` and hands one transport to
every member of a key. Matching is by NAME **and** type: same name, different message type is a
different wire.

Rename one side and you get two keys with one member each. Each is still a unique name, so each gets its
own `/{name}` topic. Nothing raises: the producer publishes into a topic no one reads, the consumer waits
on a topic no one writes. The only trace is the per-stream `Transport` log line at startup.
`core/coordination/module_coordinator.py:301-307`, `:309-320`, `:659-665`; `core/coordination/blueprints.py:119-127`

</details>

---

## 2. The prior system

**Q14.** `RelocPrior` is a `Protocol`, not a base class. Why?

<details><summary>answer</summary>

Structural typing: a prior only has to expose `name` and
`propose(global_map, local_map) -> list[np.ndarray]`. `RansacPrior` and `FiducialPrior` inherit nothing
and share no state — RANSAC is a pure function wrapper, the fiducial carries pending-fix state. A base
class would impose a lifecycle neither needs.
`mapping/relocalization/priors.py:77-86`, `:89-99`, `:160-195`

</details>

**Q15.** Why must a prior never self-select a winner?

<details><summary>answer</summary>

Ranking, gating and refinement are `refine_candidates`' job, so every fix meets ONE standard no matter
which prior proposed it — same gravity gate, same wall-only rerank, same ICP. `propose()` returns bare
4x4 matrices; there is nowhere to put a self-score. Zero candidates is a valid answer from a prior.

Note the premise: candidates are never pooled ACROSS priors. `_fire` passes exactly one prior, so this is
about one consistent standard, not about making two proposers comparable in a shared pool.
`mapping/relocalization/priors.py:77-86`, `:198-213`; `mapping/relocalization/module.py:276-278`

</details>

**Q16.** How does the config know which prior an entry configures, and what does each entry own?

<details><summary>answer</summary>

A pydantic discriminated union on the `type` literal: `RansacPriorConfig(type="ransac")` /
`FiducialPriorConfig(type="fiducial")`, annotated `Field(discriminator="type")`. That same `type` string
equals the prior object's `name`, which is what makes the accept gate a dict lookup.

Shared (`PriorConfigBase`): `enabled`, `fitness_threshold` (0.6). RANSAC adds `interval_s` (2.0) and
`min_local_points` (50_000). Fiducial adds `marker_map_file` and nothing else.
`mapping/relocalization/priors.py:44-49`, `:52-59`, `:62-67`, `:70-74`; `mapping/relocalization/module.py:115-117`

</details>

**Q17.** What does `propose()` return, and who consumes it?

<details><summary>answer</summary>

`list[np.ndarray]` — 4x4 `map_T_world` candidates placing `local_map` into `global_map`'s frame. No
wrapper type, no source field. `relocalize_with_prior` hands ONE prior's list to `refine_candidates` and
returns `(T, fitness)`, or `None` when the list is empty. There is no winning index to return: the caller
passed the prior in, so the source is `prior.name`.
`mapping/relocalization/priors.py:82-86`, `:198-213`; `mapping/relocalization/module.py:276-278`

</details>

**Q18.** RANSAC's trigger vs the fiducial's trigger.

<details><summary>answer</summary>

RANSAC is polled: every cloud reaches `_on_local_map`, and it fires when `interval_s` has elapsed AND the
cloud has ≥ `min_local_points` (starved → log, leave the timer standing). The fiducial is event-driven: it
fires on the tag-burst edge in `_on_aggregated_detections`, judged against the last cached cloud, plus a
cold-start fire on the first cloud if a burst arrived before any lidar.
`mapping/relocalization/module.py:306-327`, `:271-274`; `mapping/relocalization/priors.py:55-59`

</details>

**Q19.** Why fire the fiducial on the burst instead of waiting for the next cloud?

<details><summary>answer</summary>

The tag candidate is composed from the marker alone — no lidar needed to propose it — so waiting on the
next cloud is dead time on acquisition. It is sound because the map stream *accumulates* in the world
frame: the previous cloud scores wall fitness about as well as the newest.
`mapping/relocalization/module.py:271-274`, `:131-132`

</details>

**Q20.** Why does the fiducial prior drain `_pending` on use instead of re-offering the fix?

<details><summary>answer</summary>

A drained fix scores worse each time it is re-judged — the world has drifted since the sighting, so the
same `map_T_world` is stale. One composed fix gets exactly one trip past the judge.
`mapping/relocalization/priors.py:168-169`, `:187-195`

</details>

**Q21.** What race does `_pending_lock` prevent, exactly?

<details><summary>answer</summary>

`observe()` (detector thread) does a read-modify-write on `_pending`; `propose()` (fire thread) swaps the
dict out. Without the lock the swap tears the write — "dict changed size during iteration", or a fix
written between the read and the clear is wiped unseen — and the cycle is dropped. The lock makes
swap-and-clear one critical section.
`mapping/relocalization/priors.py:170-171`, `:178-180`, `:192-195`

</details>

**Q22.** A prior proposes nothing. What happens, and why is that not an exception?

<details><summary>answer</summary>

`relocalize_with_prior` returns `None`, and the module returns without publishing — no log, no tally.
It is the fiducial drain race: two threads fire back to back, the loser drains an already-empty
`_pending`, and the winner has already published. That is a benign no-op, so it is an in-band sentinel,
not a raise. `EmptyProposalError` was deleted; the two surviving exceptions are real refusals the caller
must log — `NoUprightCandidateError` and `InsufficientWallEvidenceError`.
`mapping/relocalization/priors.py:204-207`; `mapping/relocalization/module.py:329-334`, `:361-367`;
`mapping/relocalization/relocalize.py:43-48`

</details>

**Q23.** "Shared judge" is the wrong framing. What is actually shared between the priors, and what is not?

<details><summary>answer</summary>

Nothing is judged BETWEEN priors — `_fire` passes exactly one prior and candidates are never pooled.
What is shared is one SOLVER with one set of geometric gates: every candidate, whoever proposed it, goes
through the same `refine_candidates` — gravity tilt (`Config.gravity_tilt_max_deg`, 10°, passed at the
call site), the `MIN_WALL_POINTS` wall-evidence floor, the wall-only rerank, wall ICP, full-cloud polish.

What is per-prior is one number: `PriorConfigBase.fitness_threshold`, looked up as
`self._accept_threshold[prior.name]` — and only while tracking. With no anchor yet, every prior is held to
the same higher `acquire_fitness_threshold` instead.
`mapping/relocalization/relocalize.py:187-232`; `mapping/relocalization/priors.py:44-49`, `:198-213`;
`mapping/relocalization/module.py:115-117`, `:388-395`

</details>

**Q24.** Why are the prior objects built once in `__init__` rather than per frame?

<details><summary>answer</summary>

`FiducialPrior` holds pending-fix state across bursts — a fresh instance each frame would reset it and drop
every tag fix. `RansacPrior` is a pure source, so the module owns its poll timer instead.
The `FiducialPrior` itself is built later, in `start()`, because it needs the loaded marker map.
`mapping/relocalization/module.py:108-122`, `:191-219`

</details>

**Q25.** `FiducialPriorConfig` no longer carries `marker_length_m`. Why can that one not be given a default?

<details><summary>answer</summary>

It sets metric SCALE at PnP time. `_aruco_marker_object_points` builds the tag's object points at ±L/2 and
`cv2.solvePnP(..., SOLVEPNP_IPPE_SQUARE)` returns a `tvec` that scales linearly with L — a square viewed
from one camera is scale-ambiguous, so the printed edge length IS the scale. Wrong L and every
`map_T_world` translation is off by that ratio, with no health signal reporting it: reprojection error is
unchanged.

It is a property of the printed tag, not of the code, so no default can be right — upstream made it
`Field(..., gt=0.0)`, required. Tag geometry belongs to the detector, so the prior entry carries only
`type` + `marker_map_file` (plus inherited `enabled`/`fitness_threshold`).
`perception/fiducial/marker_detection_stream_module.py:60-62`; `perception/fiducial/marker_pose.py:56-58`, `:84`, `:98-104`;
`mapping/relocalization/priors.py:62-67`

</details>

---

## 3. Frames & math

**Q26.** What is `map_T_world` and which way does it go?

<details><summary>answer</summary>

The 4x4 that takes a point expressed in the LIO's drifting `world` frame into the prebuilt `map` frame:
`scan_in_map = T @ scan_raw`. It is what `relocalize()` returns and what every prior proposes — the
local_map → global_map direction.
`mapping/relocalization/module.py:474-484`; `mapping/relocalization/priors.py:82-86`

</details>

**Q27.** The fiducial computes `map_T_marker @ inv(world_T_marker)`. Name the result.

<details><summary>answer</summary>

`map_T_world` — the same thing RANSAC proposes, which is the point. The survey gives the tag's pose in the
map (`map_T_marker`); the detector gives the same tag aggregated in the world frame (`world_T_marker`).
Chaining through the tag cancels it: `map_T_marker · marker_T_world = map_T_world`. Naming either operand
is the wrong answer; the marker is the thing that disappears.
`mapping/relocalization/priors.py:173-180`; `mapping/relocalization/module.py:250-274`

</details>

**Q28.** What is published on `/tf`, in which direction, and how often?

<details><summary>answer</summary>

`Transform(frame_id="world", child_frame_id="map")` — i.e. `world_T_map`, the INVERSE of the accepted fix,
inverted at the publish site. Emitted on a 2 s `rx.interval` with the latest accepted transform, not per
accept.
`mapping/relocalization/module.py:474-484`, `:61`, `:164-168`, `:522-528`

</details>

**Q29.** How is the aggregated translation of a tag computed?

<details><summary>answer</summary>

Huber-weighted IRLS around the cluster medoid: 5 iterations of weight = 1 inside `huber_delta_m` (0.05 m),
decaying as `delta / r` outside, then a weighted mean. The medoid — min total pose-distance member — is the
robust seed, so a single wild glimpse cannot anchor the fit.
`perception/fiducial/apriltag_aggregation.py:170-186`, `:187-193`, `:196-224`, `:38-39`

</details>

**Q30.** And the rotation? Why can't you just average quaternions?

<details><summary>answer</summary>

`q` and `-q` are the same rotation, so a naive mean cancels — the samples are first sign-aligned to the
medoid's hemisphere. Then the Markley eigen-mean: build the Huber-weighted scatter matrix `Σ w·qqᵀ` and
take the top eigenvector, re-signed to the reference each iteration.
`perception/fiducial/apriltag_aggregation.py:208-233` (Markley 2007, cited at `:231`)

</details>

**Q31.** Offline survey aggregation fuses ALL sightings of a tag; the live path uses a 5 s window. Why the difference?

<details><summary>answer</summary>

Offline is building the static marker map — every sighting in the recording is evidence about one fixed
tag, so more is strictly better. Live, the robot is moving and the estimate must describe the current
visit, so the window is purged relative to THAT marker's newest glimpse (not wall time, so a paused marker
keeps its window).
`perception/fiducial/apriltag_aggregation.py:244-261` vs `:263-283`, `:34-35`

</details>

**Q32.** What is the gravity gate, and what happens if every candidate fails it?

<details><summary>answer</summary>

The angle between a candidate's z-axis and world z-up must be ≤ `gravity_tilt_max_deg` — the module's
`Config` field, 10.0, matching relocalize.py's `GRAVITY_TILT_MAX_DEG` default. An all-tilted pool raises
`NoUprightCandidateError` — refused, never resurrected, because a tilted winner is a
rotationally-symmetric-floor mis-solve. The module treats it as a real rejection and consumes the fix.
`mapping/relocalization/relocalize.py:38`, `:127-130`, `:201-207`; `mapping/relocalization/module.py:79-80`, `:377-383`

</details>

**Q33.** Why does the rerank score on a WALL-ONLY subset, and what happens when there aren't enough walls?

<details><summary>answer</summary>

Floor and ceiling points have vertical normals that fit any yaw — they mask a 180° flip. The subset keeps
points whose normal is within ~44° of horizontal (`|n_z| < 0.7`). Under `MIN_WALL_POINTS = 100` per cloud
it raises `InsufficientWallEvidenceError` rather than silently falling back to the full cloud.
`mapping/relocalization/relocalize.py:210-232`, `:39-40`

</details>

**Q34.** What does `fitness` mean, and is it a number in [0, 1]?

<details><summary>answer</summary>

Open3D's registration fitness: the fraction of source points with a correspondence within
`RERANK_DIST` (0.15 m = 1.5 × the 0.1 m fine voxel). So yes, 0–1, dimensionless — which is why the accept
bar is a `Field(ge=0.0, le=1.0)`.
`mapping/relocalization/relocalize.py:35-37`, `:234-239`; `mapping/relocalization/priors.py:48-49`

</details>

**Q35.** The solver runs three stages. Which stage's fitness is returned, and with which transform?

<details><summary>answer</summary>

Stage 1 ranks all upright candidates by pre-ICP wall fitness → top 10. Stage 2 runs Tukey point-to-plane
ICP on the wall clouds and picks the best; that `best_fit` is the fitness returned. Stage 3 polishes on the
FULL clouds and its transform is what is returned — so the reported fitness is the stage-2 wall number, not
a score of the returned pose.
`mapping/relocalization/relocalize.py:234-267`

</details>

**Q36.** The jump guard budgets 5 m/s. What two things is it measuring between, and why a rate rather than a radius?

<details><summary>answer</summary>

Two consecutive ACCEPTED `map_T_world` fixes — the CORRECTION, not the robot's motion. The robot can cross
the building between accepts; `map_T_world` only moves by the LIO drift the fix cancels. So 5 m/s is a
drift-correction rate, not a speed limit.

A rate because the tolerable correction grows with the time drift had to accumulate: one fixed radius is
too tight after a long gap without accepts and too loose for two fires a frame apart.

The guard cannot run at all before an anchor exists — `_last_fix_map_T_world` is `None` and there is
nothing to measure against. Acquisition is held by corroboration instead (Q52).
`mapping/relocalization/module.py:66-68`, `:125-127`, `:433-453`

</details>

**Q37.** Why is that guard yaw-only, and why is the budget floored at one second?

<details><summary>answer</summary>

Tilt is already bounded by the gravity gate, so a flip between two accepted fixes is about z — yaw is the
whole story. The budget is `max(dt_s, 1.0)` s: two priors firing back to back would otherwise get a budget
near zero and refuse exactly the cross-source correction the fiducial prior exists to publish.
`mapping/relocalization/module.py:66-68`, `:437-444`

</details>

**Q38.** The fitness bar already rejects bad poses. Why not raise it and delete the jump guard?

<details><summary>answer</summary>

Because fitness is an OVERLAP ratio, not a correctness score: Open3D defines it as
`corres_number / source.points_.size()` — the fraction of source points with a correspondence inside
`RERANK_DIST`. In a symmetric room a 180°-flipped pose overlaps the map nearly as well as the true one and
scores high. The wall-only rerank narrows that gap; it does not close it.

The two ask different questions. Fitness: does this cloud fit HERE? The jump guard: could the robot have
GOT here since the last fix? Geometric and kinematic — orthogonal, so neither substitutes for the other,
and raising the bar only costs accepts on sparse clouds.
`mapping/relocalization/relocalize.py:234-239`, `:35-37`; `mapping/relocalization/module.py:388-413` vs `:433-453`, `:429`;
Open3D `RegistrationResult.fitness` https://www.open3d.org/docs/latest/python_api/open3d.pipelines.registration.RegistrationResult.html

</details>

---

## 4. The eval

**Q39.** What does `SourceTally` count, and when is it filled?

<details><summary>answer</summary>

Per source: `accepts`, `rejects`, and the list of accepted fitnesses. Filled only when
`verbose_eval_logging` is on (`--eval`), on the fire path the module already owns — so the summary needs no
log parsing.
`mapping/relocalization/eval.py:21-27`; `mapping/relocalization/module.py:105-106`, `:398-399`, `:496-499`

</details>

**Q40.** Name every way a fire can fail to publish a fix.

<details><summary>answer</summary>

The prior proposed nothing (`relocalize_with_prior` → `None`, silent); `InsufficientWallEvidenceError`
(too few walls, throttled warn); `NoUprightCandidateError` (all candidates tilted, throttled warn);
fitness below the bar — the prior's own while tracking, the higher `acquire_fitness_threshold` while
there is no anchor (warn with `source=` and `threshold=`); an acquisition fix buffered but the anchor not
yet corroborated (silent); and the jump guard on an accepted-but-implausible step, unless enough agreeing
rejects trip `_reacquire`. Plus the catch-all `logger.exception` boundary.
`mapping/relocalization/module.py:329-345`, `:369-386`, `:388-413`, `:427-431`, `:433-464`

</details>

**Q41.** Which of those does the tally actually see?

<details><summary>answer</summary>

Only the fitness-gate reject increments `rejects`. Empty proposal, wall-evidence, gravity, uncorroborated
acquisition and jump rejections all return before the tally, so `acc + rej` is not the fire count — read
the warnings for the rest.
`mapping/relocalization/module.py:398-399` vs `:334`, `:373-383`, `:427-431`, `:444-453`

</details>

**Q42.** What is `med_fit`, and why is it blank on the TOTAL row?

<details><summary>answer</summary>

Median of that source's ACCEPTED fitnesses (`-` when a source has none). Rows are per source and the two
sources fire at different rates on different evidence, so a median over both rows describes neither —
TOTAL sums the counts and blanks the column.
`mapping/relocalization/eval.py:34-44`

</details>

**Q43.** Why is the eval a pure util rather than a Module?

<details><summary>answer</summary>

It is formatting, not a data path: no I/O, no ports, no log parsing. The module already owns the
accept/reject data and calls `format_eval_summary` once at `stop()`. A module would add a port, a process,
and a transport to render a table.
`mapping/relocalization/eval.py:15`, `:30-31`; `mapping/relocalization/module.py:177-181`

</details>

**Q44.** How does `--eval` turn it on, and why is it guarded on the module being present?

<details><summary>answer</summary>

The CLI appends `-o <instance>.verbose_eval_logging=true` for each `RelocalizationModule` atom in the
blueprint, and only if the operator hasn't already set that key. It is guarded because
`blueprint.config()` is `extra="forbid"` — an unconditional override would hard-fail `--eval` on any stack
without relocalization.
`robot/cli/dimos.py:317-321`, `:374-387`; `core/coordination/blueprints.py:249`

</details>

---

## 5. Trial judgment calls

**Q45.** This PR ships no new unit tests. Defend it — and state the counter-argument.

<details><summary>answer</summary>

Precedent: the relocalization module itself landed as arkluc's #2160, +1124/−16 with zero test files —
the surface has no in-tree test convention to extend. Counter-argument, honestly: dimos runs a codecov
PATCH gate (`patch: true`), and repo-wide the convention is co-located `test_foo.py`. This is a decision to
defend out loud, not a default.
dimos commit `4f3b7e1b` (#2160) `--stat`; `.codecov.yml:6-9`; branch diff vs `29f35555` adds no test file
(one 2-line touch to `test_marker_detection_stream_module.py`)

</details>

**Q46.** The harness "gravity-gate bug" that cost a night. What actually happened?

<details><summary>answer</summary>

Our offline harness re-anchored the relocalization submap into the FULL BODY frame (`prep.py`,
`world_pts @ inv(P)`) — a step dimos never performs; production builds the submap in the LIO's
gravity-aligned WORLD frame via the real `VoxelGridMapper`. On the tilted mid360 rig that manufactured a
52° tilt and a phantom gravity-gate bug that does not exist in production.
`CLAUDE.md:58-63`

</details>

**Q47.** Why does a PR body describe CODE and not results?

<details><summary>answer</summary>

Numbers in prose rot: move one threshold and every quoted figure is a lie — that cost seven false claims in
one draft. Evidence belongs in figures a reviewer can look at and commands they can re-run; a load-bearing
number goes in the code beside the constant it justifies.
`CLAUDE.md:179-183`

</details>

**Q48.** Why is a full `dimos --replay` never used to check your own work?

<details><summary>answer</summary>

It is 10+ minutes (survey2 is 730 s), it ties up the LCM bus, and it hangs often enough to burn a whole
agent timeout on one number. Test VALUES on constructed known-truth data instead; if wiring must be
proven, bound it with `--seek`/`--duration`. A full replay is a final deliverable step only.
`CLAUDE.md:70-82`

</details>

**Q49.** `relocalization.md` says the fitness threshold is 0.6; the pre-prior code default was 0.45. What is it now?

<details><summary>answer</summary>

0.6 — `PriorConfigBase.fitness_threshold` defaults to 0.6, per prior, and the blueprint passes no kwargs,
so both entries take the default. The point stands regardless: a fresh run beats the code, the code beats
the docs, the docs beat what anyone remembers.
`mapping/relocalization/priors.py:48-49`; `mapping/relocalization/module.py:90-91`; `CLAUDE.md:20-28`

</details>

**Q50.** "Default to adding, never change an API." Where does that show up in this diff?

<details><summary>answer</summary>

`relocalize()` keeps its `(T, fitness)` signature — the solver tail was extracted into `refine_candidates`
beneath it, so existing callers are untouched. And the module keeps a bit-for-bit pre-prior path: with
exactly one enabled RANSAC prior it calls plain `relocalize()`, so the lidar-only stack behaves as it did
on main.
`mapping/relocalization/relocalize.py:133-152`; `mapping/relocalization/module.py:347-360`

</details>

**Q51.** We wrote a compat shim mapping removed config keys to a friendly error, then deleted it. Why was it dead the day it was written?

<details><summary>answer</summary>

`_MOVED_TO_PRIORS` guarded operators who had set `fitness_threshold` / `min_local_points` at the module
level. Those keys only ever existed on this branch, and the branch is unmerged — no operator config, no
blueprint, no other repo has ever set them. Zero users by construction, not by measurement.

And the fallback was already correct: `blueprint.config()` is `extra="forbid"`, so an unknown key raises
on its own. Fowler's speculative generality — a guard against a migration that cannot happen. Deleted in
`adf6b8f0b`.
`core/coordination/blueprints.py:249`; dimos commit `adf6b8f0b`

</details>

**Q52.** Why doesn't the first accepted fix become the anchor?

<details><summary>answer</summary>

Anchor poisoning. The jump guard measures every later fix against `_last_fix_map_T_world`, so whatever
locks first becomes the yardstick for everything after it. Let one single-shot estimate through — a mirror
flip in a symmetric room, a bad surveyed marker pose — and the guard then refuses the CORRECT fixes for
disagreeing with it. A hard gate on a value that gets REPLACED has no memory of what it rejected, so the
decision can never be revisited.

So acquisition accumulates evidence instead: `acquire_corroboration` (2) independent fixes must agree
within `ANCHOR_AGREE_M` (0.5 m — inside one second of the jump budget, outside marker-pose noise), each
clearing `acquire_fitness_threshold` (0.7, above tracking's 0.6), and agreement is tested against every
buffered fix so a drifting chain never corroborates. The escape hatch runs the same way: once
`reacquire_after_rejects` (4) jump-rejected fixes agree with each other and not with the anchor, the
ANCHOR is the outlier and gets replaced.

That is the same shape as the mainstream stacks — external systems, not verified in this repo: Nav2's AMCL
keeps a particle distribution, so a wrong hypothesis loses weight as evidence accumulates and global
re-seeding can recover; Cartographer adds loop closures as pose-graph CONSTRAINTS with outlier rejection,
so one bad constraint is outvoted at optimization; RTAB-Map runs a Bayes filter over loop-closure
hypotheses and can drop a link later. Evidence that accumulates, not a commitment that replaces.
`mapping/relocalization/module.py:69`, `:81-86`, `:415-425`, `:427-431`, `:455-464`

</details>

---

## Score log

| date | section | score | missed |
|---|---|---|---|
| 2026-07-24 | round 1 — live | 2.5/5 answered | `FiducialPrior._pending` swap-under-lock failure mode (dict-changed-size-during-iteration, or a fix wiped between read and clear); `map_T_marker @ inv(world_T_marker)` resolves to `map_T_world` — named an operand, not the result. Partial: no-self-selection — right instinct (one consistent standard), invalid premise (both priors do not fire on one cycle under single-prior dispatch). Correct: offline-vs-online aggregation windowing; spotting that `EmptyProposalError` no longer exists. |
<!-- | 2026-07-23 | 2. The prior system | 8/10 | Q19, Q21 | -->
