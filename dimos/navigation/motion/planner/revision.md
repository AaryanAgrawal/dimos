# Spec revision: lattice anchoring + motion-conditioned envelope

AGREED 2026-08-06 (measurement results folded in from
[envelope_results.md](envelope_results.md)). One revision, gold + candidate +
judge move together, one re-baseline, one autoresearch re-earn. Phase 1 =
spec + gold + referee; phase 2 = the rust candidate re-earns.

## Evidence

- One lidar return 5 m behind the robot flips the door route 13.27 m ↔ 1.96 m
  (door.zenoh t=7.09; pinned as strict xfail in `referee/test_grid_invariance.py`).
- Cause: both gold (`scenarios.se2_path`) and the rust candidate anchor their
  lattices at `min(obstacle bbox, goal) − pad` — sample positions are a
  continuous function of the far corner of the cloud.
- door2.zenoh: a doorway the trunk (0.31 m) walks through reads as a wall
  because the search carries the all-gait union footprint (0.50 m) + 0.05
  precision per side = 0.60 m minimum everywhere, regardless of travel
  direction.

## Changes

1. **Absolute-lattice origin.** Grids anchor to the world frame's own lattice
   (`floor(corner / period) · period`), odom frame on the robot, scenario
   frame in the referee. Obstacles appearing/vanishing can add content but
   can never move a sample position.
2. **Pitch snapped to the map's voxel size.** `fine = voxel / 2` (0.04 for the
   go2's 0.08), `cell = 3 · fine` (0.12, unchanged). Every voxel centre lands
   exactly on a fine sample: a voxel pattern reads the same clearance wherever
   it sits, and whole-voxel translation of a scene translates the answer
   bit-exactly. Voxel size is a config constant of the deployment (never
   sniffed from data); changing it is a new spec = new baseline.
3. **Per-heading envelope.** Feasibility uses the swept box for the edge's
   body-frame drift angle, not the all-gait union; arcs add a curvature
   inflation term. Rows measured in the fitted sim
   (`simulation/envelope.py --bake`) as max swept outline over the gait cycle
   at the executable governed band (stand + 0.35 + 0.50 — the sim cannot
   execute 0.20, see min_speed below). `precision` (0.05) is untouched — it
   is the measured follower tracking floor.

   Storage on `Embodiment` (frozen dataclass, plain floats — serializes into
   the rust config blob as-is):

   ```python
   # (deg, length, width, off_x, off_y); |angle| rows — 0 = nose-first,
   # 90 = strafe, 180 = reverse. off_y is stored for POSITIVE drift and
   # mirrored by sign at lookup (the swept box lags the drift laterally by
   # up to 57 mm; a blind fold burns 15-56 mm of width). EMPTY = the union
   # length/width/center_off applies at every heading (today's behavior,
   # and the fallback for any unmeasured embodiment).
   envelope: tuple[tuple[float, float, float, float, float], ...] = ()
   # extra width per rad-per-metre of curvature (edge dyaw / edge length):
   # measured 0.0334, residuals <= 12 mm. Curvature, not per-edge yaw, so
   # the number survives lattice pitch changes unmeasured.
   arc_inflate: float = 0.0

   def envelope_at(self, drift: float) -> tuple[float, float, float, float]: ...
   def offsets(self, step=0.05, drift: float | None = None) -> np.ndarray: ...
   ```

   - **The union re-baselines to the honest 0.883 × 0.593** (the recorded
     0.852 × 0.495 was the max over a smaller command sweep — fast strafe and
     slow tight arcs were outside it). The union's jobs — judge veto,
     `half_diag`, fallback for unmeasured embodiments, body carve — are
     exactly where honest-conservative is the only acceptable property. The
     measured rows are the only non-conservative path.
   - Rows sit at the lattice's own drift angles (0, 26.6, 45, 63.4, 90 +
     reverse family) — nearest-row lookup is exact for every edge the search
     generates, no interpolation semantics.
   - No speed column: baked into the rows at measurement time (governor
     section below).
   - **min_speed (0.2) is suspect but unproven**: the SIM marches in place at
     0.2 (wider than walking at most headings), but the sim also undertracks
     slow commands vs the field (gain 0.62 vs 0.80 at 0.35). Verify on the
     real robot (command 0.2, run the tracking pass + field.py gain) before
     raising min_speed to 0.35 — a strictly-dominated crawl is a follower
     change that rides this same re-baseline if confirmed. The sim's low-speed
     undershoot is a tuning item of its own now: `simulation/FINDINGS.md` §E.

4. **Governor-time pricing in the gold cost** (adopted after phase 1.5; gold
   side only until the candidate re-earns in phase 2). An edge's tightness
   multiplier is `MAX_SPEED / governor_speed(clearance)` — what a metre there
   costs in time under the follower's own committed speed law, normalized so
   open space is 1.0 — and no longer a comfort ramp. Three reasons:

   - Planner and follower optimize the **same clock**. The follower is
     contractually slowed to `governor(clearance)` in tight places; a planner
     that priced tightness by a separate preference was choosing detours
     against a cost the robot never pays.
   - `comfort` **leaves the cost entirely**. It stays a labelling radius and
     the smoothing cap, and (with the spawn disk, same revision) it is finally
     a knob that can be turned without re-baselining the battery.
   - The charge **caps itself**: the governor floors at MIN_SPEED, so the
     multiplier tops out at MAX_SPEED/MIN_SPEED = 2.5x at contact — the same
     ceiling the comfort ramp had by hand, now derived. In between it is
     cheaper: 1.65x vs 1.92x at 0.15 m of clearance, 1.00x vs 1.19x at 0.35 m,
     and 2.50x vs 2.31x right at the precision floor, where fiction begins.

   Clearance is still read on the **union** (a preference has to be comparable
   across edges); feasibility stays per-heading. `control/profile.py` is the
   one copy of the curve, imported by the search, and its constants are in the
   gold cache key (`v11-governor-time`) — retuning the law may not serve a gold
   searched under the old one.

   *Measured, curated 16 + gen 40: doors and corridors thread more.
   `corridor` (0.9 m gap) stops detouring — 5.77 m around → 3.96 m straight
   through the middle, min truth clearance 0.242 m; `door_side` 5.55 → 3.85 m.
   `narrow_gap` (0.26 m opening) still goes around and `boxed_in` still
   refuses: pricing cannot reach what feasibility forbids. No veto flip, no
   label flip, no DQ on any of the 56 worlds; gold's pillar stays 1.0 and its
   gate goes 108.98 → 109.32 curated / 108.13 → 108.49 on the mixed roster.
   The rust candidate, which still prices comfort, drops 97.50 → 92.68 — the
   gap phase 2 exists to close.*

## Acceptance

- [x] `test_grid_invariance.py` xfail flips to passing (phase 2, when the rust
  candidate re-anchors); add whole-voxel-translation bit-exactness and
  far-point invariance over the full battery.
  *Flipped. Far-point invariance is bit-exact in the crate
  (`a_far_point_cannot_move_the_answer`) because the anchored lattice indexes
  absolutely — `k * FINE`, never `origin + i * FINE`. Whole-period translation
  is route-exact, not bit-exact, and cannot be: 0.24 is not a dyadic rational,
  so `(x + d) / FINE` and `x / FINE + d / FINE` differ in the last bit and a
  sample on a rounding boundary may tip.*
- [x] Field scenarios: door.zenoh and door2.zenoh (recorded worlds) route through
  the doorway with the forward envelope.
  *door.zenoh replays at 2.17 m mean published arc against 6.99 m before (the
  recorded run was 5.71 m), holds 14 -> 11; door2.zenoh 1.05 m against 2.24 m,
  holds unchanged at 9/9 agreeing. Both deterministic. Plan wall time falls,
  61.5 -> 54.1 ms/tick on door and 12.2 -> 11.5 on door2, so the finer pitch is
  paid for by the union-first fast path — but door is still 2.7x over the 20 ms
  budget, as it was before, and that is now the open item on this recording.*
- [x] Judge gains a per-mode envelope-violation metric (planner-assumes vs
  follower-does mismatch shows up named, not just as wall contact).
  *Phase 1.5: the planner judge sweeps the drift row and reports `env_viol`
  (union hits, the row does not); gold's 3 self-DQs became 3 named violations,
  6-27 mm. The control judge keeps the union as its pillar — it grades the
  follower, which is the thing that may leave the row — and reports the same
  metric as the share of ticks spent outside it.*
- [x] **Start witness** (adopted phase 1.5): a pose the robot actually occupies
  may always be departed. The seed's feasibility is read at the true start
  pose, not at the cell it snaps to.
  *`door_side` routes again (the snap cost it 0.083 -> 0.043 against a 0.05
  margin); a start with negative true clearance still refuses.*
- **Gold before/after review**: old vs new gold paths overlaid per world
  (curated 16 + gen 40) as a browsable artifact — Ivan reviews before phase 2.
- Referee re-baselined; runtime planner re-earns via the lab on the new spec.
  *Phase 2 landed the port, not the re-earn: curated 94.65 → 107.62 (gold
  pillar 0.842 → 0.976) and mixed gen40 92.85 → 102.22 (0.830 → 0.926) against
  frozen gates of 109.32 / 108.49. What is left is not port drift — the crate
  and `target-py` agree route-for-route on every offender (gen012 7.80 vs
  7.75 m, gen039 11.26 vs 11.15, gen007 10.91 vs 11.05) — it is the honest
  candidate's own handicap: gen012 and gen039 score ~10 because their cloud-built
  field will not thread a gap gold's box-exact SDF walks, and both scored ~10
  before this revision too. That is what the lab is for.*

## Amendment: standing is not the union (adopted 2026-08-07)

The envelope split feasibility per-heading but left every *seed-entry* test on
the union: the witness (`scenarios.py` / `planner.rs`) and the rust repair's
`free()` all ask "does the union fit" at a pose the route only ever justified
with a drift row. So the planner threads a row-passable gap, the robot follows
it in, and the next replan from mid-gap refuses — single-pose stub, follower
holds, stuck until timeout. `--score --gen 40`: 6/40 fail (5 timeouts + 1
collision); gen000 freezes at a pose reading union +0.033 < margin 0.05 while
the nose row reads +0.121. Pre-envelope this was impossible: one shape meant
"the plan accepted this pose" and "the witness accepts it" were the same test.

The doctrine error is "standing has no direction of travel, so the union is
the honest shape." Standing occupies the *static body*, not the union of all
swept walking boxes. No new bake needed: use the **intersection of all
envelope rows** — for GO2, 0.781 × 0.416 at off_x −0.039 (union is
0.883 × 0.593). It is the largest shape nested in every row, so the invariant
holds *by construction*: a pose whose row clears the margin also passes the
witness — **replanning from your own emitted route can never refuse.**

- Witness reads the intersection box (gold + rust); rust `fit_bin`/repair
  predicate likewise. Derived from the envelope at construction, mirrored-`off_y`
  union of each row's ± drift like `envelope_at` does; falls back to the
  union box for embodiments with no measured envelope (nothing changes there).
- Turn-in-place *edges* keep the union — that is real motion sweeping the full
  shape, and it correctly forbids a pirouette mid-gap.
- Gold cache version bumps (`v12-standing-witness`).

Acceptance:

- [ ] New pinned referee test: every k-th pose along every emitted path
  (gold and candidate, curated + gen), replayed as the start of a fresh query,
  yields a plan — refusal from your own route is a failure.
- [ ] `--score --gen 40 -s 'gen*'`: 0 timeouts, 0 collisions (gen014's
  collision inspected separately if it survives the fix).
- [ ] Curated + gen batteries re-earn ≥ 107.62 / 102.22 through the unchanged
  judge; gold gate moves only where refusals became routes.

## Speed: eliminated via the governor, not modelled

The envelope only binds where clearance is small, and there both tracks obey
`speed ≤ governor(clearance)` by contract. So: measure `envelope(mode, speed)`
on a 2-D grid in the fitted sim (once, to prove monotonicity), then bake one
runtime row per mode at the speeds the governor permits in tight corridors.
The search stays speed-free in the sense that matters — it never *chooses* a
speed, and the envelope it plans with carries none. It does now *price* by one
(change 4): the governor is a function of clearance, which the search already
knows, so reading it costs the search nothing and buys agreement with the
follower about what a tight metre is worth.
Hole to close: yaw rate is not clearance-governed — measure the arc row at
the deployed `max_yaw_rate`, or cap yaw rate in tight segments.

## Open questions

- Follower mode discipline in tight segments (crab correction mid-doorway
  exceeds the forward envelope): rely on stamp-encoded slow-down + judge
  metric, or add an explicit constraint?
- fine = 0.04 costs ~1.6× SDF precompute — measure against the 20 ms budget.
- Yaw resolution at doors: 16 bins mean up to 11.25° misalignment =
  0.85·sin ≈ 0.16 m of phantom width — more than the envelope recovers. More
  bins (2×/4× precompute) vs local yaw refinement in tight cells only. Move
  headings stay at 16 either way (envelope varies mm per bin).
