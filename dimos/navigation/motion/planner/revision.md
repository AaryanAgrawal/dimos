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
     change that rides this same re-baseline if confirmed.

## Acceptance

- `test_grid_invariance.py` xfail flips to passing (phase 2, when the rust
  candidate re-anchors); add whole-voxel-translation bit-exactness and
  far-point invariance over the full battery.
- Field scenarios: door.zenoh and door2.zenoh (recorded worlds) route through
  the doorway with the forward envelope.
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

## Speed: eliminated via the governor, not modelled

The envelope only binds where clearance is small, and there both tracks obey
`speed ≤ governor(clearance)` by contract. So: measure `envelope(mode, speed)`
on a 2-D grid in the fitted sim (once, to prove monotonicity), then bake one
runtime row per mode at the speeds the governor permits in tight corridors.
The search stays speed-free; the planner keeps dealing only in tolerances.
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
