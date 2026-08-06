# Spec revision: lattice anchoring + motion-conditioned envelope

Draft for discussion. One revision, gold + candidate + judge move together,
one re-baseline, one autoresearch re-earn.

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
   body-frame drift angle, not the all-gait union; arcs add a yaw-rate
   inflation term. Rows measured in the fitted sim as max swept outline over
   the gait cycle. `precision` (0.05) is untouched — it is the measured
   follower tracking floor.

   Storage on `Embodiment` (frozen dataclass, plain floats — serializes into
   the rust config blob as-is):

   ```python
   # (deg, length, width, center_off); |angle| — left/right symmetric,
   # 0 = nose-first, 90 = strafe, 180 = reverse. EMPTY = the union
   # length/width/center_off applies at every heading (today's behavior,
   # and the fallback for any unmeasured embodiment).
   envelope: tuple[tuple[float, float, float, float], ...] = ()
   arc_inflate: float = 0.0  # extra width per rad of yaw change on an edge

   def envelope_at(self, drift: float) -> tuple[float, float, float]: ...
   def offsets(self, step=0.05, drift: float | None = None) -> np.ndarray: ...
   ```

   - `length/width/center_off` keep meaning the UNION: the judge's veto,
     `half_diag`, the body carve and viz stay conservative; only the search's
     feasibility check becomes heading-aware. Forgetting to pass a drift
     angle is conservative, never unsafe.
   - Rows sit at the lattice's own drift angles (0, 26.6, 45, 63.4, 90 +
     reverse family) — nearest-row lookup is exact for every edge the search
     generates, no interpolation semantics.
   - No speed column: baked into the rows at measurement time (governor
     section below).

## Acceptance

- `test_grid_invariance.py` xfail flips to passing; add whole-voxel-translation
  bit-exactness and far-point invariance over the full battery.
- Field scenarios: door.zenoh and door2.zenoh (recorded worlds) route through
  the doorway with the forward envelope.
- Judge gains a per-mode envelope-violation metric (planner-assumes vs
  follower-does mismatch shows up named, not just as wall contact).
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
