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
3. **Per-mode envelope.** `Embodiment.envelope[mode] → (length, width,
   center_off)` for the edge classes the search already prices (forward /
   strafe / reverse / arc). Feasibility uses the mode's envelope, not the
   union; arcs get a yaw-rate inflation term. Rows measured in the fitted sim
   as max swept outline over the gait cycle. `precision` (0.05) is untouched —
   it is the measured follower tracking floor.

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
