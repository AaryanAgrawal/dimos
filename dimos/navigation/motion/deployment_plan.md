# Deploying the motion stack onto the robot

The time-critical half of the motion stack — local planner, trajectory
controller, cmd_vel mux — moves onto the Go2. Mapping stays on the laptop for
now. This is the plan, the reasoning behind the cut, and what is still missing.

## The cut

```
laptop                                   robot
------                                   -----
RayTracingVoxelMap  --- local_map ----->  MotionPlanner        (5 Hz)
MLSPlannerNative    --- planner_path -->  TrajectoryFollower   (10 Hz)
GoalRelay                                 CmdVelMux
MovementManager (click half)              OdomBodyFrame
vis_module                                go2web bridge
                    <-- lidar, odom ----
                    --- tele_cmd_vel -->
```

Everything on the right is one baked host binary, plus the bridge the robot
already runs.

### Why not the whole stack

The raycaster is the expensive module and we have not yet proven the robot can
carry it. Until then it stays on the laptop and we accept that `local_map`
crosses the link.

### What this cut does and does not buy

It does **not** remove a wire crossing from the perception→action loop. Today
the loop crosses twice (lidar out, `cmd_vel` in); after the move it still
crosses twice (lidar out, `local_map` in), and the message got much bigger.

What it buys is **jitter immunity on the last stage**: the follower ticks at a
steady 10 Hz off a locally-held path instead of the robot receiving `cmd_vel`
in bursts whenever the link hiccups. That is the actual goal — smooth, correct
motion — and it is worth the bandwidth. But it is a jitter argument, not a
latency argument, and the difference matters when deciding what to do next.

### The end state (option C, later)

Move the raycaster to the robot as well and the loop is fully local: the link
then carries lidar for visualization, goals, teleop and telemetry, none of it
in the control path. MLS can stay on the laptop — its path is small and slow,
and a stale global route is benign, because the local planner only takes a
carrot along it.

Notably this costs no extra bandwidth over the current cut: MLS consumes
`local_map` too (its `global_map` is remapped off), so the cloud crosses the
link either way — inbound now, outbound then. The blocker is purely whether
the robot's SBC can afford the raycaster. **That measurement is in flight.**

## Consequences we have to handle

### Staleness — the link was the deadman

Today a dropped link stops `cmd_vel` and the bridge watchdog halts the robot.
Once the loop runs on the robot, that accidental safety is gone: the planner
would keep replanning on a frozen map while the follower tracks the result at
cruise speed. Worse, the speed governor reads clearance from that same stale
cloud, so it stays confident.

Three guards:

| where | rule | status |
|-------|------|--------|
| `MotionPlanner` | `local_map` older than `max_map_age_s` (5 s) → publish the single-pose hold stub | **done** (`fd5c873a5`) |
| `TrajectoryFollower` | `path` older than N → zero the twist | todo |
| `CmdVelMux` | `nav_cmd_vel` stale → zero `cmd_vel` | todo, part of writing the mux |

Staleness is measured from **arrival**, not `msg.ts`: the mapper's clock is not
the robot's, and what these guard is how long since the producer was last heard
from.

### Splitting MovementManager

`MovementManager` is a click-to-goal relay *and* a velocity mux, and the two
halves land on opposite sides of the link. The seam is not clean, because
`_on_teleop` does two unrelated things: it preempts `cmd_vel` (robot side) and
calls `_cancel_goal()`, which publishes `stop_movement` (consumed by the
follower — robot side) and a NaN goal (consumed by GoalRelay/MLS — laptop
side).

So one keystroke has to land on both sides:

- **rust, robot**: in `nav_cmd_vel`, `tele_cmd_vel` → out `cmd_vel`,
  `stop_movement`
- **python, laptop**: in `clicked_point`, `tele_cmd_vel` → out `goal`,
  `way_point` (including the NaN cancel)

Both subscribe `tele_cmd_vel`. Teleop originates on the laptop, so the python
half gets it for free, and `stop_movement` stays co-located with its only
consumer. Do not route the cancel back from the rust half.

### No tf on the robot

Nothing in the robot-side set needs tf. `OdomBodyFrame` takes a **config
quaternion** (`mount_rotation`) and composes out the mount rotation — no
lookup, no subscription. The follower gets its pose straight off the odometry
message. Everything `GO2Zenoh` adds on top of the bridge — the
`odom → mid360_link` edge, the static mount tree, camera intrinsics — exists to
make rerun draw correctly, and stays on the laptop.

So: **no on-robot equivalent of GO2Zenoh is needed.** The bridge already
publishes `odometry`; the baked host subscribes it locally.

(Cleanup while we are here: `ControllerConfig.frame_id` and the
`TrajectoryController` docstring claim the controller does a live tf lookup of
that frame. It does not and never has. Delete it rather than port a lie.)

## What has to be built

Nothing on the robot side exists yet. `dimos bake --list` currently registers
only `ray_tracing` and `mls_planner` — which is to say, the only two bakeable
modules are the two that stay on the laptop.

Every crate must be a **standalone native module**, not just a bake input. The
registry format enforces this: `python` is a required key in the metadata
table, so a bake module must name a python `NativeModule` wrapper. Each crate
gets the raycaster's shape:

```
[[bin]] name = "..." path = "src/main.rs"   ← standalone shim over run_module_core
src/module.rs                                ← the struct, linkable into a host
[package.metadata.dimos.module.<id>]         ← registry entry
  python = "...:WrapperClass"                ← the standalone wrapper
```

Baking is then purely additive: develop and debug each module standalone
through its python wrapper, bake only when you want one binary on the robot.

### Build order

1. **`motion/profile/rust/`** — the z-band nearest-neighbour clearance query
   plus `encode_precision`/`decode_ceilings` (`control/profile.py`,
   `control/world.py:path_clearance`). The only real algorithm work, and both
   adapter modules need it. A uniform grid hash likely beats a KD-tree here
   given everything upstream already thinks in voxels. Same discipline as the
   existing law crates: pyo3-optional, parity-tested against the python.
2. **`motion/adapter/rust/`** — module structs wrapping `dimos-motion2-target`
   (planner) and `dimos-motion2-tc` (controller), with `main.rs` shims,
   `[[bin]]` entries, metadata, and python `NativeModule` wrappers.
   `MotionPlanner`/`TrajectoryFollower` are plain python `Module`s today, so
   even the wrapper classes are missing.
3. **`movement_manager/rust/`** — the mux, plus splitting the click half off in
   python.
4. **`odom_body_frame`** — a second module id in the existing mls_planner
   crate. Cheapest item on the list; the metadata table is keyed by id, so it
   needs no new crate.

The two existing law crates (`dimos-motion2-target`, `dimos-motion2-tc`) stay
**pure algorithm**. They are deliberately dependency-light and parity-locked to
python; do not drag `dimos-module`, `lcm-msgs` and tokio into them. The adapter
crate is the transport shell, mirroring the python layout exactly.

Open layout question: one adapter crate with two `[[bin]]` entries, or two
crates sharing the `profile` crate. Start with one and split if it chafes.

## Deployment mechanics

### The baked host runs standalone

`NativeModule` spawns a **local** subprocess (`subprocess.Popen`) — no ssh, no
remote exec. So `baked_host()` in a laptop blueprint would spawn the binary on
the laptop. The robot host therefore runs standalone: bake for aarch64, copy
it over, systemd unit, config from stdin. The laptop blueprint becomes a
separate blueprint that declares the robot's topics as external.

Cross-compilation is already supported: `--builder cross|zigbuild --target
<triple>`.

### Config is the sharp edge

`--emit-config` builds the standalone stdin blob from **python class
defaults**, not from blueprint values. For this stack that is not cosmetic:

> `OdomBodyFrameConfig.mount_rotation` defaults to identity. The real value is
> computed in the blueprint by `_mount_rotation()`. A host baked today runs
> with **no odometry leveling** — precisely the failure blueprints.py warns
> about: *"they must agree or nav steers off-heading."*

Same shape for the follower's eleven tuned `controller_config` numbers and the
planner's `replan_hz` / `goal_lookahead_m`. Silent, and it presents as a
controller bug.

Either teach bake to read a blueprint, or hand-write the stdin JSON and treat
it as the deployment artifact — but decide deliberately. `mount_rotation` is
the test case.

### Blueprint-as-arg

The eventual fix, and the reason it is worth doing: the blueprint is the only
place the tuned wiring and config actually live. The three remappings
(`MLS.path→planner_path`, `MotionPlanner.odometry→body_odometry`,
`global_map→global_map_unused`) are exactly the `--remap` flags one would
otherwise retype, where drift is a silent misconnect.

Two design points:

- **Partition, do not require purity.** A blueprint is always mixed;
  `vis_module` will never be rust. Bake a named subset and leave the rest
  external.
- **Emit the replacement blueprint** — the `baked_host(...)` call with members,
  remaps and configs — so the binary and the python driving it cannot drift.

Known limit it will surface immediately: `select_modules` refuses duplicate
ids ("one instance per host, per-instance namespacing is not implemented"),
while blueprints can carry two instances of a class under different namespaces.
Not blocking for this stack.

### Zenoh links

The baked host needs links **both** ways: locally to the bridge (`odometry`,
`cmd_vel`) and across to the laptop (`local_map`, `planner_path`,
`tele_cmd_vel`). Given the known multicast problems on the 10.55.1.x LAN,
assume explicit connect endpoints rather than scouting — especially for
something starting at boot.

## Open questions

- **Can the robot host the raycaster?** Measurement in flight. Decides whether
  we can reach the end state above.
- **Plan-to-plan discontinuity.** Once the follower ticks steadily off a local
  path, what is left to twitch is the 5 Hz replan snapping the path between
  cycles. Does the battery score path-switch discontinuity across replans? If
  not, that is the metric to add — it is a planner property, and no amount of
  deployment work fixes it.
- **Live cloud size vs battery worlds.** `TargetEpisode.plan` builds its SDF
  grid over the pose+goal+cloud bounding box at 0.05 m, so cost tracks the live
  local map's extent and density, which the synthetic scenario worlds do not
  represent. Worth pushing one recorded `local_map` through `plan()` on the
  laptop if the follower ever stutters.
