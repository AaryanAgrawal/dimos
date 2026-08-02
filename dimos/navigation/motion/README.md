# motion

Local motion: plan a path around obstacles, walk it, prove both work.

| dir           | what                                                                                         | docs                                                             |
|---------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| `planner/`    | SE(2) local planner (evolved rust crate) + its referee: worlds, gold oracle, judge           | [planner/autoresearch/README.md](planner/autoresearch/README.md) |
| `control/`    | closed-loop executor: episode runner, follower law (python + bit-exact rust), executor judge | [control/tools.md](control/tools.md)                             |
| `simulation/` | the matched Go2 MuJoCo env (physics fitted to two real recordings)                           | [simulation/tools.md](simulation/tools.md)                       |
| `adapter/`    | planner + follower as dimos modules for the go2-zenoh blueprints                             | [adapter/tools.md](adapter/tools.md)                             |

Oneliners for everything: [tools.md](tools.md).

## I/O — everything is stock dimos msgs

**MotionPlanner** (adapter)

- in: `local_map: sensor_msgs.PointCloud2` — raycaster cloud
- in: `odometry: nav_msgs.Odometry` — own pose (pointlio, `odom` frame)
- in: `planner_path: nav_msgs.Path` — global path; the goal is a carrot along it
- out: `path: nav_msgs.Path` — the local plan. Per-waypoint `ts` deltas encode
  required precision (`dt = segment / governor_speed(clearance)`, see
  `control/profile.py`); a single-pose path means "hold, no safe route"

**TrajectoryFollower** (adapter)

- in: `path: nav_msgs.Path` — stamped or plain (flat `ts` just disables the hint)
- in: `odometry: nav_msgs.Odometry`
- in: `local_map: sensor_msgs.PointCloud2` — optional, richer clearance hint;
  without it the follower decodes the path stamps instead
- in: `stop_movement: std_msgs.Bool`
- out: `nav_cmd_vel: geometry_msgs.Twist` — body-frame (vx, vy, wz) into the
  walking policy
- out: `goal_reached: std_msgs.Bool` — latched arrival

**Inner follower law** (in-process, python and rust produce identical bits):

```
update(pose: PoseStamped, path: Path, t, clearance: ndarray | None) -> Twist
```

The path timestamps are NOT a schedule — only their deltas carry information
(slow segment = tight segment = track carefully), a follower must never chase
the clock, and running slower than the encoding is always legal. Third-party
Path producers/consumers interoperate; they just don't get the precision hint.
