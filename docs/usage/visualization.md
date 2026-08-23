---
title: "Viewer Backends"
---

Dimos uses Rerun for visualizations. It can be disabled by using
`dimos --viewer none ...`.

Blueprints add Rerun stream visualization with `vis_module(...)`, which renders typed
robot streams according to `GlobalConfig.viewer`.

## Quick Start

Choose your viewer via the CLI:

```bash
# Rerun native viewer - dimos-viewer with built-in teleop + click-to-navigate
dimos run unitree-go2

# Disable visualization:
dimos --viewer none run unitree-go2
```

Control how the Rerun viewer opens with `--rerun-open` and `--rerun-web`:

```bash
# Open native desktop viewer (default)
dimos --rerun-open native run unitree-go2

# Open web viewer in browser
dimos --rerun-open web run unitree-go2

# Open both native and web
dimos --rerun-open both run unitree-go2

# No viewer (headless) — data still accessible via gRPC
dimos --rerun-open none run unitree-go2

# Serve the web viewer without auto-opening a browser
dimos --rerun-web --rerun-open native run unitree-go2
```

## Viewer Modes Explained

### Rerun Native (`rerun`, `--rerun-open native`) — Default

**What you get:**
- [dimos-viewer](https://github.com/dimensionalOS/dimos-viewer), a custom Dimensional fork of Rerun with built-in keyboard teleop and click-to-navigate
- Native desktop application (opens automatically)
- Better performance with larger maps/higher resolution
- No browser or web server required

---

### Rerun Web (`rerun`, `--rerun-open web`)

**What you get:**
- Browser-based dashboard at http://localhost:7779
- Rerun 3D viewer + command center sidebar in one page
- Teleop controls and goal setting via the web UI
- Works headless (no display required)

---

### Foxglove (`foxglove`)

**What you get:**
- A [Foxglove](https://foxglove.dev) WebSocket server on port 8765
- Every LCM/Zenoh topic carrying an LCM struct advertised as a Foxglove channel, decoded only while a panel is subscribed to it
- Teleop and click-to-navigate from the app, on `tele_cmd_vel` and `clicked_point`

Not served: `RobotState`, `JointCommand`, `MotorCommandArray`, `TrajectoryPoint`,
`TrajectoryStatus` and `EntityMarkers` (no LCM struct), and images remapped through `jpeg_lcm`.

To drive the robot, add a Teleop panel: its `geometry_msgs/Twist` reaches `tele_cmd_vel`. To send a
goal, use the 3D panel's Publish tool with type Point: its `geometry_msgs/PointStamped` reaches
`clicked_point`. The topic each panel publishes on is yours to pick, since the bridge routes on the
schema. A Teleop panel that goes away mid-drive publishes a zero Twist.

The bridge also runs standalone, so it attaches to a stack that is already running from any branch:

```bash
# terminal A — any blueprint, on any branch
dimos --viewer none run unitree-go2

# terminal B — a checkout that has the bridge
uv run dimos foxglove-bridge
```

In the Foxglove app: Open connection → Foxglove WebSocket → `ws://localhost:8765`. Use `--host` and
`--port` to move the server. `--viewer none` only saves the Rerun overhead — the bridge works with
Rerun running too.

---

## Rendering with Custom Blueprints

To enable visualization in your own blueprint, use `vis_module`:

```python skip
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.camera.module import CameraModule
from dimos.visualization.vis_module import vis_module

camera_demo = autoconnect(
    CameraModule.blueprint(),
    vis_module(viewer_backend=global_config.viewer),
)

```

Run the stack locally (this blocks until you stop the process):

```python skip
from dimos.core.coordination.module_coordinator import ModuleCoordinator

if __name__ == "__main__":
    ModuleCoordinator.build(camera_demo).loop()
```

Every LCM stream, such as `color_image` (output by CameraModule), that uses a data type (like `Image`) that has a `.to_rerun` method will get rendered (`rr.log`) using the LCM topic as the rerun entity path. In other words: to render something, simply log it to a stream and it will automatically be available in rerun.

## Performance Tuning

### Symptom: Slow Map Updates

If you notice:
- Robot appears to "walk across empty space"
- Costmap updates lag behind the robot
- Visualization stutters or freezes

This happens on lower-end hardware (NUC, older laptops) with large maps.

### Increase Voxel Size

Edit [`dimos/robot/unitree/go2/blueprints/smart/unitree_go2.py`](/dimos/robot/unitree/go2/blueprints/smart/unitree_go2.py):

```python skip
# Before (high detail, slower on large maps)
voxel_mapper(voxel_size=0.05),  # 5cm voxels

# After (lower detail, 8x faster)
voxel_mapper(voxel_size=0.1),   # 10cm voxels
```

**Trade-off:**
- Larger voxels = fewer voxels = faster updates
- But slightly less detail in the map

---

## Direct Visualization from a Module

If you want to log data to Rerun directly from inside a module (e.g. for debugging or one-off visualizations), use `rerun_init` instead of calling `rr.init()` yourself. It handles colormap registration and can optionally start a gRPC server so a viewer can connect.

```python skip
import rerun as rr
from dimos.visualization.rerun.init import rerun_init

# Basic init (no gRPC server — use when RerunBridgeModule is already running)
rerun_init()
rr.log("debug/my_points", rr.Points3D(positions=[[1, 2, 3]]))

# Start a gRPC server so a viewer can connect.  `grpc_config` is required
# whenever start_grpc=True; it carries the connect URL and the server memory cap.
rerun_init(
    start_grpc=True,
    grpc_config={
        "connect_url": "rerun+http://127.0.0.1:9999/proxy",
        "server_memory_limit": "4GB",
    },
)
# Then connect with: dimos-viewer --connect rerun+http://127.0.0.1:9999/proxy
```

When a `RerunBridgeModule` is already part of your blueprint, you typically don't need `start_grpc` — just call `rerun_init()` and log directly with `rr.log()`. The data will appear in the existing viewer.

## How to use Rerun on `dev` (and the TF/entity nuances)

Rerun on `dev` is **module-driven**: modules decide what to log, and `Blueprint.build()` sets up the shared viewer + default layout.
