#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""3d navigation on G1 with ray tracing and MLS planning.

Click a goal in the rerun viewer: Point-LIO feeds the voxel ray tracer, the MLS
planner plans over the traced surfaces, and the path follower drives the walk
through the DDS high-level velocity interface.
"""

from datetime import datetime
import os
from pathlib import Path
from typing import Any

from dimos.constants import RECORDINGS_DIR
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.hardware.sensors.lidar.pointlio.recorder import PointlioRecorder
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.navigation.basic_path_follower.module import BasicPathFollower
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.nav_3d.mls_planner.goal_relay import GoalRelay
from dimos.navigation.nav_3d.mls_planner.mls_planner_native import MLSPlannerNative
from dimos.navigation.nav_3d.mls_planner.viz import planner_visual_override, render_path
from dimos.robot.unitree.g1.config import G1
from dimos.robot.unitree.g1.effectors.high_level.dds_sdk import G1HighLevelDdsSdk
from dimos.robot.unitree.g1.g1_estop import G1EStop
from dimos.robot.unitree.g1.g1_tf_publisher import (
    MID360_HEIGHT,
    NOMINAL_BASE_HEIGHT,
    G1TfPublisher,
)
from dimos.visualization.vis_module import vis_module

voxel_size = 0.08
# Raise above 0 to draw what the planner searched over (surface, nodes, weighted edges).
planner_viz_hz = 0.0

g1_overhead_safety_margin = 0.2
g1_overhead_clearance = G1.height_clearance + g1_overhead_safety_margin
g1_max_step_height = 0.10

# Opt-in recording: set DIMOS_NAV_RECORD=1 to capture pointlio_lidar +
# pointlio_odometry into a timestamped db that plan_rrd replays from.
_RECORD = os.getenv("DIMOS_NAV_RECORD", "").lower() in ("1", "true", "yes", "on")


def _recording_dir() -> Path:
    now = datetime.now().astimezone()
    stamp = (
        now.strftime("%Y-%m-%d") + "_" + now.strftime("%I-%M%p").lower() + "-" + now.strftime("%Z")
    )
    return RECORDINGS_DIR / stamp


def _render_global_map(msg: Any) -> Any:
    return msg.to_rerun()


def _static_robot_body(rr: Any) -> list[Any]:
    """G1-shaped wireframe box on the base frame, spanning ground to head."""
    return [
        rr.Boxes3D(
            half_sizes=[0.25, G1.width_clearance / 2, 0.66],
            centers=[[0.0, 0.0, 0.66 - NOMINAL_BASE_HEIGHT]],
            colors=[(0, 255, 127)],
            fill_mode="MajorWireframe",
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


def _g1_nav_rerun_blueprint() -> Any:
    """Single 3D world view; this stack has no camera."""
    import rerun as rr
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="world",
            name="3D",
            background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
            line_grid=rrb.LineGrid3D(
                plane=rr.components.Plane3D.XY.with_distance(0.5),
            ),
            overrides={
                "world/lidar": rrb.EntityBehavior(visible=False),
            },
        ),
        rrb.TimePanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
    )


_nav_rerun_config = {
    "blueprint": _g1_nav_rerun_blueprint,
    "tf_axes": 0.5,
    "max_hz": {
        # Rate-limited at the source by global_emit_every, roughly every 5s.
        "world/global_map": 0,
        "world/local_map": 0.5,
        "world/lidar": 1,
    },
    # Ring buffer replayed to a connecting viewer. Small so connect catches up fast.
    "memory_limit": "64MB",
    # The robot box hangs off pelvis on its own entity: a static transform
    # under world/tf would override the live one.
    "static": {
        "world/robot_body": _static_robot_body,
    },
    "visual_override": {
        "world/global_map": _render_global_map,
        "world/path": render_path,
        **planner_visual_override(planner_viz_hz),
    },
}

unitree_g1_nav_3d = autoconnect(
    vis_module(viewer_backend=global_config.viewer, rerun_config=_nav_rerun_config),
    G1HighLevelDdsSdk.blueprint(),
    PointLio.blueprint(
        host_ip=os.getenv("LIDAR_HOST_IP", "192.168.123.164"),
        lidar_ip=os.getenv("LIDAR_IP", "192.168.123.120"),
    ),
    G1TfPublisher.blueprint(),
    G1EStop.blueprint(),
    RayTracingVoxelMap.blueprint(
        voxel_size=voxel_size,
        emit_every=1,
        global_emit_every=50,
        min_health=-1,
        max_health=5,
        support_min=4,
    ),
    # global_map is remapped off so the planner runs purely on the
    # incremental local_map + region_bounds pair.
    MLSPlannerNative.blueprint(
        world_frame="odom",
        voxel_size=voxel_size,
        robot_height=g1_overhead_clearance,
        surface_closing_radius=0.3,
        wall_clearance_m=0.2,
        wall_buffer_m=0.75,
        wall_buffer_weight=100.0,
        step_threshold_m=g1_max_step_height,
        step_penalty_weight=4.0,
        viz_publish_hz=planner_viz_hz,
    ).remappings([(MLSPlannerNative, "global_map", "global_map_unused")]),
    GoalRelay.blueprint(lidar_height=MID360_HEIGHT),
    BasicPathFollower.blueprint(speed=0.4, heading_gain=0.4, max_angular=0.5),
    MovementManager.blueprint(),
).global_config(n_workers=10, robot_model="unitree_g1")

# PointLio keeps its default topics here, so point the recorder's ports at them.
# Streams are recorded under the port names regardless of the topic.
if _RECORD:
    unitree_g1_nav_3d = autoconnect(
        unitree_g1_nav_3d,
        PointlioRecorder.blueprint(db_path=str(_recording_dir() / "mem2.db")).remappings(
            [
                (PointlioRecorder, "pointlio_lidar", "lidar"),
                (PointlioRecorder, "pointlio_odometry", "odometry"),
            ]
        ),
    )
