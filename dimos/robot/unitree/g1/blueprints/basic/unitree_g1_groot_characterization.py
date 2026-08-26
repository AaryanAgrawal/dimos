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

"""Operator-driven GR00T characterization with synchronized G1 recording."""

from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.memory.module import default_recording_dir
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.robot.unitree.g1.blueprints.basic.unitree_g1_groot_wbc import (
    unitree_g1_groot_wbc,
)
from dimos.robot.unitree.g1.g1_recorder import G1Recorder
from dimos.visualization.rerun.bridge import RerunBridgeModule
from dimos.visualization.rerun.websocket_server import RerunWebSocketServer
from dimos.web.cockpit import Teleop, cockpit
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule

_RECORDING_DIR = default_recording_dir()

# Full-stick Run-mode speeds measured from Point-LIO on this G1, 2026-08-26.
_FORWARD_MAX_M_S = 0.96
_BACKWARD_MAX_M_S = 0.45
_LEFT_MAX_M_S = 0.43
_RIGHT_MAX_M_S = 0.55
_CCW_MAX_RAD_S = 1.94
_CW_MAX_RAD_S = 1.21

unitree_g1_groot_characterization = (
    autoconnect(
        unitree_g1_groot_wbc,
        cockpit(
            layout=Teleop(
                forward_m_s=_FORWARD_MAX_M_S,
                backward_m_s=_BACKWARD_MAX_M_S,
                left_m_s=_LEFT_MAX_M_S,
                right_m_s=_RIGHT_MAX_M_S,
                ccw_rad_s=_CCW_MAX_RAD_S,
                cw_rad_s=_CW_MAX_RAD_S,
                speed_fraction=0.10,
                speed_fraction_step=0.05,
                max_speed_fraction=3.0,
                boost=1.0,
                publish_hz=20.0,
                watchdog_ms=300.0,
            )
        ),
        G1Recorder.blueprint(db_path=str(_RECORDING_DIR / "mem2.db")),
    )
    .remappings(
        [
            (G1Recorder, "pointlio_odometry", "odometry"),
            (G1Recorder, "pointlio_lidar", "lidar"),
            (G1Recorder, "sim_odom", "odom"),
            (G1Recorder, "sim_pointcloud", "pointcloud"),
        ]
    )
    .disabled_modules(
        RayTracingVoxelMap,
        CostMapper,
        ReplanningAStarPlanner,
        RerunBridgeModule,
        RerunWebSocketServer,
        WebsocketVisModule,
    )
    .global_config(n_workers=7, robot_model="unitree_g1", viewer="none")
)
