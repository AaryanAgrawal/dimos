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

"""Replay recorded hardware twists through the actual G1 GR00T MuJoCo stack."""

from pathlib import Path

from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.mapping.voxels.module import VoxelGridMapper
from dimos.memory.module import default_recording_dir
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.robot.unitree.g1.blueprints.basic.unitree_g1_groot_wbc import unitree_g1_groot_wbc
from dimos.robot.unitree.g1.characterization.replay import TwistRecordingReplay
from dimos.robot.unitree.g1.g1_recorder import G1Recorder
from dimos.visualization.rerun.bridge import RerunBridgeModule
from dimos.visualization.rerun.websocket_server import RerunWebSocketServer
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule

_RECORDING_DIR = default_recording_dir()

unitree_g1_groot_replay = (
    autoconnect(
        unitree_g1_groot_wbc,
        TwistRecordingReplay.blueprint(recording=Path("hardware.db")),
        G1Recorder.blueprint(db_path=str(_RECORDING_DIR / "mem2.db")),
    )
    .remappings(
        [
            (G1Recorder, "sim_odom", "odom"),
            (G1Recorder, "motor_states", "g1_joints"),
        ]
    )
    .disabled_modules(
        VoxelGridMapper,
        RayTracingVoxelMap,
        CostMapper,
        ReplanningAStarPlanner,
        MovementManager,
        RerunBridgeModule,
        RerunWebSocketServer,
        WebsocketVisModule,
    )
    .global_config(n_workers=5, robot_model="unitree_g1", viewer="none")
)
