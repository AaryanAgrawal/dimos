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

"""Drive-and-record blueprint for the G1.

Keyboard teleop from the rerun viewer's websocket walks the robot while
Point-LIO odom+lidar and the RealSense color stream are recorded into a
memory db, together with tf. The sensor mount frames from g1.urdf are
published continuously onto tf, with the base_link edge tracking the waist
joints live, so they're captured in the recording.

The lidar IPs default to the G1's internal network. Run it for a timestamped
``recordings/`` folder::

    uv run python dimos/robot/unitree/g1/blueprints/basic/unitree_g1_record.py
"""

from datetime import datetime
import os
from pathlib import Path

from dimos.constants import RECORDINGS_DIR
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.robot.unitree.g1.blueprints.primitive.unitree_g1_vis import unitree_g1_vis
from dimos.robot.unitree.g1.effectors.high_level.dds_sdk import G1HighLevelDdsSdk
from dimos.robot.unitree.g1.g1_recorder import G1Recorder
from dimos.robot.unitree.g1.g1_tf_publisher import G1TfPublisher
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def _default_recording_dir() -> Path:
    # Local time, with the machine's actual zone abbreviation (not a hardcoded PST).
    now = datetime.now().astimezone()
    stamp = (
        now.strftime("%Y-%m-%d") + "_" + now.strftime("%I-%M%p").lower() + "-" + now.strftime("%Z")
    )
    return RECORDINGS_DIR / stamp


_RECORDING_DIR = _default_recording_dir()


unitree_g1_record = autoconnect(
    MovementManager.blueprint(),
    G1HighLevelDdsSdk.blueprint(),
    PointLio.blueprint(
        frame_id="world",
        host_ip=os.getenv("LIDAR_HOST_IP", "192.168.123.164"),
        lidar_ip=os.getenv("LIDAR_IP", "192.168.123.120"),
    ).remappings(
        [
            (PointLio, "lidar", "pointlio_lidar"),
            (PointLio, "odometry", "pointlio_odometry"),
        ]
    ),
    # RGB only: the color stream anchors to the d435_link frame published by
    # G1TfPublisher via the camera's own base -> optical tf subtree.
    RealSenseCamera.blueprint(
        base_frame_id="d435_link",
        enable_depth=False,
    ).remappings(
        [
            (RealSenseCamera, "camera_info", "realsense_camera_info"),
        ]
    ),
    G1Recorder.blueprint(db_path=str(_RECORDING_DIR / "mem2.db")),
    # Continuously publishes the sensor mount frames onto tf, with the
    # torso -> base_link edge tracking the waist joints from rt/lowstate.
    G1TfPublisher.blueprint(network_interface="eth0"),
    # Rerun viewer + websocket server. Viewer keyboard teleop publishes
    # tele_cmd_vel, which feeds MovementManager.
    unitree_g1_vis,
).global_config(n_workers=12, robot_model="unitree_g1")


if __name__ == "__main__":
    _RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    coordinator = ModuleCoordinator.build(unitree_g1_record)
    coordinator.loop()
