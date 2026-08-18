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

"""Records the G1 into a memory SQLite db.

Captures Point-LIO odom + lidar (trajectory baked into ``pointlio_lidar`` via
the inherited ``@pose_setter_for``) plus the RealSense color and depth
streams. Camera frames are anchored via the mount frames the G1 tf publisher
and the camera module put on tf.
"""

from __future__ import annotations

from dimos.core.stream import In
from dimos.hardware.sensors.lidar.pointlio.recorder import PointlioRecorder
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image


class G1Recorder(PointlioRecorder):
    color_image: In[Image]
    realsense_depth_image: In[Image]
    realsense_camera_info: In[CameraInfo]
    realsense_depth_camera_info: In[CameraInfo]
