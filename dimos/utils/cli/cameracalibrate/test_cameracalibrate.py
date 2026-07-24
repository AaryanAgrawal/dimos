# Copyright 2025-2026 Dimensional Inc.
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

from pathlib import Path

import numpy as np

from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo as DimosCameraInfo
from dimos.utils.cli.cameracalibrate.cameracalibrate import write_camera_info_yaml


def test_write_camera_info_yaml_round_trips_through_dimos_camera_info(tmp_path: Path) -> None:
    """K/D/size/model written by the CLI load back bit-identical via CameraInfo.from_yaml."""
    K = np.array([[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.1, 0.05, 0.0, 0.0, 0.0])
    path = str(tmp_path / "camera_info.yaml")
    write_camera_info_yaml(
        path,
        image_width=640,
        image_height=480,
        camera_name="test_cam",
        K=K,
        D=D,
        distortion_model="plumb_bob",
    )
    info = DimosCameraInfo.from_yaml(path)
    assert info.width == 640
    assert info.height == 480
    assert info.distortion_model == "plumb_bob"
    assert np.allclose(info.get_K_matrix(), K)
    assert np.allclose(info.get_D_coeffs(), D)
