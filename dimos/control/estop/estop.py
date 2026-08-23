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

"""Rust latched e-stop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dimos.core.native_module import NativeModule, NativeModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.std_msgs.Bool import Bool


class EStopConfig(NativeModuleConfig):
    cwd: str | None = "rust"
    executable: str = "result/bin/estop"
    build_command: str | None = "nix build -L path:."
    stdin_config: bool = True

    # Tilt off gravity that reads as a fall. A standing G1 measures ~2 deg; 45 clears gait.
    max_tilt_deg: float = 45.0


class EStop(NativeModule):
    """Latches estop true when the IMU tilts past max_tilt_deg or trigger arrives true."""

    config: EStopConfig

    imu: In[Imu]
    trigger: In[Bool]
    estop: Out[Bool]


if TYPE_CHECKING:
    EStop()
