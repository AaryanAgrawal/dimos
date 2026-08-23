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

from __future__ import annotations

from dimos.control.estop.estop import EStop, EStopConfig
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.std_msgs.Bool import Bool


def test_blueprint_needs_no_arguments() -> None:
    """E-stop drops into any blueprint as one bare call: ports wired, threshold defaulted."""
    atom = EStop.blueprint().blueprints[0]
    assert [(s.name, s.type, s.direction) for s in atom.streams] == [
        ("imu", Imu, "in"),
        ("trigger", Bool, "in"),
        ("estop", Bool, "out"),
    ]
    assert EStopConfig().max_tilt_deg == 45.0
