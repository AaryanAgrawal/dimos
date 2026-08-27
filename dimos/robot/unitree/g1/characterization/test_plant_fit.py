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

"""The fit gate requires held-out, transient, and stability evidence together."""

import numpy as np

from dimos.robot.unitree.g1.characterization.plant_fit import NormalizedPlantResidual, _accepted


def _residual(value: float) -> NormalizedPlantResidual:
    return NormalizedPlantResidual(value, {"channel": value}, {"direction": value})


def test_acceptance_requires_every_gate() -> None:
    transient = np.asarray([0.10, 0.12, 0.14, 0.16, 0.18, 0.19])

    assert _accepted(_residual(0.9), _residual(0.9), transient, (0.72, 0.1, 0.1)) is True
    assert _accepted(_residual(0.9), _residual(1.0), transient, (0.72, 0.1, 0.1)) is False
    assert _accepted(_residual(0.9), _residual(0.9), transient + 0.05, (0.72, 0.1, 0.1)) is False
    assert _accepted(_residual(0.9), _residual(0.9), transient, (0.64, 0.1, 0.1)) is False
