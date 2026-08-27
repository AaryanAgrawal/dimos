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

"""Tests for the G1 MuJoCo whole-body adapter."""

from dimos.hardware.whole_body.spec import WholeBodyAdapter
from dimos.simulation.adapters.whole_body.g1 import SimMujocoG1WholeBodyAdapter


def test_g1_sim_adapter_satisfies_whole_body_contract() -> None:
    """The coordinator must accept the simulated adapter as whole-body hardware."""
    adapter = SimMujocoG1WholeBodyAdapter(address="unused.xml")

    assert isinstance(adapter, WholeBodyAdapter)
    assert adapter.get_limits() is None
