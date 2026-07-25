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

from typing import Protocol

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
from pydantic import Field

from dimos.mapping.relocalization.relocalize import refine_candidates
from dimos.protocol.service.spec import BaseConfig

# One pydantic config per prior, keyed by a Literal ``type`` into a discriminated union. Pattern from dimos/manipulation/planning/kinematics/config.py:26-57.


class PriorConfigBase(BaseConfig):
    """Fields every prior shares: the on/off toggle plus its accept bar."""

    enabled: bool = True
    # Per-prior accept gate: min wall fitness (dimensionless, 0-1) this prior's fix must clear. 0.6 because the trial's office survey produced sub-0.6 fixes that were meters off while still scoring as "fit".
    fitness_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class RelocPrior(Protocol):
    """A relocalization candidate proposer; the module owns the trigger. A prior must not self-select a winner (``refine_candidates``'s job); zero candidates is a valid response."""

    name: str

    def propose(
        self,
        global_map: o3d.geometry.PointCloud,
        local_map: o3d.geometry.PointCloud,
    ) -> list[np.ndarray]: ...


def relocalize_with_prior(
    global_map: o3d.geometry.PointCloud,
    local_map: o3d.geometry.PointCloud,
    prior: RelocPrior,
) -> tuple[np.ndarray, float] | None:
    """Judge this prior's candidates through the shared fine-ICP tail; ``None`` when it proposed none."""
    transforms = prior.propose(global_map, local_map)
    if not transforms:
        return None
    return refine_candidates(global_map, local_map, transforms)
