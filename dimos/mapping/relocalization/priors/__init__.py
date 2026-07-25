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

"""Pluggable relocalization priors: candidate proposers feeding the shared fine-ICP judge in relocalize.py (``refine_candidates``)."""

from typing import Annotated

from pydantic import Field

from dimos.mapping.relocalization.priors.base import (
    PriorConfigBase,
    RelocPrior,
    relocalize_with_prior,
)
from dimos.mapping.relocalization.priors.fiducial import FiducialPrior, FiducialPriorConfig
from dimos.mapping.relocalization.priors.ransac import RansacPrior, RansacPriorConfig

# Discriminated on ``type`` (kinematics/config.py:54 is the exemplar); assembled here, where both leaves are loaded, so base stays unaware of them.
PriorConfig = Annotated[
    RansacPriorConfig | FiducialPriorConfig,
    Field(discriminator="type"),
]

__all__ = [
    "FiducialPrior",
    "FiducialPriorConfig",
    "PriorConfig",
    "PriorConfigBase",
    "RansacPrior",
    "RansacPriorConfig",
    "RelocPrior",
    "relocalize_with_prior",
]
