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


"""The body under test: what the planner plans for and the judge measures.

Shared domain, not benchmark -- the deployed adapter reads EMBODIMENTS to
configure a live robot, and both referees condition their scoring on it. Pure
geometry and gait-cost numbers, no dependency on worlds or on the sim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Embodiment:
    """The robot under test — conditions the gold oracle, the generator's
    difficulty rules, and the judge. A net trained on varied embodiments
    deploys on a new robot by being handed a new one of these.

    comfort = obstacles-we-care-about radius (preference, tunable);
    precision = local control tracking accuracy (hard floor — clearance
    below it is fiction, planning it is planning a contact).
    """

    # Moving-body envelope measured in the fitted MuJoCo sim (union of all
    # robot geometry over stand/fwd/reverse/strafe/spin/arc/crab commands,
    # yaw-aligned base frame): the swinging legs, not the 0.31 m trunk, set
    # the width. Measured 0.852 x 0.495, centre +x offset -0.009.
    tag: str = "go2"
    length: float = 0.85
    width: float = 0.50
    center_off: float = -0.01  # body center relative to the pose point
    comfort: float = 0.4
    precision: float = 0.05
    # gait cost multipliers for the SE(2) reference (and any rollout later):
    # forward = 1; strafe/reverse scale it; yaw_w prices rotation per rad.
    strafe: float = 1.8
    reverse: float = 1.5
    yaw_w: float = 0.25
    # Vertical geometry, all measured from the surface the feet stand on. This
    # is what makes an obstacle model a property of the BODY rather than of the
    # scene: no floor estimation, the base rides a known height above the
    # ground (motion/obstacles.py).
    steppable: float = 0.20  # legs negotiate obstacles below this -- at a cost (later)
    height: float = 0.45  # above this the body passes underneath; not an obstacle
    base_height: float = 0.29  # base origin above support; frame plumbing, not semantics

    @property
    def half_diag(self) -> float:
        return math.hypot(self.length, self.width) / 2.0

    def offsets(self, step: float = 0.05) -> np.ndarray:
        """Footprint sample points, dense enough that thin slats can't slip."""
        hl, hw = self.length / 2.0, self.width / 2.0
        return np.array(
            [
                (x + self.center_off, y)
                for x in np.arange(-hl, hl + step / 2.0, step)
                for y in np.arange(-hw, hw + step / 2.0, step)
            ]
        )


GO2 = Embodiment()
EMBODIMENTS = {
    "go2": GO2,
    # payload adds 8 cm in front: longer body, centre 4 cm further forward
    "go2-payload": Embodiment(tag="go2-payload", length=0.93, center_off=0.03, comfort=0.5),
    "slim": Embodiment(tag="slim", length=2.0, width=0.24, comfort=0.3),
    # cannot crab, and has no legs to step over anything with
    "diffdrive": Embodiment(tag="diffdrive", strafe=50.0, reverse=3.0, steppable=0.0),
}
