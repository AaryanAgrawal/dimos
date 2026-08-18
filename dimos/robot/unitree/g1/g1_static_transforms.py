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

"""Static mount frames for the G1's torso-mounted Mid-360 and D435.

Published continuously onto tf (see :class:`G1StaticTf`) so the mount geometry
lands in the tf stream of a recording. Offsets are the rest-pose values from
g1.urdf. base_link is the pelvis at rest.

The published tree is rooted at mid360_link so the static edges stay off the
entity the live odom -> mid360_link edge writes. The tf buffer composes either
direction.
"""

from __future__ import annotations

from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.protocol.tf.static_tf_publisher import (
    FrameSpec,
    StaticTfPublisher,
    frames_to_edge_transforms,
)

# g1.urdf rest-pose offsets: pelvis -> torso_link via the zeroed waist chain,
# then the fixed sensor mounts on torso_link.
FRAMES: list[FrameSpec] = [
    ("base_link", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("torso_link", "base_link", (-0.0039635, 0.0, 0.044), (0.0, 0.0, 0.0)),
    ("mid360_link", "torso_link", (0.0002835, 0.00003, 0.41618), (0.0, 0.04014257279586953, 0.0)),
    ("d435_link", "torso_link", (0.0576235, 0.01753, 0.42987), (0.0, 0.8307767239493009, 0.0)),
]


def mount_transforms() -> list[Transform]:
    """The mount tree as published: rooted at mid360_link."""
    edges = {t.child_frame_id: t for t in frames_to_edge_transforms(FRAMES)}
    return [-edges["mid360_link"], -edges["torso_link"], edges["d435_link"]]


class G1StaticTf(StaticTfPublisher):
    """Publishes the G1 sensor mount tree onto tf on a fixed interval."""

    def transforms(self) -> list[Transform]:
        return mount_transforms()
