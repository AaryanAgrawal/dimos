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

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dimos.mapping.relocalization.fiducial.module import FiducialRelocalization


def pose(xyz: list[float], yaw_deg: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T


def sighting(marker_id: int, world_T_marker: np.ndarray, score: float) -> Any:
    """What the detector's aggregated_detections carries: id, world pose, score."""
    q = Rotation.from_matrix(world_T_marker[:3, :3]).as_quat()
    x, y, z = world_T_marker[:3, 3]
    center = SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=z),
        orientation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3]),
    )
    return SimpleNamespace(
        id=str(marker_id),
        bbox=SimpleNamespace(center=center),
        results_length=1,
        results=[SimpleNamespace(hypothesis=SimpleNamespace(score=score))],
    )


# Tag 7 was surveyed 3 m +x of the map origin, facing +y. Live, the robot's
# world frame sees the same tag at (1, 2) facing +x, so world and map disagree
# by exactly world_T_marker @ inv(map_T_marker).
MAP_T_MARKER = pose([3.0, -1.0, 0.5], 90.0)
WORLD_T_MARKER = pose([1.0, 2.0, 0.5], 0.0)


@pytest.fixture
def module() -> Iterator[FiducialRelocalization]:
    m = FiducialRelocalization()
    m._map_T_marker = {7: MAP_T_MARKER}
    yield m
    m.dispose()


def test_fix_composes_world_to_map_from_one_sighting(module: FiducialRelocalization) -> None:
    """A surveyed tag's live pose against its map pose is the whole fix."""
    tf = module._fix(sighting(7, WORLD_T_MARKER, score=0.9))
    assert tf is not None
    assert (tf.frame_id, tf.child_frame_id) == ("world", "map")
    np.testing.assert_allclose(
        tf.to_matrix(), WORLD_T_MARKER @ np.linalg.inv(MAP_T_MARKER), atol=1e-9
    )


def test_fix_refuses_unsurveyed_tags_and_low_scores(module: FiducialRelocalization) -> None:
    """The relocalizer holds the accept decision: unknown tag or below its own bar is no fix."""
    assert module._fix(sighting(8, WORLD_T_MARKER, score=0.9)) is None
    module.config.min_score = 0.95
    assert module._fix(sighting(7, WORLD_T_MARKER, score=0.9)) is None


def test_relocalize_once_takes_the_first_sighting_only(module: FiducialRelocalization) -> None:
    """Two tags in one burst: the first places the map, the second is not re-earned."""
    got: list[Any] = []
    module._world_to_map.subscribe(got.append)
    module._map_T_marker[9] = pose([0.0, 0.0, 0.0], 0.0)
    burst: Any = SimpleNamespace(
        detections=[sighting(7, WORLD_T_MARKER, 0.9), sighting(9, WORLD_T_MARKER, 0.9)],
        detections_length=2,
    )
    module._on_aggregated(burst)
    assert len(got) == 1 and module.placed
