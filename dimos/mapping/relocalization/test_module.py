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

from typing import get_type_hints

import numpy as np
import pytest
from reactivex import Subject
from scipy.spatial.transform import Rotation

from dimos.core.stream import In
from dimos.mapping.relocalization.lidar.module import LidarRelocalization, _yaw_only
from dimos.mapping.relocalization.module import Config, RelocalizationModule
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray


def test_submit_publishes_and_checks_frames():
    """submit no longer second-guesses fitness; the implementation already decided."""
    m = RelocalizationModule.__new__(RelocalizationModule)  # no Module.__init__: no threads
    m._world_to_map = Subject()
    m.config = Config()
    got = []
    m._world_to_map.subscribe(got.append)
    tf = Transform.from_matrix(np.eye(4), frame_id="world", child_frame_id="map")
    m.submit(tf, 0.3, "x")
    m.submit(tf, 0.9, "x")
    assert got == [tf, tf]
    with pytest.raises(AssertionError):
        m.submit(Transform.from_matrix(np.eye(4), frame_id="map", child_frame_id="world"), 1.0)


def test_relocalize_refuses_below_the_maps_own_threshold(monkeypatch):
    """One config surface: the prepared map carries the knobs and the accept decision."""
    from dimos.mapping.relocalization.lidar import module as lidar

    fix = lidar.Fix(transform=np.eye(4), fitness=0.4, rmse=0.1, margin=0.0)
    monkeypatch.setattr(lidar, "align", lambda *a, **k: fix)

    def premap(threshold):
        return lidar.PreparedMap(
            cloud=None,
            coarse=None,
            fpfh=None,
            fine=None,
            config=lidar.RelocalizeConfig(fitness_threshold=threshold),
        )

    assert lidar.relocalize(premap(0.5), None) is None
    assert lidar.relocalize(premap(0.3), None) is fix


def test_from_matrix_inverse_matches_linalg_inv():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.2, 1.3]).as_matrix()
    T[:3, 3] = [1.5, -2.0, 0.3]
    tf = Transform.from_matrix(T, frame_id="map", child_frame_id="world").inverse()
    assert (tf.frame_id, tf.child_frame_id) == ("world", "map")
    np.testing.assert_allclose(tf.to_matrix(), np.linalg.inv(T), atol=1e-9)


def test_dual_strategy_merges_ports():
    class FakeImpl(RelocalizationModule):
        detections: In[Detection3DArray]

    class Dual(LidarRelocalization, FakeImpl):
        pass

    hints = get_type_hints(Dual)
    assert {"tf", "global_map", "loaded_map", "detections"} <= hints.keys()
    assert Dual.blueprint() is not None


def test_yaw_only_keeps_yaw_and_translation_drops_tilt():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [0.12, -0.09, 0.7]).as_matrix()
    T[:3, 3] = [3.0, -4.0, 0.5]
    flat = _yaw_only(T)

    np.testing.assert_allclose(flat[:3, 3], T[:3, 3])
    # Gravity is untouched: the flattened z-axis is exactly vertical.
    np.testing.assert_allclose(flat[:3, 2], [0, 0, 1], atol=1e-12)
    assert Rotation.from_matrix(flat[:3, :3]).as_euler("xyz")[2] == pytest.approx(0.7, abs=0.02)


def test_yaw_only_is_identity_on_a_pure_yaw():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 1.1).as_matrix()
    T[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(_yaw_only(T), T, atol=1e-12)


def test_yaw_only_pivot_keeps_the_cloud_where_the_hypothesis_put_it():
    """Flattening about the origin moves a distant cloud metres; about its centre, not at all."""
    pts = np.array([[-52.0, -41.0, 4.0], [-55.0, -38.0, 2.0], [-50.0, -44.0, 3.0]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [np.radians(2.0), 0.0, 0.3]).as_matrix()
    T[:3, 3] = [1.0, -2.0, 0.1]
    moved = pts @ T[:3, :3].T + T[:3, 3]

    centre = pts.mean(axis=0)
    pivoted = _yaw_only(T, pivot=centre)
    # The cloud's centre lands exactly where the unflattened transform put it.
    np.testing.assert_allclose(
        pivoted[:3, :3] @ centre + pivoted[:3, 3], T[:3, :3] @ centre + T[:3, 3], atol=1e-9
    )
    # Gravity is still removed.
    np.testing.assert_allclose(pivoted[:3, 2], [0, 0, 1], atol=1e-12)

    at_origin = _yaw_only(T)
    origin_shift = np.linalg.norm(
        (pts @ at_origin[:3, :3].T + at_origin[:3, 3]) - moved, axis=1
    ).mean()
    pivot_shift = np.linalg.norm((pts @ pivoted[:3, :3].T + pivoted[:3, 3]) - moved, axis=1).mean()
    assert origin_shift > 1.0, origin_shift
    assert pivot_shift < 0.2, pivot_shift
