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

from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
from reactivex import Subject
from scipy.spatial.transform import Rotation

from dimos.core.stream import In
from dimos.mapping.relocalization.lidar.module import LidarRelocalization
from dimos.mapping.relocalization.module import Config, Fix, RelocalizationModule
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
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


def test_accept_inverts_the_fix():
    """A Fix maps world points into the map; the TF tree wants the frame transform."""
    T = np.eye(4)
    T[:3, 3] = [3.0, -1.0, 0.0]  # the map sits 3 m +x of where the robot thought it was
    m = RelocalizationModule.__new__(RelocalizationModule)
    m._world_to_map = Subject()
    m.config = Config()
    got = []
    m._world_to_map.subscribe(got.append)

    m.accept(Fix(transform=T, fitness=0.9), "test")
    assert (got[0].frame_id, got[0].child_frame_id) == ("world", "map")
    np.testing.assert_allclose(got[0].to_matrix(), np.linalg.inv(T), atol=1e-9)


def test_premap_defines_the_map_frame_and_waits_for_a_fix(tmp_path):
    """Loading is the base's: every strategy reads a premap and publishes it, once placed."""
    path = tmp_path / "somewhere.pc2.lcm"
    path.write_bytes(
        PointCloud2.from_numpy(np.zeros((5, 3), dtype=np.float32), timestamp=0.0).lcm_encode()
    )
    m = RelocalizationModule.__new__(RelocalizationModule)
    m._world_to_map = Subject()
    m.config = Config(publish_loaded_map=True)
    published, disposables = [], []
    m.loaded_map = SimpleNamespace(publish=published.append)
    m.register_disposable = disposables.append

    m._load_premap(str(path))
    assert m.premap is not None and len(m.premap) == 5
    assert m.premap.frame_id == "map"
    assert len(disposables) == 1  # the gated republish
    assert published == []  # ... which stays silent until a fix lands
    disposables[0].dispose()  # rx.interval runs on a thread


def test_relocalizer_refuses_below_its_own_threshold(monkeypatch):
    """One config surface: the relocalizer holds the knobs and the accept decision."""
    from dimos.mapping.relocalization.lidar import relocalize as lidar

    fix = Fix(transform=np.eye(4), fitness=0.4, rmse=0.1, margin=0.0)
    monkeypatch.setattr(lidar.LidarRelocalizer, "_prepare", lambda self, cloud: None)
    monkeypatch.setattr(lidar.LidarRelocalizer, "align", lambda self, cloud: fix)

    def relocalizer(threshold):
        return lidar.LidarRelocalizer(None, lidar.RelocalizeConfig(fitness_threshold=threshold))

    assert relocalizer(0.5).relocalize(None) is None
    assert relocalizer(0.3).relocalize(None) is fix


def test_presets_are_named_and_default_matches():
    """A config is named after the rig it was measured on, not left universal."""
    from dimos.mapping.relocalization.lidar import relocalize as lidar

    assert lidar.DEFAULT_PRESET in lidar.PRESETS
    assert lidar.PRESETS["mid360"] == lidar.RelocalizeConfig()


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
