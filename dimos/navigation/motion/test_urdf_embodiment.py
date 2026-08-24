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

"""The box a URDF describes is the box that comes back, and drift buckets snap to the lattice."""

from __future__ import annotations

from pathlib import Path

import pytest

from dimos.navigation.motion.urdf_embodiment import (
    DRIFT_DEG,
    _drift_bucket,
    embodiment_from_urdf,
    envelope_rows,
)

# One box, 0.4 x 0.2 x 0.1, centred at (0.05, 0, -0.3): every extracted scalar is known by hand.
#   x -0.15..0.25 -> length 0.400, centre +0.05
#   y -0.10..0.10 -> width  0.200
#   z -0.35..-0.25 -> height 0.100, and the lowest geometry is 0.35 below the base origin
_URDF = """<?xml version="1.0"?>
<robot name="probe">
  <link name="base_link">
    <collision>
      <origin xyz="0.05 0 -0.3" rpy="0 0 0"/>
      <geometry><box size="0.4 0.2 0.1"/></geometry>
    </collision>
  </link>
</robot>
"""


@pytest.fixture()
def urdf(tmp_path: Path) -> Path:
    path = tmp_path / "probe.urdf"
    path.write_text(_URDF)
    return path


def test_scalars_match_the_box_the_urdf_declares(urdf: Path) -> None:
    """Every scalar is read off the collision box, so a known box gives known numbers."""
    emb = embodiment_from_urdf(urdf, tag="probe")
    assert emb.tag == "probe"
    assert emb.length == pytest.approx(0.4, abs=1e-9)
    assert emb.width == pytest.approx(0.2, abs=1e-9)
    assert emb.height == pytest.approx(0.1, abs=1e-9)
    assert emb.center_off == pytest.approx(0.05, abs=1e-9)
    # The feet are the lowest geometry, so the base rides this far above the support plane.
    assert emb.base_height == pytest.approx(0.35, abs=1e-9)


def test_envelope_is_left_empty(urdf: Path) -> None:
    """A URDF cannot know where a gait swings the legs, so the swept table stays unmeasured."""
    assert embodiment_from_urdf(urdf, tag="probe").envelope == ()


def test_drift_buckets_snap_to_the_lattice() -> None:
    """A command is filed under the nearest lattice angle; a stand-still is filed nowhere."""
    assert _drift_bucket(0.15, 0.0) == 0.0
    assert _drift_bucket(0.0, 0.15) == 90.0
    assert _drift_bucket(-0.15, 0.0) == 180.0
    assert _drift_bucket(0.106, 0.106) == 45.0
    # Sign of the lateral command does not change the bucket: rows are mirrored at lookup.
    assert _drift_bucket(0.106, -0.106) == 45.0
    assert _drift_bucket(0.0, 0.0) is None
    assert set(DRIFT_DEG) >= {0.0, 45.0, 90.0, 135.0, 180.0}


def test_envelope_rows_union_each_bucket(urdf: Path) -> None:
    """Rows are the union over every sample filed under that drift angle."""
    samples = [(0.15, 0.0, {}), (0.15, 0.0, {}), (0.0, 0.15, {}), (0.0, 0.0, {})]
    rows = envelope_rows(urdf, samples)
    assert [r[0] for r in rows] == [0.0, 90.0]  # the stand-still contributed nothing
    for _deg, length, width, off_x, _off_y in rows:
        assert length == pytest.approx(0.4, abs=1e-9)
        assert width == pytest.approx(0.2, abs=1e-9)
        assert off_x == pytest.approx(0.05, abs=1e-9)
