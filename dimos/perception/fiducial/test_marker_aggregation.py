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

from typing import Any

import numpy as np

from dimos.perception.fiducial.marker_aggregation import (
    AggregationConfig,
    Glimpse,
    TagAggregator,
    gate_reason,
    noise_scale,
    robust_pose,
    tag_side_px,
)

TRUE = (1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0)
CFG = AggregationConfig()


def glimpse(ts: float = 0.0, pose: Any = TRUE, **quality: float) -> Glimpse:
    return Glimpse(ts=ts, marker_id=7, pose=tuple(pose), **quality)


def test_one_gross_outlier_does_not_move_the_huber_estimate() -> None:
    """A plain mean of 20 good glimpses plus one 9 m outlier moves ~0.4 m; Huber IRLS stays at the truth."""
    rng = np.random.default_rng(0)
    cluster = [glimpse(pose=np.add(TRUE, [*rng.normal(0, 0.01, 3), 0, 0, 0, 0])) for _ in range(20)]
    cluster.append(glimpse(pose=(6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 1.0)))
    x, y, z, *q = robust_pose(cluster, CFG)
    np.testing.assert_allclose((x, y, z), TRUE[:3], atol=0.02)
    assert abs(q[3]) > 0.999


def test_each_gate_fires_on_its_own_field_and_a_missing_field_skips_it() -> None:
    assert gate_reason(glimpse(reproj_px=3.0), CFG) == "reproj"
    assert gate_reason(glimpse(tag_px=10.0), CFG) == "small"
    assert gate_reason(glimpse(view_angle_deg=60.0), CFG) == "oblique"
    assert gate_reason(glimpse(), CFG) is None


def test_a_fused_pose_needs_min_observations_inside_the_window() -> None:
    agg = TagAggregator(lambda _msg: None, CFG)
    for ts in (0.0, 1.0):
        assert agg.observe(glimpse(ts=ts)) is None
    assert agg.fuse(7) is None
    agg.observe(glimpse(ts=2.0))
    fused = agg.fuse(7)
    assert fused is not None and fused[2] == 3
    np.testing.assert_allclose(fused[0], TRUE, atol=1e-9)
    agg.observe(glimpse(ts=2.0 + CFG.time_window_s + 1.0))  # purges the three above
    assert agg.fuse(7) is None


def test_score_is_one_at_reference_and_falls_with_range_and_blur() -> None:
    assert noise_scale(0.4, 1.0) == 1.0
    assert noise_scale(0.8, 1.0) == 4.0
    assert noise_scale(0.4, 2.0) == 4.0
    assert noise_scale(None, None) == 1.0
    agg = TagAggregator(lambda _msg: None, CFG)
    for ts in (0.0, 1.0, 2.0):
        agg.observe(glimpse(ts=ts, distance_m=0.8, reproj_px=1.0))
    fused = agg.fuse(7)
    assert fused is not None and fused[1] == 0.25


def test_tag_side_is_the_root_of_the_quad_area() -> None:
    assert tag_side_px(np.array([[0, 0], [10, 0], [10, 10], [0, 10]])) == 10.0
