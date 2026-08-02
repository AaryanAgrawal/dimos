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

import numpy as np

from dimos.navigation.motion.adapter.follower import GoalLatch, path_clearance


def test_clearance_is_band_distance_minus_half_width():
    xy = np.array([[0.0, 0.0]])
    points = np.array([[1.0, 0.0, 0.2]])  # inside the z band
    clr = path_clearance(xy, points, half_width=0.155)
    assert abs(clr[0] - (1.0 - 0.155)) < 1e-6


def test_clearance_ignores_points_outside_z_band():
    xy = np.array([[0.0, 0.0]])
    points = np.array([[0.1, 0.0, 0.01], [0.1, 0.0, 1.0]])  # floor + overhang
    assert np.isinf(path_clearance(xy, points, half_width=0.155)[0])


def test_clearance_empty_path():
    assert path_clearance(np.zeros((0, 2)), np.zeros((0, 3)), 0.1).shape == (0,)


def test_goal_latch_fires_once_then_holds():
    latch = GoalLatch(tolerance=0.2)
    latch.set_goal((1.0, 0.0))
    assert not latch.arrive((0.0, 0.0))
    assert latch.arrive((0.95, 0.0))
    assert latch.reached
    assert not latch.arrive((0.95, 0.0))


def test_goal_latch_ignores_sub_tolerance_goal_moves():
    latch = GoalLatch(tolerance=0.2)
    latch.set_goal((1.0, 0.0))
    assert latch.arrive((1.0, 0.0))
    latch.set_goal((1.05, 0.0))  # replan grid snap, same goal
    assert latch.reached
    latch.set_goal((3.0, 0.0))  # a new task
    assert not latch.reached
