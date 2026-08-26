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

"""Adjustable teleop preserves asymmetric maxima and bounded ramps."""

import pygame
import pytest

from dimos.robot.unitree.keyboard_teleop import (
    _approach,
    _bounded_fraction,
    _directional_twist,
)


@pytest.mark.parametrize(
    ("key", "component", "expected"),
    [
        (pygame.K_w, "vx", 0.48),
        (pygame.K_s, "vx", -0.225),
        (pygame.K_q, "vy", 0.215),
        (pygame.K_e, "vy", -0.275),
        (pygame.K_a, "wz", 0.97),
        (pygame.K_d, "wz", -0.605),
    ],
)
def test_directional_setpoint(key: int, component: str, expected: float) -> None:
    twist = _directional_twist(
        {key},
        forward_speed_m_s=0.96,
        backward_speed_m_s=0.45,
        left_speed_m_s=0.43,
        right_speed_m_s=0.55,
        ccw_speed_rad_s=1.94,
        cw_speed_rad_s=1.21,
        fraction=0.5,
        multiplier=1.0,
    )
    actual = {"vx": twist.linear.x, "vy": twist.linear.y, "wz": twist.angular.z}[component]
    assert actual == pytest.approx(expected)


def test_speed_fraction_is_bounded() -> None:
    assert _bounded_fraction(0.5, 2.0) == 1.0
    assert _bounded_fraction(0.5, -2.0) == 0.0
    assert _bounded_fraction(1.0, 0.5, 2.0) == 1.5
    assert _bounded_fraction(1.5, 2.0, 2.0) == 2.0


def test_approach_never_overshoots() -> None:
    assert _approach(0.0, 1.0, 0.2) == 0.2
    assert _approach(0.9, 1.0, 0.2) == 1.0
    assert _approach(0.0, -1.0, 0.2) == -0.2
