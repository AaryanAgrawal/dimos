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

"""The rust controller is a PORT, so the python is its oracle.

Every branch of the law gets sampled: straight runs, S-curves, fan clusters
of coincident waypoints, degenerate one/zero-pose paths, clearance annotated
and blind, poses on/off/behind the path, yaws swept across +-pi. Agreement is
asserted per component at 1e-9, but the observed spread is exactly zero
(`test_parity_headroom` pins that separately): the two run the same operations
in the same order against the same libm, so they agree bit for bit, and any
drift off zero is a real divergence rather than accumulated noise.
"""

import math

import numpy as np
import pytest

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path
from dimos.navigation.motion.control.controller import (
    ControllerConfig,
    PursuitController,
    make_rust,
)

TOL = 1e-9
CASES = 240


def _pose(x: float, y: float, yaw: float = 0.0) -> PoseStamped:
    return PoseStamped(
        frame_id="world",
        position=Vector3(x, y, 0.0),
        orientation=Quaternion.from_euler(Vector3(0, 0, yaw)),
    )


def _path(states: list[tuple[float, float, float]]) -> Path:
    return Path(frame_id="world", poses=[_pose(*s) for s in states])


def _straight(rng: np.random.Generator) -> list[tuple[float, float, float]]:
    n = int(rng.integers(2, 60))
    step = float(rng.uniform(0.02, 0.4))
    yaw = float(rng.uniform(-math.pi, math.pi))
    x0, y0 = rng.uniform(-3.0, 3.0, 2)
    return [(x0 + k * step * math.cos(yaw), y0 + k * step * math.sin(yaw), yaw) for k in range(n)]


def _s_curve(rng: np.random.Generator) -> list[tuple[float, float, float]]:
    n = int(rng.integers(4, 80))
    amp, freq = float(rng.uniform(0.2, 1.5)), float(rng.uniform(0.3, 2.0))
    step = float(rng.uniform(0.03, 0.25))
    out = []
    for k in range(n):
        x = k * step
        y = amp * math.sin(freq * x)
        out.append((x, y, math.atan2(amp * freq * math.cos(freq * x), 1.0)))
    return out


def _with_fans(rng: np.random.Generator) -> list[tuple[float, float, float]]:
    """An S-curve with coincident-waypoint rotations spliced in -- the branch
    that exercises both fan detection and the yaw-progress advance."""
    base = _s_curve(rng)
    out: list[tuple[float, float, float]] = []
    for k, (x, y, yaw) in enumerate(base):
        out.append((x, y, yaw))
        if k and k % int(rng.integers(3, 9)) == 0:
            turn = float(rng.uniform(-math.pi, math.pi))
            steps = int(rng.integers(2, 7))
            for m in range(1, steps + 1):
                # exactly coincident: zero arc, pure yaw step
                out.append((x, y, yaw + turn * m / steps))
    return out


def _degenerate(rng: np.random.Generator) -> list[tuple[float, float, float]]:
    n = int(rng.integers(0, 3))  # 0, 1, or 2 poses
    return [
        (
            float(rng.uniform(-2, 2)),
            float(rng.uniform(-2, 2)),
            float(rng.uniform(-math.pi, math.pi)),
        )
        for _ in range(n)
    ]


GENERATORS = (_straight, _s_curve, _with_fans, _with_fans, _degenerate)


def _cases(seed: int = 20260802, n: int = CASES):  # type: ignore[no-untyped-def]
    """(config, pose, path, clearance) tuples covering every branch."""
    rng = np.random.default_rng(seed)
    for k in range(n):
        states = GENERATORS[k % len(GENERATORS)](rng)
        path = _path(states)
        if states:
            # on the path, off it, and behind its start
            anchor = states[int(rng.integers(0, len(states)))]
            mode = k % 3
            off = (0.0, 0.0) if mode == 0 else tuple(rng.uniform(-1.5, 1.5, 2))
            base = states[0] if mode == 2 else anchor
            px, py = base[0] + off[0] - (2.0 if mode == 2 else 0.0), base[1] + off[1]
        else:
            px, py = rng.uniform(-2, 2, 2)
        pose = _pose(float(px), float(py), float(rng.uniform(-math.pi, math.pi)))

        clearance = None
        if k % 4:
            clearance = rng.uniform(0.0, 0.8, len(states))
            if k % 8 == 1:  # a wrong-length annotation must be ignored by both
                clearance = clearance[:-1]
        cfg = ControllerConfig()
        if k % 5 == 0:  # non-default gains, so the params tuple order is load-bearing
            cfg = ControllerConfig(
                lookahead=float(rng.uniform(0.1, 1.2)),
                max_speed=float(rng.uniform(0.2, 1.5)),
                max_yaw_rate=float(rng.uniform(0.4, 3.0)),
                k_pos=float(rng.uniform(0.5, 4.0)),
                k_yaw=float(rng.uniform(0.5, 4.0)),
                fan_yaw_per_m=float(rng.uniform(1.0, 6.0)),
                fan_yaw_done=float(rng.uniform(0.05, 0.6)),
                min_speed=float(rng.uniform(0.05, 0.3)),
                speed_clearance=float(rng.uniform(0.2, 0.8)),
                speed_floor_clearance=float(rng.uniform(0.01, 0.15)),
                speed_lookahead=float(rng.uniform(0.5, 4.0)),
            )
        yield cfg, pose, path, clearance


def _twists(cfg, pose, path, clearance):  # type: ignore[no-untyped-def]
    py = PursuitController(cfg).update(pose, path, 0.0, clearance)
    rs = make_rust(cfg).update(pose, path, 0.0, clearance)
    return (
        (py.linear.x, py.linear.y, py.angular.z),
        (rs.linear.x, rs.linear.y, rs.angular.z),
    )


def test_rust_matches_python() -> None:
    seen = {"fan": 0, "governed": 0, "clamped": 0, "held": 0}
    for k, case in enumerate(_cases()):
        a, b = _twists(*case)
        for c, (x, y) in enumerate(zip(a, b, strict=True)):
            assert abs(x - y) <= TOL, f"case {k} component {c}: python {x!r} vs rust {y!r}"
        cfg = case[0]
        if a == (0.0, 0.0, 0.0):
            seen["held"] += 1
        if abs(a[2]) >= cfg.max_yaw_rate - 1e-12 or math.hypot(a[0], a[1]) >= cfg.max_speed - 1e-12:
            seen["clamped"] += 1
        if case[3] is not None and len(case[3]) == len(case[2]):
            seen["governed"] += 1
        if len(case[2]) > 2 and math.hypot(a[0], a[1]) < 1e-3 and abs(a[2]) > 1e-3:
            seen["fan"] += 1
    # the sweep is only worth its tolerance if it actually reached the branches
    assert all(v > 0 for v in seen.values()), f"unexercised branches: {seen}"


def test_parity_headroom() -> None:
    """Report the real spread: it must sit far under the asserted tolerance."""
    worst = 0.0
    for case in _cases():
        a, b = _twists(*case)
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b, strict=True)))
    assert worst <= TOL, f"max component diff {worst:.3e}"
    # libm hypot/sin/cos are shared between the two, so the only expected
    # spread is zero; a non-zero worst here is a real divergence to explain
    assert worst == 0.0, f"unexpected non-zero divergence {worst:.3e}"


@pytest.mark.parametrize(
    "yaw", [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi, math.pi - 1e-12]
)
def test_wrap_boundaries(yaw: float) -> None:
    """+-pi is where a `%`-based wrap diverges from IEEE remainder."""
    path = _path([(0.0, 0.0, math.pi), (0.0, 0.0, -math.pi + 0.2), (1.0, 0.0, -math.pi + 0.2)])
    a, b = _twists(ControllerConfig(), _pose(0.0, 0.0, yaw), path, None)
    assert a == b, f"yaw={yaw!r}: python {a} vs rust {b}"


def test_registry_and_build_hint() -> None:
    from dimos.navigation.motion.control.controller import REGISTRY, load

    assert REGISTRY["pursuit-rs"].endswith(":make_rust")
    assert load("pursuit-rs") is make_rust
    assert isinstance(load("pursuit-rs")().config, ControllerConfig)


def test_corridor_reaches_goal_rust() -> None:
    """Closed-loop smoke: the extension drives a real episode to the goal."""
    from dimos.navigation.motion.control.episode import EpisodeConfig, run_episode
    from dimos.navigation.motion.planner.autoresearch.scenarios import SCENARIOS
    from dimos.navigation.motion.simulation.policy import FreePolicy
    from dimos.utils.data import get_data

    policy = FreePolicy.load(get_data("ml-trajectory-research/freewalk_mcf.bin"))
    sc = next(s for s in SCENARIOS if s.name == "corridor")
    result = run_episode(sc, make_rust(), policy, EpisodeConfig(replan_hz=0.0))
    assert result.outcome == "goal"
    assert not result.contact.any()
