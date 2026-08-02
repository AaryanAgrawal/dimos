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

"""Trajectory controllers: tf pose + Path in, body-frame Twist out.

The protocol is the deployment seam — on the robot the same object consumes
the live tf lookup of ``config.frame_id`` and the planner topic; here the
episode runner feeds it the simulated equivalents. The stub is a holonomic
pursuit law kept deliberately simple: the judge exists to measure controllers,
and the first controller only has to be measurable, not good.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
import math
from typing import Any, Protocol

import numpy as np

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path
from dimos.protocol.service.spec import BaseConfig


def _angle_diff(a: float, b: float) -> float:
    return math.remainder(a - b, math.tau)


class ControllerConfig(BaseConfig):
    frame_id: str = "base_link"  # the tf frame the controller treats as itself
    lookahead: float = 0.35  # carrot distance along the path (m)
    max_speed: float = 0.5  # planar clamp (m/s)
    max_yaw_rate: float = 1.4  # rad/s
    k_pos: float = 2.0  # body-frame position error gain (1/s)
    k_yaw: float = 2.0  # yaw error gain (1/s)
    # yaw-per-meter above this is a commanded rotation (fan), not a curve --
    # matches the referee's fan detection threshold (sim.py _fan_marks)
    fan_yaw_per_m: float = 3.0
    # while a fan segment is being executed, hold position and rotate until
    # the yaw error drops under this (rad)
    fan_yaw_done: float = 0.25
    # speed governor over the optional clearance annotation, mirroring the
    # planner-side AvoidanceConfig semantics: cruise at max_speed with
    # speed_clearance of room, creep at min_speed at the precision floor,
    # linear between; judged over the next speed_lookahead metres of path
    min_speed: float = 0.2
    speed_clearance: float = 0.35
    speed_floor_clearance: float = 0.05  # the embodiment precision floor
    speed_lookahead: float = 2.0


class TrajectoryController(Protocol):
    config: ControllerConfig

    def reset(self) -> None: ...

    def update(
        self, pose: PoseStamped, path: Path, t: float, clearance: np.ndarray | None = None
    ) -> Twist: ...


# name -> "module:factory"; arbitrary "module:factory" strings load too, so
# autoresearch candidates plug in without registering (planners/base pattern).
REGISTRY = {
    "pursuit": "dimos.navigation.motion.control.controller:make",
}


def load(name: str) -> Callable[..., TrajectoryController]:
    """Resolve a registry name or a dotted "module:factory" string."""
    target = REGISTRY.get(name, name)
    mod, _, attr = target.partition(":")
    factory: Any = getattr(import_module(mod), attr or "make")
    return factory  # type: ignore[no-any-return]


def make(config: ControllerConfig | None = None) -> PursuitController:
    return PursuitController(config)


class PursuitController:
    """Holonomic pursuit: project, look ahead, P-law in the body frame.

    The Go2 can crab, so position error maps straight to (vx, vy) — no
    car-style steering. Yaw tracks the path's own yaw (the planner encodes
    side-stepping and fans there), and fan segments become rotate-in-place:
    position holds the fan waypoint while yaw converges.
    """

    config: ControllerConfig

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self.reset()

    def reset(self) -> None:
        self._goal_reached = False

    def update(
        self, pose: PoseStamped, path: Path, t: float, clearance: np.ndarray | None = None
    ) -> Twist:
        cfg = self.config
        if len(path) < 2:
            # empty path or a single-pose veto stub: there is nothing to
            # follow -- hold position (the planner is saying "stop")
            return Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
        xy = np.array([[p.position.x, p.position.y] for p in path.poses])
        yaws = np.array([p.yaw for p in path.poses])
        px, py, pyaw = pose.position.x, pose.position.y, pose.yaw

        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.zeros(1)
        arcs = np.concatenate([[0.0], np.cumsum(seg)])

        # closest waypoint = progress along the path; inside a fan the
        # waypoints are coincident, so advance by yaw progress instead of
        # re-rotating from the fan's first pose
        i = int(np.argmin(np.linalg.norm(xy - (px, py), axis=1)))
        while (
            i + 1 < len(xy)
            and float(arcs[i + 1] - arcs[i]) < 1e-6
            and abs(_angle_diff(float(yaws[i + 1]), pyaw)) < abs(_angle_diff(float(yaws[i]), pyaw))
        ):
            i += 1

        # fan detection at the current position: yaw stepping with (near-)zero
        # displacement means the planner commands a rotation here
        j = min(i + 1, len(xy) - 1)
        ds = float(arcs[j] - arcs[i])
        dyaw = abs(_angle_diff(float(yaws[j]), float(yaws[i])))
        in_fan = j > i and dyaw > 1e-6 and dyaw / max(ds, 1e-6) > cfg.fan_yaw_per_m
        if in_fan and abs(_angle_diff(float(yaws[j]), pyaw)) > cfg.fan_yaw_done:
            target_xy = xy[i]
            target_yaw = float(yaws[j])
        else:
            s = float(arcs[i]) + cfg.lookahead
            k = min(int(np.searchsorted(arcs, s)), len(xy) - 1)
            target_xy = xy[k]
            target_yaw = float(yaws[k])

        # speed governor: cap cruise by the room ahead, when we know it
        vmax = cfg.max_speed
        if clearance is not None and len(clearance) == len(xy):
            ahead = clearance[(arcs >= arcs[i]) & (arcs <= arcs[i] + cfg.speed_lookahead)]
            room = float(np.min(ahead)) if len(ahead) else float(clearance[i])
            frac = (room - cfg.speed_floor_clearance) / max(
                cfg.speed_clearance - cfg.speed_floor_clearance, 1e-6
            )
            vmax = cfg.min_speed + (cfg.max_speed - cfg.min_speed) * min(max(frac, 0.0), 1.0)

        # body-frame error -> velocity
        ex, ey = target_xy[0] - px, target_xy[1] - py
        c, s_ = math.cos(-pyaw), math.sin(-pyaw)
        bx, by = c * ex - s_ * ey, s_ * ex + c * ey
        vx, vy = cfg.k_pos * bx, cfg.k_pos * by
        speed = math.hypot(vx, vy)
        if speed > vmax:
            vx, vy = vx / speed * vmax, vy / speed * vmax
        wz = float(
            np.clip(cfg.k_yaw * _angle_diff(target_yaw, pyaw), -cfg.max_yaw_rate, cfg.max_yaw_rate)
        )
        return Twist(Vector3(vx, vy, 0.0), Vector3(0.0, 0.0, wz))
