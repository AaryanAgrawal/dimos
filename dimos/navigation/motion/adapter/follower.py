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

"""TrajectoryFollower: the motion controller as a dimos module.

A thin transport shell around the pluggable ``TrajectoryController`` — the
controller stays a pure pose+path -> twist law (the piece that later ports
to rust); this module owns subscriptions, the control clock, the on-robot
clearance annotation and goal arrival. Clearance is recomputed from the
local map per (path, map) pair, the same room hint the control battery's
judge hands the controller in sim.
"""

from __future__ import annotations

import math
from threading import Event, RLock, Thread
import time
from typing import Any

from dimos_lcm.std_msgs import Bool  # type: ignore[import-untyped]
import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.motion.control.controller import (
    ControllerConfig,
    TrajectoryController,
    load,
)
from dimos.navigation.motion.control.profile import ceilings_to_clearance, decode_ceilings
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Cloud slice that can touch the body, as the planner sees it (target.py).
Z_BAND = (0.05, 0.45)


def path_clearance(xy: np.ndarray, points: np.ndarray, half_width: float) -> np.ndarray:
    """Per-waypoint room hint (m): nearest z-band point minus the half-width.

    A speed hint for the controller, not a safety contract. Empty band or
    empty path = infinite room.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    band = pts[(pts[:, 2] > Z_BAND[0]) & (pts[:, 2] < Z_BAND[1])][:, :2]
    if not len(band) or not len(xy):
        return np.full(len(xy), np.inf)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(band).query(xy)
    return np.asarray(d, dtype=float) - half_width


class GoalLatch:
    """Arrival edge detector: fires once per goal, then holds until it moves."""

    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance
        self._goal: tuple[float, float] | None = None
        self._reached = False

    @property
    def reached(self) -> bool:
        return self._reached

    def set_goal(self, xy: tuple[float, float]) -> None:
        # moves under the arrival tolerance are the same goal — replans snap
        # the path end to the search grid, and re-chasing that is jitter
        if self._goal is None or math.dist(xy, self._goal) > self.tolerance:
            self._goal = xy
            self._reached = False

    def arrive(self, xy: tuple[float, float]) -> bool:
        """True exactly once: the tick this position first reaches the goal."""
        if self._goal is None or self._reached:
            return False
        if math.dist(xy, self._goal) < self.tolerance:
            self._reached = True
            return True
        return False


class TrajectoryFollowerConfig(ModuleConfig):
    controller: str = "pursuit"  # registry name or "module:factory"
    controller_config: ControllerConfig = Field(default_factory=ControllerConfig)
    control_frequency: float = 10.0
    goal_tolerance: float = 0.20  # planar distance that counts as arrival (m)
    annotate_clearance: bool = True  # hand the controller the path room hint
    half_width: float = 0.155  # embodiment half-width for the clearance hint (go2)


class TrajectoryFollower(Module):
    """Track the planned path; stop and latch goal_reached on arrival."""

    config: TrajectoryFollowerConfig

    path: In[Path]
    odometry: In[Odometry]
    local_map: In[PointCloud2]
    stop_movement: In[Bool]

    nav_cmd_vel: Out[Twist]
    goal_reached: Out[Bool]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = RLock()
        self._pose: PoseStamped | None = None
        self._path: Path | None = None
        self._cloud: PointCloud2 | None = None
        self._clearance: np.ndarray | None = None
        self._clearance_key: tuple[int, int] | None = None
        self._latch = GoalLatch(self.config.goal_tolerance)
        self._controller: TrajectoryController | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._controller = load(self.config.controller)(self.config.controller_config)
        self._controller.reset()
        self.register_disposable(Disposable(self.path.subscribe(self._on_path)))
        self.register_disposable(Disposable(self.odometry.subscribe(self._on_odometry)))
        self.register_disposable(Disposable(self.local_map.subscribe(self._on_local_map)))
        if self.stop_movement.transport is not None:
            self.register_disposable(Disposable(self.stop_movement.subscribe(self._on_stop)))
        self._thread = Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self.nav_cmd_vel.publish(Twist())
        super().stop()

    def _on_path(self, msg: Path) -> None:
        with self._lock:
            self._path = msg
            if len(msg.poses) >= 2:
                # the plan ends at the goal; a single-pose stub is a refusal,
                # never an arrival target
                self._latch.set_goal((msg.poses[-1].position.x, msg.poses[-1].position.y))

    def _on_odometry(self, msg: Odometry) -> None:
        with self._lock:
            self._pose = msg.to_pose_stamped()

    def _on_local_map(self, msg: PointCloud2) -> None:
        with self._lock:
            self._cloud = msg

    def _on_stop(self, msg: Bool) -> None:
        if msg.data:
            with self._lock:
                self._path = None
            self.nav_cmd_vel.publish(Twist())

    def _control_loop(self) -> None:
        period = 1.0 / self.config.control_frequency
        while not self._stop_event.is_set():
            started = time.perf_counter()
            with self._lock:
                pose, path = self._pose, self._path
            if pose is not None and path is not None:
                self._step(pose, path)
            elapsed = time.perf_counter() - started
            self._stop_event.wait(max(0.0, period - elapsed))

    def _step(self, pose: PoseStamped, path: Path) -> None:
        assert self._controller is not None
        xy = (pose.position.x, pose.position.y)
        if self._latch.arrive(xy):
            self.nav_cmd_vel.publish(Twist())
            self.goal_reached.publish(Bool(True))
            logger.info("Goal reached")
            return
        if self._latch.reached:
            self.nav_cmd_vel.publish(Twist())
            return
        tw = self._controller.update(pose, path, time.monotonic(), self._clearance_for(path))
        self.nav_cmd_vel.publish(tw)

    def _clearance_for(self, path: Path) -> np.ndarray | None:
        if not self.config.annotate_clearance:
            return None
        with self._lock:
            cloud = self._cloud
        if cloud is None:
            # no local map: fall back to the precision the planner stamped
            # into the path's own timestamps (control/profile.py dialect)
            ceilings = decode_ceilings(path)
            return ceilings_to_clearance(ceilings) if ceilings is not None else None
        key = (id(path), id(cloud))
        if key != self._clearance_key:
            wp = np.array([[p.position.x, p.position.y] for p in path.poses]).reshape(-1, 2)
            self._clearance = path_clearance(wp, cloud.points_f32(), self.config.half_width)
            self._clearance_key = key
        return self._clearance
