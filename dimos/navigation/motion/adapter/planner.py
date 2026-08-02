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

"""MotionPlanner: the autoresearch planner as a dimos local planner module.

Bridges the referee-side ``PlannerEpisode`` protocol onto module streams:
the raycaster's ``local_map`` is the cloud, leveled body odometry is the
pose, and the goal is a carrot — ``goal_lookahead_m`` of arc along the MLS
global path (``planner_path``), clamped to its end. Replans on a fixed
cadence (receding horizon, as the control battery runs it) and republishes
the result as a nav Path. A refusal comes out as the planner made it — a
single-pose stub the follower reads as "hold" — while MLS reroutes globally.
"""

from __future__ import annotations

from threading import Event, RLock, Thread
import time
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.motion.planner.autoresearch.geometry import AvoidanceConfig
from dimos.navigation.motion.planner.autoresearch.planners.base import PlannerEpisode, load
from dimos.navigation.motion.planner.autoresearch.scenarios import EMBODIMENTS, Scenario
from dimos.navigation.motion.planner.autoresearch.types import (
    Path as RefereePath,
    PointCloud2 as RefereeCloud,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def to_nav_path(ref: RefereePath, ts: float = 0.0, frame_id: str = "odom") -> Path:
    """Referee path -> dimos nav_msgs Path (the type the follower consumes)."""
    poses = [
        PoseStamped(
            ts=ts,
            frame_id=frame_id,
            position=Vector3(p.position.x, p.position.y, p.position.z),
            orientation=Quaternion(
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
            ),
        )
        for p in ref.poses
    ]
    return Path(ts=ts, frame_id=frame_id, poses=poses)


def carrot_along(
    path_xy: np.ndarray, robot_xy: tuple[float, float], lookahead: float
) -> tuple[float, float]:
    """`lookahead` metres of arc along the path from the waypoint closest to
    the robot, clamped to the path end."""
    xy = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    i = int(np.argmin(np.linalg.norm(xy - robot_xy, axis=1)))
    remaining = lookahead
    for j in range(i, len(xy) - 1):
        seg = xy[j + 1] - xy[j]
        seg_len = float(np.linalg.norm(seg))
        if seg_len >= remaining:
            point = xy[j] + (remaining / seg_len) * seg
            return (float(point[0]), float(point[1]))
        remaining -= seg_len
    return (float(xy[-1][0]), float(xy[-1][1]))


class MotionPlannerConfig(ModuleConfig):
    planner: str = "target"  # referee registry name or "module:factory"
    embodiment: str = "go2"
    replan_hz: float = 5.0  # the control battery's reality default
    goal_lookahead_m: float = 5.0  # carrot arc along the global path
    world_frame: str = "odom"
    # Calibration: added to cloud z before planning, so the planner's body
    # z-band (0.05..0.45 above the floor) lands where the floor actually is
    # when the map's z origin is not ground level.
    cloud_z_offset: float = 0.0


class MotionPlanner(Module):
    """Receding-horizon local planning over the live local map."""

    config: MotionPlannerConfig

    local_map: In[PointCloud2]
    odometry: In[Odometry]
    planner_path: In[Path]

    path: Out[Path]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = RLock()
        self._cloud: PointCloud2 | None = None
        self._pose: tuple[float, float, float] | None = None
        self._global_xy: np.ndarray | None = None
        self._episode: PlannerEpisode | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        sc = Scenario("live", [], goal=(0.0, 0.0), emb=EMBODIMENTS[self.config.embodiment])
        self._episode = load(self.config.planner)(sc, AvoidanceConfig())
        self._episode.reset()
        self.register_disposable(Disposable(self.local_map.subscribe(self._on_local_map)))
        self.register_disposable(Disposable(self.odometry.subscribe(self._on_odometry)))
        self.register_disposable(Disposable(self.planner_path.subscribe(self._on_planner_path)))
        self._thread = Thread(target=self._plan_loop, daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        super().stop()

    def _on_local_map(self, msg: PointCloud2) -> None:
        with self._lock:
            self._cloud = msg

    def _on_odometry(self, msg: Odometry) -> None:
        with self._lock:
            self._pose = (msg.position.x, msg.position.y, msg.orientation.euler[2])

    def _on_planner_path(self, msg: Path) -> None:
        # MLS emits an empty path when it finds no route: no carrot, hold the
        # last local plan rather than chase a stale one.
        xy = np.array([[p.position.x, p.position.y] for p in msg.poses]).reshape(-1, 2)
        with self._lock:
            self._global_xy = xy if len(xy) else None
        if self._episode is not None:
            # a new task: warm starts and hysteresis from the old route are
            # stale (no-op for the stateless rust target)
            self._episode.reset()

    def _plan_loop(self) -> None:
        period = 1.0 / self.config.replan_hz
        while not self._stop_event.is_set():
            started = time.perf_counter()
            with self._lock:
                cloud, pose, global_xy = self._cloud, self._pose, self._global_xy
            if cloud is not None and pose is not None and global_xy is not None:
                goal = carrot_along(global_xy, (pose[0], pose[1]), self.config.goal_lookahead_m)
                self._plan_once(cloud, pose, goal)
            elapsed = time.perf_counter() - started
            self._stop_event.wait(max(0.0, period - elapsed))

    def _plan_once(
        self, cloud: PointCloud2, pose: tuple[float, float, float], goal: tuple[float, float]
    ) -> None:
        assert self._episode is not None
        pts = cloud.points_f32()
        if self.config.cloud_z_offset != 0.0:
            pts = pts + np.array([0.0, 0.0, self.config.cloud_z_offset], dtype=np.float32)
        ref_cloud = RefereeCloud.from_numpy(pts, frame_id=self.config.world_frame)
        try:
            ref = self._episode.plan(ref_cloud, pose, goal)
        except Exception:
            logger.exception("planner failed; keeping the last published path")
            return
        self.path.publish(to_nav_path(ref, ts=time.time(), frame_id=self.config.world_frame))
