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

"""Lidar relocalization runtime: the live voxel map against the premap's points.

What is here is what only a pointcloud matcher needs - the ``global_map``
input, how often to attempt a match, how sparse a cloud is too sparse, and
the aligner. Loading the premap and publishing what comes back is the base
module's; the alignment itself is ``relocalize.py``'s.
"""

from __future__ import annotations

import time
from typing import Any

from reactivex import operators as ops

from dimos.core.core import rpc
from dimos.core.stream import In
from dimos.mapping.relocalization.lidar.relocalize import LidarRelocalizer, RelocalizeConfig
from dimos.mapping.relocalization.module import Config, RelocalizationModule
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

logger = setup_logger()


class LidarConfig(Config):
    reloc_interval: float = 2.0
    # Skip a cloud too sparse to be worth a match, in points of the *voxel
    # map* the mapper emits - not raw sensor points. A mid360 sweep is only
    # ~2.8k points and two of them voxel down to ~3.5k, which is already
    # enough to relocalize; a rig whose mapper emits far more raises this in
    # its own config subclass.
    min_local_points: int = 2_000
    relocalize: RelocalizeConfig = RelocalizeConfig()


class LidarRelocalization(RelocalizationModule):
    """Coarse FPFH+RANSAC then ICP of the live voxel map against a pointcloud premap."""

    config: LidarConfig
    global_map: In[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._relocalizer: LidarRelocalizer | None = None
        self._last_skip_log = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        if self.premap is None:
            return
        # Downsampling the premap and computing its normals and FPFH is the
        # pipeline's dominant cost, so it is a startup cost, not a per-fix one.
        self._relocalizer = LidarRelocalizer(self.premap.pointcloud, self.config.relocalize)
        self.register_disposable(
            backpressure(
                self.global_map.observable().pipe(  # type: ignore[no-untyped-call]
                    ops.throttle_first(self.config.reloc_interval),
                    ops.do_action(self._maybe_log_skip),
                    ops.filter(self._has_enough_points),
                )
            ).subscribe(self._relocalize)
        )

    def _maybe_log_skip(self, msg: PointCloud2) -> None:
        if self._has_enough_points(msg):
            return
        now = time.monotonic()
        if now - self._last_skip_log > 5.0:
            logger.warning(
                f"relocalize skipped: n_pts={len(msg)} "
                f"< min_local_points={self.config.min_local_points}"
            )
            self._last_skip_log = now

    def _has_enough_points(self, msg: PointCloud2) -> bool:
        return len(msg) >= self.config.min_local_points

    def _relocalize(self, msg: PointCloud2) -> None:
        assert self._relocalizer is not None
        t0 = time.monotonic()
        try:
            fix = self._relocalizer.relocalize(msg.pointcloud)
        except Exception:
            logger.exception("relocalize() failed")
            return
        dt = time.monotonic() - t0
        if fix is None:
            logger.info(
                f"relocalize lidar: refused after {dt:.1f}s n_pts={len(msg)} "
                f"(below fitness_threshold={self.config.relocalize.fitness_threshold})"
            )
            return
        logger.info(
            f"relocalize lidar: time_cost={dt:.1f}s n_pts={len(msg)} "
            f"fitness={fix.fitness:.3f} rmse={fix.rmse:.3f} margin={fix.margin:.3f}"
        )
        self.accept(fix, "lidar")
