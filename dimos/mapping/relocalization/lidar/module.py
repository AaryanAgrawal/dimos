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

"""Lidar relocalization runtime: feed the live voxel map to the aligner."""

from __future__ import annotations

import time
from typing import Any

import reactivex as rx
from reactivex import operators as ops

from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.mapping.relocalization.lidar.relocalize import (
    PreparedMap,
    RelocalizeConfig,
    prepare,
    relocalize,
)
from dimos.mapping.relocalization.module import (
    FRAME_MAP,
    FRAME_WORLD,
    PUBLISH_INTERVAL,
    Config,
    RelocalizationModule,
)
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.data import resolve_named_path
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

logger = setup_logger()

MAP_SUFFIX = ".pc2.lcm"


class LidarConfig(Config):
    map_file: str | None = (
        None  # premap stem or path, e.g. `--map-file=go2_hongkong_office_twopass_map`
    )
    publish_loaded_map: bool = False
    reloc_interval: float = 2.0
    min_local_points: int = 50_000
    relocalize: RelocalizeConfig = RelocalizeConfig()


class LidarRelocalization(RelocalizationModule):
    """Coarse FPFH+RANSAC then ICP of the live voxel map against a pointcloud premap."""

    config: LidarConfig
    global_map: In[PointCloud2]
    loaded_map: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._premap: PointCloud2 | None = None
        self._prepared: PreparedMap | None = None
        self._last_skip_log = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        if not self.config.map_file:
            logger.info("Relocalization module disabled (no map_file configured)")
            return

        path = resolve_named_path(self.config.map_file, MAP_SUFFIX)
        self._premap = PointCloud2.lcm_decode(path.read_bytes())
        self._premap.frame_id = FRAME_MAP
        # The premap never changes, so its downsampling, normals and FPFH are
        # a startup cost rather than a per-fix one.
        self._prepared = prepare(self._premap.pointcloud, self.config.relocalize)

        self.register_disposable(
            backpressure(
                self.global_map.observable().pipe(  # type: ignore[no-untyped-call]
                    ops.throttle_first(self.config.reloc_interval),
                    ops.do_action(self._maybe_log_skip),
                    ops.filter(self._has_enough_points),
                )
            ).subscribe(self._relocalize)
        )

        if self.config.publish_loaded_map:
            premap = self._premap
            self.register_disposable(
                rx.interval(PUBLISH_INTERVAL).subscribe(lambda _: self.loaded_map.publish(premap))
            )

        logger.info(f"Relocalization module started: map_file={self.config.map_file!r}")

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
        assert self._premap is not None
        t0 = time.monotonic()
        try:
            assert self._prepared is not None
            fix = relocalize(self._prepared, msg.pointcloud)
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
        # relocalize() returns T with p_map = T @ p_world; the TF tree wants world -> map.
        tf = Transform.from_matrix(
            fix.transform, frame_id=FRAME_MAP, child_frame_id=FRAME_WORLD
        ).inverse()
        self.submit(tf, fix.fitness, "lidar")
