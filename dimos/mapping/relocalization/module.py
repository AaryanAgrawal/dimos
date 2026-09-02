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

"""Relocalization: place the live ``world`` frame inside a prior map's ``map`` frame.

This file is the contract: every relocalizer, whatever it matches, has a
prior map and answers with a :class:`Fix`, so it publishes the same two
things - ``tf`` and the placed prior map on ``loaded_map``. Both live here,
along with :meth:`RelocalizationModule.accept` to turn a fix into a
transform and :meth:`RelocalizationModule.set_premap` to hand over the map.

It deliberately owns no *input* and no map *format*. Matching lidar against
a pointcloud premap, apriltags against a table of tag poses and GPS against
a datum share no port type, no file format and no reason to attempt a fix at
the same moment - so each implementation declares its own ``In`` ports and
prior-map config and drives itself. See ``lidar/module.py``, which is the
pointcloud runtime; a GPS one subclasses this directly and inherits none of
it.

Whether a fix is good enough is the implementation's call, made against its
own config. A second threshold here would be a second place to configure one
decision, and the two would drift.

A dual strategy is a subclass of two implementations: ports merge across the
MRO and ``start()`` chains through ``super()``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import reactivex as rx
from reactivex import Subject, operators as ops

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

FRAME_MAP = "map"
FRAME_WORLD = "world"

PUBLISH_INTERVAL = 2.0  # TF republish period


class Fix(NamedTuple):
    """What one relocalization attempt concluded."""

    # 4x4 placing live points into the prior map: p_map = transform @ p_world.
    transform: np.ndarray
    fitness: float  # in [0, 1], strategy-defined
    # Diagnostics: logged, never acted on. `rmse` is how tightly the matched
    # points sit where fitness only counts how many matched; `margin` is how
    # far the winning hypothesis beat the runner-up, near zero for a place
    # the map matches in several spots equally well.
    rmse: float = 0.0
    margin: float = 0.0


class Config(ModuleConfig):
    publish_loaded_map: bool = False


class RelocalizationModule(Module):
    config: Config
    tf: Out[TFMessage]
    loaded_map: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._world_to_map: Subject[Transform] = Subject()
        self._premap: PointCloud2 | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(
            rx.interval(PUBLISH_INTERVAL)
            .pipe(ops.with_latest_from(self._world_to_map))
            .subscribe(lambda pair: self.tf.publish(TFMessage(pair[1].now())))
        )

    def set_premap(self, premap: PointCloud2) -> None:
        """Adopt the loaded prior map. Implementations call this once, from ``start()``.

        Loading is the implementation's job - only it knows the format - but
        what happens next is the same everywhere: the map defines the ``map``
        frame, and republishing it is gated on a fix, because until one lands
        there is nothing to resolve that frame against.
        """
        premap.frame_id = FRAME_MAP
        self._premap = premap
        if self.config.publish_loaded_map:
            self.register_disposable(
                rx.interval(PUBLISH_INTERVAL)
                .pipe(ops.with_latest_from(self._world_to_map))
                .subscribe(lambda _: self.loaded_map.publish(premap))
            )

    def accept(self, fix: Fix, source: str = "") -> None:
        """Publish a fix an implementation already decided to believe."""
        # fix.transform maps world points into the map; the TF tree wants the
        # frame transform, which is its inverse.
        self.submit(
            Transform.from_matrix(
                fix.transform, frame_id=FRAME_MAP, child_frame_id=FRAME_WORLD
            ).inverse(),
            fix.fitness,
            source,
        )

    def submit(self, tf: Transform, fitness: float, source: str = "") -> None:
        """Publish a ``world -> map`` fix; fitness in [0, 1] is impl-defined."""
        assert (tf.frame_id, tf.child_frame_id) == (FRAME_WORLD, FRAME_MAP), (
            f"relocalize {source}: expected {FRAME_WORLD!r} -> {FRAME_MAP!r}, "
            f"got {tf.frame_id!r} -> {tf.child_frame_id!r}"
        )
        logger.info(
            f"relocalize {source}: fitness={fitness:.3f} "
            f"TF {FRAME_WORLD!r} -> {FRAME_MAP!r} t={tf.translation}"
        )
        self._world_to_map.on_next(tf)
