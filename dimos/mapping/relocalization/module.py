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

"""Relocalization base: publish the ``world -> map`` transform.

A relocalizer estimates the rigid transform between the live ``world`` frame
and a prior map's ``map`` frame. Implementations live in ``impl/`` and
subclass :class:`RelocalizationModule`: declare your own ``In`` ports and
prior-map config, call ``super().start()``, subscribe, and call
:meth:`RelocalizationModule.submit` on every fix. The base gates on
keeps the last accepted fix and republishes it on ``tf`` every
``PUBLISH_INTERVAL``. Deciding whether a fix is worth submitting belongs to
the implementation and its own config, not to a second threshold here.

A dual strategy is a subclass of two implementations: ports merge across the
MRO and ``start()`` chains through ``super()``.
"""

from typing import Any

import reactivex as rx
from reactivex import Subject, operators as ops

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

FRAME_MAP = "map"
FRAME_WORLD = "world"

PUBLISH_INTERVAL = 2.0  # TF republish period


class Config(ModuleConfig):
    pass


class RelocalizationModule(Module):
    config: Config
    tf: Out[TFMessage]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._world_to_map: Subject[Transform] = Subject()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(
            rx.interval(PUBLISH_INTERVAL)
            .pipe(ops.with_latest_from(self._world_to_map))
            .subscribe(lambda pair: self.tf.publish(TFMessage(pair[1].now())))
        )

    def submit(self, tf: Transform, fitness: float, source: str = "") -> None:
        """Publish a ``world -> map`` fix; fitness in [0, 1] is impl-defined.

        Whether a fix is good enough is the implementation's call, made
        against its own config - the lidar one refuses inside
        :func:`~dimos.mapping.relocalization.lidar.module.relocalize`, which
        returns nothing rather than a fix it does not believe. A second
        threshold here would be a second place to configure the same
        decision, and the two would drift.
        """
        assert (tf.frame_id, tf.child_frame_id) == (FRAME_WORLD, FRAME_MAP), (
            f"relocalize {source}: expected {FRAME_WORLD!r} -> {FRAME_MAP!r}, "
            f"got {tf.frame_id!r} -> {tf.child_frame_id!r}"
        )
        logger.info(
            f"relocalize {source}: fitness={fitness:.3f} "
            f"TF {FRAME_WORLD!r} -> {FRAME_MAP!r} t={tf.translation}"
        )
        self._world_to_map.on_next(tf)
