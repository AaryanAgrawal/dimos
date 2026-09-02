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

"""Relocalization against a premap, driven by a recording instead of a robot.

    dimos run relocalize-mid360

Replays a mid360 walk's registered scans at wall-clock rate into the same
mapper the Go2 stack runs, so :class:`LidarRelocalization` sees exactly the
``global_map`` it would on hardware and has to find the walk inside a premap
built from it. Watch it in Rerun: `world/loaded_map` appears only once a fix
lands, and lands on top of `world/global_map` when the fix is right.

The recording and the premap are the eval's dataset (``lidar/eval.py``,
``lidar/readme.md``), so a demo that looks wrong and an eval that scores
badly are the same bug.
"""

from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.mapping.relocalization.lidar.module import LidarRelocalization
from dimos.mapping.voxels.module import VoxelGridMapper
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.data import resolve_named_path
from dimos.utils.logging_config import setup_logger
from dimos.visualization.vis_module import vis_module

logger = setup_logger()

# The walk, and a premap built from the same walk's second half. Overlapping
# but not identical: the relocalizer has to place the live scans, not
# recognize a copy of them.
DATASET = "recording_go2_mid360_2026-05-29_4-45pm-PST_corrected"


class RecordingPlayerConfig(ModuleConfig):
    dataset: str = DATASET  # recording stem or path; `.db`, LFS-fetched on miss
    stream: str = "fastlio_lidar"  # already registered into the world frame
    speed: float = 1.0
    seek: float | None = None
    duration: float | None = None
    loop: bool = False


class RecordingPlayer(Module):
    """Publish one stream of a recording on ``lidar``, paced by its own timestamps."""

    config: RecordingPlayerConfig
    lidar: Out[PointCloud2]

    @rpc
    def start(self) -> None:
        super().start()
        path = resolve_named_path(self.config.dataset, ".db")
        store = self.register_disposable(SqliteStore(path=str(path), must_exist=True))
        store.start()
        replay = store.replay(
            speed=self.config.speed,
            seek=self.config.seek,
            duration=self.config.duration,
            loop=self.config.loop,
        )
        stream: Any = replay.stream(self.config.stream)
        logger.info(
            f"Replaying {path.name}:{self.config.stream} "
            f"({stream.count()} frames at {self.config.speed}x)"
        )
        self.register_disposable(stream.observable().subscribe(self.lidar.publish))


relocalize_mid360 = autoconnect(
    RecordingPlayer.blueprint(),
    # The Go2 stack's mapper, at its settings, so `global_map` is the message
    # the relocalizer meets in production.
    VoxelGridMapper.blueprint(emit_every=5),
    LidarRelocalization.blueprint(map_file=DATASET, publish_loaded_map=True),
    vis_module("rerun"),
).global_config(n_workers=5, robot_model="relocalize_mid360")
