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

The recording and the premap are the eval's dataset (``lidar/tune.py``,
``lidar/tune.md``), so a demo that looks wrong and an eval that scores
badly are the same bug.
"""

from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.mapping.relocalization.lidar.module import LidarRelocalization
from dimos.mapping.relocalization.module import FRAME_WORLD
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
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
    stream: str = "fastlio_lidar"
    # Frame to stamp the sensor cloud and the tf that places it. Only a label:
    # the raycaster looks up `world -> <this>` and never reads the name.
    sensor_frame: str = "lidar_link"
    speed: float = 1.0
    seek: float | None = None
    duration: float | None = None
    loop: bool = False


class RecordingPlayer(Module):
    """Replay a recording's lidar as a live sensor would: cloud plus the tf that places it.

    A recorded scan comes in one of two shapes, and the mapper downstream
    wants neither of them directly. Point-LIO stores the sensor-frame cloud
    with its pose alongside; FAST-LIO stores the cloud already registered
    into the world. Both are published here as the sensor frame plus a
    ``world -> sensor`` transform, so the raycaster registers them itself and
    knows where the rays started from - which is the whole point of using it
    over a mapper that just stacks clouds.
    """

    config: RecordingPlayerConfig
    lidar: Out[PointCloud2]
    tf: Out[TFMessage]

    @rpc
    def start(self) -> None:
        super().start()
        path = resolve_named_path(self.config.dataset, ".db")
        # A separate store for the pose scan: reading a stream to exhaustion
        # leaves the replay's own iteration on a closed database.
        scan = SqliteStore(path=str(path), must_exist=True)
        scan.start()
        try:
            poses = {
                obs.ts: obs.pose
                for obs in scan.stream(self.config.stream, PointCloud2)
                if obs.pose is not None
            }
        finally:
            scan.dispose()

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
            f"Replaying {path.name}:{self.config.stream} ({stream.count()} frames "
            f"at {self.config.speed}x, {len(poses)} posed)"
        )
        self.register_disposable(stream.observable().subscribe(self._publish(poses)))

    def _publish(self, poses: dict[float, Any]) -> Any:
        frame = self.config.sensor_frame

        def publish(cloud: PointCloud2) -> None:
            pose = poses.get(cloud.ts)
            if pose is None:  # unposed scan: nothing can place it
                return
            tf = Transform.from_pose(FRAME_WORLD, pose)
            tf.child_frame_id = frame
            # The recording's stamp, not the wall clock: the raycaster matches
            # a cloud to a transform by stamp, within a scan period.
            tf.ts = cloud.ts
            if cloud.frame_id == FRAME_WORLD:
                # Already registered by the recording's LIO. Undo it, so the
                # raycaster does the placing and both recording shapes take
                # one path.
                cloud = cloud.transform(tf.inverse())
            cloud.frame_id = frame
            self.tf.publish(TFMessage(tf))
            self.lidar.publish(cloud)

        return publish


def _fine_points(cloud: Any) -> Any:
    """The premap is millimetre-scale; draw it at that size, not the 5 cm default."""
    return cloud.to_rerun(voxel_size=0.0015)


# Off until asked for. The question this demo answers is whether the premap
# and the live map land on top of each other, and the raw scans, the live
# global map and the raycaster's region box all sit in the same space and
# bury it. Toggle any of them back on in the viewer.
HIDDEN = ("world/global_map", "world/lidar", "world/region_bounds")


def _view() -> Any:
    """The default 3D view, with the noisy entities off."""
    import rerun as rr
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="world",
            background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
            line_grid=rrb.LineGrid3D(plane=rr.components.Plane3D.XY.with_distance(0.5)),
            overrides={path: rrb.EntityBehavior(visible=False) for path in HIDDEN},
        ),
    )


relocalize_mid360 = autoconnect(
    RecordingPlayer.blueprint(),
    # The raycaster, not VoxelGridMapper: it registers each cloud through tf
    # and clears the space the rays passed through, which is also what the
    # eval accumulates its local maps with. Same mapper on both sides means a
    # demo that looks wrong and an eval that scores badly are one bug.
    RayTracingVoxelMap.blueprint(voxel_size=0.1, world_frame=FRAME_WORLD, global_emit_every=5),
    LidarRelocalization.blueprint(map_file=DATASET, publish_loaded_map=True),
    vis_module(
        "rerun",
        {"visual_override": {"world/loaded_map": _fine_points}, "blueprint": _view},
    ),
).global_config(n_workers=5, robot_model="relocalize_mid360")
