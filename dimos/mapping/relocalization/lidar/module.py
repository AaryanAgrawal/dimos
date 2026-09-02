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

"""Lidar relocalization: align the live voxel map to a pointcloud premap."""

import time
from typing import Any, NamedTuple

import numpy as np
import reactivex as rx
from reactivex import operators as ops

from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.mapping.relocalization.module import (
    FRAME_MAP,
    FRAME_WORLD,
    PUBLISH_INTERVAL,
    Config,
    RelocalizationModule,
)
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.protocol.service.spec import BaseConfig
from dimos.utils.data import resolve_named_path
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

logger = setup_logger()

MAP_SUFFIX = ".pc2.lcm"


class Prepared(NamedTuple):
    """A map's config-dependent preprocessing, which never changes with the query.

    The premap is fixed for the life of a relocalizer, so down-sampling it,
    estimating its normals and computing its FPFH features belongs outside
    the per-call path - it dominates the cost otherwise, and a live module
    would redo it on every fix.
    """

    coarse: Any
    fpfh: Any
    fine: Any


class Fix(NamedTuple):
    """What one relocalization attempt concluded."""

    transform: np.ndarray
    fitness: float
    # Inlier RMSE: how tightly the matched points sit, where fitness only
    # counts how many matched. Unlike fitness it does not inflate when the
    # correspondence distance is widened.
    rmse: float
    # Best minus runner-up fitness across restarts. A place that matches
    # many parts of the map equally well scores near zero here however
    # confident any single hypothesis looks; with one restart it is 0.
    margin: float


def _yaw_only(T: np.ndarray, pivot: np.ndarray | None = None) -> np.ndarray:
    """``T`` with roll and pitch dropped, rotating about ``pivot``.

    The yaw is read off the transformed x-axis rather than by decomposing
    Euler angles, so a hypothesis tilted right past vertical degrades
    instead of hitting a gimbal singularity.

    ``pivot`` matters more than the flattening itself. Swapping the rotation
    while keeping the translation pivots the cloud about the world origin,
    and these clouds sit a hundred metres from it - half a degree of tilt
    then throws them 0.36 m, past the distance ICP will look, so the
    correction destroys the hypothesis it was meant to clean up. Pivoting
    about the cloud's own centre removes the tilt and leaves the cloud where
    the hypothesis put it. ``None`` keeps the origin, which is only right
    when the cloud is already centred there.
    """
    yaw = np.arctan2(T[1, 0], T[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pivot = np.zeros(3) if pivot is None else np.asarray(pivot, dtype=float)

    flat = np.eye(4)
    flat[:3, :3] = rotation
    # The pivot lands where the original transform put it; only the tilt goes.
    flat[:3, 3] = (T[:3, :3] @ pivot + T[:3, 3]) - rotation @ pivot
    return flat


class RelocalizeConfig(BaseConfig):
    """Knobs of :func:`relocalize`. Every field was a literal in its body.

    The defaults are trial 229 of the ``go2-sf-area1`` study - the candidate
    that held its hit rate on probes it was never tuned against, where
    others dropped forty points. Its two neighbours on the front agree with
    it to within a few percent on every field, so this is a plateau rather
    than a spike, which is the kind worth shipping.

    They were measured on one outdoor mid360 walk against a premap of the
    same neighbourhood. A different sensor, or a room-sized map, wants its
    own instance and its own study - ``dimos.mapping.relocalization.lidar.eval
    tune``, then ``verify`` before believing any of it.
    """

    voxel_coarse: float = 0.59  # FPFH + RANSAC scale
    voxel_fine: float = 0.30  # ICP scale
    # Normal and feature neighbourhoods, in multiples of the working voxel.
    normal_radius_factor: float = 1.72
    fpfh_radius_factor: float = 4.13
    normal_max_nn: int = 30
    fpfh_max_nn: int = 100
    # RANSAC correspondence distance, in multiples of voxel_coarse.
    coarse_dist_factor: float = 2.73
    ransac_iters: int = 1_578_291
    ransac_confidence: float = 0.999
    ransac_n: int = 3
    mutual_filter: bool = True
    edge_length: float = 0.70
    # ICP correspondence distance, in multiples of voxel_fine - the distance
    # of the *last* stage. Earlier stages double it each step back, so a
    # hypothesis landing further out than the final threshold still has a
    # stage wide enough to see its correspondences. One stage is a single
    # pass at icp_dist_factor.
    icp_dist_factor: float = 0.55
    icp_stages: int = 2
    icp_max_iter: int = 200
    # Normal *sign* is arbitrary out of estimate_normals, and FPFH is built
    # from angles involving it, so two clouds scanned from different passes
    # describe the same corner differently. Pointing every normal into the
    # same half-space makes the descriptors comparable. Needs a shared up
    # axis, which gravity_aligned already asserts.
    orient_normals: bool = True
    # Both clouds gravity-aligned, so the transform between them is a pure
    # yaw and any roll or pitch RANSAC proposes is error by construction.
    # On: the constraint is simply true when both maps come from a
    # lidar-inertial odometry, and it removes a degree of freedom the answer
    # cannot use. The tuning measured it as a wash once _yaw_only pivots
    # about the cloud rather than the origin (before that fix it was
    # actively harmful, which is why the search buried it) - a wash on a
    # walk whose hypotheses rarely come out tilted is not evidence against
    # the cases where they do. Turn it off for a premap whose frame is not
    # known to be level.
    gravity_aligned: bool = True
    # RANSAC is stochastic and its best hypothesis is sometimes simply
    # wrong; ICP then polishes a wrong answer. Restarts take the best of
    # several, and their spread is what `margin` reports. One suffices at
    # these settings, where the iteration budget above already finds the
    # place; restarts were what rescued the older, sparser search.
    ransac_restarts: int = 1
    # ICP fitness a fix must clear to be published. The two populations sit
    # far apart - on the measured walk, fixes that found the right place
    # score 0.58-0.76 and places absent from the premap score 0.13-0.17 - so
    # this sits in the empty middle rather than snug against either. Being
    # strict is not free: a relocalizer gets another cloud every couple of
    # seconds, so a rejected fix is a retry, while an accepted wrong one is
    # a TF the whole stack believes.
    fitness_threshold: float = 0.5


def prepare(cloud: Any, config: RelocalizeConfig | None = None) -> Prepared:
    """A map's voxel-downsampled forms and FPFH features, ready to match against.

    Call once per map per config and hand the result to :func:`relocalize`;
    for the premap that turns the pipeline's dominant cost into a startup
    cost. :func:`relocalize` does it inline when not given one.
    """
    import open3d as o3d  # type: ignore[import-untyped]

    cfg = config or RelocalizeConfig()
    reg = o3d.pipelines.registration

    def normals(pcd: Any, voxel: float) -> Any:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * cfg.normal_radius_factor, max_nn=cfg.normal_max_nn
            )
        )
        if cfg.orient_normals:
            pcd.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
        return pcd

    coarse = normals(cloud.voxel_down_sample(cfg.voxel_coarse), cfg.voxel_coarse)
    fpfh = reg.compute_fpfh_feature(
        coarse,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.voxel_coarse * cfg.fpfh_radius_factor, max_nn=cfg.fpfh_max_nn
        ),
    )
    fine = normals(cloud.voxel_down_sample(cfg.voxel_fine), cfg.voxel_fine)
    return Prepared(coarse=coarse, fpfh=fpfh, fine=fine)


def align(
    global_map: Any,
    local_map: Any,
    config: RelocalizeConfig | None = None,
    prepared: Prepared | None = None,
) -> Fix:
    """Open3D clouds in; the 4x4 T placing ``local_map`` into ``global_map`` (p_map = T @ p_local).

    Always answers, however poor the answer. :func:`relocalize` is the one
    to call - this is for a caller that wants to see a refused fix, such as
    the eval measuring how far a rejected hypothesis actually landed.

    Coarse FPFH+RANSAC to find the place, then point-to-plane ICP over
    widening-to-narrowing distances to settle it. Strategies + evals live in
    https://github.com/leshy/relocalization-test (this began as ``align_fast``).

    ``prepared`` skips the target's preprocessing, which is the same on
    every call for a fixed premap.
    """
    import open3d as o3d  # type: ignore[import-untyped]

    cfg = config or RelocalizeConfig()
    reg = o3d.pipelines.registration
    target = prepared or prepare(global_map, cfg)
    source = prepare(local_map, cfg)
    dist = cfg.voxel_coarse * cfg.coarse_dist_factor

    def hypothesis() -> Any:
        return reg.registration_ransac_based_on_feature_matching(
            source.coarse,
            target.coarse,
            source.fpfh,
            target.fpfh,
            mutual_filter=cfg.mutual_filter,
            max_correspondence_distance=dist,
            estimation_method=reg.TransformationEstimationPointToPoint(False),
            ransac_n=cfg.ransac_n,
            checkers=[
                reg.CorrespondenceCheckerBasedOnEdgeLength(cfg.edge_length),
                reg.CorrespondenceCheckerBasedOnDistance(dist),
            ],
            criteria=reg.RANSACConvergenceCriteria(cfg.ransac_iters, cfg.ransac_confidence),
        )

    def refine(coarse_T: np.ndarray) -> Any:
        """Point-to-plane ICP from wide to narrow, so a distant guess is still reachable.

        A single pass at the final distance cannot pull in a hypothesis
        further out than that distance - its correspondences are already
        wrong - which is why each earlier stage doubles it.
        """
        result = None
        for stage in reversed(range(max(cfg.icp_stages, 1))):
            result = reg.registration_icp(
                source.fine,
                target.fine,
                cfg.voxel_fine * cfg.icp_dist_factor * (2**stage),
                coarse_T,
                reg.TransformationEstimationPointToPlane(),
                reg.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iter),
            )
            coarse_T = result.transformation
        return result

    scored: list[Any] = []
    for _ in range(max(cfg.ransac_restarts, 1)):
        coarse_T = np.asarray(hypothesis().transformation)
        if cfg.gravity_aligned:
            coarse_T = _yaw_only(coarse_T, pivot=np.asarray(source.coarse.points).mean(axis=0))
        scored.append(refine(coarse_T))

    scored.sort(key=lambda r: r.fitness, reverse=True)
    best = scored[0]
    margin = best.fitness - scored[1].fitness if len(scored) > 1 else 0.0
    return Fix(
        transform=np.asarray(best.transformation),
        fitness=float(best.fitness),
        rmse=float(best.inlier_rmse),
        margin=float(margin),
    )


def relocalize(
    global_map: Any,
    local_map: Any,
    config: RelocalizeConfig | None = None,
    prepared: Prepared | None = None,
) -> Fix | None:
    """The fix, or ``None`` when nothing cleared ``config.fitness_threshold``.

    Refusing is a real answer and the common one for a place the premap
    never saw. Everything the decision rests on - the aligner's knobs and
    the threshold - lives on the one :class:`RelocalizeConfig`, so a caller
    configures this in a single place and just checks whether it got a fix.
    """
    cfg = config or RelocalizeConfig()
    fix = align(global_map, local_map, cfg, prepared)
    return fix if fix.fitness >= cfg.fitness_threshold else None


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
        self._prepared: Prepared | None = None
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
            fix = relocalize(
                self._premap.pointcloud,
                msg.pointcloud,
                self.config.relocalize,
                prepared=self._prepared,
            )
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
