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

"""Placing a live lidar cloud into a prior map: the aligner, and nothing else.

Coarse FPFH+RANSAC to find the place, then point-to-plane ICP over
widening-to-narrowing distances to settle it. No streams, no modules, no
clock - :mod:`dimos.mapping.relocalization.lidar.module` is the runtime
around this, and ``eval.py`` measures it.

    premap = prepare(cloud, RelocalizeConfig())
    fix = relocalize(premap, live_cloud)   # a Fix, or None if it refused
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from dimos.protocol.service.spec import BaseConfig

if TYPE_CHECKING:
    # open3d is imported lazily inside the functions that need it - it is a
    # heavy import and a module-scope one would cost every process that
    # merely touches this file. These names are for reading the signatures;
    # open3d ships no stubs, so mypy still widens them to Any.
    from open3d.geometry import PointCloud
    from open3d.pipelines.registration import Feature, RegistrationResult


class PreparedMap(NamedTuple):
    """A prior map with the preprocessing a match needs, and the config it used.

    The premap is fixed for the life of a relocalizer, so down-sampling it,
    estimating its normals and computing its FPFH features belongs outside
    the per-call path - it dominates the cost otherwise, and a live module
    would redo it on every fix.

    Carrying the config rather than taking it again per call is what makes
    that safe: the derived forms below only mean anything under the settings
    that produced them, and a second config argument at match time would be
    a way to compare a map downsampled at one voxel against a query
    downsampled at another, silently.
    """

    cloud: PointCloud  # the map itself, as handed to prepare()
    coarse: PointCloud  # downsampled at voxel_coarse, with normals
    fpfh: Feature  # FPFH descriptors of `coarse`
    fine: PointCloud  # downsampled at voxel_fine, with normals
    config: RelocalizeConfig


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


def prepare(cloud: PointCloud, config: RelocalizeConfig | None = None) -> PreparedMap:
    """A map's voxel-downsampled forms and FPFH features, ready to match against.

    Call once per map and hand the result to :func:`relocalize` for every
    query against it; for the premap that turns the pipeline's dominant cost
    into a startup cost.
    """
    import open3d as o3d  # type: ignore[import-untyped]

    cfg = config or RelocalizeConfig()
    reg = o3d.pipelines.registration

    def normals(pcd: PointCloud, voxel: float) -> PointCloud:
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
    return PreparedMap(cloud=cloud, coarse=coarse, fpfh=fpfh, fine=fine, config=cfg)


def align(premap: PreparedMap, local_map: PointCloud) -> Fix:
    """Open3D clouds in; the 4x4 T placing ``local_map`` into ``global_map`` (p_map = T @ p_local).

    Always answers, however poor the answer. :func:`relocalize` is the one
    to call - this is for a caller that wants to see a refused fix, such as
    the eval measuring how far a rejected hypothesis actually landed.

    Coarse FPFH+RANSAC to find the place, then point-to-plane ICP over
    widening-to-narrowing distances to settle it. Strategies + evals live in
    https://github.com/leshy/relocalization-test (this began as ``align_fast``).

    """
    import open3d as o3d  # type: ignore[import-untyped]

    cfg = premap.config
    reg = o3d.pipelines.registration
    source = prepare(local_map, cfg)
    dist = cfg.voxel_coarse * cfg.coarse_dist_factor

    def hypothesis() -> RegistrationResult:
        return reg.registration_ransac_based_on_feature_matching(
            source.coarse,
            premap.coarse,
            source.fpfh,
            premap.fpfh,
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

    def refine(coarse_T: np.ndarray) -> RegistrationResult:
        """Point-to-plane ICP from wide to narrow, so a distant guess is still reachable.

        A single pass at the final distance cannot pull in a hypothesis
        further out than that distance - its correspondences are already
        wrong - which is why each earlier stage doubles it.
        """
        result = None
        for stage in reversed(range(max(cfg.icp_stages, 1))):
            result = reg.registration_icp(
                source.fine,
                premap.fine,
                cfg.voxel_fine * cfg.icp_dist_factor * (2**stage),
                coarse_T,
                reg.TransformationEstimationPointToPlane(),
                reg.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iter),
            )
            coarse_T = result.transformation
        return result

    scored: list[RegistrationResult] = []
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


def relocalize(premap: PreparedMap, local_map: PointCloud) -> Fix | None:
    """The fix, or ``None`` when nothing cleared the map's ``fitness_threshold``.

    Refusing is a real answer and the common one for a place the premap
    never saw. Everything the decision rests on - the aligner's knobs and
    the threshold - travels on the map that :func:`prepare` returned, so a
    caller configures this in one place and just checks whether it got a fix.
    """
    fix = align(premap, local_map)
    return fix if fix.fitness >= premap.config.fitness_threshold else None
