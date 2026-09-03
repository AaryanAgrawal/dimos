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

"""One robust pose per tag visit: gate each glimpse, window per tag, Huber IRLS to one, publish once."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.perception.detection.type.detection3d.imageDetections3D import ImageDetections3D
from dimos.protocol.service.spec import BaseConfig
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.memory.type.observation import Observation
    from dimos.perception.detection.type.detection3d.marker import Detection3DMarker

logger = setup_logger()

Pose7 = tuple[float, float, float, float, float, float, float]  # (x, y, z, qx, qy, qz, qw): m, xyzw

HUBER_ITERATIONS = 5  # IRLS weights settle within ~5 rounds at huber_delta_m scale
# The glimpse where noise_scale == 1: close and sharp. Both terms are squared, so twice the range
# or twice the blur is four times the variance.
REF_DISTANCE_M = 0.4
REF_REPROJ_PX = 1.0
MIN_SCALE_DISTANCE_M = 0.2  # nearer than this the range term stops shrinking the variance
MIN_SCALE_REPROJ_PX = 0.5  # sharper than this the blur term stops shrinking the variance
MIN_NOISE_SCALE = 0.25  # floor, so no glimpse claims near-zero variance


class AggregationConfig(BaseConfig):
    """Per-glimpse gates and the window each tag is fused over; gate values are jnav's (#2587), not re-tuned here."""

    # deg, line of sight vs tag normal. Past this a planar tag foreshortens until IPPE's two solutions meet and the pose flips.
    max_view_angle_deg: float = 45.0
    # px RMS corner reprojection: 2x REF_REPROJ_PX, the sharp-glimpse reference.
    max_reproj_px: float = 2.0
    # px per tag side. 36h11 is 8 cells across, so 3 px per cell, the floor for a clean decode.
    min_tag_px: float = 24.0
    # A medoid over 2 has nothing to reject; 3 is the first count where an outlier is outvoted.
    min_observations: int = 3
    # m per rad in the pose distance: at REF_DISTANCE_M a 1 rad tag tilt moves the implied robot pose ~0.4 m.
    rotation_weight_m_per_rad: float = 0.5
    # m residual past which a glimpse is down-weighted: the translation scatter of in-gate glimpses at REF_DISTANCE_M.
    huber_delta_m: float = 0.05
    # s back from the newest glimpse. ~10 glimpses at the detector's 0.5 s quality window, and LIO drift over it stays cm-scale.
    time_window_s: float = 5.0


@dataclass(frozen=True)
class Glimpse:
    """One sighting of a tag; a None quality field disables that gate only."""

    ts: float
    marker_id: int
    pose: Pose7  # world_T_marker
    distance_m: float | None = None
    view_angle_deg: float | None = None
    reproj_px: float | None = None
    tag_px: float | None = None


def matrix_from_pose7(pose: tuple[float, ...]) -> np.ndarray:
    """(x, y, z, qx, qy, qz, qw) -> 4x4."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    return T


def tag_side_px(corners_px: np.ndarray) -> float:
    """Tag side in px: sqrt of the corner quad's shoelace area."""
    x, y = np.asarray(corners_px, dtype=np.float64).reshape(4, 2).T
    return math.sqrt(0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1))))


def glimpse(det: Detection3DMarker, ts: float) -> Glimpse:
    """One detection's gate inputs; range and view angle only when the camera transform rode along."""
    c, o = det.center, det.orientation
    pose: Pose7 = (c.x, c.y, c.z, o.x, o.y, o.z, o.w)
    distance_m = view_angle_deg = None
    if det.transform is not None:  # smoothing_window > 0 averages the camera transform away
        optical_T_marker = np.linalg.inv(det.transform.to_matrix()) @ matrix_from_pose7(pose)
        t = optical_T_marker[:3, 3]
        distance_m = float(np.linalg.norm(t))
        cos = abs(float(t @ optical_T_marker[:3, 2])) / (distance_m + 1e-9)  # tag normal is its z
        view_angle_deg = math.degrees(math.acos(min(1.0, cos)))
    return Glimpse(
        ts,
        det.marker_id,
        pose,
        distance_m,
        view_angle_deg,
        det.reprojection_error,
        tag_side_px(det.corners_px),
    )


def noise_scale(distance_m: float | None, reproj_px: float | None) -> float:
    """Variance inflation for one fused pose, quadratic in range x blur; None reads as the reference."""
    d = REF_DISTANCE_M if distance_m is None else max(distance_m, MIN_SCALE_DISTANCE_M)
    r = REF_REPROJ_PX if reproj_px is None else max(reproj_px, MIN_SCALE_REPROJ_PX)
    return max((d / REF_DISTANCE_M) ** 2 * (r / REF_REPROJ_PX) ** 2, MIN_NOISE_SCALE)


def gate_reason(g: Glimpse, config: AggregationConfig) -> str | None:
    """Why this glimpse is out, or None; a missing quality field skips its gate rather than failing shut."""
    if g.reproj_px is not None and g.reproj_px > config.max_reproj_px:
        return "reproj"
    if g.tag_px is not None and g.tag_px < config.min_tag_px:
        return "small"
    if g.view_angle_deg is not None and g.view_angle_deg > config.max_view_angle_deg:
        return "oblique"
    return None


def _pose_distance(a: Pose7, b: Pose7, rotation_weight_m_per_rad: float) -> float:
    """Translation plus weighted rotation between two poses, in m."""
    translation_m = float(np.linalg.norm(np.subtract(a[:3], b[:3])))
    rotation_rad = 2.0 * math.acos(min(1.0, abs(float(np.dot(a[3:], b[3:])))))
    return translation_m + rotation_weight_m_per_rad * rotation_rad


def _medoid(poses: list[Pose7], rotation_weight_m_per_rad: float) -> Pose7:
    """The pose closest to all the others: the robust seed the IRLS refines from."""
    costs = [sum(_pose_distance(p, q, rotation_weight_m_per_rad) for q in poses) for p in poses]
    return poses[int(np.argmin(costs))]


def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """1 inside delta, delta / r past it. https://en.wikipedia.org/wiki/Huber_loss"""
    return np.asarray(np.minimum(1.0, delta / np.maximum(residuals, 1e-12)))


def _huber_translation(translations: np.ndarray, seed: np.ndarray, delta_m: float) -> np.ndarray:
    """IRLS weighted mean from the seed."""
    estimate = seed.copy()
    for _ in range(HUBER_ITERATIONS):
        w = _huber_weights(np.linalg.norm(translations - estimate, axis=1), delta_m)
        estimate = (w[:, None] * translations).sum(0) / w.sum()
    return estimate


def _huber_rotation(quaternions: np.ndarray, seed: np.ndarray, delta_rad: float) -> np.ndarray:
    """IRLS Markley eigen-mean from the seed. https://ntrs.nasa.gov/citations/20070017872"""
    # q and -q are one rotation: align every sample to the seed's hemisphere or the mean cancels.
    signs = np.sign(quaternions @ seed)
    signs[signs == 0] = 1.0
    q = quaternions * signs[:, None]
    estimate = seed.copy()
    for _ in range(HUBER_ITERATIONS):
        w = _huber_weights(2.0 * np.arccos(np.clip(np.abs(q @ estimate), 0.0, 1.0)), delta_rad)
        scatter = (w[:, None, None] * np.einsum("ni,nj->nij", q, q)).sum(0)
        estimate = np.linalg.eigh(scatter)[1][:, -1]
        if estimate @ seed < 0:
            estimate = -estimate
    return estimate


def robust_pose(cluster: list[Glimpse], config: AggregationConfig) -> Pose7:
    """One pose for a cluster: the medoid, refined by Huber IRLS on translation and rotation."""
    poses = [g.pose for g in cluster]
    seed = _medoid(poses, config.rotation_weight_m_per_rad)
    if len(poses) < 2:
        return seed
    arr = np.array(poses, dtype=np.float64)
    t = _huber_translation(arr[:, :3], np.array(seed[:3]), config.huber_delta_m)
    # delta in rad, so the rotation residual shares the translation delta's robustness scale
    delta_rad = config.huber_delta_m / max(config.rotation_weight_m_per_rad, 1e-9)
    q = _huber_rotation(arr[:, 3:], np.array(seed[3:]), delta_rad)
    return (*t.tolist(), *q.tolist())  # type: ignore[return-value]


def _median(values: list[float | None]) -> float | None:
    """Median over the members carrying the field; median, so one in-gate outlier cannot inflate the health signal."""
    present = [v for v in values if v is not None]
    return float(np.median(present)) if present else None


class TagAggregator:
    """The detector's tap: gate each glimpse, window per tag, publish one fused pose per visit."""

    def __init__(
        self,
        publish: Callable[[Detection3DArray], Any],
        config: AggregationConfig,
        world_frame: str = "world",
    ) -> None:
        self._publish = publish
        self._config = config
        self._world_frame = world_frame
        self._window: dict[int, list[Glimpse]] = defaultdict(list)
        self._published: set[int] = (
            set()
        )  # tags whose current visit already published; re-armed when the window thins

    def observe(self, g: Glimpse) -> str | None:
        """Keep the glimpse in its tag's window, or return why it was gated out."""
        reason = gate_reason(g, self._config)
        if reason is not None:
            return reason
        window = self._window[g.marker_id]
        window.append(g)
        # Purge against this tag's newest glimpse, not wall time, so a tag out of view keeps its window.
        self._window[g.marker_id] = [o for o in window if g.ts - o.ts <= self._config.time_window_s]
        return None

    def fuse(self, marker_id: int) -> tuple[Pose7, float, int] | None:
        """(pose, score, n) for a tag's window, or None under min_observations."""
        window = self._window.get(marker_id, [])
        if len(window) < self._config.min_observations:
            return None
        scale = noise_scale(
            _median([g.distance_m for g in window]), _median([g.reproj_px for g in window])
        )
        return robust_pose(window, self._config), min(1.0, 1.0 / scale), len(window)

    def __call__(self, obs: Observation[Detection3DMarker | None]) -> None:
        det = obs.data
        if det is None or self.observe(glimpse(det, obs.ts)) is not None:
            return  # a tag-free frame (DetectMarkers emit_empty_frames) or a gated glimpse
        fused = self.fuse(det.marker_id)
        if fused is None:
            self._published.discard(
                det.marker_id
            )  # window thinned: the next full window is a new visit
            return
        if det.marker_id in self._published:
            return  # same visit: publishing per frame would fire ~2x/s
        self._published.add(det.marker_id)
        self._publish(self._message(det, *fused))

    def _message(
        self, det: Detection3DMarker, pose: Pose7, score: float, n: int
    ) -> Detection3DArray:
        """One-detection array carrying the fused world_T_marker, scored."""
        x, y, z, qx, qy, qz, qw = pose
        fused = replace(
            det,
            center=Vector3(x, y, z),
            orientation=Quaternion(qx, qy, qz, qw),
            transform=None,  # fused across frames: no one camera transform holds
            confidence=score,  # -> hypothesis.score, 0-1
        )
        logger.debug("tag fused", marker_id=det.marker_id, n=n, score=round(score, 2))
        return ImageDetections3D(det.image, [fused]).to_ros_detection3d_array(
            frame_id=self._world_frame
        )
