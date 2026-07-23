# Copyright 2025-2026 Dimensional Inc.
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

"""Interactive camera calibration for dimos (ROS CameraInfo YAML output)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, TypedDict, cast
import warnings

# Default OpenCL off: on Apple Silicon, CPU chessboard detection is often faster and more stable here.
# Use setdefault so an explicit OPENCV_OPENCL_RUNTIME from the environment still wins.
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import numpy as np
import typer
import yaml

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg"})

_DEFAULT_CHECK_RMS_THRESHOLD_PX = 1.0  # median reprojection RMS a deployed calib must stay at/below
_DEFAULT_CHECK_DRIFT_THRESHOLD_FRAC = (
    0.05  # relative intrinsics drift (deployed vs fresh) called large
)
# A planar chessboard yields one homography per view; >=3 views are needed to separate fx/fy/cx/cy
# plus distortion in a fresh cv2.calibrateCamera. Fisheye may need more -- handled by cv2.error below.
_MIN_DRIFT_CALIBRATION_FRAMES = 3

# ChArUco interpolates one corner per interior square-square boundary; a partial view still
# calibrates, but solvePnP needs >= 4, and calibrateCameraCharuco / cv2.fisheye.calibrate are
# ill-conditioned on very few. Require 6 so the per-view pose and the drift solve stay well-posed.
_MIN_CHARUCO_CORNERS = 6
_DEFAULT_CHARUCO_DICT = "DICT_4X4_50"
_DEFAULT_MARKER_RATIO = 0.8  # markerLength / squareLength for a standard ChArUco print


class CalibrationResultDict(TypedDict):
    """Structured return from ``calibrate_from_frames`` (and base of ``run_calibration``)."""

    K: np.ndarray
    D: np.ndarray
    rms: float
    image_size: tuple[int, int]
    n_used: int
    pattern_size: tuple[int, int]
    pattern_label: str


class CalibrationRunResultDict(CalibrationResultDict, total=False):
    """Optional paths written when ``run_calibration`` is asked to emit files."""

    out_path: Path
    preview_path: Path


class CheckDriftDict(TypedDict, total=False):
    """Fresh-calibration drift block from ``check_calibration``.

    ``ran``/``reason`` are always set; the numeric fields exist only when ``ran`` is True.
    Deltas are ``deployed - fresh`` in pixels; ``max_rel_intrinsics_drift`` is the largest of
    the four per-parameter relative drifts (each ``|delta| / fresh``).
    """

    ran: bool
    reason: str
    fresh_rms_px: float
    delta_fx_px: float
    delta_fy_px: float
    delta_cx_px: float
    delta_cy_px: float
    max_rel_intrinsics_drift: float
    distortion_abs_delta: list[float]
    drift_small: bool
    # Fresh intrinsics as JSON-serializable lists (3x3 row-major K, 1-d D) so the same solve that
    # produced the drift numbers can be re-emitted as a CameraInfo YAML without re-calibrating.
    fresh_K: list[list[float]]
    fresh_D: list[float]


class CalibrationCheckResultDict(TypedDict):
    """Structured return from ``check_calibration``.

    ``verdict`` is ``"OK"`` or ``"DEGRADED"``. ``median_reproj_rms_px``/``p90_reproj_rms_px``
    aggregate the per-view reprojection RMS of the detected corners under the DEPLOYED intrinsics.
    """

    verdict: str
    median_reproj_rms_px: float
    p90_reproj_rms_px: float
    n_frames_used: int
    rms_threshold_px: float
    per_view_rms_px: list[float]
    distortion_model: str
    image_size_wh: tuple[int, int]
    cols: int
    rows: int
    square_size_m: float
    drift: CheckDriftDict
    recommendation: str


class CalibrationCheckRunResultDict(CalibrationCheckResultDict, total=False):
    """Provenance + output-path fields added by ``run_check`` on top of the core result."""

    camera_info_path: str
    source: str
    resolution_warning: str
    out_path: Path


class Source(str, Enum):
    """Frame source supported by the calibration CLI."""

    webcam = "webcam"
    folder = "folder"
    topic = "topic"


class BoardType(str, Enum):
    """Calibration target. ``charuco`` (cv2.aruco) tolerates partial occlusion -- each ArUco marker
    identifies its corner -- where a plain ``chessboard`` needs the whole board visible.
    """

    chessboard = "chessboard"
    charuco = "charuco"


class DistortionModel(str, Enum):
    """Lens model. ``plumb_bob`` = ``cv2.calibrateCamera`` (5-coeff radial-tangential, near-pinhole);
    ``fisheye`` = ``cv2.fisheye.calibrate`` (4-coeff Kannala-Brandt) for wide-angle lenses like the
    Go2 front camera, written to YAML under the ROS-canonical name ``equidistant``.
    """

    plumb_bob = "plumb_bob"
    fisheye = "fisheye"

    def to_ros_name(self) -> str:
        return "equidistant" if self is DistortionModel.fisheye else self.value


app = typer.Typer(
    help="Calibrate camera intrinsics and write ROS CameraInfo YAML.",
    no_args_is_help=True,
)


def write_camera_info_yaml(
    path: str,
    *,
    image_width: int,
    image_height: int,
    camera_name: str,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray | None = None,
    P: np.ndarray | None = None,
    distortion_model: str = "plumb_bob",
) -> None:
    """Write ROS-style CameraInfo YAML loadable by dimos CameraInfo helpers.

    The emitted schema is accepted by ``CameraInfo.from_yaml``,
    ``load_camera_info``, and ``load_camera_info_opencv``.
    """
    k = np.asarray(K, dtype=np.float64).reshape(3, 3)
    d = np.asarray(D, dtype=np.float64).ravel()
    k_flat = k.ravel(order="C").tolist()
    d_flat = d.tolist()

    if R is None:
        r_flat = np.eye(3, dtype=np.float64).ravel(order="C").tolist()
    else:
        r_flat = np.asarray(R, dtype=np.float64).reshape(3, 3).ravel(order="C").tolist()

    if P is None:
        fx = k_flat[0]
        fy = k_flat[4]
        cx = k_flat[2]
        cy = k_flat[5]
        p_flat = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    else:
        p_flat = np.asarray(P, dtype=np.float64).reshape(3, 4).ravel(order="C").tolist()

    n_dist = len(d_flat)
    payload = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_name": camera_name,
        "distortion_model": distortion_model,
        "camera_matrix": {"rows": 3, "cols": 3, "data": k_flat},
        "distortion_coefficients": {"rows": 1, "cols": int(n_dist), "data": d_flat},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": r_flat},
        "projection_matrix": {"rows": 3, "cols": 4, "data": p_flat},
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def load_frames_from_folder(path: str) -> list[np.ndarray]:
    """Load ``*.png``, ``*.jpg``, and ``*.jpeg`` images from a directory.

    Files are ordered by filename (lexicographic sort of basenames). Raises if the path
    is not a directory or if any matching file fails to decode with ``cv2.imread``.
    """
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {path}")

    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    out: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Could not read image: {p}")
        out.append(img)
    return out


_CAMERACALIBRATE_WINDOW = "dimos cameracalibrate"
_MAX_CONSECUTIVE_WEBCAM_READ_FAILURES = 30
_OPENCV_CALIBRATION_RUNTIME_CONFIGURED = False


@dataclass(frozen=True)
class _ChessboardDetection:
    corners: np.ndarray
    cols: int
    rows: int
    label: str


@dataclass(frozen=True)
class _WebcamCapture:
    frames: list[np.ndarray]
    image_points: list[np.ndarray]
    pattern: tuple[int, int, str] | None
    # Populated only on a ChArUco capture; None for the checkerboard path.
    charuco_detections: list[_CharucoDetection] | None = None


@dataclass(frozen=True)
class CharucoBoardSpec:
    """A built ChArUco board plus the geometry needed to report it.

    ``board`` is a ``cv2.aruco.CharucoBoard`` built with lengths in METERS, so
    ``board.getChessboardCorners()`` returns object points already in meters. ``squares_x`` /
    ``squares_y`` are square counts (not inner corners); the board yields
    ``(squares_x - 1) * (squares_y - 1)`` interior ChArUco corners.
    """

    board: Any  # cv2.aruco.CharucoBoard (no stub type is exported by cv2)
    dict_name: str
    squares_x: int
    squares_y: int
    square_size_m: float
    marker_size_m: float

    @cached_property
    def detector(self) -> Any:  # cv2.aruco.CharucoDetector (no cv2 stub)
        """One detector per board, reused across every frame (constructing per frame is wasteful)."""
        return cv2.aruco.CharucoDetector(self.board)


@dataclass(frozen=True)
class _CharucoDetection:
    """One frame's ChArUco detection: interpolated corners + ids + matching board object points.

    ``charuco_corners_px`` is ``(N, 1, 2)`` float32 pixel corners, ``charuco_ids`` is ``(N, 1)``
    int32, and ``object_points_m`` is ``(N, 3)`` float32 board points (meters) selected by id from
    ``board.getChessboardCorners()``. ``N`` varies per view (partial boards are fine).
    """

    charuco_corners_px: np.ndarray
    charuco_ids: np.ndarray
    object_points_m: np.ndarray


def build_charuco_board(
    *,
    dict_name: str,
    squares_x: int,
    squares_y: int,
    square_size_m: float,
    marker_size_m: float | None = None,
    marker_ratio: float = _DEFAULT_MARKER_RATIO,
) -> CharucoBoardSpec:
    """Build a ``cv2.aruco.CharucoBoard`` from a predefined dictionary name (cv2.aruco / OpenCV ArUco).

    ``marker_size_m`` wins if given; otherwise it is ``square_size_m * marker_ratio``. Lengths are
    in meters so the resulting object points are metric. Raises ``ValueError`` (got-vs-want) on a
    bad dictionary name or a non-physical marker/square ratio.
    """
    if squares_x < 2 or squares_y < 2:
        raise ValueError(f"charuco needs squares_x/squares_y >= 2, got {squares_x}x{squares_y}")
    if square_size_m <= 0:
        raise ValueError(f"square_size_m must be > 0, got {square_size_m}")

    dict_id = getattr(cv2.aruco, dict_name, None)
    if not isinstance(dict_id, int):
        raise ValueError(
            f"unknown aruco dictionary {dict_name!r}; want a cv2.aruco.DICT_* name "
            "(e.g. DICT_4X4_50, DICT_5X5_100, DICT_APRILTAG_36h11)"
        )
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    resolved_marker_m = (
        float(marker_size_m) if marker_size_m is not None else float(square_size_m) * marker_ratio
    )
    if not 0.0 < resolved_marker_m < float(square_size_m):
        raise ValueError(
            f"marker size must satisfy 0 < marker < square; got marker={resolved_marker_m} m, "
            f"square={square_size_m} m (check --marker-size-m / --marker-ratio)"
        )

    board = cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(square_size_m),
        resolved_marker_m,
        dictionary,
    )
    return CharucoBoardSpec(
        board=board,
        dict_name=dict_name,
        squares_x=int(squares_x),
        squares_y=int(squares_y),
        square_size_m=float(square_size_m),
        marker_size_m=resolved_marker_m,
    )


def _detect_charuco(gray: np.ndarray, spec: CharucoBoardSpec) -> _CharucoDetection | None:
    """Detect + interpolate ChArUco corners in one frame (cv2.aruco / OpenCV ArUco).

    Uses ``cv2.aruco.CharucoDetector.detectBoard`` (cached on the spec). Returns ``None`` when
    fewer than ``_MIN_CHARUCO_CORNERS`` corners are recovered. Object points come straight from
    ``board.getChessboardCorners()[ids]`` -- no re-implemented marker or corner math.
    """
    g = _as_grayscale_uint8(gray)
    board = spec.board

    detect_board = cast("Callable[..., tuple[Any, Any, Any, Any]]", spec.detector.detectBoard)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detect_board(g)

    if charuco_ids is None or charuco_corners is None or len(charuco_ids) < _MIN_CHARUCO_CORNERS:
        return None

    ids_flat = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    all_object_points_m = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    object_points_m = all_object_points_m[ids_flat]
    corners_px = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    return _CharucoDetection(corners_px, ids_flat.reshape(-1, 1), object_points_m)


def _charuco_detections_from_frames(
    frames: list[np.ndarray], spec: CharucoBoardSpec
) -> list[_CharucoDetection]:
    """Detect ChArUco corners in each frame (skipping frames with too few corners)."""
    out: list[_CharucoDetection] = []
    for frame in frames:
        f = np.asarray(frame)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        detection = _detect_charuco(gray, spec)
        if detection is not None:
            out.append(detection)
    return out


def _calibrate_charuco(
    charuco_corners_per_view: list[np.ndarray],
    charuco_ids_per_view: list[np.ndarray],
    board: Any,
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run ``cv2.aruco.calibrateCameraCharuco`` (plumb-bob) over per-view ChArUco corners.

    Returns ``(rms, K, D)``; ``K`` is 3x3 and ``D`` is 1-d, same shape as ``_calibrate_pinhole``.
    """
    _calibrate = cast("Callable[..., Any]", cv2.aruco.calibrateCameraCharuco)
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = _calibrate(
        charuco_corners_per_view,
        charuco_ids_per_view,
        board,
        image_size,
        None,
        None,
    )
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    return float(rms), K, D


def _calibrate_charuco_fisheye(
    detections: list[_CharucoDetection],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fisheye (equidistant) intrinsics from ChArUco per-view correspondences.

    Per view the correspondence is object points = ``board.getChessboardCorners()[ids]`` (meters,
    already selected in ``_detect_charuco``) against the detected ``charucoCorners``. The solve
    reuses the same ``cv2.fisheye.calibrate`` machinery as the chessboard fisheye path.
    """
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    for det in detections:
        # cv2.fisheye.calibrate is strict: float64, explicit middle axis (N,1,3)/(N,1,2) per view.
        objp = np.asarray(det.object_points_m, dtype=np.float64).reshape(-1, 1, 3)
        imgp = np.asarray(det.charuco_corners_px, dtype=np.float64).reshape(-1, 1, 2)
        if objp.shape[0] < _MIN_CHARUCO_CORNERS:
            continue
        objpoints.append(objp)
        imgpoints.append(imgp)

    if not objpoints:
        raise ValueError(
            f"fisheye charuco calibration wants views with >= {_MIN_CHARUCO_CORNERS} "
            f"corners; got {len(detections)} detected view(s), none with enough corners."
        )

    try:
        # Boundary: convergence failure surfaces as cv2.error; raise loudly with got-vs-want.
        return _calibrate_fisheye(objpoints, imgpoints, image_size)
    except cv2.error as exc:
        raise ValueError(
            f"cv2.fisheye.calibrate did not converge on {len(objpoints)} charuco view(s); "
            f"want more views with wider, corner-covering board spread (OpenCV: {exc})"
        ) from exc


def calibrate_from_frames_charuco(
    frames: list[np.ndarray],
    spec: CharucoBoardSpec,
    *,
    distortion_model: DistortionModel | str = DistortionModel.plumb_bob,
) -> CalibrationResultDict:
    """Calibrate intrinsics from frames showing a ChArUco board (cv2.aruco).

    Each frame where ``_detect_charuco`` recovers >= ``_MIN_CHARUCO_CORNERS`` corners contributes
    one view. All frames must share the same resolution. ``distortion_model`` picks the solver:
    ``plumb_bob`` (``cv2.aruco.calibrateCameraCharuco``, 5 coeffs) or ``fisheye``
    (``cv2.fisheye.calibrate``, 4-coeff Kannala-Brandt) for genuine fisheye lenses like the Go2
    front camera. Returns the same ``CalibrationResultDict`` as ``calibrate_from_frames``.
    """
    if not frames:
        raise ValueError("frames must be non-empty")

    model = DistortionModel(distortion_model)

    h0, w0 = np.asarray(frames[0]).shape[:2]
    if any(np.asarray(f).shape[:2] != (h0, w0) for f in frames):
        raise ValueError("All frames must have the same shape.")

    detections = _charuco_detections_from_frames(frames, spec)
    if not detections:
        raise ValueError("ChArUco board not found in any frame.")

    if model is DistortionModel.fisheye:
        rms, K, D = _calibrate_charuco_fisheye(detections, (w0, h0))
    else:
        rms, K, D = _calibrate_charuco(
            [d.charuco_corners_px for d in detections],
            [d.charuco_ids for d in detections],
            spec.board,
            (w0, h0),
        )
    return {
        "K": K,
        "D": D,
        "rms": float(rms),
        "image_size": (int(w0), int(h0)),
        "n_used": len(detections),
        "pattern_size": (spec.squares_x, spec.squares_y),
        "pattern_label": f"charuco {spec.dict_name} {spec.squares_x}x{spec.squares_y} squares",
    }


def _pattern_candidates(cols: int, rows: int) -> list[tuple[int, int, str]]:
    """Return plausible inner-corner pattern sizes, exact request first."""
    # Board size may be given as inner corners or as square counts; also try swapped axes (portrait).
    candidates = [
        (cols, rows, "requested inner corners"),
        (rows, cols, "requested inner corners, rotated"),
    ]
    if cols > 1 and rows > 1:
        candidates.extend(
            [
                (cols - 1, rows - 1, "requested square count"),
                (rows - 1, cols - 1, "requested square count, rotated"),
            ]
        )

    out: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for cand_cols, cand_rows, label in candidates:
        if cand_cols < 1 or cand_rows < 1:
            continue
        key = (cand_cols, cand_rows)
        if key in seen:
            continue
        seen.add(key)
        out.append((cand_cols, cand_rows, label))
    return out


def _as_grayscale_uint8(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        # Corner finders expect uint8 range; normalize wider dtypes before detection.
        out_norm = np.empty(g.shape, dtype=np.uint8)
        g = cv2.normalize(g, out_norm, 0, 255, cv2.NORM_MINMAX)
    return np.ascontiguousarray(g)


def _configure_opencv_calibration_runtime() -> None:
    global _OPENCV_CALIBRATION_RUNTIME_CONFIGURED
    if _OPENCV_CALIBRATION_RUNTIME_CONFIGURED:
        return

    # Process-global OpenCV settings; run once before any chessboard detection in this module.
    cv2.setUseOptimized(True)
    if hasattr(cv2, "ocl"):
        try:
            cv2.ocl.setUseOpenCL(False)
        except cv2.error:
            pass

    _OPENCV_CALIBRATION_RUNTIME_CONFIGURED = True


def _find_chessboard_corners_sb(
    gray: np.ndarray,
    cols: int,
    rows: int,
    *,
    exhaustive: bool,
) -> np.ndarray | None:
    _configure_opencv_calibration_runtime()
    find_sb = getattr(cv2, "findChessboardCornersSB", None)
    if find_sb is None:
        return None

    g = _as_grayscale_uint8(gray)
    pattern_size = (cols, rows)
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    if exhaustive:
        # CALIB_CB_EXHAUSTIVE and CALIB_CB_ACCURACY: higher CPU, better recall on difficult frames.
        # Live preview passes exhaustive=False; exhaustive=True is reserved for the offline fallback path.
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            sb_flags |= cv2.CALIB_CB_ACCURACY

    ok, corners = cast("Callable[..., tuple[bool, Any]]", find_sb)(g, pattern_size, sb_flags)
    if not ok or corners is None:
        return None
    return np.asarray(corners, dtype=np.float32).reshape(cols * rows, 1, 2)


def _find_chessboard_corners_realtime(
    gray: np.ndarray,
    cols: int,
    rows: int,
) -> np.ndarray | None:
    # Preview path: SB without exhaustive flags so most misses stay cheap (latency budget).
    corners = _find_chessboard_corners_sb(gray, cols, rows, exhaustive=False)
    if corners is not None:
        return corners

    if getattr(cv2, "findChessboardCornersSB", None) is not None:
        return None

    return _find_chessboard_corners_exact(gray, cols, rows)


def _find_chessboard_corners_exact(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    g = _as_grayscale_uint8(gray)
    pattern_size = (cols, rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    g_cv = cast("Any", np.ascontiguousarray(g))
    _find_corners = cast("Callable[..., tuple[bool, Any]]", cv2.findChessboardCorners)
    ok, corners = _find_corners(g_cv, pattern_size, flags)
    if ok and corners is not None:
        # Classic detector already gave pixel locations; refine to sub-pixel accuracy.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined = cv2.cornerSubPix(g_cv, corners, (11, 11), (-1, -1), criteria)
        return cast("np.ndarray", refined)

    # Classic detector missed: try SB again with exhaustive flags (slower, higher recall).
    return _find_chessboard_corners_sb(g_cv, cols, rows, exhaustive=True)


def _find_chessboard_detection(
    gray: np.ndarray,
    cols: int,
    rows: int,
    *,
    realtime: bool = False,
    candidates: list[tuple[int, int, str]] | None = None,
) -> _ChessboardDetection | None:
    detector = _find_chessboard_corners_realtime if realtime else _find_chessboard_corners_exact
    for cand_cols, cand_rows, label in candidates or _pattern_candidates(cols, rows):
        corners = detector(gray, cand_cols, cand_rows)
        if corners is not None:
            return _ChessboardDetection(corners, cand_cols, cand_rows, label)
    return None


def _draw_capture_status(
    preview: np.ndarray,
    *,
    detection: _ChessboardDetection | None,
    accepted_count: int,
    target_count: int,
) -> None:
    status = f"Accepted {accepted_count}/{target_count}"
    if detection is None:
        detail = "No chessboard detected - SPACE ignored"
        color = (0, 0, 255)
    else:
        detail = f"Detected {detection.cols}x{detection.rows} ({detection.label}) - SPACE saves"
        color = (0, 180, 0)

    cv2.rectangle(preview, (0, 0), (preview.shape[1], 58), (0, 0, 0), thickness=-1)
    cv2.putText(preview, status, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(preview, detail, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# --- Coverage overlay (interactive-capture visual hint) -----------------------
# The live preview is binned into a GRID x GRID grid; a cell clears once the detected board has
# swept over it, so the still-drawn cells show which image regions the board has not reached yet.
# A visual hint only -- SPACE captures whatever is in view, no gating.
_COVERAGE_GRID = 6  # image binned into GRID x GRID cells for the coverage overlay


def _touched_cells(corners_px: np.ndarray, image_wh: tuple[int, int]) -> frozenset[tuple[int, int]]:
    """Grid cells (each axis 0..GRID-1) that contain at least one board corner."""
    grid = _COVERAGE_GRID
    w = max(int(image_wh[0]), 1)
    h = max(int(image_wh[1]), 1)
    pts = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    cx = np.clip((pts[:, 0] / w * grid).astype(np.int64), 0, grid - 1)
    cy = np.clip((pts[:, 1] / h * grid).astype(np.int64), 0, grid - 1)
    return frozenset(zip(cx.tolist(), cy.tolist(), strict=True))


def _draw_coverage_overlay(preview: np.ndarray, covered_cells: set[tuple[int, int]]) -> None:
    """Outline every still-uncovered GRID x GRID cell on ``preview``; covered cells are omitted.

    A light capture hint: each cell the detected board has not yet swept over is drawn as a thin
    rectangle, so the operator sees at a glance which image regions still need the board. Cells in
    ``covered_cells`` are left clear ("deleted"). Draw-only -- it never gates SPACE capture.
    """
    grid = _COVERAGE_GRID
    h, w = preview.shape[:2]
    for cx in range(grid):
        for cy in range(grid):
            if (cx, cy) in covered_cells:
                continue
            x0 = round(cx * w / grid)
            y0 = round(cy * h / grid)
            x1 = round((cx + 1) * w / grid)
            y1 = round((cy + 1) * h / grid)
            cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 200, 255), thickness=1)


def _interactive_capture(
    next_frame: Callable[[], np.ndarray | None],
    target_count: int,
    cols: int,
    rows: int,
    *,
    no_display: bool,
) -> _WebcamCapture:
    """Interactive chessboard preview + SPACE-accept / q-quit loop.

    ``next_frame()`` returns the latest BGR/grayscale frame, or ``None`` to skip the iteration.
    ``no_display`` skips ``imshow``/teardown but still calls ``cv2.waitKey`` so tests can inject
    keys. A light ``_draw_coverage_overlay`` marks image regions the board has not yet swept over;
    it is a visual hint only -- SPACE accepts whatever board is in view.
    """
    if target_count < 1:
        raise ValueError("target_count must be >= 1")

    accepted: list[np.ndarray] = []
    accepted_corners: list[np.ndarray] = []
    last_detected: tuple[int, int, str] | None = None
    locked_pattern: tuple[int, int, str] | None = None
    locked_exact_probe = False
    pattern_candidates = _pattern_candidates(cols, rows)
    pattern_candidate_index = 0
    covered_cells: set[tuple[int, int]] = set()

    try:
        while len(accepted) < target_count:
            frame = next_frame()
            if frame is None:
                continue

            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            if locked_pattern is None:
                candidate = pattern_candidates[pattern_candidate_index]
                pattern_candidate_index = (pattern_candidate_index + 1) % len(pattern_candidates)
                # Until pattern lock: one candidate per frame so we do not evaluate every pattern every frame.
                detection = _find_chessboard_detection(
                    gray,
                    cols,
                    rows,
                    realtime=True,
                    candidates=[candidate],
                )
                if detection is not None:
                    locked_pattern = (detection.cols, detection.rows, detection.label)
                    locked_exact_probe = True
            else:
                locked_cols, locked_rows, locked_label = locked_pattern
                if locked_exact_probe:
                    # Locked board: prefer full detector; on miss, drop to realtime until corners reappear.
                    corners = find_chessboard_corners(gray, locked_cols, locked_rows)
                    if corners is None:
                        locked_exact_probe = False
                else:
                    corners = _find_chessboard_corners_realtime(gray, locked_cols, locked_rows)
                    if corners is not None:
                        locked_exact_probe = True
                detection = (
                    _ChessboardDetection(corners, locked_cols, locked_rows, locked_label)
                    if corners is not None
                    else None
                )
            preview = np.asarray(frame).copy()
            if detection is not None:
                detected = (detection.cols, detection.rows, detection.label)
                if detected != last_detected:
                    last_detected = detected
                covered_cells |= _touched_cells(
                    detection.corners, (preview.shape[1], preview.shape[0])
                )
                cv2.drawChessboardCorners(
                    preview,
                    (detection.cols, detection.rows),
                    detection.corners,
                    True,
                )
            _draw_capture_status(
                preview,
                detection=detection,
                accepted_count=len(accepted),
                target_count=target_count,
            )

            if not no_display:
                _draw_coverage_overlay(preview, covered_cells)
                cv2.imshow(_CAMERACALIBRATE_WINDOW, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and detection is not None:
                accepted.append(np.asarray(frame).copy())
                accepted_corners.append(np.asarray(detection.corners, dtype=np.float32).copy())
            elif key == ord("q"):
                break

        if len(accepted) < target_count:
            raise RuntimeError(
                f"Capture ended with {len(accepted)} of {target_count} frames "
                f"(quit early, missing detections on SPACE, or read failures)."
            )

        return _WebcamCapture(accepted, accepted_corners, locked_pattern)

    finally:
        if not no_display:
            try:
                cv2.destroyWindow(_CAMERACALIBRATE_WINDOW)
            except cv2.error:
                pass
            cv2.waitKey(1)


def _interactive_capture_charuco(
    next_frame: Callable[[], np.ndarray | None],
    target_count: int,
    spec: CharucoBoardSpec,
    *,
    no_display: bool,
) -> _WebcamCapture:
    """ChArUco variant of ``_interactive_capture``: same SPACE-accept / q-quit UX, charuco detector.

    Detection is per-frame ``_detect_charuco`` (the board fixes the geometry, so no pattern-size
    guessing). Accepted frames store their ``_CharucoDetection`` for reuse at accept time. The same
    light ``_draw_coverage_overlay`` hint keys off the ChArUco corner pixels.
    """
    if target_count < 1:
        raise ValueError("target_count must be >= 1")

    accepted: list[np.ndarray] = []
    accepted_detections: list[_CharucoDetection] = []
    covered_cells: set[tuple[int, int]] = set()

    try:
        while len(accepted) < target_count:
            frame = next_frame()
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            detection = _detect_charuco(gray, spec)

            preview = np.asarray(frame).copy()
            if detection is not None:
                # drawDetectedCornersCharuco expects a BGR image; upgrade grayscale previews.
                if preview.ndim == 2:
                    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                covered_cells |= _touched_cells(
                    detection.charuco_corners_px, (preview.shape[1], preview.shape[0])
                )
                cv2.aruco.drawDetectedCornersCharuco(
                    preview, detection.charuco_corners_px, detection.charuco_ids
                )
            n_corners = 0 if detection is None else int(detection.charuco_ids.shape[0])
            status = f"Accepted {len(accepted)}/{target_count}"
            if detection is None:
                detail = "No ChArUco board detected - SPACE ignored"
                color = (0, 0, 255)
            else:
                detail = f"Detected {n_corners} charuco corners - SPACE saves"
                color = (0, 180, 0)
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 58), (0, 0, 0), thickness=-1)
            cv2.putText(
                preview, status, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(preview, detail, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if not no_display:
                _draw_coverage_overlay(preview, covered_cells)
                cv2.imshow(_CAMERACALIBRATE_WINDOW, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and detection is not None:
                accepted.append(np.asarray(frame).copy())
                accepted_detections.append(detection)
            elif key == ord("q"):
                break

        if len(accepted) < target_count:
            raise RuntimeError(
                f"Capture ended with {len(accepted)} of {target_count} frames "
                f"(quit early, missing detections on SPACE, or read failures)."
            )

        return _WebcamCapture(accepted, [], None, charuco_detections=accepted_detections)

    finally:
        if not no_display:
            try:
                cv2.destroyWindow(_CAMERACALIBRATE_WINDOW)
            except cv2.error:
                pass
            cv2.waitKey(1)


def _capture_frames_from_webcam(
    device_index: int,
    target_count: int,
    cols: int,
    rows: int,
    *,
    no_display: bool = False,
    charuco_spec: CharucoBoardSpec | None = None,
) -> _WebcamCapture:
    """Open the webcam and drive ``_interactive_capture`` (or the charuco variant) over its frames.

    Raises if the device cannot be opened or fails to read ``_MAX_CONSECUTIVE_WEBCAM_READ_FAILURES``
    times; the SPACE-accept / q-quit UX is that of ``_interactive_capture``.
    """
    if target_count < 1:
        raise ValueError("target_count must be >= 1")

    cap: cv2.VideoCapture | None = None
    consecutive_read_failures = 0

    try:
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera device_index={device_index!r}")

        def _next() -> np.ndarray | None:
            nonlocal consecutive_read_failures
            assert cap is not None  # narrow for type-checker
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= _MAX_CONSECUTIVE_WEBCAM_READ_FAILURES:
                    raise RuntimeError(
                        "Failed to read from camera "
                        f"device_index={device_index!r} for "
                        f"{_MAX_CONSECUTIVE_WEBCAM_READ_FAILURES} consecutive attempts."
                    )
                return None
            consecutive_read_failures = 0
            return frame  # type: ignore[no-any-return]

        if charuco_spec is not None:
            return _interactive_capture_charuco(
                _next,
                target_count,
                charuco_spec,
                no_display=no_display,
            )
        return _interactive_capture(
            _next,
            target_count,
            cols,
            rows,
            no_display=no_display,
        )

    finally:
        if cap is not None:
            cap.release()


def capture_frames_from_webcam(
    device_index: int,
    target_count: int,
    cols: int,
    rows: int,
    *,
    no_display: bool = False,
) -> list[np.ndarray]:
    """Capture ``target_count`` BGR frames from a webcam when the board is visible."""
    return _capture_frames_from_webcam(
        device_index,
        target_count,
        cols,
        rows,
        no_display=no_display,
    ).frames


def _capture_frames_from_topic(
    topic_uri: str,
    target_count: int,
    cols: int,
    rows: int,
    *,
    no_display: bool = False,
    timeout_sec: float = 60.0,
    charuco_spec: CharucoBoardSpec | None = None,
) -> _WebcamCapture:
    """Capture frames from an LCM/SHM image topic with the same interactive UX.

    ``topic_uri`` follows the pubsub registry format ``"<proto>:<topic>"``, e.g.
    ``"jpeg_lcm:/color_image"`` or ``"pshm:color_image"``. The publisher must
    emit ``sensor_msgs.Image`` messages; ``Image.to_opencv()`` normalizes the
    payload to BGR before detection. Raises ``RuntimeError`` if no frames arrive
    within ``timeout_sec``.
    """
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.protocol.pubsub.registry import subscribe_pubsub_uri

    if target_count < 1:
        raise ValueError("target_count must be >= 1")

    latest_frame: list[np.ndarray | None] = [None]
    last_received_ts: list[float] = [time.time()]
    lock = threading.Lock()

    def _on_image(msg: Any) -> None:
        try:
            arr = msg.to_opencv()
        except (AttributeError, ValueError):
            return
        with lock:
            latest_frame[0] = np.asarray(arr)
            last_received_ts[0] = time.time()

    transport, unsub = subscribe_pubsub_uri(topic_uri, _on_image, msg_type=Image)

    def _next() -> np.ndarray | None:
        with lock:
            frame = latest_frame[0]
            ts = last_received_ts[0]
        if frame is None:
            if time.time() - ts > timeout_sec:
                raise RuntimeError(
                    f"No frames received on topic {topic_uri!r} within {timeout_sec:.1f}s."
                )
            # Yield so the LCM/SHM callback thread can run; avoid busy spin.
            time.sleep(0.01)
            return None
        return frame

    try:
        if charuco_spec is not None:
            return _interactive_capture_charuco(
                _next,
                target_count,
                charuco_spec,
                no_display=no_display,
            )
        return _interactive_capture(
            _next,
            target_count,
            cols,
            rows,
            no_display=no_display,
        )
    finally:
        # Best-effort teardown: swallow per-transport quirks so cleanup
        # never masks the original error from _interactive_capture.
        try:
            unsub()
        except Exception:
            pass
        try:
            transport.stop()
        except Exception:
            pass


def capture_frames_from_topic(
    topic_uri: str,
    target_count: int,
    cols: int,
    rows: int,
    *,
    no_display: bool = False,
    timeout_sec: float = 60.0,
) -> list[np.ndarray]:
    """Capture ``target_count`` frames from an LCM/SHM image topic."""
    return _capture_frames_from_topic(
        topic_uri,
        target_count,
        cols,
        rows,
        no_display=no_display,
        timeout_sec=timeout_sec,
    ).frames


def find_chessboard_corners(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """Detect inner chessboard corners and refine them with sub-pixel accuracy.

    ``cols`` and ``rows`` are the counts of **inner** corners along each axis, matching
    ``cv2.findChessboardCorners(..., patternSize=(cols, rows))``.

    Returns:
        Float array of shape ``(cols * rows, 1, 2)`` on success, else ``None``.
    """
    return _find_chessboard_corners_exact(gray, cols, rows)


def _select_calibration_pattern(
    frames: list[np.ndarray],
    cols: int,
    rows: int,
) -> tuple[int, int, str]:
    candidates = _pattern_candidates(cols, rows)
    best_cols, best_rows, best_label = cols, rows, "requested inner corners"
    best_count = -1
    for cand_cols, cand_rows, label in candidates:
        count = 0
        for frame in frames:
            # Many frames: score each candidate with the lightweight detector first (same idea as preview).
            corners = _find_chessboard_corners_realtime(frame, cand_cols, cand_rows)
            if corners is not None:
                count += 1
        if count > best_count:
            best_cols, best_rows, best_label = cand_cols, cand_rows, label
            best_count = count

    if best_count <= 0:
        for cand_cols, cand_rows, label in candidates:
            count = 0
            for frame in frames:
                corners = find_chessboard_corners(frame, cand_cols, cand_rows)
                if corners is not None:
                    count += 1
            if count > best_count:
                best_cols, best_rows, best_label = cand_cols, cand_rows, label
                best_count = count

        if best_count <= 0:
            raise ValueError("Chessboard not found in any frame.")
    return best_cols, best_rows, best_label


def _calibrate_pinhole(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run ``cv2.calibrateCamera`` (plumb-bob)."""
    _calibrate = cast("Callable[..., Any]", cv2.calibrateCamera)
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = _calibrate(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    return float(rms), K, D


def _calibrate_fisheye(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run ``cv2.fisheye.calibrate`` (4-coeff Kannala-Brandt).

    ``objpoints`` must be a list of ``(N, 1, 3)`` arrays and ``imgpoints`` a list of
    ``(N, 1, 2)`` arrays (the fisheye solver is strict about the extra middle axis).
    """
    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    n_views = len(objpoints)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_views)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_views)]
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    _calibrate = cast("Callable[..., Any]", cv2.fisheye.calibrate)
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = _calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        D,
        rvecs,
        tvecs,
        flags,
        criteria,
    )
    K_out = np.asarray(camera_matrix, dtype=np.float64)
    D_out = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    return float(rms), K_out, D_out


def calibrate_from_frames(
    frames: list[np.ndarray],
    cols: int,
    rows: int,
    square_size_m: float,
    *,
    pattern_hint: tuple[int, int, str] | None = None,
    image_points_hint: list[np.ndarray] | None = None,
    distortion_model: DistortionModel | str = DistortionModel.plumb_bob,
) -> CalibrationResultDict:
    """Calibrate intrinsics from grayscale or BGR frames containing a chessboard.

    Each frame where ``find_chessboard_corners`` succeeds contributes one view.
    All frames must share the same resolution.

    ``distortion_model`` picks the solver: ``plumb_bob`` (``cv2.calibrateCamera``,
    5 coeffs) or ``fisheye`` (``cv2.fisheye.calibrate``, 4 coeffs).

    Returns:
        ``{"K", "D", "rms", "image_size", "n_used"}`` with ``K`` (3x3) and ``D`` (1-d),
        ``rms`` reprojection RMSE from OpenCV, ``image_size`` ``(width, height)``, and
        ``n_used`` the number of frames that yielded detections.
    """
    if not frames:
        raise ValueError("frames must be non-empty")

    model = DistortionModel(distortion_model)

    if pattern_hint is None:
        actual_cols, actual_rows, pattern_label = _select_calibration_pattern(frames, cols, rows)
    else:
        actual_cols, actual_rows, pattern_label = pattern_hint

    # Object points on Z=0 with XY spacing square_size_m. cv2.fisheye.calibrate
    # demands an explicit middle axis on each view; cv2.calibrateCamera is fine
    # with the flat (N, 3) shape.
    objp_flat = np.zeros((actual_rows * actual_cols, 3), dtype=np.float32)
    objp_flat[:, :2] = np.mgrid[0:actual_cols, 0:actual_rows].T.reshape(-1, 2).astype(np.float32)
    objp_flat *= float(square_size_m)
    objp_view = objp_flat.reshape(-1, 1, 3) if model is DistortionModel.fisheye else objp_flat

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    first = np.asarray(frames[0])
    h0, w0 = first.shape[:2]

    if image_points_hint is not None and len(image_points_hint) != len(frames):
        raise ValueError("image_points_hint length must match frames length.")

    for i, frame in enumerate(frames):
        f = np.asarray(frame)
        if f.shape[:2] != (h0, w0):
            raise ValueError("All frames must have the same shape.")

        corners_found: np.ndarray
        if image_points_hint is not None:
            # Corners from the frame at SPACE accept time; avoids re-detecting a different instant.
            corners_found = np.asarray(image_points_hint[i], dtype=np.float32).reshape(
                actual_rows * actual_cols,
                1,
                2,
            )
        else:
            gray_in: np.ndarray
            if f.ndim == 3:
                gray_in = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            else:
                gray_in = np.asarray(f)
            corners_opt = find_chessboard_corners(gray_in, actual_cols, actual_rows)
            if corners_opt is None:
                continue
            corners_found = corners_opt
        objpoints.append(objp_view)
        imgpoints.append(corners_found.astype(np.float32))

    if not objpoints:
        raise ValueError("Chessboard not found in any frame.")

    if model is DistortionModel.fisheye:
        rms, K, D = _calibrate_fisheye(objpoints, imgpoints, (w0, h0))
    else:
        rms, K, D = _calibrate_pinhole(objpoints, imgpoints, (w0, h0))

    return {
        "K": K,
        "D": D,
        "rms": float(rms),
        "image_size": (int(w0), int(h0)),
        "n_used": len(objpoints),
        "pattern_size": (int(actual_cols), int(actual_rows)),
        "pattern_label": pattern_label,
    }


def write_preview_overlay_png(
    frames: list[np.ndarray],
    cols: int,
    rows: int,
    path: Path,
) -> Path:
    """Write a preview PNG with detected chessboard corners drawn on one input frame."""
    for frame in frames:
        f = np.asarray(frame)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        corners = find_chessboard_corners(gray, cols, rows)
        if corners is None:
            continue

        preview = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else np.asarray(frame).copy()
        cv2.drawChessboardCorners(preview, (cols, rows), corners, True)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), preview):
            raise ValueError(f"Could not write preview image: {path}")
        return path

    raise ValueError("Chessboard not found in any frame for preview overlay.")


def write_charuco_preview_overlay_png(
    frames: list[np.ndarray],
    spec: CharucoBoardSpec,
    path: Path,
) -> Path:
    """Write a preview PNG with detected ChArUco corners drawn on one input frame (cv2.aruco)."""
    for frame in frames:
        f = np.asarray(frame)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        detection = _detect_charuco(gray, spec)
        if detection is None:
            continue

        preview = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else np.asarray(frame).copy()
        cv2.aruco.drawDetectedCornersCharuco(
            preview, detection.charuco_corners_px, detection.charuco_ids
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), preview):
            raise ValueError(f"Could not write preview image: {path}")
        return path

    raise ValueError("ChArUco board not found in any frame for preview overlay.")


def run_calibration(
    *,
    source: Source | str,
    device_index: int,
    images: Path | None,
    topic: str | None,
    topic_timeout_sec: float,
    cols: int,
    rows: int,
    square_size_m: float,
    out: Path | None,
    preview_out: Path | None,
    camera_name: str,
    target_count: int,
    no_display: bool,
    distortion_model: DistortionModel | str = DistortionModel.plumb_bob,
    board: BoardType | str = BoardType.chessboard,
    dict_name: str = _DEFAULT_CHARUCO_DICT,
    squares_x: int | None = None,
    squares_y: int | None = None,
    marker_size_m: float | None = None,
    marker_ratio: float = _DEFAULT_MARKER_RATIO,
) -> CalibrationRunResultDict:
    """Capture frames from ``source``, calibrate, and (optionally) write CameraInfo YAML + preview.

    ``board`` picks chessboard vs charuco and ``distortion_model`` picks the solver -- charuco
    fisheye routes through ``cv2.fisheye.calibrate`` over the per-view corner correspondences.
    """
    source_value = Source(source)
    model = DistortionModel(distortion_model)
    board_type = BoardType(board)
    if square_size_m <= 0:
        raise ValueError("square_size_m must be > 0")

    charuco_spec: CharucoBoardSpec | None = None
    if board_type is BoardType.charuco:
        if squares_x is None or squares_y is None:
            raise ValueError("--squares-x and --squares-y are required when --board charuco")
        charuco_spec = build_charuco_board(
            dict_name=dict_name,
            squares_x=squares_x,
            squares_y=squares_y,
            square_size_m=square_size_m,
            marker_size_m=marker_size_m,
            marker_ratio=marker_ratio,
        )
    else:
        if cols < 1:
            raise ValueError("cols must be >= 1")
        if rows < 1:
            raise ValueError("rows must be >= 1")

    if source_value is Source.folder:
        if images is None:
            raise ValueError("--images is required when --source folder")
        frames = load_frames_from_folder(str(images))
        pattern_hint = None
        image_points_hint = None
    elif source_value is Source.topic:
        if topic is None:
            raise ValueError(
                "--topic is required when --source topic (e.g. --topic jpeg_lcm:/color_image)"
            )
        capture = _capture_frames_from_topic(
            topic,
            target_count,
            cols,
            rows,
            no_display=no_display,
            timeout_sec=topic_timeout_sec,
            charuco_spec=charuco_spec,
        )
        frames = capture.frames
        pattern_hint = capture.pattern
        image_points_hint = capture.image_points
    else:
        capture = _capture_frames_from_webcam(
            device_index,
            target_count,
            cols,
            rows,
            no_display=no_display,
            charuco_spec=charuco_spec,
        )
        frames = capture.frames
        pattern_hint = capture.pattern
        image_points_hint = capture.image_points

    if charuco_spec is not None:
        cal = calibrate_from_frames_charuco(frames, charuco_spec, distortion_model=model)
    else:
        cal = calibrate_from_frames(
            frames,
            cols,
            rows,
            square_size_m,
            pattern_hint=pattern_hint,
            image_points_hint=image_points_hint,
            distortion_model=model,
        )
    result: CalibrationRunResultDict = {
        "K": cal["K"],
        "D": cal["D"],
        "rms": cal["rms"],
        "image_size": cal["image_size"],
        "n_used": cal["n_used"],
        "pattern_size": cal["pattern_size"],
        "pattern_label": cal["pattern_label"],
    }
    image_width, image_height = result["image_size"]

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        write_camera_info_yaml(
            str(out),
            image_width=int(image_width),
            image_height=int(image_height),
            camera_name=camera_name,
            K=np.asarray(result["K"], dtype=np.float64),
            D=np.asarray(result["D"], dtype=np.float64),
            distortion_model=model.to_ros_name(),
        )
        result["out_path"] = out

    if preview_out is not None:
        preview_out.parent.mkdir(parents=True, exist_ok=True)
        pattern_cols, pattern_rows = result["pattern_size"]
        # Preview is best-effort: a board-detection failure here must not mask
        # the fact that the YAML was already written above.
        try:
            if charuco_spec is not None:
                write_charuco_preview_overlay_png(frames, charuco_spec, preview_out)
            else:
                write_preview_overlay_png(frames, int(pattern_cols), int(pattern_rows), preview_out)
            result["preview_path"] = preview_out
        except ValueError as exc:
            warnings.warn(
                f"Preview PNG skipped ({exc}). Camera info YAML was still written to {out}.",
                stacklevel=2,
            )

    return result


def _board_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    """Checkerboard inner-corner object points on Z=0, XY spacing ``square_size_m`` (meters).

    Same construction as ``calibrate_from_frames`` so corner ordering matches ``find_chessboard_corners``.
    Returns an ``(cols * rows, 3)`` float32 array.
    """
    objp_m = np.zeros((rows * cols, 3), dtype=np.float32)
    objp_m[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objp_m *= float(square_size_m)
    return objp_m


def _reproject_rms_px(
    objp_m: np.ndarray,
    corners_px: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    *,
    fisheye: bool,
) -> float | None:
    """RMS reprojection error (px) of ``corners_px`` under the deployed ``K``/``D``.

    solvePnP recovers ``camera_T_board`` from the corners, then the object points are
    reprojected under the same intrinsics; the RMS of (reprojected - detected) is returned.
    A full board grid has no mirror ambiguity, so this RMS is a valid single-frame signal.
    Returns ``None`` if solvePnP fails (degenerate view).

    Reprojection via cv2.projectPoints (OpenCV) -- the verified projection, not hand-rolled math.
    """
    corners = np.asarray(corners_px, dtype=np.float64).reshape(-1, 1, 2)
    if fisheye:
        # No fisheye solvePnP exists: undistort corners into an ideal pinhole (P=K), then a
        # plain solvePnP with zero distortion recovers the board pose.
        undistorted_px = cv2.fisheye.undistortPoints(corners, K, D, R=np.eye(3), P=K)
        ok, rvec, tvec = cv2.solvePnP(objp_m, undistorted_px, K, None)
    else:
        ok, rvec, tvec = cv2.solvePnP(objp_m, corners, K, D)
    if not ok:
        return None

    if fisheye:
        projected, _ = cv2.fisheye.projectPoints(
            objp_m.reshape(-1, 1, 3), rvec.reshape(1, 1, 3), tvec.reshape(1, 1, 3), K, D
        )
    else:
        projected, _ = cv2.projectPoints(objp_m, rvec, tvec, K, D)

    residual_px = np.asarray(projected, dtype=np.float64).reshape(-1, 2) - corners.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual_px**2, axis=1))))


def _assess_drift(
    objp_m: np.ndarray,
    imgpoints_px: list[np.ndarray],
    K_deployed: np.ndarray,
    D_deployed: np.ndarray,
    *,
    image_size_wh: tuple[int, int],
    fisheye: bool,
    drift_threshold_frac: float,
    run_drift: bool,
) -> CheckDriftDict:
    """Run a fresh calibration on the detected corners and compare intrinsics to the deployed ones.

    Uses the same lens model as the deployed calib so distortion vectors are comparable. The fresh
    solve is diagnostic-only: a solver failure (``cv2.error``) is converted to ``ran=False`` so it
    can never sink the reprojection verdict.
    """
    n_views = len(imgpoints_px)
    if not run_drift:
        return {"ran": False, "reason": "drift check disabled (--no-drift)"}
    if n_views < _MIN_DRIFT_CALIBRATION_FRAMES:
        return {
            "ran": False,
            "reason": (
                f"need >= {_MIN_DRIFT_CALIBRATION_FRAMES} frames for a fresh calibration, "
                f"have {n_views}"
            ),
        }

    objp_each = objp_m.reshape(-1, 1, 3) if fisheye else objp_m
    objpoints = [objp_each for _ in range(n_views)]
    try:
        if fisheye:
            fresh_rms_px, K_fresh, D_fresh = _calibrate_fisheye(
                objpoints, imgpoints_px, image_size_wh
            )
        else:
            fresh_rms_px, K_fresh, D_fresh = _calibrate_pinhole(
                objpoints, imgpoints_px, image_size_wh
            )
    except cv2.error as exc:
        return {"ran": False, "reason": f"fresh calibration failed: {exc}"}

    return _intrinsics_drift_block(
        K_deployed,
        D_deployed,
        K_fresh,
        D_fresh,
        fresh_rms_px=fresh_rms_px,
        drift_threshold_frac=drift_threshold_frac,
    )


def _intrinsics_drift_block(
    K_deployed: np.ndarray,
    D_deployed: np.ndarray,
    K_fresh: np.ndarray,
    D_fresh: np.ndarray,
    *,
    fresh_rms_px: float,
    drift_threshold_frac: float,
) -> CheckDriftDict:
    """Compare deployed vs fresh intrinsics into a ``CheckDriftDict`` (board-type agnostic).

    Deltas are ``deployed - fresh`` (px); ``max_rel_intrinsics_drift`` is the largest per-parameter
    relative drift over fx/fy/cx/cy. The distortion delta is empty when the coefficient counts
    differ (deployed and fresh lens models disagree).
    """
    Kd = np.asarray(K_deployed, dtype=np.float64).reshape(3, 3)
    Kf = np.asarray(K_fresh, dtype=np.float64).reshape(3, 3)
    delta_fx_px = float(Kd[0, 0] - Kf[0, 0])
    delta_fy_px = float(Kd[1, 1] - Kf[1, 1])
    delta_cx_px = float(Kd[0, 2] - Kf[0, 2])
    delta_cy_px = float(Kd[1, 2] - Kf[1, 2])
    max_rel = max(
        abs(delta_fx_px) / max(abs(float(Kf[0, 0])), 1e-9),
        abs(delta_fy_px) / max(abs(float(Kf[1, 1])), 1e-9),
        abs(delta_cx_px) / max(abs(float(Kf[0, 2])), 1e-9),
        abs(delta_cy_px) / max(abs(float(Kf[1, 2])), 1e-9),
    )

    Dd = np.asarray(D_deployed, dtype=np.float64).ravel()
    Df = np.asarray(D_fresh, dtype=np.float64).ravel()
    dist_abs_delta = (
        np.abs(Dd - Df).tolist() if Dd.shape == Df.shape else []
    )  # empty when coeff counts differ (models disagree)

    return {
        "ran": True,
        "reason": "",
        "fresh_rms_px": float(fresh_rms_px),
        "delta_fx_px": delta_fx_px,
        "delta_fy_px": delta_fy_px,
        "delta_cx_px": delta_cx_px,
        "delta_cy_px": delta_cy_px,
        "max_rel_intrinsics_drift": float(max_rel),
        "distortion_abs_delta": [float(x) for x in dist_abs_delta],
        "drift_small": bool(max_rel <= drift_threshold_frac),
        "fresh_K": Kf.tolist(),
        "fresh_D": [float(x) for x in Df.tolist()],
    }


def check_calibration(
    image_points_per_view: list[np.ndarray],
    *,
    cols: int,
    rows: int,
    square_size_m: float,
    K_deployed: np.ndarray,
    D_deployed: np.ndarray,
    distortion_model: str,
    image_size_wh: tuple[int, int],
    rms_threshold_px: float = _DEFAULT_CHECK_RMS_THRESHOLD_PX,
    drift_threshold_frac: float = _DEFAULT_CHECK_DRIFT_THRESHOLD_FRAC,
    run_drift: bool = True,
) -> CalibrationCheckResultDict:
    """Score how well a DEPLOYED CameraInfo explains real detected checkerboard corners.

    ``image_points_per_view`` is one ``(cols * rows, 1, 2)`` corner array per frame (from the
    existing detector). For each view solvePnP recovers the board pose under the deployed
    intrinsics and the reprojection RMS is computed; the median/p90 across views is the headline
    signal. When enough views are present a fresh calibration is run and its intrinsics compared
    (drift). Verdict is OK iff median RMS <= ``rms_threshold_px`` AND drift is small.
    """
    if not image_points_per_view:
        raise ValueError("image_points_per_view must be non-empty")
    if cols < 1 or rows < 1:
        raise ValueError("cols and rows must be >= 1")
    if square_size_m <= 0:
        raise ValueError("square_size_m must be > 0")

    # is_fisheye_model (incl. kannala_brandt) is the verified source of truth; import it lazily so
    # the calibrate path stays free of dimos_lcm (marker_pose -> CameraInfo) -- only --check gets here.
    from dimos.perception.fiducial.marker_pose import is_fisheye_model

    fisheye = is_fisheye_model(distortion_model)
    K = np.asarray(K_deployed, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D_deployed, dtype=np.float64).reshape(-1, 1)
    objp_m = _board_object_points(cols, rows, square_size_m)
    n_corners = cols * rows

    per_view_rms_px: list[float] = []
    imgpoints_px: list[np.ndarray] = []
    for corners in image_points_per_view:
        corners_px = np.asarray(corners, dtype=np.float32).reshape(n_corners, 1, 2)
        imgpoints_px.append(corners_px)
        rms_px = _reproject_rms_px(objp_m, corners_px, K, D, fisheye=fisheye)
        if rms_px is not None:
            per_view_rms_px.append(rms_px)

    drift = _assess_drift(
        objp_m,
        imgpoints_px,
        K,
        D,
        image_size_wh=image_size_wh,
        fisheye=fisheye,
        drift_threshold_frac=drift_threshold_frac,
        run_drift=run_drift,
    )

    return _finalize_check_result(
        per_view_rms_px=per_view_rms_px,
        drift=drift,
        rms_threshold_px=rms_threshold_px,
        distortion_model=distortion_model,
        image_size_wh=image_size_wh,
        cols=cols,
        rows=rows,
        square_size_m=square_size_m,
        recommendation_flags="--cols C --rows R --square-size-m S",
    )


def _finalize_check_result(
    *,
    per_view_rms_px: list[float],
    drift: CheckDriftDict,
    rms_threshold_px: float,
    distortion_model: str,
    image_size_wh: tuple[int, int],
    cols: int,
    rows: int,
    square_size_m: float,
    recommendation_flags: str,
) -> CalibrationCheckResultDict:
    """Aggregate per-view RMS + drift into the verdict + result dict (shared by both board types).

    Verdict is OK iff median reprojection RMS <= ``rms_threshold_px`` AND drift is small (a
    skipped/failed drift solve does not force DEGRADED). ``recommendation_flags`` fills the
    board-appropriate flag hint in the recalibrate suggestion.
    """
    if not per_view_rms_px:
        raise ValueError("solvePnP failed on every view; cannot assess reprojection error.")

    median_reproj_rms_px = float(np.median(per_view_rms_px))
    p90_reproj_rms_px = float(np.percentile(per_view_rms_px, 90))

    drift_small = bool(drift.get("drift_small", True)) if drift["ran"] else True
    rms_ok = median_reproj_rms_px <= rms_threshold_px
    verdict = "OK" if (rms_ok and drift_small) else "DEGRADED"
    recommendation = (
        ""
        if verdict == "OK"
        else "recalibrate this unit (dimos cameracalibrate --source <topic|folder> "
        f"{recommendation_flags} --out <new.yaml>)"
    )

    return {
        "verdict": verdict,
        "median_reproj_rms_px": median_reproj_rms_px,
        "p90_reproj_rms_px": p90_reproj_rms_px,
        "n_frames_used": len(per_view_rms_px),
        "rms_threshold_px": float(rms_threshold_px),
        "per_view_rms_px": per_view_rms_px,
        "distortion_model": distortion_model,
        "image_size_wh": (int(image_size_wh[0]), int(image_size_wh[1])),
        "cols": int(cols),
        "rows": int(rows),
        "square_size_m": float(square_size_m),
        "drift": drift,
        "recommendation": recommendation,
    }


def _assess_drift_charuco(
    detections: list[_CharucoDetection],
    board: Any,
    K_deployed: np.ndarray,
    D_deployed: np.ndarray,
    *,
    image_size_wh: tuple[int, int],
    fisheye: bool,
    drift_threshold_frac: float,
    run_drift: bool,
) -> CheckDriftDict:
    """Fresh ChArUco calibration vs the deployed intrinsics, in the DEPLOYED lens model.

    A fisheye deployment is refit with ``cv2.fisheye.calibrate`` (``_calibrate_charuco_fisheye``)
    so the fresh intrinsics carry the same model as the deployment -- the recalibration offer never
    hands a plumb-bob YAML to a fisheye lens. A plumb-bob deployment uses ``calibrateCameraCharuco``.
    The distortion delta is comparable when the models (hence coeff counts) match. Diagnostic-only:
    a solver failure becomes ``ran=False`` and can never sink the reprojection verdict.
    """
    n_views = len(detections)
    if not run_drift:
        return {"ran": False, "reason": "drift check disabled (--no-drift)"}
    if n_views < _MIN_DRIFT_CALIBRATION_FRAMES:
        return {
            "ran": False,
            "reason": (
                f"need >= {_MIN_DRIFT_CALIBRATION_FRAMES} frames for a fresh calibration, "
                f"have {n_views}"
            ),
        }

    try:
        if fisheye:
            fresh_rms_px, K_fresh, D_fresh = _calibrate_charuco_fisheye(detections, image_size_wh)
        else:
            fresh_rms_px, K_fresh, D_fresh = _calibrate_charuco(
                [d.charuco_corners_px for d in detections],
                [d.charuco_ids for d in detections],
                board,
                image_size_wh,
            )
    # ValueError: _calibrate_charuco_fisheye's got-vs-want convergence sentinel (diagnostic-only).
    except (cv2.error, ValueError) as exc:
        return {"ran": False, "reason": f"fresh calibration failed: {exc}"}

    return _intrinsics_drift_block(
        K_deployed,
        D_deployed,
        K_fresh,
        D_fresh,
        fresh_rms_px=fresh_rms_px,
        drift_threshold_frac=drift_threshold_frac,
    )


def check_calibration_charuco(
    detections: list[_CharucoDetection],
    *,
    board_spec: CharucoBoardSpec,
    K_deployed: np.ndarray,
    D_deployed: np.ndarray,
    distortion_model: str,
    image_size_wh: tuple[int, int],
    rms_threshold_px: float = _DEFAULT_CHECK_RMS_THRESHOLD_PX,
    drift_threshold_frac: float = _DEFAULT_CHECK_DRIFT_THRESHOLD_FRAC,
    run_drift: bool = True,
) -> CalibrationCheckResultDict:
    """Score a DEPLOYED CameraInfo against detected ChArUco corners (same verdict path as chessboard).

    Per view, solvePnP recovers the board pose from that view's ``object_points_m`` <->
    ``charuco_corners_px`` correspondence under the deployed intrinsics, then the reprojection RMS
    is computed (``_reproject_rms_px``, cv2.projectPoints / fisheye.projectPoints). Median/p90 and
    the drift comparison feed ``_finalize_check_result`` unchanged. The fresh drift calibration
    matches the deployed lens model: fisheye deployment -> fisheye refit, else plumb-bob.
    """
    if not detections:
        raise ValueError("detections must be non-empty")

    # is_fisheye_model (incl. kannala_brandt) is the verified source of truth; import it lazily so
    # the calibrate path stays free of dimos_lcm (marker_pose -> CameraInfo) -- only --check gets here.
    from dimos.perception.fiducial.marker_pose import is_fisheye_model

    fisheye = is_fisheye_model(distortion_model)
    K = np.asarray(K_deployed, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D_deployed, dtype=np.float64).reshape(-1, 1)

    per_view_rms_px: list[float] = []
    for detection in detections:
        objp_m = np.asarray(detection.object_points_m, dtype=np.float64).reshape(-1, 3)
        rms_px = _reproject_rms_px(objp_m, detection.charuco_corners_px, K, D, fisheye=fisheye)
        if rms_px is not None:
            per_view_rms_px.append(rms_px)

    drift = _assess_drift_charuco(
        detections,
        board_spec.board,
        K,
        D,
        image_size_wh=image_size_wh,
        fisheye=fisheye,
        drift_threshold_frac=drift_threshold_frac,
        run_drift=run_drift,
    )

    return _finalize_check_result(
        per_view_rms_px=per_view_rms_px,
        drift=drift,
        rms_threshold_px=rms_threshold_px,
        distortion_model=distortion_model,
        image_size_wh=image_size_wh,
        cols=board_spec.squares_x,
        rows=board_spec.squares_y,
        square_size_m=board_spec.square_size_m,
        recommendation_flags=(
            f"--board charuco --dict {board_spec.dict_name} "
            f"--squares-x {board_spec.squares_x} --squares-y {board_spec.squares_y} "
            f"--square-size-m {board_spec.square_size_m}"
        ),
    )


def _git_rev_short() -> str:
    """Best-effort short git rev for provenance; ``"unknown"`` if git is unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    rev = completed.stdout.strip()
    return rev or "unknown"


def _default_check_out_path(source: Source, images: Path | None) -> Path:
    """Default JSON path: next to the folder source, else the current directory."""
    name = "calibration_check.json"
    if source is Source.folder and images is not None:
        return Path(images) / name
    return Path.cwd() / name


def _write_check_json(path: Path, result: CalibrationCheckRunResultDict) -> None:
    """Write the check numbers + provenance as JSON (a rerun has every input it needs)."""
    payload = {
        "tool": "dimos cameracalibrate --check",
        "verdict": result["verdict"],
        "median_reproj_rms_px": result["median_reproj_rms_px"],
        "p90_reproj_rms_px": result["p90_reproj_rms_px"],
        "n_frames_used": result["n_frames_used"],
        "rms_threshold_px": result["rms_threshold_px"],
        "recommendation": result["recommendation"],
        "resolution_warning": result.get("resolution_warning", ""),
        "drift": dict(result["drift"]),
        "provenance": {
            "camera_info_path": result["camera_info_path"],
            "distortion_model": result["distortion_model"],
            "source": result["source"],
            "cols": result["cols"],
            "rows": result["rows"],
            "square_size_m": result["square_size_m"],
            "image_size_wh": list(result["image_size_wh"]),
            "git_rev": _git_rev_short(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_recalibration_from_check(
    result: CalibrationCheckRunResultDict,
    out_yaml: Path,
) -> Path:
    """Write the fresh intrinsics ``--check`` already computed to a CameraInfo YAML (no re-solve).

    Reuses ``result['drift']['fresh_K']/['fresh_D']`` so no second calibration is run. The fresh
    solve (chessboard or ChArUco) uses the deployed lens model, so its ROS name carries straight
    over -- a fisheye deployment yields an ``equidistant`` YAML, never a plumb-bob mismatch. Raises
    ``ValueError`` (got-vs-want) when the drift solve did not run (too few frames), i.e. there are
    no fresh intrinsics to emit.
    """
    drift = result["drift"]
    if not drift.get("ran") or "fresh_K" not in drift:
        raise ValueError(
            "cannot recalibrate: the fresh-calibration drift solve did not run "
            f"(reason: {drift.get('reason', 'unknown')}); need enough detected frames"
        )
    K_fresh = np.asarray(drift["fresh_K"], dtype=np.float64).reshape(3, 3)
    D_fresh = np.asarray(drift["fresh_D"], dtype=np.float64).ravel()
    fresh_model = result["distortion_model"]
    w, h = result["image_size_wh"]
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    write_camera_info_yaml(
        str(out_yaml),
        image_width=int(w),
        image_height=int(h),
        camera_name="recalibrated",
        K=K_fresh,
        D=D_fresh,
        distortion_model=fresh_model,
    )
    return out_yaml


def run_check(
    *,
    source: Source | str,
    device_index: int,
    images: Path | None,
    topic: str | None,
    topic_timeout_sec: float,
    cols: int,
    rows: int,
    square_size_m: float,
    camera_info_path: Path,
    rms_threshold_px: float,
    drift_threshold_frac: float,
    run_drift: bool,
    out: Path | None,
    target_count: int,
    no_display: bool,
    board: BoardType | str = BoardType.chessboard,
    dict_name: str = _DEFAULT_CHARUCO_DICT,
    squares_x: int | None = None,
    squares_y: int | None = None,
    marker_size_m: float | None = None,
    marker_ratio: float = _DEFAULT_MARKER_RATIO,
) -> CalibrationCheckRunResultDict:
    """Ingest frames from the requested source, detect boards, and check a deployed CameraInfo.

    ``board`` selects checkerboard (``check_calibration``) or charuco
    (``check_calibration_charuco``); only the per-frame corner source differs -- both feed the same
    reprojection/drift verdict. Writes the check JSON to ``out`` (or a default next to the source).
    """
    from dimos.msgs.sensor_msgs.CameraInfo import (
        CameraInfo,  # heavy (dimos_lcm); only --check needs it
    )

    source_value = Source(source)
    board_type = BoardType(board)
    if square_size_m <= 0:
        raise ValueError("square_size_m must be > 0")
    if not Path(camera_info_path).is_file():
        raise ValueError(f"--camera-info not found: {camera_info_path}")

    charuco_spec: CharucoBoardSpec | None = None
    if board_type is BoardType.charuco:
        if squares_x is None or squares_y is None:
            raise ValueError("--squares-x and --squares-y are required when --board charuco")
        charuco_spec = build_charuco_board(
            dict_name=dict_name,
            squares_x=squares_x,
            squares_y=squares_y,
            square_size_m=square_size_m,
            marker_size_m=marker_size_m,
            marker_ratio=marker_ratio,
        )
    else:
        if cols < 1:
            raise ValueError("cols must be >= 1")
        if rows < 1:
            raise ValueError("rows must be >= 1")

    camera_info = CameraInfo.from_yaml(str(camera_info_path))
    K_deployed = camera_info.get_K_matrix()
    D_deployed = camera_info.get_D_coeffs()
    distortion_model = camera_info.distortion_model or "plumb_bob"

    image_points_per_view: list[np.ndarray] = []
    charuco_detections: list[_CharucoDetection] = []
    actual_cols, actual_rows = cols, rows

    if source_value is Source.folder:
        if images is None:
            raise ValueError("--images is required when --source folder")
        frames = load_frames_from_folder(str(images))
        if charuco_spec is not None:
            charuco_detections = _charuco_detections_from_frames(frames, charuco_spec)
        else:
            actual_cols, actual_rows, _label = _select_calibration_pattern(frames, cols, rows)
            for frame in frames:
                f = np.asarray(frame)
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
                corners = find_chessboard_corners(gray, actual_cols, actual_rows)
                if corners is not None:
                    image_points_per_view.append(corners)
    else:
        if source_value is Source.topic:
            if topic is None:
                raise ValueError(
                    "--topic is required when --source topic (e.g. --topic jpeg_lcm:/color_image)"
                )
            capture = _capture_frames_from_topic(
                topic,
                target_count,
                cols,
                rows,
                no_display=no_display,
                timeout_sec=topic_timeout_sec,
                charuco_spec=charuco_spec,
            )
        else:
            capture = _capture_frames_from_webcam(
                device_index,
                target_count,
                cols,
                rows,
                no_display=no_display,
                charuco_spec=charuco_spec,
            )
        frames = capture.frames
        if charuco_spec is not None:
            charuco_detections = list(capture.charuco_detections or [])
        else:
            image_points_per_view = list(capture.image_points)
            actual_cols, actual_rows = (
                (capture.pattern[0], capture.pattern[1]) if capture.pattern else (cols, rows)
            )

    if not frames:
        raise ValueError("No frames ingested for --check.")

    first = np.asarray(frames[0])
    h0, w0 = first.shape[:2]
    image_size_wh = (int(w0), int(h0))

    if charuco_spec is not None:
        if not charuco_detections:
            raise ValueError("ChArUco board not detected in any frame; nothing to check.")
        core = check_calibration_charuco(
            charuco_detections,
            board_spec=charuco_spec,
            K_deployed=K_deployed,
            D_deployed=D_deployed,
            distortion_model=distortion_model,
            image_size_wh=image_size_wh,
            rms_threshold_px=rms_threshold_px,
            drift_threshold_frac=drift_threshold_frac,
            run_drift=run_drift,
        )
    else:
        if not image_points_per_view:
            raise ValueError("Chessboard not detected in any frame; nothing to check.")
        core = check_calibration(
            image_points_per_view,
            cols=actual_cols,
            rows=actual_rows,
            square_size_m=square_size_m,
            K_deployed=K_deployed,
            D_deployed=D_deployed,
            distortion_model=distortion_model,
            image_size_wh=image_size_wh,
            rms_threshold_px=rms_threshold_px,
            drift_threshold_frac=drift_threshold_frac,
            run_drift=run_drift,
        )
    # ``core`` (required keys only) is a structurally-valid run result; the extra provenance keys
    # are total=False. cast is needed because mypy does not narrow a ** TypedDict spread.
    result = cast("CalibrationCheckRunResultDict", dict(core))
    result["source"] = source_value.value
    result["camera_info_path"] = str(camera_info_path)

    # The deployed intrinsics are only valid at their own resolution; a mismatch makes the
    # reprojection numbers meaningless, so surface it loudly rather than silently comparing.
    if (
        camera_info.width
        and camera_info.height
        and (int(camera_info.width), int(camera_info.height)) != image_size_wh
    ):
        result["resolution_warning"] = (
            f"deployed CameraInfo is {camera_info.width}x{camera_info.height} but frames are "
            f"{w0}x{h0}; reprojection RMS is not meaningful across resolutions."
        )

    out_path = out if out is not None else _default_check_out_path(source_value, images)
    _write_check_json(out_path, result)
    result["out_path"] = out_path
    return result


def run_check_report(
    *,
    source: Source | str,
    device_index: int,
    images: Path | None,
    topic: str | None,
    topic_timeout_sec: float,
    cols: int,
    rows: int,
    square_size_m: float,
    camera_info: Path,
    rms_threshold_px: float,
    drift_threshold_frac: float,
    check_drift: bool,
    out: Path | None,
    target_count: int,
    no_display: bool,
    board: BoardType | str = BoardType.chessboard,
    dict_name: str = _DEFAULT_CHARUCO_DICT,
    squares_x: int | None = None,
    squares_y: int | None = None,
    marker_size_m: float | None = None,
    marker_ratio: float = _DEFAULT_MARKER_RATIO,
) -> None:
    """CLI boundary for ``--check``: run the check, print the evidence table, echo the JSON path.

    Shared by the standalone ``app`` and the ``dimos cameracalibrate`` command so both report
    identically (mirrors how both call ``run_calibration``). On a DEGRADED verdict with the fresh
    drift solve available, an interactive TTY offers the operator a ``[y/N]`` prompt to write the
    fresh calibration to a new CameraInfo YAML.
    """
    try:
        result = run_check(
            source=source,
            device_index=device_index,
            images=images,
            topic=topic,
            topic_timeout_sec=topic_timeout_sec,
            cols=cols,
            rows=rows,
            square_size_m=square_size_m,
            camera_info_path=camera_info,
            rms_threshold_px=rms_threshold_px,
            drift_threshold_frac=drift_threshold_frac,
            run_drift=check_drift,
            out=out,
            target_count=target_count,
            no_display=no_display,
            board=board,
            dict_name=dict_name,
            squares_x=squares_x,
            squares_y=squares_y,
            marker_size_m=marker_size_m,
            marker_ratio=marker_ratio,
        )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"CALIBRATION CHECK: {result['verdict']}")
    typer.echo(
        f"  deployed CameraInfo: {result.get('camera_info_path', camera_info)} "
        f"({result['distortion_model']})"
    )
    typer.echo(
        f"  reprojection RMS px: median={result['median_reproj_rms_px']:.3f} "
        f"p90={result['p90_reproj_rms_px']:.3f}  "
        f"(n={result['n_frames_used']} frames, threshold={result['rms_threshold_px']:.3f})"
    )
    drift = result["drift"]
    if drift["ran"]:
        typer.echo(f"  fresh calibration RMS px: {drift['fresh_rms_px']:.3f}")
        typer.echo(
            "  intrinsics drift (deployed - fresh): "
            f"dfx={drift['delta_fx_px']:+.2f} dfy={drift['delta_fy_px']:+.2f} "
            f"dcx={drift['delta_cx_px']:+.2f} dcy={drift['delta_cy_px']:+.2f} px "
            f"(max rel {drift['max_rel_intrinsics_drift']:.2%})"
        )
        dist_delta = [round(x, 4) for x in drift.get("distortion_abs_delta", [])]
        typer.echo(f"  distortion |delta|: {dist_delta}")
    else:
        typer.echo(f"  drift check: skipped ({drift['reason']})")

    resolution_warning = result.get("resolution_warning", "")
    if resolution_warning:
        typer.echo(f"  WARNING: {resolution_warning}")
    if result["recommendation"]:
        typer.echo(f"  -> {result['recommendation']}")
    if "out_path" in result:
        typer.echo(f"Wrote check JSON to {result['out_path']}")

    if result["verdict"] == "DEGRADED" and result["drift"].get("ran") and sys.stdin.isatty():
        # Interactive offer: reuse the fresh calibration the drift solve already computed.
        if typer.confirm(
            "Deployed calibration is DEGRADED. Write the fresh calibration from these "
            "frames to a new CameraInfo YAML?",
            default=False,
        ):
            default_yaml = Path(result["out_path"]).with_name("recalibrated.yaml")
            target = Path(typer.prompt("Output YAML path", default=str(default_yaml)))
            written = write_recalibration_from_check(result, target)
            typer.echo(f"Wrote fresh recalibration YAML to {written}")


@app.command()
def calibrate(
    source: Source = typer.Option(..., "--source", help="Frame source: webcam, folder, or topic"),
    device_index: int = typer.Option(0, "--device-index", help="Webcam device index"),
    images: Path | None = typer.Option(
        None, "--images", help="Directory of calibration images for --source folder"
    ),
    topic: str | None = typer.Option(
        None,
        "--topic",
        help=(
            "Pubsub URI for --source topic (proto:channel), "
            "e.g. 'jpeg_lcm:/color_image' or 'pshm:color_image'."
        ),
    ),
    topic_timeout_sec: float = typer.Option(
        60.0,
        "--topic-timeout-sec",
        help="Abort --source topic if no frames arrive within this many seconds.",
    ),
    board: BoardType = typer.Option(
        BoardType.chessboard,
        "--board",
        help="Calibration target: 'chessboard' (--cols/--rows) or 'charuco' (--dict/--squares-x/-y).",
    ),
    cols: int | None = typer.Option(
        None, "--cols", help="Inner chessboard corner columns (--board chessboard)"
    ),
    rows: int | None = typer.Option(
        None, "--rows", help="Inner chessboard corner rows (--board chessboard)"
    ),
    square_size_m: float = typer.Option(
        ..., "--square-size-m", help="Square size in meters (chessboard square or charuco square)"
    ),
    charuco_dict: str = typer.Option(
        _DEFAULT_CHARUCO_DICT,
        "--dict",
        help="--board charuco: predefined aruco dictionary name, e.g. DICT_4X4_50.",
    ),
    squares_x: int | None = typer.Option(
        None, "--squares-x", help="--board charuco: number of squares along X (not inner corners)."
    ),
    squares_y: int | None = typer.Option(
        None, "--squares-y", help="--board charuco: number of squares along Y (not inner corners)."
    ),
    marker_ratio: float = typer.Option(
        _DEFAULT_MARKER_RATIO,
        "--marker-ratio",
        help="--board charuco: markerLength/squareLength (ignored if --marker-size-m is given).",
    ),
    marker_size_m: float | None = typer.Option(
        None,
        "--marker-size-m",
        help="--board charuco: aruco marker size in meters (overrides --marker-ratio).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help=(
            "Calibrate: ROS CameraInfo YAML output path. --check: JSON report output path "
            "(default: next to the source)."
        ),
    ),
    preview_out: Path | None = typer.Argument(
        None, help="Optional preview PNG output path. Requires --out."
    ),
    camera_name: str = typer.Option("webcam", "--camera-name", help="Camera name in YAML"),
    target_count: int = typer.Option(20, "--target-count", help="Accepted webcam frame count"),
    no_display: bool = typer.Option(False, "--no-display", help="Disable OpenCV preview windows"),
    distortion_model: DistortionModel = typer.Option(
        DistortionModel.plumb_bob,
        "--distortion-model",
        help=(
            "Lens model: 'plumb_bob' (cv2.calibrateCamera, 5 coeffs) for near-pinhole "
            "lenses, or 'fisheye' (cv2.fisheye.calibrate, 4 coeffs) for wide-angle / "
            "fisheye lenses. Fisheye writes ROS 'equidistant' to the YAML."
        ),
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help=(
            "CHECK mode: verify a DEPLOYED CameraInfo (--camera-info) against the boards on this "
            "source -- reports reprojection RMS and, via a fresh calibration on the same frames, "
            "intrinsics drift."
        ),
    ),
    camera_info: Path | None = typer.Option(
        None,
        "--camera-info",
        help="--check: deployed CameraInfo YAML to test (required with --check).",
    ),
    rms_threshold_px: float = typer.Option(
        _DEFAULT_CHECK_RMS_THRESHOLD_PX,
        "--rms-threshold-px",
        help="--check: median reprojection RMS (px) at/below which the deployed calibration passes.",
    ),
    no_drift: bool = typer.Option(
        False,
        "--no-drift",
        help="--check: skip the fresh-calibration drift comparison (reprojection RMS only).",
    ),
) -> None:
    """Calibrate camera intrinsics and write ROS CameraInfo YAML.

    With ``--check`` the command scores how well ``--camera-info`` explains the boards on this
    source (running a fresh calibration on the same frames for the drift comparison) rather than
    producing a calibration: it writes a JSON report to ``--out`` (or a default next to the source),
    and on a DEGRADED result offers to write a fresh CameraInfo YAML from the same frames.
    """
    if board is BoardType.chessboard and (cols is None or rows is None):
        raise typer.BadParameter("--cols and --rows are required when --board chessboard")
    if board is BoardType.charuco and (squares_x is None or squares_y is None):
        raise typer.BadParameter("--squares-x and --squares-y are required when --board charuco")
    resolved_cols = int(cols) if cols is not None else 0
    resolved_rows = int(rows) if rows is not None else 0

    if check:
        if camera_info is None:
            raise typer.BadParameter("--camera-info is required with --check")
        run_check_report(
            source=source,
            device_index=device_index,
            images=images,
            topic=topic,
            topic_timeout_sec=topic_timeout_sec,
            cols=resolved_cols,
            rows=resolved_rows,
            square_size_m=square_size_m,
            camera_info=camera_info,
            rms_threshold_px=rms_threshold_px,
            drift_threshold_frac=_DEFAULT_CHECK_DRIFT_THRESHOLD_FRAC,
            check_drift=not no_drift,
            out=out,
            target_count=target_count,
            no_display=no_display,
            board=board,
            dict_name=charuco_dict,
            squares_x=squares_x,
            squares_y=squares_y,
            marker_size_m=marker_size_m,
            marker_ratio=marker_ratio,
        )
        return

    if preview_out is not None and out is None:
        raise typer.BadParameter("preview output requires --out")

    try:
        result = run_calibration(
            source=source,
            device_index=device_index,
            images=images,
            topic=topic,
            topic_timeout_sec=topic_timeout_sec,
            cols=resolved_cols,
            rows=resolved_rows,
            square_size_m=square_size_m,
            out=out,
            preview_out=preview_out,
            camera_name=camera_name,
            target_count=target_count,
            no_display=no_display,
            distortion_model=distortion_model,
            board=board,
            dict_name=charuco_dict,
            squares_x=squares_x,
            squares_y=squares_y,
            marker_size_m=marker_size_m,
            marker_ratio=marker_ratio,
        )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"RMS: {float(result['rms']):.6f} px ({int(result['n_used'])} frame(s) used)")
    typer.echo(
        f"Detected pattern: {tuple(result.get('pattern_size', (cols, rows)))} "
        f"({result.get('pattern_label', 'requested inner corners')})"
    )
    if out is not None:
        typer.echo(f"Wrote camera info YAML to {out}")
    if preview_out is not None:
        typer.echo(f"Wrote preview overlay PNG to {preview_out}")


def main(args: list[str] | None = None) -> None:
    """CLI entry point."""
    app(args=args)
