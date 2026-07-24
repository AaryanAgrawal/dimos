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
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
import warnings

# Default OpenCL off: on Apple Silicon, CPU chessboard detection is often faster and more stable.
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import numpy as np
import typer
import yaml

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg"})
_DEFAULT_CHECK_RMS_THRESHOLD_PX = 1.0  # median reprojection RMS a deployed calib must stay at/below
_DEFAULT_CHECK_DRIFT_THRESHOLD_FRAC = 0.05  # relative intrinsics drift (deployed vs fresh)
_MIN_DRIFT_CALIBRATION_FRAMES = 3  # >= 3 planar views to separate fx/fy/cx/cy plus distortion
_MIN_CHARUCO_CORNERS = 6  # solvePnP needs >= 4; 6 keeps per-view pose and drift solve well-posed
_DEFAULT_CHARUCO_DICT = "DICT_4X4_50"
_DEFAULT_MARKER_RATIO = 0.8  # markerLength / squareLength for a standard ChArUco print
_COVERAGE_GRID = 6  # image binned into GRID x GRID cells for the capture coverage overlay
_WINDOW = "dimos cameracalibrate"
_MAX_WEBCAM_READ_FAILURES = 30


@dataclass(frozen=True)
class _Detection:
    """One frame's board detection; ``charuco_ids`` is None for a plain chessboard."""

    corners_px: np.ndarray  # (N, 1, 2) float32 pixel corners
    object_points_m: np.ndarray  # (N, 3) float32 board-frame points, meters
    charuco_ids: np.ndarray | None = None  # (N, 1) int32


@dataclass(frozen=True)
class _CharucoSpec:
    """Built ``cv2.aruco.CharucoBoard`` (meters) + one reused detector + report geometry."""

    board: Any  # cv2.aruco.CharucoBoard (no cv2 stub)
    detector: Any  # cv2.aruco.CharucoDetector
    dict_name: str
    squares_x: int
    squares_y: int
    square_size_m: float


def write_camera_info_yaml(
    path: str,
    *,
    image_width: int,
    image_height: int,
    camera_name: str,
    K: np.ndarray,
    D: np.ndarray,
    distortion_model: str = "plumb_bob",
) -> None:
    """Write ROS-style CameraInfo YAML loadable by ``CameraInfo.from_yaml``."""
    k = np.asarray(K, dtype=np.float64).reshape(3, 3)
    d = np.asarray(D, dtype=np.float64).ravel().tolist()
    # P = [K | 0] and R = identity: a monocular calibration has no rectification.
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    p_flat = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    payload = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_name": camera_name,
        "distortion_model": distortion_model,
        "camera_matrix": {"rows": 3, "cols": 3, "data": k.ravel().tolist()},
        "distortion_coefficients": {"rows": 1, "cols": len(d), "data": d},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": np.eye(3).ravel().tolist()},
        "projection_matrix": {"rows": 3, "cols": 4, "data": p_flat},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def find_chessboard_corners(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """Detect (cols, rows) inner corners: SB detector first, classic + cornerSubPix fallback."""
    ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok and corners is not None:
        return np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok or corners is None:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return np.asarray(refined, dtype=np.float32).reshape(-1, 1, 2)


def _detect_charuco(gray: np.ndarray, spec: _CharucoSpec) -> _Detection | None:
    """Interpolate ChArUco corners in one frame; None below ``_MIN_CHARUCO_CORNERS``."""
    corners, ids, _marker_corners, _marker_ids = spec.detector.detectBoard(gray)
    if ids is None or corners is None or len(ids) < _MIN_CHARUCO_CORNERS:
        return None
    ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
    # Object points come straight from the board geometry, selected by corner id.
    objp_m = np.asarray(spec.board.getChessboardCorners(), dtype=np.float32)[ids_flat]
    corners_px = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return _Detection(corners_px, objp_m, ids_flat.reshape(-1, 1))


def _board_setup(
    board: str,
    cols: int,
    rows: int,
    square_size_m: float,
    dict_name: str,
    squares_x: int | None,
    squares_y: int | None,
    marker_size_m: float | None,
    marker_ratio: float,
) -> tuple[_CharucoSpec | None, Callable[..., Any], Callable[..., Any]]:
    """Validate board flags; return ``(charuco_spec_or_None, detect(frame), draw(preview, det))``."""
    if square_size_m <= 0:
        raise ValueError(f"square_size_m must be > 0, got {square_size_m}")
    if board == "charuco":
        if squares_x is None or squares_y is None:
            raise ValueError("--squares-x and --squares-y are required when --board charuco")
        if squares_x < 2 or squares_y < 2:
            raise ValueError(f"charuco needs squares_x/squares_y >= 2, got {squares_x}x{squares_y}")
        dict_id = getattr(cv2.aruco, dict_name, None)
        if not isinstance(dict_id, int):
            raise ValueError(
                f"unknown aruco dictionary {dict_name!r}; want a cv2.aruco.DICT_* name "
                "(e.g. DICT_4X4_50, DICT_5X5_100, DICT_APRILTAG_36h11)"
            )
        marker_m = (
            float(marker_size_m) if marker_size_m is not None else square_size_m * marker_ratio
        )
        if not 0.0 < marker_m < square_size_m:
            raise ValueError(
                f"marker size must satisfy 0 < marker < square; got marker={marker_m} m, "
                f"square={square_size_m} m (check --marker-size-m / --marker-ratio)"
            )
        charuco_board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_size_m,
            marker_m,
            cv2.aruco.getPredefinedDictionary(dict_id),
        )
        spec = _CharucoSpec(
            board=charuco_board,
            detector=cv2.aruco.CharucoDetector(charuco_board),
            dict_name=dict_name,
            squares_x=squares_x,
            squares_y=squares_y,
            square_size_m=square_size_m,
        )

        def detect(frame: np.ndarray) -> _Detection | None:
            return _detect_charuco(_gray(frame), spec)

        def draw(preview: np.ndarray, det: _Detection) -> None:
            cv2.aruco.drawDetectedCornersCharuco(preview, det.corners_px, det.charuco_ids)

        return spec, detect, draw

    if board != "chessboard":
        raise ValueError(f"board must be 'chessboard' or 'charuco', got {board!r}")
    if cols < 1 or rows < 1:
        raise ValueError(f"cols and rows must be >= 1, got {cols}x{rows}")
    # Inner-corner grid on Z=0, XY spacing square_size_m (meters); same ordering as the detector.
    objp_m = np.zeros((rows * cols, 3), dtype=np.float32)
    objp_m[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp_m *= square_size_m

    def detect(frame: np.ndarray) -> _Detection | None:
        corners = find_chessboard_corners(_gray(frame), cols, rows)
        return None if corners is None else _Detection(corners, objp_m)

    def draw(preview: np.ndarray, det: _Detection) -> None:
        cv2.drawChessboardCorners(preview, (cols, rows), det.corners_px, True)

    return None, detect, draw


def _touched_cells(corners_px: np.ndarray, image_wh: tuple[int, int]) -> set[tuple[int, int]]:
    """Coverage-grid cells containing at least one detected corner."""
    pts = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    cx = np.clip(
        (pts[:, 0] / max(image_wh[0], 1) * _COVERAGE_GRID).astype(int), 0, _COVERAGE_GRID - 1
    )
    cy = np.clip(
        (pts[:, 1] / max(image_wh[1], 1) * _COVERAGE_GRID).astype(int), 0, _COVERAGE_GRID - 1
    )
    return set(zip(cx.tolist(), cy.tolist(), strict=True))


def _draw_coverage_overlay(preview: np.ndarray, covered_cells: set[tuple[int, int]]) -> None:
    """Outline grid cells the board has not swept yet; a visual hint only, never gates SPACE."""
    h, w = preview.shape[:2]
    for cx in range(_COVERAGE_GRID):
        for cy in range(_COVERAGE_GRID):
            if (cx, cy) in covered_cells:
                continue
            x0, y0 = round(cx * w / _COVERAGE_GRID), round(cy * h / _COVERAGE_GRID)
            x1, y1 = round((cx + 1) * w / _COVERAGE_GRID), round((cy + 1) * h / _COVERAGE_GRID)
            cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 200, 255), thickness=1)


def _interactive_capture(
    next_frame: Callable[[], np.ndarray | None],
    target_count: int,
    detect: Callable[[np.ndarray], _Detection | None],
    draw: Callable[[np.ndarray, _Detection], None],
    *,
    no_display: bool,
) -> tuple[list[np.ndarray], list[_Detection]]:
    """SPACE accepts a frame when the board is detected, q quits; returns (frames, detections)."""
    if target_count < 1:
        raise ValueError("target_count must be >= 1")
    frames: list[np.ndarray] = []
    detections: list[_Detection] = []
    covered_cells: set[tuple[int, int]] = set()
    try:
        while len(frames) < target_count:
            frame = next_frame()
            if frame is None:
                continue
            det = detect(frame)
            preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
            if det is not None:
                covered_cells |= _touched_cells(
                    det.corners_px, (preview.shape[1], preview.shape[0])
                )
                draw(preview, det)
                detail, color = (
                    f"Detected {det.corners_px.shape[0]} corners - SPACE saves",
                    (0, 180, 0),
                )
            else:
                detail, color = "No board detected - SPACE ignored", (0, 0, 255)
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 58), (0, 0, 0), thickness=-1)
            status = f"Accepted {len(frames)}/{target_count}"
            cv2.putText(
                preview, status, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(preview, detail, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if not no_display:
                _draw_coverage_overlay(preview, covered_cells)
                cv2.imshow(_WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and det is not None:
                frames.append(frame.copy())
                detections.append(det)
            elif key == ord("q"):
                break
        if len(frames) < target_count:
            raise RuntimeError(
                f"Capture ended with {len(frames)} of {target_count} frames "
                "(quit early, missing detections on SPACE, or read failures)."
            )
        return frames, detections
    finally:
        if not no_display:
            try:
                cv2.destroyWindow(_WINDOW)
            except cv2.error:
                pass
            cv2.waitKey(1)


def _ingest(
    *,
    source: str,
    device_index: int,
    images: Path | None,
    topic: str | None,
    topic_timeout_sec: float,
    target_count: int,
    no_display: bool,
    detect: Callable[[np.ndarray], _Detection | None],
    draw: Callable[[np.ndarray, _Detection], None],
) -> tuple[list[np.ndarray], list[_Detection]]:
    """Frames + detections from the requested source (folder is offline, others interactive)."""
    if source == "folder":
        if images is None:
            raise ValueError("--images is required when --source folder")
        if not images.is_dir():
            raise ValueError(f"Not a directory: {images}")
        frames = []
        for p in sorted(
            f for f in images.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        ):
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"Could not read image: {p}")
            frames.append(img)
        if any(f.shape[:2] != frames[0].shape[:2] for f in frames):
            raise ValueError("All frames must have the same shape.")
        return frames, [d for d in map(detect, frames) if d is not None]

    if source == "topic":
        if topic is None:
            raise ValueError(
                "--topic is required when --source topic (e.g. --topic jpeg_lcm:/color_image)"
            )
        from dimos.msgs.sensor_msgs.Image import Image
        from dimos.protocol.pubsub.registry import subscribe_pubsub_uri

        latest: list[np.ndarray | None] = [None]
        started = time.time()
        lock = threading.Lock()

        def _on_image(msg: Any) -> None:
            try:
                arr = msg.to_opencv()
            except (AttributeError, ValueError):
                return
            with lock:
                latest[0] = np.asarray(arr)

        transport, unsub = subscribe_pubsub_uri(topic, _on_image, msg_type=Image)

        def _next_topic() -> np.ndarray | None:
            with lock:
                frame = latest[0]
            if frame is None:
                if time.time() - started > topic_timeout_sec:
                    raise RuntimeError(
                        f"No frames received on topic {topic!r} within {topic_timeout_sec:.1f}s."
                    )
                time.sleep(0.01)  # yield to the subscriber callback thread
                return None
            return frame

        try:
            return _interactive_capture(
                _next_topic, target_count, detect, draw, no_display=no_display
            )
        finally:
            # Best-effort teardown; never mask the capture error.
            try:
                unsub()
            except Exception:
                pass
            try:
                transport.stop()
            except Exception:
                pass

    if source != "webcam":
        raise ValueError(f"source must be 'webcam', 'folder', or 'topic', got {source!r}")
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera device_index={device_index!r}")
    failures = 0

    def _next_webcam() -> np.ndarray | None:
        nonlocal failures
        ok, frame = cap.read()
        if not ok or frame is None:
            failures += 1
            if failures >= _MAX_WEBCAM_READ_FAILURES:
                raise RuntimeError(
                    f"Failed to read from camera device_index={device_index!r} for "
                    f"{_MAX_WEBCAM_READ_FAILURES} consecutive attempts."
                )
            return None
        failures = 0
        return frame

    try:
        return _interactive_capture(_next_webcam, target_count, detect, draw, no_display=no_display)
    finally:
        cap.release()


def _calibrate_views(
    detections: list[_Detection],
    image_size_wh: tuple[int, int],
    distortion_model: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Intrinsics from per-view correspondences: ``cv2.calibrateCamera`` / ``cv2.fisheye.calibrate``."""
    if distortion_model == "fisheye":
        # cv2.fisheye.calibrate is strict: float64 with an explicit middle axis per view.
        objpoints = [d.object_points_m.astype(np.float64).reshape(-1, 1, 3) for d in detections]
        imgpoints = [d.corners_px.astype(np.float64).reshape(-1, 1, 2) for d in detections]
        K0 = np.zeros((3, 3), dtype=np.float64)
        D0 = np.zeros((4, 1), dtype=np.float64)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in detections]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in detections]
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        try:
            rms, K, D, _rvecs, _tvecs = cv2.fisheye.calibrate(
                objpoints, imgpoints, image_size_wh, K0, D0, rvecs, tvecs, flags, criteria
            )
        except cv2.error as exc:
            raise ValueError(
                f"cv2.fisheye.calibrate did not converge on {len(detections)} view(s); "
                f"want more views with wider, corner-covering board spread (OpenCV: {exc})"
            ) from exc
    else:
        objpoints = [d.object_points_m for d in detections]
        imgpoints = [d.corners_px.astype(np.float32) for d in detections]
        rms, K, D, _rvecs, _tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size_wh, None, None
        )
    return float(rms), np.asarray(K, dtype=np.float64), np.asarray(D, dtype=np.float64).reshape(-1)


def run_calibration(
    *,
    source: str,
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
    distortion_model: str = "plumb_bob",
    board: str = "chessboard",
    dict_name: str = _DEFAULT_CHARUCO_DICT,
    squares_x: int | None = None,
    squares_y: int | None = None,
    marker_size_m: float | None = None,
    marker_ratio: float = _DEFAULT_MARKER_RATIO,
) -> dict[str, Any]:
    """Capture frames, calibrate, and (optionally) write CameraInfo YAML + preview PNG."""
    if distortion_model not in ("plumb_bob", "fisheye"):
        raise ValueError(
            f"distortion_model must be 'plumb_bob' or 'fisheye', got {distortion_model!r}"
        )
    spec, detect, draw = _board_setup(
        board,
        cols,
        rows,
        square_size_m,
        dict_name,
        squares_x,
        squares_y,
        marker_size_m,
        marker_ratio,
    )
    frames, detections = _ingest(
        source=source,
        device_index=device_index,
        images=images,
        topic=topic,
        topic_timeout_sec=topic_timeout_sec,
        target_count=target_count,
        no_display=no_display,
        detect=detect,
        draw=draw,
    )
    if not detections:
        raise ValueError("Calibration board not found in any frame.")
    h0, w0 = frames[0].shape[:2]
    rms, K, D = _calibrate_views(detections, (int(w0), int(h0)), distortion_model)
    if spec is not None:
        pattern_size = (spec.squares_x, spec.squares_y)
        pattern_label = f"charuco {spec.dict_name} {spec.squares_x}x{spec.squares_y} squares"
    else:
        pattern_size, pattern_label = (cols, rows), "requested inner corners"
    result: dict[str, Any] = {
        "K": K,
        "D": D,
        "rms": rms,
        "image_size": (int(w0), int(h0)),
        "n_used": len(detections),
        "pattern_size": pattern_size,
        "pattern_label": pattern_label,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        # Fisheye is written under the ROS-canonical name for the Kannala-Brandt model.
        write_camera_info_yaml(
            str(out),
            image_width=int(w0),
            image_height=int(h0),
            camera_name=camera_name,
            K=K,
            D=D,
            distortion_model="equidistant" if distortion_model == "fisheye" else distortion_model,
        )
        result["out_path"] = out
    if preview_out is not None:
        # Best-effort: a preview failure must not mask the YAML already written above.
        for frame in frames:
            det = detect(frame)
            if det is None:
                continue
            preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
            draw(preview, det)
            preview_out.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(preview_out), preview):
                result["preview_path"] = preview_out
            break
        if "preview_path" not in result:
            warnings.warn(
                f"Preview PNG skipped. Camera info YAML was still written to {out}.", stacklevel=2
            )
    return result


def _reproject_rms_px(
    det: _Detection, K: np.ndarray, D: np.ndarray, *, fisheye: bool
) -> float | None:
    """Per-view reprojection RMS (px) under deployed intrinsics; None if solvePnP fails."""
    objp_m = np.asarray(det.object_points_m, dtype=np.float64).reshape(-1, 3)
    corners = np.asarray(det.corners_px, dtype=np.float64).reshape(-1, 1, 2)
    if fisheye:
        # No fisheye solvePnP exists: undistort into an ideal pinhole (P=K), then plain solvePnP.
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


def _drift_block(
    detections: list[_Detection],
    K_deployed: np.ndarray,
    D_deployed: np.ndarray,
    *,
    image_size_wh: tuple[int, int],
    model: str,
    drift_threshold_frac: float,
    run_drift: bool,
) -> dict[str, Any]:
    """Fresh calibration on the same views vs deployed intrinsics; solver failure -> ran=False."""
    if not run_drift:
        return {"ran": False, "reason": "drift check disabled (--no-drift)"}
    if len(detections) < _MIN_DRIFT_CALIBRATION_FRAMES:
        return {
            "ran": False,
            "reason": (
                f"need >= {_MIN_DRIFT_CALIBRATION_FRAMES} frames for a fresh calibration, "
                f"have {len(detections)}"
            ),
        }
    try:
        fresh_rms_px, Kf, Df = _calibrate_views(detections, image_size_wh, model)
    except (cv2.error, ValueError) as exc:
        return {"ran": False, "reason": f"fresh calibration failed: {exc}"}
    Kd = np.asarray(K_deployed, dtype=np.float64).reshape(3, 3)
    # Deltas are deployed - fresh (px); max_rel is the largest per-parameter relative drift.
    deltas = {
        "delta_fx_px": float(Kd[0, 0] - Kf[0, 0]),
        "delta_fy_px": float(Kd[1, 1] - Kf[1, 1]),
        "delta_cx_px": float(Kd[0, 2] - Kf[0, 2]),
        "delta_cy_px": float(Kd[1, 2] - Kf[1, 2]),
    }
    fresh = [Kf[0, 0], Kf[1, 1], Kf[0, 2], Kf[1, 2]]
    max_rel = max(
        abs(d) / max(abs(float(f)), 1e-9) for d, f in zip(deltas.values(), fresh, strict=True)
    )
    Dd = np.asarray(D_deployed, dtype=np.float64).ravel()
    # Empty when coefficient counts differ (deployed and fresh lens models disagree).
    dist_abs_delta = np.abs(Dd - Df).tolist() if Dd.shape == Df.shape else []
    return {
        "ran": True,
        "reason": "",
        "fresh_rms_px": float(fresh_rms_px),
        **deltas,
        "max_rel_intrinsics_drift": float(max_rel),
        "distortion_abs_delta": [float(x) for x in dist_abs_delta],
        "drift_small": bool(max_rel <= drift_threshold_frac),
        "fresh_K": Kf.tolist(),
        "fresh_D": [float(x) for x in Df.tolist()],
    }


def run_check_report(
    *,
    source: str,
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
    board: str = "chessboard",
    dict_name: str = _DEFAULT_CHARUCO_DICT,
    squares_x: int | None = None,
    squares_y: int | None = None,
    marker_size_m: float | None = None,
    marker_ratio: float = _DEFAULT_MARKER_RATIO,
) -> None:
    """``--check``: score a DEPLOYED CameraInfo against detected boards, write JSON, print evidence.

    Verdict is OK iff the median reprojection RMS stays at/below ``rms_threshold_px`` AND the
    fresh-calibration drift (same frames, deployed lens model) is small. On DEGRADED, an
    interactive TTY offers to write the fresh calibration to a new CameraInfo YAML (no re-solve).
    """
    from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo  # heavy; only --check needs it

    # is_fisheye_model (incl. kannala_brandt) is the verified source of truth for model names.
    from dimos.perception.fiducial.marker_pose import is_fisheye_model

    try:
        if not camera_info.is_file():
            raise ValueError(f"--camera-info not found: {camera_info}")
        spec, detect, draw = _board_setup(
            board,
            cols,
            rows,
            square_size_m,
            dict_name,
            squares_x,
            squares_y,
            marker_size_m,
            marker_ratio,
        )
        info = CameraInfo.from_yaml(str(camera_info))
        K = np.asarray(info.get_K_matrix(), dtype=np.float64).reshape(3, 3)
        D = np.asarray(info.get_D_coeffs(), dtype=np.float64).reshape(-1, 1)
        distortion_model = info.distortion_model or "plumb_bob"
        fisheye = is_fisheye_model(distortion_model)

        frames, detections = _ingest(
            source=source,
            device_index=device_index,
            images=images,
            topic=topic,
            topic_timeout_sec=topic_timeout_sec,
            target_count=target_count,
            no_display=no_display,
            detect=detect,
            draw=draw,
        )
        if not detections:
            raise ValueError("Calibration board not detected in any frame; nothing to check.")
        h0, w0 = frames[0].shape[:2]
        image_size_wh = (int(w0), int(h0))

        per_view_rms_px = [
            rms
            for det in detections
            if (rms := _reproject_rms_px(det, K, D, fisheye=fisheye)) is not None
        ]
        if not per_view_rms_px:
            raise ValueError("solvePnP failed on every view; cannot assess reprojection error.")
        # Fresh refit in the DEPLOYED lens model so the intrinsics/distortion are comparable.
        drift = _drift_block(
            detections,
            K,
            D,
            image_size_wh=image_size_wh,
            model="fisheye" if fisheye else "plumb_bob",
            drift_threshold_frac=drift_threshold_frac,
            run_drift=check_drift,
        )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    median_rms_px = float(np.median(per_view_rms_px))
    p90_rms_px = float(np.percentile(per_view_rms_px, 90))
    drift_small = bool(drift.get("drift_small", True)) if drift["ran"] else True
    verdict = "OK" if (median_rms_px <= rms_threshold_px and drift_small) else "DEGRADED"
    if spec is not None:
        flags = (
            f"--board charuco --dict {spec.dict_name} --squares-x {spec.squares_x} "
            f"--squares-y {spec.squares_y} --square-size-m {spec.square_size_m}"
        )
    else:
        flags = "--cols C --rows R --square-size-m S"
    recommendation = (
        ""
        if verdict == "OK"
        else f"recalibrate this unit (dimos cameracalibrate --source <topic|folder> {flags} --out <new.yaml>)"
    )
    # Intrinsics are only valid at their own resolution; a mismatch makes the RMS meaningless.
    resolution_warning = ""
    if info.width and info.height and (int(info.width), int(info.height)) != image_size_wh:
        resolution_warning = (
            f"deployed CameraInfo is {info.width}x{info.height} but frames are "
            f"{w0}x{h0}; reprojection RMS is not meaningful across resolutions."
        )

    if out is not None:
        out_path = out
    elif source == "folder" and images is not None:
        out_path = images / "calibration_check.json"
    else:
        out_path = Path.cwd() / "calibration_check.json"
    try:
        git_rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_rev = ""
    payload = {
        "tool": "dimos cameracalibrate --check",
        "verdict": verdict,
        "median_reproj_rms_px": median_rms_px,
        "p90_reproj_rms_px": p90_rms_px,
        "n_frames_used": len(per_view_rms_px),
        "rms_threshold_px": float(rms_threshold_px),
        "recommendation": recommendation,
        "resolution_warning": resolution_warning,
        "drift": drift,
        "provenance": {
            "camera_info_path": str(camera_info),
            "distortion_model": distortion_model,
            "source": source,
            "cols": cols,
            "rows": rows,
            "square_size_m": square_size_m,
            "image_size_wh": list(image_size_wh),
            "git_rev": git_rev or "unknown",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    typer.echo(f"CALIBRATION CHECK: {verdict}")
    typer.echo(f"  deployed CameraInfo: {camera_info} ({distortion_model})")
    typer.echo(
        f"  reprojection RMS px: median={median_rms_px:.3f} p90={p90_rms_px:.3f}  "
        f"(n={len(per_view_rms_px)} frames, threshold={rms_threshold_px:.3f})"
    )
    if drift["ran"]:
        typer.echo(f"  fresh calibration RMS px: {drift['fresh_rms_px']:.3f}")
        typer.echo(
            "  intrinsics drift (deployed - fresh): "
            f"dfx={drift['delta_fx_px']:+.2f} dfy={drift['delta_fy_px']:+.2f} "
            f"dcx={drift['delta_cx_px']:+.2f} dcy={drift['delta_cy_px']:+.2f} px "
            f"(max rel {drift['max_rel_intrinsics_drift']:.2%})"
        )
        typer.echo(f"  distortion |delta|: {[round(x, 4) for x in drift['distortion_abs_delta']]}")
    else:
        typer.echo(f"  drift check: skipped ({drift['reason']})")
    if resolution_warning:
        typer.echo(f"  WARNING: {resolution_warning}")
    if recommendation:
        typer.echo(f"  -> {recommendation}")
    typer.echo(f"Wrote check JSON to {out_path}")

    if verdict == "DEGRADED" and drift["ran"] and sys.stdin.isatty():
        # Interactive offer: reuse the fresh calibration the drift solve already computed.
        if typer.confirm(
            "Deployed calibration is DEGRADED. Write the fresh calibration from these "
            "frames to a new CameraInfo YAML?",
            default=False,
        ):
            target = Path(
                typer.prompt(
                    "Output YAML path", default=str(out_path.with_name("recalibrated.yaml"))
                )
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            # The fresh solve used the deployed lens model, so its ROS model name carries over.
            write_camera_info_yaml(
                str(target),
                image_width=image_size_wh[0],
                image_height=image_size_wh[1],
                camera_name="recalibrated",
                K=np.asarray(drift["fresh_K"], dtype=np.float64).reshape(3, 3),
                D=np.asarray(drift["fresh_D"], dtype=np.float64).ravel(),
                distortion_model=distortion_model,
            )
            typer.echo(f"Wrote fresh recalibration YAML to {target}")
