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

"""Tests for ChArUco support in the cameracalibrate CLI (calibrate + --check, plumb_bob + fisheye).

All synthetic: a ChArUco board's corners are projected through a KNOWN ground-truth K/D at
deterministic poses (seeded rng) via ``cv2.projectPoints`` (plumb_bob) or ``cv2.fisheye.projectPoints``
(equidistant); the folder tests warp a rendered board so the REAL cv2.aruco detector runs. No live
capture, no replay, no display.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner
import yaml

from dimos.utils.cli.cameracalibrate.cameracalibrate import (
    _MIN_CHARUCO_CORNERS,
    CharucoBoardSpec,
    _calibrate_charuco_fisheye,
    _capture_frames_from_webcam,
    _CharucoDetection,
    app,
    build_charuco_board,
    calibrate_from_frames_charuco,
    check_calibration_charuco,
    write_camera_info_yaml,
)

_WIDTH, _HEIGHT = 640, 480
_SQUARES_X, _SQUARES_Y = 8, 6
_SQUARE_SIZE_M = 0.03
_MARKER_RATIO = 0.8
_DICT = "DICT_4X4_50"
_K_TRUE = np.array(
    [[512.0, 0.0, 318.5], [0.0, 508.0, 242.3], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
_D_TRUE = np.array([-0.08, 0.02, 0.0, 0.0, 0.0], dtype=np.float64)  # small plumb_bob distortion
_D_ZERO = np.zeros(5, dtype=np.float64)
# Fisheye ground truth for the equidistant recovery test: fx == fy, principal point at center,
# 4-coeff Kannala-Brandt (k1..k4), a small physical barrel.
_K_FISHEYE = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
_D_FISHEYE = np.array([0.02, -0.01, 0.004, -0.001], dtype=np.float64)


def _board_spec() -> CharucoBoardSpec:
    return build_charuco_board(
        dict_name=_DICT,
        squares_x=_SQUARES_X,
        squares_y=_SQUARES_Y,
        square_size_m=_SQUARE_SIZE_M,
        marker_ratio=0.8,
    )


def _synthetic_charuco_detections(
    spec: CharucoBoardSpec,
    *,
    K: np.ndarray = _K_TRUE,
    D: np.ndarray = _D_TRUE,
    count: int = 14,
    noise_px: float = 0.03,
    seed: int = 7,
) -> list[_CharucoDetection]:
    """Project the FULL ChArUco board through ``K``/``D`` at deterministic poses -> detections.

    Truth object points come straight from ``board.getChessboardCorners()`` and the pixels from
    ``cv2.projectPoints`` -- no re-implemented marker/corner math. All ids are present (full board),
    with a tiny seeded sub-pixel noise so the reprojection RMS is realistically nonzero.
    """
    object_points_m = np.asarray(spec.board.getChessboardCorners(), dtype=np.float32)
    n_corners = object_points_m.shape[0]
    ids = np.arange(n_corners, dtype=np.int32).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    out: list[_CharucoDetection] = []
    while len(out) < count:
        rvec = rng.uniform(-0.25, 0.25, size=3).astype(np.float64)
        tvec = np.array(
            [rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(0.4, 0.6)],
            dtype=np.float64,
        )
        imgpts, _ = cv2.projectPoints(object_points_m, rvec, tvec, K, D)
        pts_px = np.asarray(imgpts, dtype=np.float64).reshape(-1, 2)
        if (
            pts_px[:, 0].min() < 10
            or pts_px[:, 0].max() > _WIDTH - 10
            or pts_px[:, 1].min() < 10
            or pts_px[:, 1].max() > _HEIGHT - 10
        ):
            continue
        pts_px += rng.normal(0.0, noise_px, size=pts_px.shape)
        out.append(
            _CharucoDetection(
                pts_px.astype(np.float32).reshape(-1, 1, 2),
                ids.copy(),
                object_points_m.copy(),
            )
        )
    return out


def _check_kwargs(K: np.ndarray, D: np.ndarray, spec: CharucoBoardSpec) -> dict[str, object]:
    return {
        "board_spec": spec,
        "K_deployed": K,
        "D_deployed": D,
        "distortion_model": "plumb_bob",
        "image_size_wh": (_WIDTH, _HEIGHT),
    }


def _synthetic_charuco_fisheye_detections(
    spec: CharucoBoardSpec, *, count: int = 15, seed: int = 0
) -> list[_CharucoDetection]:
    """Project the board's corners through the KNOWN fisheye ``_K_FISHEYE``/``_D_FISHEYE`` at seeded poses.

    Object points come straight from ``board.getChessboardCorners()`` (meters); pixels from
    ``cv2.fisheye.projectPoints`` -- no re-implemented projection. Every fourth view is a PARTIAL
    board (random id subset above the min-corner gate) so the per-view variable-N handling in
    ``_calibrate_charuco_fisheye`` is exercised. No pixel noise: recovery is essentially exact.
    """
    obj_all = np.asarray(spec.board.getChessboardCorners(), dtype=np.float64)
    n_corners = obj_all.shape[0]
    all_ids = np.arange(n_corners, dtype=np.int32)
    rng = np.random.default_rng(seed)
    out: list[_CharucoDetection] = []
    while len(out) < count:
        rvec = rng.uniform(-0.35, 0.35, size=3).reshape(1, 1, 3)
        tvec = np.array(
            [rng.uniform(-0.06, 0.06), rng.uniform(-0.05, 0.05), rng.uniform(0.32, 0.55)],
            dtype=np.float64,
        ).reshape(1, 1, 3)
        sel = (
            np.sort(rng.choice(n_corners, size=20, replace=False)).astype(np.int32)
            if len(out) % 4 == 3
            else all_ids
        )
        objp = obj_all[sel].reshape(-1, 1, 3)
        imgpts, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, _K_FISHEYE, _D_FISHEYE)
        px = np.asarray(imgpts, dtype=np.float64).reshape(-1, 2)
        # Keep the whole (partial) board strictly in-frame so the solve stays well-conditioned.
        if px[:, 0].min() < 5 or px[:, 0].max() > _WIDTH - 5:
            continue
        if px[:, 1].min() < 5 or px[:, 1].max() > _HEIGHT - 5:
            continue
        out.append(
            _CharucoDetection(
                px.astype(np.float32).reshape(-1, 1, 2),
                sel.reshape(-1, 1),
                obj_all[sel].astype(np.float32),
            )
        )
    return out


# --- build_charuco_board validation -----------------------------------------------------------


def test_build_charuco_board_metric_object_points() -> None:
    """Board built with meter lengths -> object points span (squares-1) * square_size in meters."""
    spec = _board_spec()
    obj = np.asarray(spec.board.getChessboardCorners())
    assert obj.shape == ((_SQUARES_X - 1) * (_SQUARES_Y - 1), 3)
    # Interior corners span (squares_x - 1) gaps of one square along X.
    assert obj[:, 0].max() == pytest.approx((_SQUARES_X - 1) * _SQUARE_SIZE_M, rel=1e-5)
    assert spec.marker_size_m == pytest.approx(_SQUARE_SIZE_M * 0.8, rel=1e-9)


def test_build_charuco_board_rejects_unknown_dict() -> None:
    with pytest.raises(ValueError, match="unknown aruco dictionary"):
        build_charuco_board(dict_name="DICT_NOPE", squares_x=8, squares_y=6, square_size_m=0.03)


def test_build_charuco_board_rejects_marker_ge_square() -> None:
    with pytest.raises(ValueError, match="0 < marker < square"):
        build_charuco_board(
            dict_name=_DICT, squares_x=8, squares_y=6, square_size_m=0.03, marker_size_m=0.05
        )


# --- check_calibration_charuco: the required verdict flip -------------------------------------


@pytest.mark.parametrize(("fx_scale", "verdict"), [(1.0, "OK"), (1.08, "DEGRADED")])
def test_check_charuco_verdict_tracks_intrinsics(fx_scale: float, verdict: str) -> None:
    """Deployed K/D == truth -> OK, low RMS, small drift; fx off 8% -> DEGRADED, high RMS, drift."""
    spec = _board_spec()
    detections = _synthetic_charuco_detections(spec, count=14)
    K = _K_TRUE.copy()
    K[0, 0] *= fx_scale
    result = check_calibration_charuco(detections, **_check_kwargs(K, _D_TRUE, spec))  # type: ignore[arg-type]

    assert result["verdict"] == verdict
    assert result["n_frames_used"] == 14
    assert result["drift"]["ran"] is True
    if verdict == "OK":
        assert result["median_reproj_rms_px"] < 1.0
        assert result["drift"]["drift_small"] is True
    else:
        assert result["median_reproj_rms_px"] > 1.0
        assert result["drift"]["drift_small"] is False
        assert result["drift"]["max_rel_intrinsics_drift"] > 0.05
        assert "charuco" in result["recommendation"]


def test_check_charuco_empty_detections_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        check_calibration_charuco([], **_check_kwargs(_K_TRUE, _D_TRUE, _board_spec()))  # type: ignore[arg-type]


# --- folder end-to-end over a rendered board (REAL cv2.aruco detection path) -------------------


def _synthetic_detectable_charuco_frames(
    spec: CharucoBoardSpec, count: int = 12
) -> list[np.ndarray]:
    """Warp a RENDERED ChArUco board to known poses so the real detector finds corners."""
    board_img = spec.board.generateImage((_WIDTH * 2, _HEIGHT * 2))
    detector = cv2.aruco.CharucoDetector(spec.board)
    flat_corners, flat_ids, _mc, _mi = detector.detectBoard(board_img)
    assert flat_ids is not None and len(flat_ids) >= 6
    src = np.asarray(flat_corners, dtype=np.float32).reshape(-1, 2)
    ids_flat = np.asarray(flat_ids, dtype=np.int32).reshape(-1)
    object_points_m = np.asarray(spec.board.getChessboardCorners(), dtype=np.float32)[ids_flat]

    rng = np.random.default_rng(3)
    frames: list[np.ndarray] = []
    for _ in range(600):
        if len(frames) >= count:
            break
        rvec = rng.uniform(-0.18, 0.18, size=3).astype(np.float64)
        tvec = np.array(
            [rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), rng.uniform(0.45, 0.6)],
            dtype=np.float64,
        )
        imgpts, _ = cv2.projectPoints(object_points_m, rvec, tvec, _K_TRUE, _D_ZERO)
        dst = imgpts.reshape(-1, 2).astype(np.float32)
        if (
            dst[:, 0].min() < 20
            or dst[:, 0].max() > _WIDTH - 20
            or dst[:, 1].min() < 20
            or dst[:, 1].max() > _HEIGHT - 20
        ):
            continue
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        if H is None:
            continue
        warped = cv2.warpPerspective(board_img, H, (_WIDTH, _HEIGHT), borderValue=255)
        cc, ci, _mc2, _mi2 = detector.detectBoard(warped)
        if ci is not None and len(ci) >= 6:
            frames.append(warped)
    assert len(frames) >= count, f"only rendered {len(frames)} detectable charuco frames"
    return frames[:count]


def test_calibrate_from_frames_charuco_recovers_intrinsics() -> None:
    """Full detect + calibrateCameraCharuco (default plumb_bob) recovers K near the projection K."""
    spec = _board_spec()
    frames = _synthetic_detectable_charuco_frames(spec, count=12)
    result = calibrate_from_frames_charuco(frames, spec)

    assert result["n_used"] == 12
    assert result["image_size"] == (_WIDTH, _HEIGHT)
    K = np.asarray(result["K"], dtype=np.float64)
    # Zero-distortion warped renders -> fx/fy/cx/cy within a few percent of the true projection.
    assert K[0, 0] == pytest.approx(_K_TRUE[0, 0], rel=0.05)
    assert K[1, 1] == pytest.approx(_K_TRUE[1, 1], rel=0.05)


def test_calibrate_charuco_fisheye_recovers_known_intrinsics() -> None:
    """Charuco+fisheye recovers the CONSTRUCTED fisheye K/D through ``_calibrate_charuco_fisheye``.

    Feeds known-truth per-view (objectPoints, imagePoints) through the real solver and asserts
    fx/fy/cx/cy and the 4 Kannala-Brandt coefficients land on the constructed truth (noiseless).
    """
    spec = _board_spec()
    detections = _synthetic_charuco_fisheye_detections(spec, count=15)

    rms, K_est, D_est = _calibrate_charuco_fisheye(detections, (_WIDTH, _HEIGHT))

    assert rms < 0.05, f"noiseless fisheye scene should reproject near-exactly, got rms={rms}"
    assert K_est[0, 0] == pytest.approx(_K_FISHEYE[0, 0], rel=1e-3)
    assert K_est[1, 1] == pytest.approx(_K_FISHEYE[1, 1], rel=1e-3)
    assert abs(K_est[0, 2] - _K_FISHEYE[0, 2]) < 0.5
    assert abs(K_est[1, 2] - _K_FISHEYE[1, 2]) < 0.5
    D_flat = np.asarray(D_est, dtype=np.float64).ravel()
    assert D_flat.shape == (4,), "fisheye (Kannala-Brandt) emits exactly 4 coefficients"
    assert np.allclose(D_flat, _D_FISHEYE, atol=5e-3), f"D recovery {D_flat} vs truth {_D_FISHEYE}"


def test_calibrate_charuco_fisheye_thin_views_raise_got_vs_want() -> None:
    """Views under the min-corner gate leave nothing to solve -> got-vs-want ``ValueError``."""
    spec = _board_spec()
    obj_all = np.asarray(spec.board.getChessboardCorners(), dtype=np.float64)
    thin_n = _MIN_CHARUCO_CORNERS - 1  # one short of the gate on every view
    rng = np.random.default_rng(1)
    thin: list[_CharucoDetection] = []
    for _ in range(4):
        sel = np.sort(rng.choice(obj_all.shape[0], size=thin_n, replace=False)).astype(np.int32)
        px = np.zeros((thin_n, 1, 2), dtype=np.float32)  # positions irrelevant; the gate is on N
        thin.append(_CharucoDetection(px, sel.reshape(-1, 1), obj_all[sel].astype(np.float32)))

    with pytest.raises(ValueError, match="none with enough corners"):
        _calibrate_charuco_fisheye(thin, (_WIDTH, _HEIGHT))


@pytest.mark.parametrize(("model", "n_coeffs"), [("fisheye", 4), ("plumb_bob", 5)])
def test_calibrate_from_frames_charuco_routes_by_model(model: str, n_coeffs: int) -> None:
    """``distortion_model`` routes the charuco path to the equidistant (4-coeff) or plumb_bob (5) solver."""
    spec = _board_spec()
    frames = _synthetic_detectable_charuco_frames(spec, count=12)

    result = calibrate_from_frames_charuco(frames, spec, distortion_model=model)

    assert np.asarray(result["D"], dtype=np.float64).ravel().shape == (n_coeffs,)
    assert result["n_used"] == 12
    assert result["image_size"] == (_WIDTH, _HEIGHT)


def _write_frames(images_dir: Path, frames: list[np.ndarray]) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        assert cv2.imwrite(str(images_dir / f"frame_{i:02d}.png"), frame)


def _write_deployed_yaml(path: Path, K: np.ndarray, D: np.ndarray) -> None:
    write_camera_info_yaml(
        str(path),
        image_width=_WIDTH,
        image_height=_HEIGHT,
        camera_name="synthetic_charuco_deployed",
        K=K,
        D=D,
        distortion_model="plumb_bob",
    )


def _charuco_cli_args(images: Path, deployed: Path, out_json: Path) -> list[str]:
    return [
        "--check",
        "--board",
        "charuco",
        "--source",
        "folder",
        "--images",
        str(images),
        "--camera-info",
        str(deployed),
        "--dict",
        _DICT,
        "--squares-x",
        str(_SQUARES_X),
        "--squares-y",
        str(_SQUARES_Y),
        "--square-size-m",
        str(_SQUARE_SIZE_M),
        "--marker-ratio",
        "0.8",
        "--out",
        str(out_json),
        "--no-display",
    ]


@pytest.mark.parametrize(("fx_scale", "verdict"), [(1.0, "OK"), (1.08, "DEGRADED")])
def test_cli_check_charuco_folder_verdict(tmp_path: Path, fx_scale: float, verdict: str) -> None:
    """`--check --board charuco --source folder` end-to-end: matched -> OK, fx off 8% -> DEGRADED."""
    spec = _board_spec()
    frames = _synthetic_detectable_charuco_frames(spec, count=12)
    images = tmp_path / "images"
    _write_frames(images, frames)
    K = _K_TRUE.copy()
    K[0, 0] *= fx_scale
    deployed = tmp_path / "deployed.yaml"
    _write_deployed_yaml(deployed, K, _D_ZERO)
    out_json = tmp_path / "check.json"

    result = CliRunner().invoke(app, _charuco_cli_args(images, deployed, out_json))

    assert result.exit_code == 0, result.output
    assert f"CALIBRATION CHECK: {verdict}" in result.output
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == verdict
    if verdict == "OK":
        assert payload["n_frames_used"] == 12
        assert payload["median_reproj_rms_px"] < 1.0


@pytest.mark.parametrize(
    ("model", "ros_model", "n_coeffs"),
    [("fisheye", "equidistant", 4), ("plumb_bob", "plumb_bob", 5)],
)
def test_cli_calibrate_charuco_emits_model_yaml(
    tmp_path: Path, model: str, ros_model: str, n_coeffs: int
) -> None:
    """`calibrate --board charuco --distortion-model M` writes the matching ROS model + coeff count."""
    spec = _board_spec()
    frames = _synthetic_detectable_charuco_frames(spec, count=12)
    images = tmp_path / "images"
    _write_frames(images, frames)
    out_yaml = tmp_path / "charuco_calib.yaml"

    result = CliRunner().invoke(
        app,
        [
            "--board", "charuco", "--source", "folder", "--images", str(images),
            "--distortion-model", model, "--dict", _DICT,
            "--squares-x", str(_SQUARES_X), "--squares-y", str(_SQUARES_Y),
            "--square-size-m", str(_SQUARE_SIZE_M), "--marker-ratio", str(_MARKER_RATIO),
            "--out", str(out_yaml), "--no-display",
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    payload = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert payload["distortion_model"] == ros_model
    assert payload["distortion_coefficients"]["cols"] == n_coeffs
    assert len(payload["distortion_coefficients"]["data"]) == n_coeffs
    assert payload["image_width"] == _WIDTH


def test_cli_help_lists_charuco_flags() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    output_plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    # Rich truncates long option names at narrow widths, so assert on truncation-safe substrings.
    for flag in ["--board", "--dict", "--squares-x", "--squares-y", "--marker-rat"]:
        assert flag in output_plain, output_plain


def test_cli_charuco_check_requires_squares(tmp_path: Path) -> None:
    """--board charuco without --squares-x/-y is a clean BadParameter, not a crash."""
    deployed = tmp_path / "deployed.yaml"
    _write_deployed_yaml(deployed, _K_TRUE, _D_ZERO)
    result = CliRunner().invoke(
        app,
        [
            "--check",
            "--board",
            "charuco",
            "--source",
            "folder",
            "--images",
            str(tmp_path),
            "--camera-info",
            str(deployed),
            "--square-size-m",
            str(_SQUARE_SIZE_M),
            "--no-display",
        ],
    )
    assert result.exit_code != 0
    output_plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("\n", " ")
    output_plain = re.sub(r"\s+", " ", output_plain)
    assert "squares" in output_plain, output_plain


# --- interactive webcam capture (charuco SPACE path + coverage overlay) --------


def test_capture_frames_from_webcam_charuco_space_accepts_and_draws_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ChArUco webcam capture accepts a detected board on SPACE and draws the coverage overlay."""
    spec = _board_spec()
    frame_gray = _synthetic_detectable_charuco_frames(spec, count=1)[0]
    bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    class _Cap:
        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            return True, bgr.copy()

        def release(self) -> None:
            return None

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_a, **_k: _Cap())
    monkeypatch.setattr(cv2, "waitKey", lambda _delay=0: ord(" "))
    mock_imshow = MagicMock()
    monkeypatch.setattr(cv2, "imshow", mock_imshow)
    monkeypatch.setattr(cv2, "destroyWindow", MagicMock())

    # no_display=False exercises the imshow + overlay draw path; the single frame is accepted twice.
    capture = _capture_frames_from_webcam(0, 2, 0, 0, no_display=False, charuco_spec=spec)

    assert len(capture.frames) == 2
    assert capture.charuco_detections is not None
    assert len(capture.charuco_detections) == 2
    mock_imshow.assert_called()
