# feat(cameracalibrate): charuco boards and a --check mode

## Problem

`dimos cameracalibrate` handled a chessboard only, and there was no way to tell whether a shipped calibration had drifted. Charuco boards recover intrinsics from partial/occluded views a plain chessboard can't, and a drifted calibration silently poisons every downstream pose estimate.

## What this adds

- **Charuco board support** — calibrate from a charuco board, not just a chessboard. Robust to occlusion and partial views (each marker is uniquely identified). Feeds the *existing* `cv2.calibrateCamera` / `cv2.fisheye.calibrate` solvers.
- **`--check` mode** — re-score an existing calibration against fresh frames: per-view solvePnP reprojection RMS + drift vs a fresh calibration in the deployed lens model → an `OK` / `DEGRADED` verdict and a JSON report.
- **Light coverage overlay** — during capture, a hint overlay that clears where the board has already swept (space to capture, `q` to quit). Hint only; never gates capture.

Additive: chessboard detection, both calibration solvers, `write_camera_info_yaml`, and the capture loop are untouched.

## Breaking Changes

---
- Board size is given as **inner corners** exactly (no square-count / rotated auto-guess).
- `--check` on a charuco target solves via `cv2.calibrateCamera` over the id-selected correspondences (same math as `calibrateCameraCharuco`).
---

## Core changes — why

- **`utils/cli/cameracalibrate/cameracalibrate.py`** — one detector path per board, one `cv2.calibrateCamera` / `cv2.fisheye.calibrate` over per-view correspondences, `write_camera_info_yaml` for the saved intrinsics. Uses verified OpenCV throughout; `is_fisheye_model` is imported from `marker_pose`, not re-implemented.
- **`robot/cli/dimos.py`** — the `cameracalibrate` command wiring (flags → `run_calibration` / `run_check_report`).

## How to Test

**Hardware.** Point a camera at a charuco (or chessboard), capture ~15 views, calibrate, then `--check` the result against fresh frames:

```
uv run dimos cameracalibrate --source webcam --board charuco --cols 9 --rows 6
uv run dimos cameracalibrate --check --calibration <cam>.yaml --source webcam
```

Test to read: `test_cameracalibrate.py::test_write_camera_info_yaml_round_trips_through_dimos_camera_info` — the saved YAML loads back identically through `DimosCameraInfo`. Calibration math itself is verified OpenCV.

## AI assistance

Claude Code (Opus 4.8 / Fable 5) — implementation + simplification; all changes reviewed.

## Checklist

- [ ] Scoped to one problem.
- [ ] Ran `uv run pytest` + pre-commit for the changed files.
- [ ] Reviewed every line.
- [ ] Disclosed AI assistance.
- [ ] Read + approved the CLA.
