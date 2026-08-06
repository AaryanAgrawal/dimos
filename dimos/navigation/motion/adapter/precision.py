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

"""How good is the control? The precision-vs-speed frontier of a recording.

For every odometry sample with an active plan: the body's actual speed
(central-difference of the tf-resolved track) against its cross-track error.
Binned by speed, the p50/p95 percentiles per bin are the empirical curve
"at speed v this controller holds X m" — and the headline is the fastest bin
whose p95 still beats the embodiment's precision floor. Several recordings
overlay on one plot, so a control change reads as one curve moving.

    python -m dimos.navigation.motion.adapter.precision rec1.mcap rec2.mcap
    python -m dimos.navigation.motion.adapter.precision rec.mcap --from 10 --to 60
"""

from __future__ import annotations

import argparse
from pathlib import Path as FsPath

import numpy as np

from dimos.navigation.motion.adapter.diagnose import (
    Recording,
    _before,
    is_hold,
    load_recording,
    parse_instant,
)
from dimos.navigation.motion.embodiment import EMBODIMENTS

SPEED_BIN = 0.1  # m/s per bin
MIN_SAMPLES = 25  # a bin with fewer samples is noise, not a measurement
SMOOTH_HALF = 2  # central-difference half-window (odom samples, ~33 ms each)


def frontier(rec: Recording) -> np.ndarray:
    """(n, 2) [speed, cross-track error] for every sample with an active plan."""
    plan_ts = np.array([t for t, _ in rec.plans])
    rows = []
    n = len(rec.odom_ts)
    for i in range(n):
        pose, ts = rec.poses[i], rec.odom_ts[i]
        if pose is None or ts not in rec.window:
            continue
        k = _before(ts, plan_ts)
        if k < 0 or is_hold(rec.plans[k][1]):
            continue
        a, b = max(0, i - SMOOTH_HALF), min(n - 1, i + SMOOTH_HALF)
        pa, pb = rec.poses[a], rec.poses[b]
        if pa is None or pb is None or rec.odom_ts[b] <= rec.odom_ts[a]:
            continue
        step = np.array(pb[:2]) - np.array(pa[:2])
        speed = float(np.linalg.norm(step) / (rec.odom_ts[b] - rec.odom_ts[a]))
        xy = rec.plans[k][1]
        p = np.array(pose[:2])
        s0, s1 = xy[:-1, :2], xy[1:, :2]
        seg = s1 - s0
        tt = np.clip(
            np.einsum("ij,ij->i", p - s0, seg) / (np.einsum("ij,ij->i", seg, seg) + 1e-12), 0, 1
        )
        err = float(np.min(np.linalg.norm(p - (s0 + tt[:, None] * seg), axis=1)))
        rows.append((speed, err))
    return np.array(rows).reshape(-1, 2)


def binned(fr: np.ndarray) -> list[tuple[float, int, float, float, float]]:
    """Per speed bin: (centre, n, p50, p90, p95) for bins with enough samples."""
    out = []
    edges = np.arange(0.0, fr[:, 0].max() + SPEED_BIN, SPEED_BIN) if len(fr) else np.array([0.0])
    for lo in edges[:-1]:
        m = (fr[:, 0] >= lo) & (fr[:, 0] < lo + SPEED_BIN)
        if m.sum() < MIN_SAMPLES:
            continue
        e = fr[m, 1]
        out.append(
            (
                float(lo + SPEED_BIN / 2),
                int(m.sum()),
                float(np.percentile(e, 50)),
                float(np.percentile(e, 90)),
                float(np.percentile(e, 95)),
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recordings", nargs="+", help="mcap recordings to overlay")
    ap.add_argument("--from", dest="start", default=None, help="window start (s or HH:MM:SS)")
    ap.add_argument("--to", dest="end", default=None, help="window end")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--embodiment", default="go2")
    ap.add_argument("--out", default="recordings/precision.svg", help="plot path")
    args = ap.parse_args()

    from dimos.memory2.vis import color
    from dimos.memory2.vis.plot.elements import HLine, Series
    from dimos.memory2.vis.plot.plot import Plot, TimeAxis

    floor = EMBODIMENTS[args.embodiment].precision
    start = parse_instant(args.start) if args.start else None
    end = parse_instant(args.end) if args.end else None
    palette = [color.blue, color.green, color.orange, color.purple, color.red]

    # x is SPEED, not time: raw numeric axis, or the formatter prints "0s".."1s"
    p = Plot(time_axis=TimeAxis.raw)
    p.add(HLine(y=floor, color=color.red.hex(), opacity=0.6))
    for idx, path in enumerate(args.recordings):
        rec = load_recording(path, args.base_frame, 5.0, start, end)
        fr = frontier(rec)
        rows = binned(fr)
        name = FsPath(path).stem.removesuffix(".zenoh")
        print(f"\n{name}: {len(fr)} samples with an active plan")
        print("  speed    n    p50    p90    p95")
        held = 0.0
        for centre, n, p50, p90, p95 in rows:
            mark = "  <- floor" if p95 <= floor else ""
            if p95 <= floor:
                held = centre + SPEED_BIN / 2
            print(f"  {centre:5.2f} {n:5d}  {p50:.3f}  {p90:.3f}  {p95:.3f}{mark}")
        print(
            f"  holds the {floor:.2f} m floor (p95) up to {held:.2f} m/s"
            if held
            else f"  never holds the {floor:.2f} m floor at p95"
        )
        c = palette[idx % len(palette)]
        p.add(
            Series(
                ts=[r[0] for r in rows],
                values=[r[4] for r in rows],
                label=f"{name} p95",
                color=c.hex(),
                axis="cross-track (m) vs speed (m/s)",
            )
        )
        p.add(
            Series(
                ts=[r[0] for r in rows],
                values=[r[2] for r in rows],
                label=f"{name} p50",
                color=c.hex(),
                opacity=0.45,
                axis="cross-track (m) vs speed (m/s)",
            )
        )
    FsPath(args.out).parent.mkdir(parents=True, exist_ok=True)
    p.to_svg(args.out)
    print(f"\nplot: {args.out}  (x = speed m/s, y = cross-track m, red line = precision floor)")


if __name__ == "__main__":
    main()
