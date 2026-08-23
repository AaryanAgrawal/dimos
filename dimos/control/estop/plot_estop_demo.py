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

"""Replay the committed G1 fall fixture through the e-stop latch and plot what the module does.

python -m dimos.control.estop.plot_estop_demo [out_dir]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import statistics

import matplotlib
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from dimos.control.estop.estop import EStopConfig

matplotlib.use("Agg")  # headless: this script only ever writes files

FIXTURE = Path(__file__).parent / "g1_fall_imu.csv"
UPRIGHT_COLOR = "#2a78d6"
FALLEN_COLOR = "#e34948"
RULE_COLOR = "#52514e"
BIN_WIDTH_DEG = 2.5


def tilt_deg(qx: float, qy: float) -> float:
    """Angle between the body z axis and gravity, off R[2][2], so yaw invariant."""
    return math.degrees(math.acos(min(1.0, max(-1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))))


def read_fixture(path: Path) -> tuple[list[float], list[float]]:
    """Sample times and tilts; only data lines start with a digit, as the Rust test filters."""
    rows = [line.split(",") for line in path.read_text().splitlines() if line[:1].isdigit()]
    return [float(r[0]) for r in rows], [tilt_deg(float(r[2]), float(r[3])) for r in rows]


def latch_holds(tripped: list[bool]) -> list[bool]:
    """Hold true from the first true onward, mirroring the module's one way Latch."""
    held = False
    out: list[bool] = []
    for flag in tripped:
        held = held or flag
        out.append(held)
    return out


def rise_indices(held: list[bool]) -> list[int]:
    """Samples where the latch publishes, meaning its false to true transitions."""
    prev = [False, *held[:-1]]
    return [i for i, (p, h) in enumerate(zip(prev, held, strict=True)) if h and not p]


def mode_bin(values: list[float], bins: list[float]) -> tuple[float, int]:
    """Center and count of the fullest bin, so a log axis is never the only read of height."""
    counts = [sum(1 for v in values if lo <= v < lo + BIN_WIDTH_DEG) for lo in bins[:-1]]
    return bins[counts.index(max(counts))] + BIN_WIDTH_DEG / 2, max(counts)


@dataclass(frozen=True)
class Replay:
    """The fixture graded by the same threshold and latch the Rust module implements."""

    t_s: list[float]
    tilts_deg: list[float]
    held: list[bool]
    max_tilt_deg: float
    rises: list[int]

    @property
    def upright_deg(self) -> list[float]:
        return [d for d in self.tilts_deg if d <= self.max_tilt_deg]

    @property
    def fallen_deg(self) -> list[float]:
        return [d for d in self.tilts_deg if d > self.max_tilt_deg]

    @property
    def recoveries(self) -> int:
        """Runs where tilt drops back under the threshold, which only a latch survives."""
        pairs = zip(self.tilts_deg, self.tilts_deg[1:], strict=False)  # offset by one on purpose
        return sum(1 for a, b in pairs if a > self.max_tilt_deg >= b)

    @property
    def recovery_low(self) -> tuple[float, float]:
        """Most upright the robot gets after tripping, the state a non latched check would clear."""
        i = min(range(self.rises[0], len(self.tilts_deg)), key=lambda j: self.tilts_deg[j])
        return self.t_s[i], self.tilts_deg[i]


def replay(path: Path, max_tilt_deg: float) -> Replay:
    """Run every fixture sample through the module's threshold and latch."""
    t_s, tilts_deg = read_fixture(path)
    held = latch_holds([d > max_tilt_deg for d in tilts_deg])
    return Replay(t_s, tilts_deg, held, max_tilt_deg, rise_indices(held))


def _panel_tilt(ax: Axes, r: Replay) -> None:
    """Measured tilt with the threshold, the fallen stretches, the trip, and the recovery low."""
    trip, (low_t_s, low_deg) = r.rises[0], r.recovery_low
    ax.plot(r.t_s, r.tilts_deg, color=UPRIGHT_COLOR, lw=1.6)
    fallen = [d > r.max_tilt_deg for d in r.tilts_deg]
    ax.fill_between(r.t_s, 0, r.tilts_deg, where=fallen, color=FALLEN_COLOR, alpha=0.15, lw=0)
    ax.axhline(r.max_tilt_deg, color=RULE_COLOR, ls="--", lw=1.2)
    ax.text(
        r.t_s[-1],
        r.max_tilt_deg + 1.5,
        f"{r.max_tilt_deg:.0f} deg threshold",
        ha="right",
        va="bottom",
        fontsize=9,
        color=RULE_COLOR,
    )
    ax.plot(r.t_s[trip], r.tilts_deg[trip], "o", color=FALLEN_COLOR, ms=8, zorder=5)
    ax.annotate(
        f"trip at {r.t_s[trip]:.2f} s, {r.tilts_deg[trip]:.1f} deg",
        (r.t_s[trip], r.tilts_deg[trip]),
        textcoords="offset points",
        xytext=(10, -16),
        fontsize=9,
        color=FALLEN_COLOR,
    )
    ax.plot(low_t_s, low_deg, "o", color=UPRIGHT_COLOR, ms=7, zorder=5)
    ax.annotate(
        f"back up at {low_deg:.1f} deg, latch still true",
        (low_t_s, low_deg),
        textcoords="offset points",
        xytext=(-6, 14),
        ha="right",
        fontsize=9,
        color=UPRIGHT_COLOR,
    )
    ax.set_ylabel("tilt off gravity (deg)")
    ax.set_ylim(0, max(r.tilts_deg) * 1.18)


def _panel_latch(ax: Axes, r: Replay) -> None:
    """The published e-stop signal, which rises once and never returns."""
    ax.step(r.t_s, [int(h) for h in r.held], where="post", color=FALLEN_COLOR, lw=2)
    ax.axvline(r.t_s[r.rises[0]], color=RULE_COLOR, ls=":", lw=1)
    ax.set_yticks([0, 1], ["false", "true"])
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("estop")


def plot_fall(r: Replay, out_path: Path) -> None:
    """Save the replay figure: tilt above, the latched output below."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9.5, 5.8), height_ratios=[3, 1])
    _panel_tilt(axes[0], r)
    _panel_latch(axes[1], r)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"One trip at {r.t_s[r.rises[0]]:.2f} s, then the latch holds true through a "
        f"recovery to {r.recovery_low[1]:.1f} deg and a second fall",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _annotate_margin(ax: Axes, r: Replay) -> None:
    """The gap the threshold keeps from ordinary upright tilt, plus what fills that gap."""
    median_deg = statistics.median(r.upright_deg)
    ax.annotate(
        "",
        (r.max_tilt_deg, 200),
        xytext=(median_deg, 200),
        arrowprops={"arrowstyle": "<->", "color": RULE_COLOR, "lw": 1},
    )
    ax.text(
        (median_deg + r.max_tilt_deg) / 2,
        280,
        f"{r.max_tilt_deg - median_deg:.1f} deg from the upright median to the threshold",
        ha="center",
        fontsize=9,
        color=RULE_COLOR,
    )
    ax.text(
        40,
        13,
        "the thin bins here are the two falls\npassing through, not normal standing",
        ha="center",
        va="bottom",
        fontsize=9,
        color=RULE_COLOR,
    )


def plot_margin(r: Replay, out_path: Path) -> None:
    """Save the separation figure: upright and fallen tilt populations around the threshold."""
    bins = [i * BIN_WIDTH_DEG for i in range(int(max(r.tilts_deg) / BIN_WIDTH_DEG) + 2)]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(r.upright_deg, bins=bins, color=UPRIGHT_COLOR, label="upright, latch idle")
    ax.hist(r.fallen_deg, bins=bins, color=FALLEN_COLOR, label="fallen, latch tripped")
    ax.axvline(r.max_tilt_deg, color=RULE_COLOR, ls="--", lw=1.2)
    upright_mode_deg, _ = mode_bin(r.upright_deg, bins)
    for values, color in [(r.upright_deg, UPRIGHT_COLOR), (r.fallen_deg, FALLEN_COLOR)]:
        center_deg, count = mode_bin(values, bins)
        ax.text(center_deg, count * 1.3, f"{count} samples", ha="center", fontsize=9, color=color)
    _annotate_margin(ax, r)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 3000)
    ax.set_xlabel("tilt off gravity (deg)")
    ax.set_ylabel("samples per 2.5 deg bin (log)")
    ax.grid(alpha=0.25, lw=0.6, axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(
        f"The upright mode sits {r.max_tilt_deg - upright_mode_deg:.0f} deg below the "
        f"threshold, and only the falls themselves come near it",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def report(r: Replay) -> None:
    """Print every number the two figures claim, plus the Rust test's own assertions."""
    trip = r.rises[0]
    peak = max(r.tilts_deg)
    median_deg = statistics.median(r.upright_deg)
    print(f"samples              {len(r.tilts_deg)}  over {r.t_s[0]:.4f} to {r.t_s[-1]:.4f} s")
    print(f"threshold            {r.max_tilt_deg:.1f} deg (EStopConfig default)")
    print(f"trip time            {r.t_s[trip]:.4f} s")
    print(f"tilt at trip         {r.tilts_deg[trip]:.2f} deg")
    print(f"peak tilt            {peak:.2f} deg at {r.t_s[r.tilts_deg.index(peak)]:.4f} s")
    print(f"upright median       {median_deg:.2f} deg  (n={len(r.upright_deg)})")
    print(f"margin               {r.max_tilt_deg - median_deg:.2f} deg to the threshold")
    print(f"fallen samples       {len(r.fallen_deg)}  min {min(r.fallen_deg):.2f} deg")
    print(f"latch held           {sum(r.held)} of {len(r.held)} samples")
    print(f"recovery low         {r.recovery_low[1]:.2f} deg at {r.recovery_low[0]:.4f} s")
    print(f"recoveries survived  {r.recoveries}")
    print(f"rise count           {len(r.rises)}  at {[round(r.t_s[i], 4) for i in r.rises]} s")
    for name, got, want in [
        ("rows", len(r.tilts_deg), 1401),
        ("fired", [round(r.t_s[i], 4) for i in r.rises], [8.37]),
        ("recoveries", r.recoveries, 1),
    ]:
        verdict = "agrees" if got == want else "*** DISAGREES ***"
        print(f"rust test {name:12} {verdict}  got {got}, asserts {want}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path.cwd())
    out_dir = parser.parse_args().out_dir
    r = replay(FIXTURE, EStopConfig().max_tilt_deg)
    report(r)
    plot_fall(r, out_dir / "estop_rust_fall.png")
    plot_margin(r, out_dir / "estop_rust_margin.png")
    print(f"wrote                {out_dir / 'estop_rust_fall.png'}")
    print(f"wrote                {out_dir / 'estop_rust_margin.png'}")


if __name__ == "__main__":
    main()
