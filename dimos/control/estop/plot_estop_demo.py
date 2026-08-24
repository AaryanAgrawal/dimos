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

"""Replay the G1 fall fixture at recorded timing through the real Rust e-stop.

python -m dimos.control.estop.plot_estop_demo [output.png]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dimos.control.estop.estop import EStop, EStopConfig
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.std_msgs.Bool import Bool

FIXTURE = Path(__file__).parent / "g1_fall_imu.csv"
BLUE = "#2a78d6"
RED = "#e34948"
GRAY = "#52514e"


@dataclass(frozen=True)
class Sample:
    """One recorded sample time and its SIMULATED orientation in w, x, y, z order."""

    t_s: float
    qw: float
    qx: float
    qy: float
    qz: float

    @property
    def tilt_deg(self) -> float:
        """Tilt is calculated only for the graph; the Rust module makes the decision."""
        r22 = 1.0 - 2.0 * (self.qx * self.qx + self.qy * self.qy)
        return math.degrees(math.acos(min(1.0, max(-1.0, r22))))


@dataclass(frozen=True)
class ReplayResult:
    """Wall-clock replay timing and the real module's received true outputs."""

    elapsed_s: float
    true_times_s: list[float]


def read_samples(path: Path) -> list[Sample]:
    """Read fixture rows while preserving their recorded timestamps."""
    with path.open() as fixture:
        rows = csv.DictReader(line for line in fixture if not line.startswith("#"))
        return [
            Sample(*(float(row[key]) for key in ("t_s", "qw", "qx", "qy", "qz"))) for row in rows
        ]


def _wait_until(deadline_s: float) -> None:
    """Wait until one replay deadline without accumulating per-sample drift."""
    while (remaining_s := deadline_s - time.monotonic()) > 0:
        time.sleep(min(remaining_s, 0.005))


def _publish_at_recorded_timing(
    samples: list[Sample], transport: LCMTransport[Imu], start_s: float
) -> float:
    """Publish every orientation through dimos at its recorded offset."""
    for sample in samples:
        _wait_until(start_s + sample.t_s)
        orientation = Quaternion(sample.qx, sample.qy, sample.qz, sample.qw)
        transport.broadcast(None, Imu(orientation=orientation, ts=sample.t_s))
    return time.monotonic() - start_s


def run_replay(samples: list[Sample]) -> ReplayResult:
    """Run the native module over LCM and collect its actual output at 1x speed."""
    topic = f"/estop/replay/{os.getpid()}"
    imu = LCMTransport(f"{topic}/imu", Imu)
    trigger = LCMTransport(f"{topic}/trigger", Bool)
    estop = LCMTransport(f"{topic}/estop", Bool)
    module = EStop(build_command=None)
    for name, transport in (("imu", imu), ("trigger", trigger), ("estop", estop)):
        module.set_transport(name, transport)

    start_s: float | None = None
    true_times_s: list[float] = []

    def collect(msg: Bool) -> None:
        if msg.data and start_s is not None:
            true_times_s.append(time.monotonic() - start_s)

    unsubscribe = estop.subscribe(collect)
    try:
        module.start()
        time.sleep(0.5)  # the native subscriber must be ready before the replay clock starts
        start_s = time.monotonic()
        elapsed_s = _publish_at_recorded_timing(samples, imu, start_s)
        _wait_until(start_s + samples[-1].t_s + 0.1)
    finally:
        module.stop()
        unsubscribe()
        for transport in (imu, trigger, estop):
            with contextlib.suppress(Exception):
                transport.stop()
    return ReplayResult(elapsed_s, true_times_s)


def plot(samples: list[Sample], result: ReplayResult, out_path: Path) -> None:
    """Plot recorded tilt beside the native module's observed output."""
    times_s = [sample.t_s for sample in samples]
    tilts_deg = [sample.tilt_deg for sample in samples]
    first_true_s = result.true_times_s[0]
    threshold_deg = EStopConfig().max_tilt_deg
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5.8), height_ratios=[3, 1])
    axes[0].plot(times_s, tilts_deg, color=BLUE, lw=1.5, label="G1 measured tilt")
    axes[0].axhline(threshold_deg, color=RED, ls="--", lw=1.2, label="45 deg limit")
    axes[0].axvline(first_true_s, color=GRAY, ls=":", lw=1.1)
    axes[0].set_ylabel("tilt off gravity (deg)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].step([times_s[0], first_true_s, times_s[-1]], [0, 1, 1], where="post", color=RED, lw=2)
    axes[1].scatter(result.true_times_s, [1] * len(result.true_times_s), color=RED, s=5)
    axes[1].set_yticks([0, 1], ["no true", "true"])
    axes[1].set_xlabel("replay time (s)")
    axes[1].set_ylabel("E-stop output")
    for axis in axes:
        axis.grid(alpha=0.25, lw=0.6)
    figure.suptitle("SIMULATED orientation replayed at 1x through the real Rust EStop module")
    figure.text(
        0.5,
        0.01,
        "Tilt was measured on the G1; orientation was reconstructed. Dots are true messages received over LCM.",
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


def report(samples: list[Sample], result: ReplayResult, out_path: Path) -> None:
    """Print the replay facts needed to reproduce and interpret the graph."""
    threshold_deg = EStopConfig().max_tilt_deg
    threshold_t_s = next(sample.t_s for sample in samples if sample.tilt_deg > threshold_deg)
    print("SIMULATED orientation: measured G1 tilt re-encoded as quaternion")
    print(f"samples={len(samples)} recorded_span_s={samples[-1].t_s - samples[0].t_s:.4f}")
    print(f"wall_replay_s={result.elapsed_s:.4f} threshold_input_s={threshold_t_s:.4f}")
    print(
        f"first_true_output_s={result.true_times_s[0]:.4f} true_messages={len(result.true_times_s)}"
    )
    print(f"wrote={out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("estop_fall_replay.png"))
    out_path = parser.parse_args().output.resolve()
    samples = read_samples(FIXTURE)
    result = run_replay(samples)
    if not result.true_times_s:
        raise RuntimeError("EStop published no true output during the fall replay")
    plot(samples, result, out_path)
    report(samples, result, out_path)


if __name__ == "__main__":
    main()
