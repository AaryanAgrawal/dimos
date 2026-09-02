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

"""Replay a G1 mem2 recording through the shipped protective checks and print every trip."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dimos.memory.cli.dataset import open_store
from dimos.robot.unitree.g1.protective import FLAIL_JOINT_SPEED_RAD_S, stop_reason, tilt_deg


def _streams(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Motor timestamps, dq [n, 29], IMU timestamps and IMU quaternions as (w,x,y,z) [m, 4]."""
    store = open_store(path)
    try:
        motors: list[Any] = list(store.stream("motor_states").order_by("ts"))
        imu: list[Any] = list(store.stream("imu").order_by("ts"))
    finally:
        store.stop()
    quat = [
        (r.data.orientation.w, r.data.orientation.x, r.data.orientation.y, r.data.orientation.z)
        for r in imu
    ]
    return (
        np.array([r.ts for r in motors]),
        np.array([list(r.data.velocity) for r in motors]),
        np.array([r.ts for r in imu]),
        np.array(quat),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    args = ap.parse_args()
    t, dq, imu_t, quat = _streams(args.recording)
    nearest = np.clip(np.searchsorted(imu_t, t), 0, len(imu_t) - 1)
    fast = (np.abs(dq) > FLAIL_JOINT_SPEED_RAD_S).sum(axis=1)
    tilt = np.array([tilt_deg(q) for q in quat[nearest]])
    reasons = [stop_reason(quat[j], dq[i]) for i, j in enumerate(nearest)]
    trips = [i for i, reason in enumerate(reasons) if reason]
    span = t[-1] - t[0]
    print(
        f"{args.recording.name}: {len(t)} motor samples over {span:.1f} s at {len(t) / span:.0f} Hz"
    )
    print(
        f"max tilt {tilt.max():.1f} deg, max |dq| {np.abs(dq).max():.2f} rad/s, "
        f"most joints past {FLAIL_JOINT_SPEED_RAD_S:g} rad/s at once {fast.max()}"
    )
    if not trips:
        print("no trip")
        return
    first = trips[0]
    print(f"first trip at t={t[first] - t[0]:.3f} s: {reasons[first]}; {len(trips)} samples trip")


if __name__ == "__main__":
    main()
