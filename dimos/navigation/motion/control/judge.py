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

"""The executor judge: score what the robot DID, not what was planned.

Referee-shaped on purpose — ``total = gate * (100*progress + 10*tracking +
1*composure)``, max 111 — so planner and controller numbers read on one scale.
Priorities are lexicographic by magnitude: composure never buys back a missed
goal, and nothing buys back a crash or a fall (gate 0).

The referee scores a *plan* against the gold maneuver; this judge scores an
*execution* against physics: the gate reads MuJoCo contacts, progress reads
arrival against the plan's own arc at cruise speed, tracking reads the
distance between where the body went and the line it was following.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dimos.navigation.motion.control.episode import EpisodeResult
from dimos.navigation.motion.planner.autoresearch.scenarios import Scenario
from dimos.navigation.motion.simulation.walk import COMMAND_SLEW

CRUISE = 0.35  # m/s the stub can hold through curves; progress=1 at this pace
GRACE_S = 2.0  # flat allowance: settle, first-step lag, terminal deceleration
TRACK_SCALE = 0.25  # cross-track p95 (m) at which the tracking pillar hits 0
TILT_SCALE = 0.35  # tilt p99 (rad, ~20 deg) at which stability hits 0


def cross_track(pos_xy: np.ndarray, path_xy: np.ndarray) -> np.ndarray:
    """Distance from each executed point to the planned polyline."""
    if len(path_xy) == 0 or len(pos_xy) == 0:
        return np.zeros(0)
    if len(path_xy) == 1:
        return np.asarray(np.linalg.norm(pos_xy - path_xy[0], axis=1))
    a, b = path_xy[:-1], path_xy[1:]
    ab = b - a
    denom = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    # (n_pos, n_seg) projection parameter, clamped to the segment
    ap = pos_xy[:, None, :] - a[None, :, :]
    s = np.clip(np.einsum("nij,ij->ni", ap, ab) / denom, 0.0, 1.0)
    closest = a[None, :, :] + s[..., None] * ab[None, :, :]
    return np.asarray(np.min(np.linalg.norm(pos_xy[:, None, :] - closest, axis=2), axis=1))


def _plan_xy(result: EpisodeResult) -> np.ndarray:
    return np.array([[p.position.x, p.position.y] for p in result.plan.poses]).reshape(-1, 2)


def _saturation(result: EpisodeResult) -> float:
    """Fraction of active ticks where the hardware slew clipped the request."""
    if len(result.twist_cmd) < 2:
        return 0.0
    gap = np.abs(result.twist_cmd - result.used_cmd)
    return float(np.mean(np.any(gap > COMMAND_SLEW[None, :] * 1.001, axis=1)))


def score_episode(result: EpisodeResult) -> dict[str, Any]:
    """One executed episode -> pillars, total, and the raw stats behind them."""
    sc = result.scenario
    refuse_world = sc.expect == "refuse"
    out: dict[str, Any] = {
        "name": sc.name,
        "outcome": result.outcome,
        "dq": result.outcome in ("collision", "fall"),
        "time_to_goal": result.time_to_goal,
    }

    # Gate: physics vetoes everything else.
    if out["dq"]:
        out.update(progress=0.0, tracking=0.0, composure=0.0, total=0.0)
        return out

    # Progress: arriving, at pace. Refusal is arrival when truth is sealed.
    plan_xy = _plan_xy(result)
    arc = (
        float(np.sum(np.linalg.norm(np.diff(plan_xy, axis=0), axis=1))) if len(plan_xy) > 1 else 0.0
    )
    if refuse_world:
        progress = 1.0 if result.outcome in ("refused", "timeout") else 0.0
    elif result.outcome == "goal" and result.time_to_goal is not None:
        expected = arc / CRUISE + GRACE_S
        progress = min(1.0, expected / max(result.time_to_goal, 1e-6))
    else:
        progress = 0.0

    # Tracking: the executed body line vs the plan it was following.
    xt = cross_track(result.pos[:, :2], plan_xy) if len(result.pos) else np.zeros(0)
    xt_p95 = float(np.percentile(xt, 95)) if len(xt) else 0.0
    if refuse_world and result.outcome in ("refused", "timeout"):
        tracking = 1.0  # nothing to track; the correct output was to stay put
    else:
        tracking = max(0.0, 1.0 - xt_p95 / TRACK_SCALE)

    # Composure: upright and inside the command envelope.
    tilt_p99 = float(np.percentile(result.tilt, 99)) if len(result.tilt) else 0.0
    stability = max(0.0, 1.0 - tilt_p99 / TILT_SCALE)
    sat = _saturation(result)
    composure = 0.5 * stability + 0.5 * (1.0 - sat)

    out.update(
        progress=round(progress, 4),
        tracking=round(tracking, 4),
        composure=round(composure, 4),
        xtrack_p95=round(xt_p95, 4),
        xtrack_max=round(float(np.max(xt)) if len(xt) else 0.0, 4),
        tilt_p99=round(tilt_p99, 4),
        saturation=round(sat, 4),
        plan_ms=round(float(np.max(result.plan_ms)), 2) if result.plan_ms else 0.0,
        total=round(100.0 * progress + 10.0 * tracking + composure, 2),
    )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [r["total"] for r in rows]
    worst = min(rows, key=lambda r: r["total"]) if rows else None
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    return {
        "score": round(float(np.mean(totals)), 2) if totals else math.nan,
        "worst": {"name": worst["name"], "total": worst["total"]} if worst else None,
        "dq": sum(1 for r in rows if r["dq"]),
        "outcomes": outcomes,
        "progress": round(float(np.mean([r["progress"] for r in rows])), 4) if rows else math.nan,
        "tracking": round(float(np.mean([r["tracking"] for r in rows])), 4) if rows else math.nan,
        "composure": round(float(np.mean([r["composure"] for r in rows])), 4) if rows else math.nan,
        "worlds": len(rows),
    }


def print_row(row: dict[str, Any], sc: Scenario) -> None:
    ttg = f"{row['time_to_goal']:5.1f}s" if row["time_to_goal"] is not None else "    --"
    print(
        f"{row['name']:<18s} {row['outcome']:<9s} {ttg}"
        f"  xt95 {row.get('xtrack_p95', 0.0):5.2f}  tilt99 {row.get('tilt_p99', 0.0):5.2f}"
        f"  sat {row.get('saturation', 0.0):4.2f}  {row['total']:6.2f}  {sc.note}"
    )
