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

"""Score a simulated rollout against the recording that produced its commands.

One call runs the policy under the recorded command schedule, summarizes both
sides, and measures each statistic's own noise by repeating the rollout from
perturbed initial poses. A statistic is only worth fitting when the sim-real
difference clearly exceeds that noise, so :class:`Report` carries the ratio.

    from dimos.navigation.motion.trajectory.research.evaluate import evaluate
    print(evaluate(DATASET, POLICY).table())

``physics`` overrides leg-joint parameters on the compiled model, which is the
hook a parameter search drives:

    evaluate(DATASET, POLICY, physics={"armature": 0.03, "damping": 2.0})
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from dimos.navigation.motion.trajectory.research import (
    metrics,
    model as go2_model,
    walk as walk_mod,
)
from dimos.navigation.motion.trajectory.research.policy import FreePolicy
from dimos.navigation.motion.trajectory.research.vive import base_track, mount_rotation

# Leg dofs in qvel/dof indexing: 6 free-joint dofs, then the twelve joints.
LEG_DOFS = slice(6, 18)

PERTURB_RAD = 0.05
"""Initial-pose spread used to measure a statistic's noise floor.

Small on purpose -- about 3 degrees. The gait is chaotic enough that this
already decorrelates position within seconds, so it measures the noise a
parameter search has to beat rather than a plausible modelling error.
"""

# Set by the anchor, not measured: the recorded height is only as good as the
# tracker offset, so height_mean cannot be compared. See vive.py.
NOT_COMPARABLE = ("height_mean",)


@contextlib.contextmanager
def _physics(overrides: dict[str, float] | None) -> Iterator[None]:
    """Temporarily patch leg-joint parameters onto every model that gets built."""
    if not overrides:
        yield
        return
    unknown = set(overrides) - {
        "armature",
        "damping",
        "frictionloss",
        "trunk_mass_scale",
        "trunk_inertia_scale",
    }
    if unknown:
        raise ValueError(f"unknown physics override(s): {sorted(unknown)}")

    original = go2_model.load

    def patched(menagerie: Path | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
        model, data = original(menagerie)
        if "armature" in overrides:
            model.dof_armature[LEG_DOFS] = overrides["armature"]
        if "damping" in overrides:
            model.dof_damping[LEG_DOFS] = overrides["damping"]
        if "frictionloss" in overrides:
            model.dof_frictionloss[LEG_DOFS] = overrides["frictionloss"]
        # A heavier or more rotationally sluggish trunk is the competing
        # explanation for the turn lag: inertia delays *and* smooths the
        # response, where transport delay only shifts it.
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if "trunk_mass_scale" in overrides:
            model.body_mass[trunk] *= overrides["trunk_mass_scale"]
        if "trunk_inertia_scale" in overrides:
            model.body_inertia[trunk] *= overrides["trunk_inertia_scale"]
        return model, data

    go2_model.load = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        go2_model.load = original  # type: ignore[assignment]


def _noise_floor(
    policy: FreePolicy, sched: Any, start: float, seconds: float, seeds: int
) -> dict[str, float]:
    runs = []
    base = policy.default_pose.copy()
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        object.__setattr__(policy, "default_pose", base + rng.normal(0, PERTURB_RAD, 12))
        try:
            t2 = walk_mod.walk(policy, schedule=sched, seconds=seconds, start=start)
        finally:
            object.__setattr__(policy, "default_pose", base)
        runs.append(_summarize_run(t2.t, t2.pos, t2.quat, sched, start))
    return metrics.chaos_spread(runs)


def measure_noise(
    dataset: str | Path,
    policy_bin: str | Path,
    *,
    start: float = 6.0,
    seconds: float = 30.0,
    seeds: int = 4,
) -> dict[str, float]:
    """Each statistic's noise floor, to be measured once and reused by a search."""
    policy = FreePolicy.load(policy_bin)
    return _noise_floor(policy, walk_mod.read_control_log(dataset), start, seconds, seeds)


@dataclass
class Report:
    sim: metrics.Summary
    real: metrics.Summary
    noise: dict[str, float]
    seconds: float
    start: float
    physics: dict[str, float] = field(default_factory=dict)

    def snr(self) -> dict[str, float]:
        """Sim-real difference over the statistic's own noise, per statistic."""
        s, r = self.sim.as_dict(), self.real.as_dict()
        out = {}
        for k in s:
            if k in NOT_COMPARABLE:
                continue
            n = self.noise[k]
            out[k] = abs(s[k] - r[k]) / n if n > 1e-9 else float("inf")
        return out

    def loss(self) -> float:
        """Noise-weighted distance, for a parameter search to minimize.

        Each statistic is scaled by its own noise floor, so a term cannot be
        won by driving a quantity the simulator cannot resolve anyway.
        """
        return float(np.sqrt(np.mean([v**2 for v in self.snr().values()])))

    def table(self) -> str:
        s, r = self.sim.as_dict(), self.real.as_dict()
        snr = self.snr()
        head = f"{self.seconds:.0f}s from t={self.start:.0f}s"
        if self.physics:
            head += "  " + " ".join(f"{k}={v:g}" for k, v in sorted(self.physics.items()))
        lines = [head, f"{'statistic':>14} {'sim':>9} {'real':>9} {'noise':>9} {'SNR':>7}"]
        for k in sorted(snr, key=lambda k: -snr[k]):
            lines.append(f"{k:>14} {s[k]:9.3f} {r[k]:9.3f} {self.noise[k]:9.3f} {snr[k]:7.1f}")
        for k in NOT_COMPARABLE:
            lines.append(f"{k:>14} {s[k]:9.3f} {r[k]:9.3f} {'--':>9} {'n/a':>7}")
        lines.append(f"{'loss':>14} {self.loss():9.2f}")
        return "\n".join(lines)


def _summarize_run(
    t: np.ndarray, pos: np.ndarray, quat: np.ndarray, sched: Any, offset: float
) -> metrics.Summary:
    ct, cc = sched
    idx = np.clip(np.searchsorted(ct, t + offset, side="right") - 1, 0, len(ct) - 1)
    return metrics.summarize(t, pos, quat, cc[idx])


def evaluate(
    dataset: str | Path,
    policy_bin: str | Path,
    *,
    start: float = 6.0,
    seconds: float = 30.0,
    mount_yaw: float = 94.0,
    tracker_z: float = 0.207,
    anchor_height: float = 0.28,
    seeds: int = 4,
    physics: dict[str, float] | None = None,
    command_delay: float = 0.0,
    noise: dict[str, float] | None = None,
) -> Report:
    """Run the policy under the recording's commands and score it against them.

    ``noise`` skips the perturbed rollouts. A search should call
    :func:`measure_noise` once and reuse the result: it costs ``seeds``
    rollouts, it is a property of the system rather than of any one
    parameter set, and holding it fixed keeps the loss comparable between
    trials. That is a 5x saving per trial at the default four seeds.
    """
    policy = FreePolicy.load(policy_bin)
    sched = walk_mod.read_control_log(dataset)

    with _physics(physics):
        track = walk_mod.walk(
            policy, schedule=sched, seconds=seconds, start=start, command_delay=command_delay
        )
        sim = _summarize_run(track.t, track.pos, track.quat, sched, start)
        if noise is None:
            noise = _noise_floor(policy, sched, start, seconds, seeds)

    gt, gp, gq = base_track(
        dataset,
        tracker_offset=np.array([0.0, 0.0, tracker_z]),
        mount=mount_rotation(mount_yaw),
        anchor_at=start,
        anchor_pos=np.array([0.0, 0.0, anchor_height]),
    )
    sel = (gt >= start) & (gt < start + seconds)
    real = _summarize_run(gt[sel] - start, gp[sel], gq[sel], sched, start)

    return Report(
        sim=sim,
        real=real,
        noise=noise,
        seconds=seconds,
        start=start,
        physics={**(physics or {}), **({"command_delay": command_delay} if command_delay else {})},
    )
