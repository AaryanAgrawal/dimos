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
from dimos.navigation.motion.trajectory.research.vive import base_track, mount_rotation, quat_to_mat

# Leg dofs in qvel/dof indexing: 6 free-joint dofs, then the twelve joints.
LEG_DOFS = slice(6, 18)

PERTURB_RAD = 0.05
"""Initial-pose spread used to measure a statistic's noise floor.

Small on purpose -- about 3 degrees. The gait is chaotic enough that this
already decorrelates position within seconds, so it measures the noise a
parameter search has to beat rather than a plausible modelling error.
"""

# The recorded height lives in the Vive room frame, whose floor is unknown,
# so the mean cannot be compared. See vive.py.
NOT_COMPARABLE = ("height_mean",)

# Foot geom names in the menagerie MJCF. Their contacts are already condim=6
# (tangential + torsional + rolling friction, priority 1 -- the "condim=1"
# in the go2 default class only governs the calf capsules), so what is open
# to fitting is the friction values, not the contact dimensionality.
FOOT_GEOMS = ("FL", "FR", "RL", "RR")

# Leg-space statistics, scored command-to-command against policy/lowcmd. The
# judge's base statistics cannot see the legs at all -- the sim matched every
# one of them while visibly high-stepping its front feet.
LEG_STATS = ("front_lift", "rear_lift")

_FK_CACHE: tuple[mujoco.MjModel, mujoco.MjData, list[int]] | None = None


def commanded_clearance(targets: np.ndarray, base_z: float = 0.318) -> np.ndarray:
    """Foot clearance (n, 4) implied by joint targets, FK with the base level.

    Kinematics only, so it works identically on the simulator's commanded
    targets and on the recording's ``policy/lowcmd`` -- comparing command to
    command sidesteps both the unrecorded real leg state and the instability
    of open-loop replay.
    """
    global _FK_CACHE
    if _FK_CACHE is None:
        model, data = go2_model.load()
        feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS]
        _FK_CACHE = (model, data, feet)
    model, data, feet = _FK_CACHE
    out = np.empty((len(targets), 4))
    for i, q in enumerate(targets):
        data.qpos[:] = 0.0
        data.qpos[2] = base_z
        data.qpos[3] = 1.0
        data.qpos[7:19] = q
        # mj_kinematics exists at runtime; the bundled mujoco stubs omit it.
        mujoco.mj_kinematics(model, data)  # type: ignore[attr-defined]
        out[i] = data.geom_xpos[feet, 2]
    return out - 0.022  # foot sphere radius


def leg_stats(targets: np.ndarray) -> dict[str, float]:
    """p95 commanded foot lift, front and rear pairs -- the swing apex."""
    c = commanded_clearance(targets)
    return {
        "front_lift": float(np.percentile(c[:, :2], 95)),
        "rear_lift": float(np.percentile(c[:, 2:], 95)),
    }


# Best-known configuration: the point the multi-objective search collapsed to
# on himloco01 (see FINDINGS). Every statistic at or below its noise floor.
FITTED_PHYSICS = {
    "armature": 0.01395,
    "damping": 0.2381,
    "frictionloss": 0.7372,
    "trunk_mass_scale": 1.326,
    "trunk_inertia_scale": 1.487,
    "foot_friction": 0.5692,
    "foot_friction_torsional": 0.003138,
}
FITTED_COMMAND_DELAY = 0.0231
FITTED_ACTUATOR_TAU = 0.0289


def virtual_tracker(
    pos: np.ndarray, quat: np.ndarray, *, mount_yaw: float, tracker_z: float
) -> np.ndarray:
    """Positions with z replaced by a virtual tracker's height.

    Height statistics are compared in *sensor space*: the real side keeps the
    raw tracker height (``base_track(sensor_z=True)``), and the sim side mounts
    a virtual tracker on its base with the same guessed offset. Inverting the
    guess on the real data instead put 11.4 mm of its z std against 5.6 mm from
    the tracker itself; done this way the guess distorts both sides identically
    and mostly cancels.
    """
    off_base = mount_rotation(mount_yaw).T @ np.array([0.0, 0.0, tracker_z])
    out = pos.copy()
    out[:, 2] = pos[:, 2] - np.einsum("nij,j->ni", quat_to_mat(quat), off_base)[:, 2]
    return out


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
        "foot_friction",
        "foot_friction_torsional",
        "trunk_com_x",
        "leg_mass_scale",
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
        # Payload placement, not just payload mass: the lidar and tracker sit
        # forward and top of the trunk, which body_mass scaling cannot express.
        if "trunk_com_x" in overrides:
            model.body_ipos[trunk][0] += overrides["trunk_com_x"]
        # Real legs carry covers and cabling the MJCF omits; heavier swing
        # inertia damps how high a foot flies for the same action.
        if "leg_mass_scale" in overrides:
            for prefix in FOOT_GEOMS:
                for part in ("thigh", "calf"):
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{part}")
                    model.body_mass[bid] *= overrides["leg_mass_scale"]
                    model.body_inertia[bid] *= overrides["leg_mass_scale"]
        # geom_friction columns are (tangential, torsional, rolling); the foot
        # has priority 1, so its values dictate the foot-floor contact pair.
        # Torsional is what resists pivoting the stance feet in a turn.
        for name in FOOT_GEOMS:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if "foot_friction" in overrides:
                model.geom_friction[gid, 0] = overrides["foot_friction"]
            if "foot_friction_torsional" in overrides:
                model.geom_friction[gid, 1] = overrides["foot_friction_torsional"]
        return model, data

    go2_model.load = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        go2_model.load = original  # type: ignore[assignment]


def _noise_floor(
    policy: FreePolicy,
    sched: Any,
    start: float,
    seconds: float,
    seeds: int,
    mount_yaw: float = 94.0,
    tracker_z: float = 0.207,
) -> dict[str, float]:
    runs = []
    leg_runs: list[dict[str, float]] = []
    base = policy.default_pose.copy()
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        object.__setattr__(policy, "default_pose", base + rng.normal(0, PERTURB_RAD, 12))
        try:
            t2 = walk_mod.walk(policy, schedule=sched, seconds=seconds, start=start)
        finally:
            object.__setattr__(policy, "default_pose", base)
        p2 = virtual_tracker(t2.pos, t2.quat, mount_yaw=mount_yaw, tracker_z=tracker_z)
        runs.append(_summarize_run(t2.t, p2, t2.quat, sched, start))
        leg_runs.append(leg_stats(t2.target))
    spread = metrics.chaos_spread(runs)
    for k in LEG_STATS:
        vals = [lr[k] for lr in leg_runs]
        spread[k] = float(max(vals) - min(vals))
    return spread


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
    sim_legs: dict[str, float] = field(default_factory=dict)
    real_legs: dict[str, float] = field(default_factory=dict)

    def snr(self) -> dict[str, float]:
        """Sim-real difference over the statistic's own noise, per statistic."""
        s = {**self.sim.as_dict(), **self.sim_legs}
        r = {**self.real.as_dict(), **self.real_legs}
        out = {}
        for k in s:
            if k in NOT_COMPARABLE or k not in r:
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
        s = {**self.sim.as_dict(), **self.sim_legs}
        r = {**self.real.as_dict(), **self.real_legs}
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
    actuator_tau: float = 0.0,
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
            policy,
            schedule=sched,
            seconds=seconds,
            start=start,
            command_delay=command_delay,
            actuator_tau=actuator_tau,
        )
        sim_p = virtual_tracker(track.pos, track.quat, mount_yaw=mount_yaw, tracker_z=tracker_z)
        sim = _summarize_run(track.t, sim_p, track.quat, sched, start)
        if noise is None:
            noise = _noise_floor(policy, sched, start, seconds, seeds, mount_yaw, tracker_z)

    sim_legs = leg_stats(track.target)
    try:
        lt, lq = walk_mod.read_policy_lowcmd(dataset)
        lsel = (lt >= start) & (lt < start + seconds)
        real_legs = leg_stats(lq[lsel])
    except ValueError:  # recording without policy/lowcmd: legs stay unscored
        real_legs = {}

    gt, gp, gq = base_track(
        dataset,
        tracker_offset=np.array([0.0, 0.0, tracker_z]),
        mount=mount_rotation(mount_yaw),
        anchor_at=start,
        anchor_pos=np.array([0.0, 0.0, anchor_height]),
        sensor_z=True,
    )
    sel = (gt >= start) & (gt < start + seconds)
    real = _summarize_run(gt[sel] - start, gp[sel], gq[sel], sched, start)

    return Report(
        sim=sim,
        real=real,
        noise=noise,
        seconds=seconds,
        start=start,
        sim_legs=sim_legs,
        real_legs=real_legs,
        physics={
            **(physics or {}),
            **({"command_delay": command_delay} if command_delay else {}),
            **({"actuator_tau": actuator_tau} if actuator_tau else {}),
        },
    )
