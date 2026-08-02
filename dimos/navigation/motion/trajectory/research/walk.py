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

"""Run a FREE policy on the flat-ground MuJoCo Go2.

Commands come either from a constant (vx, vy, vyaw) or from a recording's
``control_log``, which is what makes a run comparable to its ``vive_pose``.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import json
from pathlib import Path

import mujoco
import numpy as np

from dimos.navigation.motion.trajectory.research import model as go2_model
from dimos.navigation.motion.trajectory.research.policy import FreePolicy

CONTROL_DT = 0.02  # 50 Hz policy rate; not stored in the blob (cfg "dt")

# Per-joint torque limits, also absent from the blob (cfg "torque_limits").
# Slightly tighter than the MJCF ctrlrange, so they bind first.
TORQUE_LIMITS = np.array([23.0, 23.0, 35.0] * 4)


@dataclass
class Track:
    """Simulated base trajectory, sampled at the policy rate."""

    t: np.ndarray
    pos: np.ndarray = field(repr=False)  # (n, 3)
    quat: np.ndarray = field(repr=False)  # (n, 4) wxyz
    cmd: np.ndarray = field(repr=False)  # (n, 3) vx, vy, vyaw applied


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])


def read_control_log(dataset: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t, cmd)`` — seconds from run start, and (n, 3) vx/vy/vyaw.

    Only ``action == "walk"`` entries carry a velocity; everything else
    (pitch, gait_height, engage events) is ignored here.
    """
    from mcap.reader import make_reader

    ts: list[float] = []
    cmds: list[list[float]] = []
    with Path(dataset).open("rb") as f:
        for _schema, channel, msg in make_reader(f).iter_messages(topics=["control_log"]):
            if channel.topic != "control_log":
                continue
            d = json.loads(msg.data)
            if d.get("action") != "walk":
                continue
            ts.append(msg.log_time / 1e9)
            cmds.append([d.get("vx", 0.0), d.get("vy", 0.0), d.get("vyaw", 0.0)])
    if not ts:
        raise ValueError(f"{dataset}: no walk commands in control_log")
    t = np.array(ts)
    return t - t[0], np.array(cmds)


def walk(
    policy: FreePolicy,
    *,
    command: np.ndarray | None = None,
    schedule: tuple[np.ndarray, np.ndarray] | None = None,
    seconds: float | None = None,
    start: float = 0.0,
    settle: float = 0.5,
    menagerie: Path | None = None,
    view: bool = False,
    speed: float = 1.0,
    ghost: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Track:
    """Step the policy in MuJoCo.

    ``command`` holds vx/vy/vyaw fixed; ``schedule`` is a ``(t, cmd)`` pair as
    returned by :func:`read_control_log`, held zero-order between samples.
    Exactly one must be given.

    ``ghost`` is a ``(t, pos, quat)`` recorded base_link track (see
    :func:`vive.base_track`) drawn as a translucent box alongside the robot.
    """
    if (command is None) == (schedule is None):
        raise ValueError("pass exactly one of command= or schedule=")
    if schedule is not None:
        sched_t, sched_cmd = schedule
        duration = float(sched_t[-1]) - start if seconds is None else seconds
    else:
        duration = 8.0 if seconds is None else seconds

    if ghost is None:
        model, data = go2_model.load(menagerie)
    else:
        model, data = go2_model.load_with_ghost(menagerie)
    sim_dt = model.opt.timestep
    decim = max(1, round(CONTROL_DT / sim_dt))

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")  # type: ignore[attr-defined]
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    data.qpos[7:19] = policy.default_pose
    mujoco.mj_forward(model, data)

    hist: collections.deque[np.ndarray] = collections.deque(maxlen=policy.hist)
    last_action = np.zeros(policy.act_dim)
    target = policy.default_pose.copy()

    def observe(cmd: np.ndarray) -> np.ndarray:
        q = data.qpos[7:19]
        dq = data.qvel[6:18]
        raw = np.concatenate(
            [cmd, data.qvel[3:6], projected_gravity(data.qpos[3:7]), q, dq, last_action]
        )
        return policy.normalize(raw)

    def cmd_at(t: float) -> np.ndarray:
        if schedule is None:
            assert command is not None
            return command
        held: np.ndarray = sched_cmd[max(0, int(np.searchsorted(sched_t, t, side="right")) - 1)]
        return held

    for _ in range(policy.hist):
        hist.append(observe(cmd_at(0.0)))

    ts: list[float] = []
    pos: list[np.ndarray] = []
    quat: list[np.ndarray] = []
    used: list[np.ndarray] = []

    viewer_cm = None
    if view:
        from mujoco import viewer as mj_viewer

        viewer_cm = mj_viewer.launch_passive(model, data)
    viewer = viewer_cm.__enter__() if viewer_cm is not None else None

    try:
        import time

        wall = time.perf_counter()
        for step in range(int(duration / sim_dt)):
            t = step * sim_dt
            if step % decim == 0:
                cmd = cmd_at(t)
                if t >= settle:
                    hist.append(observe(cmd))
                    # deque is oldest..newest; the nets want newest first.
                    p_obs = np.concatenate(list(hist)[::-1])
                    last_action, target = policy.act(p_obs, cmd)
                if ghost is not None:
                    g_t, g_p, g_q = ghost
                    i = max(0, int(np.searchsorted(g_t, t + start, side="right")) - 1)
                    data.mocap_pos[0] = g_p[i]
                    data.mocap_quat[0] = g_q[i]

                ts.append(t)
                pos.append(data.qpos[0:3].copy())
                quat.append(data.qpos[3:7].copy())
                used.append(cmd)

            tau = policy.kp * (target - data.qpos[7:19]) - policy.kd * data.qvel[6:18]
            data.ctrl[:] = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
            mujoco.mj_step(model, data)

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                wall += sim_dt / max(speed, 1e-6)
                lag = wall - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
                else:
                    wall = time.perf_counter()
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)

    return Track(t=np.array(ts), pos=np.array(pos), quat=np.array(quat), cmd=np.array(used))
