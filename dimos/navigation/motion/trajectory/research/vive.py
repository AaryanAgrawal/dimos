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

"""Recorded Vive tracker poses, expressed as a base_link trajectory.

Two conventions were read off the data rather than assumed (see
``test_vive.py`` for the checks that pin them):

* the quaternion is **wxyz** — under that reading the tracker's z axis holds
  0.997 +/- 0.003 alignment with world z across a run, i.e. the robot stays
  upright; xyzw gives 0.725 +/- 0.279, which is nonsense for a walking robot
* the frame is **z-up** — over a 55 s walk the z range is 0.09 m against
  1.0 and 1.6 m in x and y

What is *not* known is where the tracker sits relative to base_link. It is
mounted roughly 15 cm above the body, so :data:`DEFAULT_TRACKER_OFFSET` is
``(0, 0, -0.15)`` in tracker frame — a starting guess to be corrected by eye
against the ghost, not a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# base_link expressed in the tracker's frame. Guess: 15 cm straight down.
DEFAULT_TRACKER_OFFSET = np.array([0.0, 0.0, -0.15])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty((*q.shape[:-1], 3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """(3, 3) -> (4,) wxyz."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0) * 2
    q = np.empty(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (m[j, i] + m[i, j]) / s
    q[k + 1] = (m[k, i] + m[i, k]) / s
    return q


def read_vive_pose(dataset: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(t, pos, quat)`` — seconds from run start, (n, 3), (n, 4) wxyz."""
    from mcap.reader import make_reader

    ts: list[float] = []
    pos: list[list[float]] = []
    quat: list[list[float]] = []
    with Path(dataset).open("rb") as f:
        for _schema, channel, msg in make_reader(f).iter_messages(topics=["vive/pose"]):
            if channel.topic != "vive/pose":
                continue
            d = json.loads(msg.data)
            ts.append(msg.log_time / 1e9)
            pos.append(d["p"])
            quat.append(d["q"])
    if not ts:
        raise ValueError(f"{dataset}: no vive/pose messages")
    t = np.array(ts)
    return t - t[0], np.array(pos), np.array(quat)


def base_track(
    dataset: str | Path,
    *,
    tracker_offset: np.ndarray | None = None,
    anchor_pos: np.ndarray | None = None,
    anchor_quat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recorded base_link pose, re-anchored so t=0 sits at the sim's start pose.

    Anchoring at t=0 is what makes this comparable without knowing the Vive
    room's extrinsics: the unknown constant rotation and origin cancel, leaving
    the motion *relative to where the robot started*. The tracker offset does
    not cancel — it is a lever arm, so it still shows up as soon as the body
    rotates.
    """
    off = DEFAULT_TRACKER_OFFSET if tracker_offset is None else np.asarray(tracker_offset, float)
    t, p, q = read_vive_pose(dataset)
    rot = quat_to_mat(q)

    # tracker -> base_link, in the Vive frame
    base_p = p + np.einsum("nij,j->ni", rot, off)

    # re-express relative to the first sample
    r0t = rot[0].T
    rel_p = np.einsum("ij,nj->ni", r0t, base_p - base_p[0])
    rel_r = np.einsum("ij,njk->nik", r0t, rot)

    if anchor_quat is not None:
        a = quat_to_mat(np.asarray(anchor_quat, float))
        rel_p = np.einsum("ij,nj->ni", a, rel_p)
        rel_r = np.einsum("ij,njk->nik", a, rel_r)
    if anchor_pos is not None:
        rel_p = rel_p + np.asarray(anchor_pos, float)

    return t, rel_p, np.array([mat_to_quat(r) for r in rel_r])
