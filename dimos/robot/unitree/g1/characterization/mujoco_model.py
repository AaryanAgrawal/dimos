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

"""The production G1 MuJoCo model composition used by offline evaluation."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.utils.data import get_data

_G1_MJCF = Path(__file__).resolve().parents[1] / "assets" / "g1_29dof.xml"


@dataclass(frozen=True)
class G1MujocoBinding:
    """MuJoCo addresses in recorded G1 motor order."""

    root_qpos: int
    root_qvel: int
    joint_qpos: NDArray[np.int64]
    joint_qvel: NDArray[np.int64]
    actuators: NDArray[np.int64]


def build_g1_mujoco_spec() -> mujoco.MjSpec:
    """Compose the same empty scene, robot MJCF, and mesh tree as the blueprint."""
    scene = Path(get_data("mujoco_sim")) / "scene_empty.xml"
    meshdir = Path(get_data("g1_urdf")) / "meshes"
    spec = mujoco.MjSpec.from_file(str(scene))
    robot = mujoco.MjSpec.from_file(str(_G1_MJCF))
    robot.meshdir = str(meshdir)
    spec.option.timestep = robot.option.timestep
    spec.attach(robot, frame=spec.worldbody.add_frame())
    return spec


def _model_id(model: mujoco.MjModel, object_type: object, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        result = mujoco.mj_name2id(model, object_type, f"/{name}")
    if result < 0:
        raise KeyError(f"MuJoCo model has no {name!r}")
    return int(result)


def g1_mujoco_binding(model: mujoco.MjModel, motor_names: tuple[str, ...]) -> G1MujocoBinding:
    """Resolve root, joint, and actuator indices without assuming model order."""
    root_joint = _model_id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    qpos, qvel, actuator = [], [], []
    for hardware_name in motor_names:
        bare = hardware_name.rsplit("/", 1)[-1]
        model_name = bare if bare.endswith("_joint") else f"{bare}_joint"
        joint_id = _model_id(model, mujoco.mjtObj.mjOBJ_JOINT, model_name)
        actuator_id = _model_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model_name)
        qpos.append(int(model.jnt_qposadr[joint_id]))
        qvel.append(int(model.jnt_dofadr[joint_id]))
        actuator.append(actuator_id)
    return G1MujocoBinding(
        root_qpos=int(model.jnt_qposadr[root_joint]),
        root_qvel=int(model.jnt_dofadr[root_joint]),
        joint_qpos=np.asarray(qpos, dtype=np.int64),
        joint_qvel=np.asarray(qvel, dtype=np.int64),
        actuators=np.asarray(actuator, dtype=np.int64),
    )
