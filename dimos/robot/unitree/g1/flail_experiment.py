"""SIMULATED: the GR00T G1 policy lifted or pushed over in MuJoCo, with or without the protective stop."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import mujoco
import numpy as np

from dimos.control.components import HardwareComponent, HardwareType
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.control.tasks.g1_groot_wbc_task.g1_groot_wbc_task import (
    _DEFAULT_POSITIONS_29,
    G1_GROOT_KD,
    G1_GROOT_KP,
    g1_arms,
    g1_joints,
    g1_legs_waist,
)
from dimos.control.tasks.trajectory_task.trajectory_task import joint_trajectory_task
from dimos.hardware.whole_body.spec import WholeBodyConfig
import dimos.simulation.adapters.whole_body.g1 as sim_g1
from dimos.simulation.engines.mujoco_sim_module import MujocoSimModule
from dimos.simulation.engines.robot_sim_binding import RobotSimSpec, mjcf_joint_names_from_hardware
from dimos.utils.data import get_data

_ROBOT_MJCF = Path(__file__).with_name("assets") / "g1_29dof.xml"
_SETTLE_S = 0.5  # driven stance on the hoist after arming, before the hoist lets go
_STAND_S = 2.0  # free stance before the lift / push / walk command
_LIFT_M, _LIFT_S, _HOLD_S = 0.5, 0.5, 6.0
_PUSH_N = 300.0  # 35 kg G1 tips inside 0.5 s at 300 N lateral; the balance policy cannot hold it
_PUSH_STOP_DEG, _PUSH_MAX_S, _AFTER_IMPACT_S = 30.0, 1.5, 5.0
_FIELDS = ("t", "q", "dq", "tau", "q_cmd", "quat", "gyro", "accel", "height", "foot", "body")


def _tilt_deg(quat: Any) -> float:
    return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2), -1, 1))))


def _hoist_mjcf(meshdir: Path) -> Path:
    """Test rig: the robot MJCF plus a mocap hook at the torso, weld (rigid) and pin (free) both inactive."""
    spec = mujoco.MjSpec.from_file(str(_ROBOT_MJCF))
    spec.meshdir = str(meshdir)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    pos, quat = (" ".join(f"{v:.6f}" for v in x) for x in (data.xpos[torso], data.xquat[torso]))
    path = Path(tempfile.gettempdir()) / "g1_29dof_hoist.xml"
    xml = _ROBOT_MJCF.read_text().replace('meshdir="meshes"', f'meshdir="{meshdir}"')
    xml = xml.replace(
        "</worldbody>", f'<body name="hook" mocap="true" pos="{pos}" quat="{quat}"/></worldbody>'
    )
    path.write_text(
        xml.replace(
            "</mujoco>",
            (
                '<equality><weld name="hoist_weld" body1="hook" body2="torso_link" active="false"/>'
                '<connect name="hoist_pin" body1="hook" body2="torso_link" anchor="0 0 0" active="false"/>'
                "</equality></mujoco>"
            ),
        )
    )
    return path


class _Scenario:
    """Before-hook drives the scenario off sim time; after-hook records every physics step."""

    def __init__(
        self, name: str, sim: MujocoSimModule, coord: ControlCoordinator, rng: Any
    ) -> None:
        self.name, self.sim, self.coord, self.done = name, sim, coord, False
        assert sim._engine is not None and sim._sim_hooks is not None
        self.engine, self.hooks = sim._engine, sim._sim_hooks
        self.rows: dict[str, list[Any]] = {k: [] for k in _FIELDS}
        self.t_release = self.t_impact = np.nan
        self.hold: tuple[Any, Any] | None = None  # torso (position, quaternion) the hoist holds
        model, qadr, binding = (
            self.engine.model,
            self.engine.root_qpos_adr,
            self.engine.robot_binding,
        )
        assert qadr is not None and binding is not None
        self.qadr = qadr
        self.pelvis = binding.root_body_id
        self.torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.hook = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hook")]
        self.weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "hoist_weld")  # type: ignore[attr-defined]
        self.pin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "hoist_pin")  # type: ignore[attr-defined]
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)
        ]
        self.feet = {i for i, n in enumerate(names) if n.endswith("ankle_roll_link")}
        angle = rng.choice([-1.0, 1.0]) * np.pi / 2 + rng.uniform(
            -0.5, 0.5
        )  # lateral, +-30 deg jitter
        self.push_dir = np.array([np.cos(angle), np.sin(angle), 0.0])

    def hoist(self, data: Any, lift_m: float = 0.0, rigid: bool = True) -> None:
        """Move the mocap hook to the held torso pose (+lift) and engage the weld or the pin."""
        if self.hold is None:
            self.hold = (data.xpos[self.torso].copy(), data.xquat[self.torso].copy())
        data.mocap_pos[self.hook] = self.hold[0] + np.array([0.0, 0.0, lift_m])
        data.mocap_quat[self.hook] = self.hold[1]
        data.eq_active[self.weld if rigid else self.pin] = 1

    def before(self, engine: Any) -> None:
        self.hooks.pre_step(engine)
        data, t = engine.data, engine.data.time
        task: Any = self.coord._tasks.get("groot_wbc")
        if np.isnan(self.t_release):
            if task is not None and task._armed:
                self.t_release = t + _SETTLE_S
            return self.hoist(data)
        if t < self.t_release:
            return self.hoist(data)
        t_event = self.t_release + _STAND_S
        if t < t_event:
            data.eq_active[self.weld] = data.eq_active[self.pin] = 0
            self.hold = None
            return
        if self.name == "walk":
            task.set_velocity_command(0.5, 0.0, 0.0, time.perf_counter())
            self.done = t >= t_event + 5.0
        elif self.name.startswith("lift"):
            self.hoist(data, _LIFT_M * min(1.0, (t - t_event) / _LIFT_S), rigid=self.name == "lift")
            self.done = t >= t_event + _LIFT_S + _HOLD_S
        elif self.name == "fall":
            pushing = _tilt_deg(data.qpos[self.qadr + 3 : self.qadr + 7]) < _PUSH_STOP_DEG
            pushing = pushing and t < t_event + _PUSH_MAX_S and np.isnan(self.t_impact)
            data.xfrc_applied[self.pelvis, :3] = _PUSH_N * self.push_dir if pushing else 0.0
            self.done = t >= (t if np.isnan(self.t_impact) else self.t_impact) + _AFTER_IMPACT_S
        else:
            self.done = t >= t_event + 5.0

    def after(self, engine: Any) -> None:
        self.sim._publish_shm_and_lcm(engine)
        data, r = engine.data, self.rows
        foot = body = 0
        for c in data.contact[: data.ncon]:
            if self.floor in (c.geom1, c.geom2):
                other = engine.model.geom_bodyid[c.geom2 if c.geom1 == self.floor else c.geom1]
                foot, body = foot + (other in self.feet), body + (other not in self.feet)
        if body and np.isnan(self.t_impact) and data.time >= self.t_release + _STAND_S:
            self.t_impact = data.time
        cmd = self.hooks._latest_pd_pos_target
        gyro, accel = (
            data.sensordata[self.sim._imu_gyro_slice],
            data.sensordata[self.sim._imu_accel_slice],
        )
        values = (
            data.time,
            engine.joint_positions,
            engine.joint_velocities,
            engine.joint_efforts,
            np.full(29, np.nan) if cmd is None else cmd,
            data.qpos[self.qadr + 3 : self.qadr + 7],
            gyro,
            accel,
            data.qpos[self.qadr + 2],
            foot,
            body,
        )
        for k, v in zip(_FIELDS, values, strict=True):
            r[k].append(np.array(v, dtype=np.float64))


def main() -> None:
    global _LIFT_S
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario", choices=["stand", "walk", "lift", "lift_free", "fall"], required=True
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lift-seconds", type=float, default=_LIFT_S)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-stop", action="store_true", help="the BEFORE case: never damp the robot")
    args = ap.parse_args()
    _LIFT_S = args.lift_seconds
    if args.no_stop:
        sim_g1.stop_reason = lambda *_args, **_kwargs: ""  # type: ignore[attr-defined]
    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    version = mujoco.__version__  # type: ignore[attr-defined]
    print(
        f"SIMULATED MuJoCo {version} seed={args.seed} scenario={args.scenario} "
        f"stop={not args.no_stop} git={rev}"
    )
    meshdir = get_data("g1_urdf") / "meshes"
    robot_mjcf = _hoist_mjcf(meshdir)
    spec = RobotSimSpec(
        robot_id="g1",
        hardware_joints=tuple(g1_joints),
        root_body_names=("pelvis",),
        root_joint_names=("floating_base_joint",),
        require_floating_base=True,
        model_joint_names=mjcf_joint_names_from_hardware(tuple(g1_joints)),
        imu_gyro_names=("imu-angular-velocity",),
        imu_accel_names=("imu-linear-acceleration",),
        require_imu=True,
    )
    sim = MujocoSimModule(
        scene_xml=get_data("mujoco_sim") / "scene_empty.xml",
        robot_mjcf=robot_mjcf,
        robot_meshdir=meshdir,
        dof=29,
        headless=True,
        enable_color=False,
        enable_depth=False,
        reset_joint_positions=list(_DEFAULT_POSITIONS_29),
        robot_sim_spec=spec,
    )
    coord = ControlCoordinator(
        tick_rate=50.0,
        publish_joint_state=False,
        hardware=[
            HardwareComponent(
                hardware_id="g1",
                hardware_type=HardwareType.WHOLE_BODY,
                joints=g1_joints,
                adapter_type="sim_mujoco_g1",
                address=robot_mjcf,
                wb_config=WholeBodyConfig(kp=tuple(G1_GROOT_KP), kd=tuple(G1_GROOT_KD)),
            )
        ],
        tasks=[
            TaskConfig(
                name="groot_wbc",
                type="g1_groot_wbc",
                joint_names=g1_legs_waist,
                priority=50,
                auto_start=True,
                params={
                    "model_path": get_data("groot"),
                    "hardware_id": "g1",
                    "auto_arm": True,
                    "default_ramp_seconds": 0.0,
                    "decimation": 1,
                },
            ),
            joint_trajectory_task(
                g1_arms,
                priority=10,
                velocity_limits=dict.fromkeys(g1_arms, 1.0),
                hold_position_when_idle=True,
            ),
        ],
    )
    sim.start()
    scenario = _Scenario(args.scenario, sim, coord, np.random.default_rng(args.seed))
    scenario.engine.set_step_hooks(before=scenario.before, after=scenario.after)
    coord.start()
    while not scenario.done:
        time.sleep(0.05)
    coord.stop()
    sim.stop()
    arrays = {k: np.array(v) for k, v in scenario.rows.items()}
    np.savez_compressed(
        args.out,
        **arrays,  # type: ignore[arg-type]
        scenario=args.scenario,
        seed=args.seed,
        git=rev,
        mujoco=version,
        t_release=scenario.t_release,
        t_event=scenario.t_release + _STAND_S,
        t_impact=scenario.t_impact,
        push_dir=scenario.push_dir,
        joint_names=np.array(g1_joints),
    )
    print(
        f"saved {args.out}: {len(arrays['t'])} steps, t_release={scenario.t_release:.3f}, "
        f"t_impact={scenario.t_impact:.3f}, max|dq|={np.abs(arrays['dq']).max():.2f} rad/s"
    )


if __name__ == "__main__":
    main()
