"""Scripted pick-and-place controller for the SO-101 (ground-truth poses + mink IK).

Pure MuJoCo + IK, no policy. Drives the arm through a state machine
(approach -> descend -> grasp -> lift -> over-tray -> release), using mink
differential IK on a robot-only model to turn end-effector waypoints into joint
targets, then commands them through the env's position actuators.

Used to (a) generate successful demos for finetuning and (b) be watched live in
the mjpython viewer. Success = brick ends up inside the tray (geometric check).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mujoco
import mink

from drishti.sim.so101_env import ARM_JOINTS, SO101Env

ROBOT_XML = Path(__file__).resolve().parents[2] / "sim" / "so101" / "so101_new_calib.xml"

GRIP_OPEN = 1.5     # radians (within ctrlrange [-0.17, 1.745])
GRIP_CLOSED = 0.05


class IK:
    """mink IK on a robot-only model (no cube freejoint to corrupt the solve)."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        self.cfg = mink.Configuration(self.model)
        self._adr = [int(self.model.joint(j).qposadr[0]) for j in ARM_JOINTS]
        # at home (qpos=0) the gripper points down — use that as the grasp orientation
        self.cfg.update(np.zeros(self.model.nq))
        self.down_rot = mink.SO3.from_matrix(
            self.cfg.data.site("gripperframe").xmat.reshape(3, 3).copy())

    def solve(self, q_arm6, target_pos, ori_cost=0.0, iters=400, dt=0.02):
        q = np.zeros(self.model.nq)
        for a, val in zip(self._adr, q_arm6):
            q[a] = val
        self.cfg.update(q)
        # fresh task each call (mutating orientation_cost on a reused task doesn't take)
        task = mink.FrameTask("gripperframe", "site", position_cost=1.0,
                              orientation_cost=ori_cost, lm_damping=1e-2)
        if ori_cost > 0:
            task.set_target(mink.SE3.from_rotation_and_translation(
                self.down_rot, np.asarray(target_pos, float)))
        else:
            task.set_target(mink.SE3.from_translation(np.asarray(target_pos, float)))
        for _ in range(iters):
            v = mink.solve_ik(self.cfg, [task], dt, solver="quadprog", damping=1e-3)
            self.cfg.integrate_inplace(v, dt)
        return np.array([self.cfg.q[a] for a in self._adr])

    def ee_pos(self):
        return self.cfg.data.site("gripperframe").xpos.copy()


def run_episode(env: SO101Env, ik: IK, record=True, render_cams=("up", "side", "wrist"),
                steps_per_phase=14, on_step=None):
    """Run one scripted pick-place. Returns (traj, success).

    Position-only IK (reliable on the 5-DoF arm). The grasp is kinematic: while the
    gripper is closed the cube tracks the grasp point, and is released (falls) on open.
    `on_step(env)` is called after each control step (for the live viewer / sub-step feed).
    """
    env.reset()
    cube = env.cube_pos()
    tray = np.array([0.13, -0.14], float)  # tray centre (matches task_scene.xml)

    waypoints = [
        (cube + [0, 0, 0.10], GRIP_OPEN),                 # approach above brick
        (cube + [0, 0, 0.01], GRIP_OPEN),                 # descend onto brick
        (cube + [0, 0, 0.01], GRIP_CLOSED),               # grasp
        (cube + [0, 0, 0.14], GRIP_CLOSED),               # lift
        ([tray[0], tray[1], 0.16], GRIP_CLOSED),          # over tray
        ([tray[0], tray[1], 0.10], GRIP_OPEN),            # release over tray
    ]

    traj = {"state": [], "action": [], "images": {c: [] for c in render_cams}}
    q_arm = env.state()[:6].astype(float)

    def _do(target):
        if record:
            traj["state"].append(env.state().tolist())
            traj["action"].append(target.tolist())
            for c in render_cams:
                traj["images"][c].append(env.render(c, 640, 480))
        env.step(target)
        if target[5] == GRIP_CLOSED:          # kinematic grasp: cube follows the gripper
            env.set_cube(env.gripper_tcp())
        if on_step is not None:
            on_step(env)

    for (pos, grip) in waypoints:
        arm = ik.solve(q_arm[:6], pos, ori_cost=0.0)
        target = np.concatenate([arm[:5], [grip]])
        for _ in range(steps_per_phase):
            _do(target)
        q_arm = env.state()[:6].astype(float)

    for _ in range(20):  # settle: the released brick drops into the tray
        _do(target)

    success = _cube_in_tray(env.cube_pos(), np.array([tray[0], tray[1], 0.0]))
    return traj, success


def _cube_in_tray(cube_pos, box) -> bool:
    dx, dy = cube_pos[0] - box[0], cube_pos[1] - box[1]
    return abs(dx) < 0.045 and abs(dy) < 0.03 and cube_pos[2] < 0.05


def _smoke():
    """Run one episode headless, render a frame per phase to inspect the grasp."""
    from PIL import Image
    env = SO101Env(n_substeps=20)
    ik = IK()
    traj, success = run_episode(env, ik, render_cams=("up",), steps_per_phase=12)
    out = Path("data/scripted_smoke"); out.mkdir(parents=True, exist_ok=True)
    ups = traj["images"]["up"]
    # one frame at the end of each phase (every steps_per_phase)
    for ph in range(6):
        idx = min((ph + 1) * 12 - 1, len(ups) - 1)
        Image.fromarray(ups[idx]).save(out / f"phase{ph}.png")
    print(f"episode done | success={success} | frames={len(ups)} | final cube={np.round(env.cube_pos(),3)}")


if __name__ == "__main__":
    _smoke()
