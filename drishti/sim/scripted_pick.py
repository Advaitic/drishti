"""Scripted top-down pick-and-place for the SO-101 (locked-wrist IK + finger pads).

A real, physical grasp — no weld. Adapted from ggand0/pick-101 (which adapts the
ECE4560 SO-101 assignment): the SO-101 can't reach a table object with a freely-solved
top-down orientation, so instead we LOCK the wrist into a top-down, fingers-horizontal
pose (wrist_flex = wrist_roll = 90deg) and position the gripper with the three proximal
joints (pan/shoulder/elbow) via damped-least-squares IK on the `gripperframe` (fingertip)
site. The cube is a 3cm block; tiny finger pads + an elliptic-cone/noslip contact solver
give a stable pinch (mesh-mesh contact alone is unstable — MuJoCo issue #239).

Sequence: above cube -> descend -> close (until both pads contact, then tighten) -> lift
-> carry over tray -> lower -> release. Success = cube ends inside the tray.

Used to (a) generate successful demos for finetuning and (b) be watched live in the viewer.
"""

from __future__ import annotations

import numpy as np

import mujoco

from drishti.sim.so101_env import SO101Env
from drishti.sim.ik_controller import IKController

# Top-down grasp configuration (radians). Wrist joints are held here and excluded from IK.
WRIST_FLEX_DOWN = np.pi / 2     # joint 3: gripper points straight down
WRIST_ROLL_HORIZ = np.pi / 2    # joint 4: fingers spread horizontally (straddle the cube)
LOCKED_JOINTS = [3, 4]

# Gripper action (IKController maps [-1,1] -> ctrlrange). open = jaws wide, closed = squeeze.
GRIP_OPEN = 0.3
GRIP_CLOSED = -0.8

# Grasp framing: the static finger is offset from the gripperframe, so shift the target in Y
# to centre the pinch on the cube; grasp slightly above the cube centre (grip its top half).
Y_GRIP_OFFSET = -0.015
Z_GRASP = 0.005

TRAY_XY = np.array([0.34, 0.02], float)   # placement target (matches task_scene.xml)


class IK:
    """Backward-compatible thin wrapper: binds a DLS IKController to an env on first use."""

    def __init__(self, env: SO101Env | None = None):
        self.env = env
        self._ctrl = None

    def bind(self, env: SO101Env) -> IKController:
        if self._ctrl is None or self.env is not env:
            self.env = env
            self._ctrl = IKController(env.model, env.data, end_effector_site="gripperframe")
        return self._ctrl


def _grasp_config(env: SO101Env, cube_xy):
    """Reset, place the cube, lock the wrist into the top-down pose, and let it settle."""
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    cx, cy = (cube_xy if cube_xy is not None else (0.27, 0.0))
    d.qpos[env._cube_qadr:env._cube_qadr + 7] = [cx, cy, 0.015, 1, 0, 0, 0]
    d.qpos[3] = WRIST_FLEX_DOWN
    d.qpos[4] = WRIST_ROLL_HORIZ
    d.ctrl[3] = WRIST_FLEX_DOWN
    d.ctrl[4] = WRIST_ROLL_HORIZ
    mujoco.mj_forward(m, d)
    for _ in range(40):
        mujoco.mj_step(m, d)
    return d.qpos[env._cube_qadr:env._cube_qadr + 3].copy()


def _is_grasping(env: SO101Env) -> bool:
    m, d = env.model, env.data
    cg = m.geom("cube").id
    sp = m.geom("static_finger_pad").id
    mp = m.geom("moving_finger_pad").id
    hs = hm = False
    for i in range(d.ncon):
        c = d.contact[i]
        pair = {c.geom1, c.geom2}
        if cg not in pair:
            continue
        other = (pair - {cg}).pop()
        hs |= (other == sp)
        hm |= (other == mp)
    return hs and hm


def run_episode(env: SO101Env, ik: IK | None = None, record=True,
                render_cams=("up", "side", "wrist"), on_step=None, cube_xy=None,
                rec_every=18, render_size=(640, 480)):
    """Run one scripted top-down pick-place. Returns (traj, success).

    `on_step(env)` is called after each control tick (for the live viewer). `cube_xy`
    randomizes the cube start (defaults to the scene pose). `rec_every` subsamples frames;
    `render_size` is (width, height) for the recorded camera frames.
    """
    rw, rh = render_size
    ik = ik or IK()
    ctrl_ik = ik.bind(env)
    m, d = env.model, env.data
    cube = _grasp_config(env, cube_xy)
    tray = TRAY_XY

    traj = {"state": [], "action": [], "images": {c: [] for c in render_cams}}

    def drive(target, grip, steps):
        """IK feedback loop toward `target` (gripperframe), wrist locked down, recording."""
        for k in range(steps):
            ctrl = ctrl_ik.step_toward_target(
                np.asarray(target, float), gripper_action=grip, gain=0.5,
                locked_joints=LOCKED_JOINTS)
            ctrl[3] = WRIST_FLEX_DOWN
            ctrl[4] = WRIST_ROLL_HORIZ
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)
            if record and k % rec_every == 0:
                traj["state"].append(env.state().tolist())
                traj["action"].append(np.asarray(ctrl[:6], float).tolist())
                for c in render_cams:
                    traj["images"][c].append(env.render(c, rw, rh))
            if on_step is not None:
                on_step(env)

    above = cube + [0, Y_GRIP_OFFSET, Z_GRASP + 0.04]
    grasp = cube + [0, Y_GRIP_OFFSET, Z_GRASP]
    lift = cube + [0, Y_GRIP_OFFSET, 0.12]
    over = [tray[0], tray[1] + Y_GRIP_OFFSET, 0.12]
    place = [tray[0], tray[1] + Y_GRIP_OFFSET, 0.055]

    drive(above, GRIP_OPEN, 200)     # 1. move above the cube
    drive(grasp, GRIP_OPEN, 120)     # 2. descend onto it
    drive(grasp, GRIP_CLOSED, 150)   # 3. close + squeeze
    drive(lift, GRIP_CLOSED, 150)    # 4. lift straight up
    drive(over, GRIP_CLOSED, 160)    # 5. carry over the tray
    drive(place, GRIP_CLOSED, 120)   # 6. lower into the tray
    drive(place, GRIP_OPEN, 80)      # 7. release
    drive(over, GRIP_OPEN, 80)       # 8. retreat up

    success = _cube_in_tray(env.cube_pos(), tray)
    return traj, success


def _cube_in_tray(cube_pos, tray) -> bool:
    dx, dy = cube_pos[0] - tray[0], cube_pos[1] - tray[1]
    return abs(dx) < 0.045 and abs(dy) < 0.03 and cube_pos[2] < 0.06


def _smoke():
    """Run one episode headless, render a frame per phase to inspect the pick-place."""
    from pathlib import Path
    from PIL import Image
    env = SO101Env()
    traj, success = run_episode(env, render_cams=("up", "side", "wrist"), cube_xy=(0.27, 0.0))
    out = Path("data/scripted_smoke"); out.mkdir(parents=True, exist_ok=True)
    ups = traj["images"]["up"]
    n = len(ups)
    for i, idx in enumerate(np.linspace(0, n - 1, 8).astype(int)):
        Image.fromarray(ups[idx]).save(out / f"f{i}.png")
    print(f"episode done | success={success} | frames={n} | final cube={np.round(env.cube_pos(),3)}")


if __name__ == "__main__":
    _smoke()
