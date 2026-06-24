"""SO-101 MuJoCo environment — a thin wrapper for the closed-loop driver.

Loads the vendored SO-101 scene (arm + floor + cube + up/side cameras), exposes
reset / state / render / step. The 6 actuators (shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper) are position-controlled in radians;
`step(action6)` clamps to each actuator's ctrlrange and sub-steps the sim.

Mirrors the structure of the old sim/runner.py (Franka), adapted to SO-101.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import mujoco

ARM_JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
]

# repo_root/sim/so101/task_scene.xml  (this file is drishti/sim/so101_env.py)
DEFAULT_SCENE = Path(__file__).resolve().parents[2] / "sim" / "so101" / "task_scene.xml"


class SO101Env:
    def __init__(self, scene_path: Optional[str] = None, n_substeps: int = 10):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path or DEFAULT_SCENE))
        self.data = mujoco.MjData(self.model)
        self.n_substeps = n_substeps
        self._qadr = np.array([int(self.model.joint(j).qposadr[0]) for j in ARM_JOINTS])
        self.ctrlrange = self.model.actuator_ctrlrange.copy()  # (nu, 2)
        self._cube_qadr = int(self.model.joint("cube_free").qposadr[0])
        self._home = np.zeros(6, dtype=np.float64)
        self._renderers: dict = {}

    def reset(self, cube_xy=None, seed: Optional[int] = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)  # cube → its XML-defined pose
        for adr, val in zip(self._qadr, self._home):
            self.data.qpos[adr] = val
        if cube_xy is not None:  # optional override (e.g. for randomized starts)
            cx, cy = cube_xy
            self.data.qpos[self._cube_qadr:self._cube_qadr + 7] = [cx, cy, 0.0125, 1, 0, 0, 0]
        self.data.ctrl[:6] = self._home
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self) -> np.ndarray:
        """6-d joint position vector [5 arm joints + gripper], radians."""
        return self.data.qpos[self._qadr].astype(np.float32).copy()

    def render(self, camera: str = "up", width: int = 640, height: int = 480) -> np.ndarray:
        r = self._renderers.get((width, height))
        if r is None:
            r = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[(width, height)] = r
        r.update_scene(self.data, camera=camera)
        return r.render()  # (H, W, 3) uint8

    def step(self, action6, n_substeps: Optional[int] = None) -> np.ndarray:
        a = np.asarray(action6, dtype=np.float64).reshape(-1)[:6]
        a = np.clip(a, self.ctrlrange[:6, 0], self.ctrlrange[:6, 1])
        self.data.ctrl[:6] = a
        for _ in range(n_substeps or self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def cube_pos(self) -> np.ndarray:
        return self.data.qpos[self._cube_qadr:self._cube_qadr + 3].astype(np.float32).copy()


def _smoke() -> None:
    """Load the scene, render up + side cameras to PNGs to verify framing."""
    from PIL import Image

    env = SO101Env()
    print(f"model ok: nq={env.model.nq} nu={env.model.nu} cameras="
          f"{[env.model.camera(i).name for i in range(env.model.ncam)]}")
    print(f"ctrlrange:\n{np.round(env.ctrlrange[:6], 3)}")
    env.reset()
    out = Path("data/so101_smoke")
    out.mkdir(parents=True, exist_ok=True)
    for cam in ("up", "side", "wrist"):
        img = env.render(cam, 640, 480)
        Image.fromarray(img).save(out / f"{cam}.png")
        print(f"  rendered {cam}: {img.shape} -> {out / f'{cam}.png'}")
    print("state:", np.round(env.state(), 3))
    print("ok")


if __name__ == "__main__":
    _smoke()
