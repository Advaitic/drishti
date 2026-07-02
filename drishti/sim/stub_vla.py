"""StubVLA — a deterministic, CPU-only stand-in for MolmoAct2 (Stage 0 plumbing tests).

It implements the same `VLA` surface the rollout server drives — `predict(images, task, state_rad)
-> (H,6)`, `sample(images, task, state_rad, n) -> (n,H,6)`, `reasoning() -> str` — so
`rollout_server` runs identically with the stub or the real model. It does NOT simulate real
grasping (that is MolmoAct2's job in Stage 1); it exists to exercise the server/env/reward wiring:

  * "flail" (default): jittery actions and high-variance action samples -> the ACE detector spikes,
    the cube is never moved -> the episode fails. This drives the `status="spike"` path.
  * "solve": holds a steady pose (low-variance samples -> no spike) AND nudges the cube toward the
    tray each call, so `_cube_in_tray` becomes true within a few frames -> `status="success"`.

The stub flips flail -> solve the moment the language `task` changes from the first one it saw — i.e.
when the corrector injects a corrective prompt. So a baseline rollout (no correction) fails, and a
corrected rollout succeeds: exactly the reward-responds-to-correction wiring Stage 0 must prove.
"""

from __future__ import annotations

import numpy as np

from drishti.sim.so101_env import SO101Env
from drishti.sim.scripted_pick import TRAY_XY


class StubVLA:
    def __init__(self, env: SO101Env, *, success_on_correction: bool = True,
                 flail_scale: float = 1.2, seed: int = 0):
        self.env = env
        self.success_on_correction = success_on_correction
        self.flail_scale = float(flail_scale)
        self.rng = np.random.default_rng(seed)
        self._base_task = None            # the first task seen == the uncorrected instruction
        self._placed = False
        self._reasoning = ""

    # --- mode: solve once the task has been corrected (changed from the baseline) ---
    def _solving(self, task: str) -> bool:
        if self._base_task is None:
            self._base_task = task
        return self.success_on_correction and task != self._base_task

    def predict(self, images, task, state_rad) -> np.ndarray:
        state = np.asarray(state_rad, dtype=float).reshape(-1)[:6]
        if self._solving(task):
            self._place_cube_in_tray()
            self._reasoning = "moving the cube into the box"
            return state[None, :]                        # (1,6) hold pose while the cube is placed
        self._reasoning = "unsure where the cube is; hunting"
        jitter = self.rng.normal(0.0, 0.15, size=6)
        return (state + jitter)[None, :]                 # (1,6) small wander

    def sample(self, images, task, state_rad, n: int = 6) -> np.ndarray:
        state = np.asarray(state_rad, dtype=float).reshape(-1)[:6]
        if self._solving(task):                           # confident -> identical samples -> ACE == 0
            return np.repeat(state[None, None, :], int(n), axis=0)
        wide = self.rng.normal(0.0, self.flail_scale, size=(int(n), 1, 6))  # confused -> high ACE
        return state[None, None, :] + wide

    def reasoning(self) -> str:
        return self._reasoning

    # --- test-only shortcut: drop the cube cleanly into the tray ONCE so `_cube_in_tray` fires ---
    # (Teleporting every frame shoved the cube into the tray walls and stalled the contact solver.)
    def _place_cube_in_tray(self) -> None:
        if self._placed:
            return
        env = self.env
        qa, va = env._cube_qadr, env._cube_vadr
        env.data.qpos[qa:qa + 7] = [TRAY_XY[0], TRAY_XY[1], 0.035, 1, 0, 0, 0]
        env.data.qvel[va:va + 6] = 0.0                   # no launch velocity -> settles, no penetration
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        self._placed = True
