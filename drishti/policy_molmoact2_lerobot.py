"""Finetuned MolmoAct2 (LeRobot checkpoint) as a VLA for the rollout server / recorder.

Wraps LeRobot's `MolmoAct2Policy` + its pre/post processors (loaded from a finetuned checkpoint) behind the
same `VLA` surface the rollout server drives — `predict` / `sample` / `reasoning`. Needs CUDA and a lerobot
`--extra molmoact2` env; runs on the pod, not the Mac.

Two things it handles that the raw checkpoint doesn't:
- **Joint-frame correction (R4, action side).** LeRobot's molmoact2 postprocessor does NOT cleanly invert
  `joint_signs/offsets` for the ACTION output, so shoulder_lift(1) and elbow_flex(2) come out in the wrong
  frame. Empirically (teacher-forced on the demos): `env_j1 = 90 - pred`, `env_j2 = pred + 90`, others
  pass through. `_frame_fix` applies this. (Clean fix = rebuild the dataset in the model frame + identity
  transform; deferred.)
- **Buffered execution (`n_action_steps=4`, matching the eval that grasped 3/8).** `predict` pulls from the
  policy's `_action_queue` via `select_action`, so it COMMITS to 4 steps of a planned descent before
  re-planning — the follow-through a grasp needs. (Re-planning a fresh stochastic chunk every frame,
  `n_action_steps=1`, jitters and never completes the grasp — that gave 0/2 in a recorder smoke.)
- **ACE sampling is decoupled from execution.** `sample` calls `predict_action_chunk` directly (fresh flow
  noise per call) which does NOT touch `_action_queue`, so measuring uncertainty never perturbs the arm's
  committed trajectory. Call `reset()` once per episode to clear the queue.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEG2RAD = np.pi / 180.0
CAMS_VLA = ("up", "wrist")
DEFAULT_TASK = "Pick up the cube and place it in the box"   # matches the dataset's task string (capital P)


def _frame_fix(a) -> np.ndarray:
    """Policy action output (degrees) -> env-frame degrees. Empirical per-joint correction (see module doc)."""
    a = np.asarray(a, dtype=float).reshape(-1)[:6].copy()
    a[1] = 90.0 - a[1]
    a[2] = a[2] + 90.0
    return a


class MolmoAct2LeRobotVLA:
    def __init__(self, ckpt: str, device: str = "cuda", n_action_steps: int = 4,
                 num_inference_steps: int = 10):
        import torch  # noqa: F401
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
        from lerobot.policies import make_pre_post_processors
        self.policy = MolmoAct2Policy.from_pretrained(ckpt).eval().to(device)
        self.policy.config.inference_action_mode = "continuous"
        self.policy.config.n_action_steps = n_action_steps
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": device}})
        self.device = device
        self._reasoning = ""
        self._last_head_std = np.zeros(6, dtype=float)   # fast ACE: per-dim spread of the last chunk

    def _obs(self, images, task, state_rad) -> dict:
        import torch
        o: dict[str, Any] = {}
        for cam, img in zip(CAMS_VLA, images):
            t = torch.from_numpy(np.asarray(img, np.uint8)).permute(2, 0, 1).float().div(255).unsqueeze(0)
            o[f"observation.images.{cam}"] = t.to(self.device)
        state_deg = np.asarray(state_rad, dtype=float).reshape(-1)[:6] / DEG2RAD
        o["observation.state"] = torch.from_numpy(state_deg.astype(np.float32)).unsqueeze(0).to(self.device)
        o["task"] = task or DEFAULT_TASK
        return o

    def reset(self) -> None:
        """Clear the action-step queue; call once at the start of each episode."""
        self.policy.reset()

    def predict(self, images, task, state_rad) -> np.ndarray:
        """One env action from the BUFFERED policy (receding horizon: execute n_action_steps of a planned
        chunk, then re-plan). Replicates `select_action`'s queue but also captures the regenerated chunk's
        per-dim horizon spread as **fast ACE** — the SmolVLA `action_head_std` analog, ZERO extra forwards."""
        import torch
        q = self.policy._action_queue
        batch = self.pre(self._obs(images, task, state_rad))
        with torch.inference_mode():
            if len(q) == 0:
                chunk = self.policy.predict_action_chunk(batch)          # (1, T, D) fresh plan
                n = int(self.policy.config.n_action_steps)
                q.extend(chunk[:, :n].transpose(0, 1))                   # buffer the next n actions
                # fast ACE: spread of the planned trajectory over its horizon, per joint (raw model frame;
                # the joint transform is affine so it preserves the spread, and calibrate z-normalizes it).
                self._last_head_std = np.abs(chunk[0].float().cpu().numpy()).std(axis=0)[:6]
            a = self.post(q.popleft())
        return (_frame_fix(a.squeeze(0).float().cpu().numpy()) * DEG2RAD)[None, :]   # (1, 6) env rad

    def action_head_std(self) -> list:
        """Per-dim spread of the last planned chunk (fast, single-forward ACE). Reused between re-plans."""
        return [float(x) for x in self._last_head_std]

    def sample(self, images, task, state_rad, n: int = 6) -> np.ndarray:
        """n fresh-noise first-actions via predict_action_chunk (does NOT touch the execution queue) — for ACE."""
        import torch
        batch = self.pre(self._obs(images, task, state_rad))
        outs = []
        with torch.inference_mode():
            for _ in range(int(n)):
                chunk = self.policy.predict_action_chunk(batch)          # (1, T, D) fresh flow noise, buffer-free
                a = self.post(chunk[:, 0, :])                            # first action, unnormalized -> (1, D)
                outs.append(_frame_fix(a.squeeze(0).float().cpu().numpy()) * DEG2RAD)
        return np.stack(outs, axis=0)[:, None, :]                        # (n, 1, 6) env rad

    def reasoning(self) -> str:
        return self._reasoning


def load_molmoact2_lerobot(ckpt: str, device: str = "cuda", **kw) -> MolmoAct2LeRobotVLA:
    return MolmoAct2LeRobotVLA(ckpt, device=device, **kw)
