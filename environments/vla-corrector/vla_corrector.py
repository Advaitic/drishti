"""`vla-corrector` — a verifiers MultiTurnEnv that trains a corrective policy over a VLA worker.

The heavy half (MuJoCo sim + MolmoAct2 + the uncertainty cascade) runs on a pod behind the drishti
`rollout_server`; THIS env is a thin HTTP client of it, carrying no torch/mujoco. Each rollout:

  setup_state:  start an episode, run the VLA closed-loop to the first uncertainty spike, and seed the
                first failure report into the prompt (so the model always sees a failure before acting).
  env_response: the model's message is a corrective instruction -> inject it, resume the VLA to the next
                spike (return the next report) or a terminal outcome (signal stop via `final_env_response`).
  Rubric:       reward = task success (+ optional shaping). Reward is computed here, in the trainer's
                process, from `state["success"]` — never trusting the pod's advisory reward.

One turn == one intervention. Multimodal-ready: when `include_image` is on, the failure report is sent as
OpenAI `image_url` content parts (the spike frames) alongside the text — for a Qwen3-VL corrector.

Confirmed against verifiers==0.1.14 (async env_response/setup_state; termination via the `final_env_response`
`@vf.stop` sentinel; `state["info"]` carries the per-episode seed/cube/task from the dataset row).
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx
import verifiers as vf
from datasets import Dataset

SYSTEM = (
    "You supervise a robot arm's vision-language policy performing a pick-and-place. "
    "When the policy's uncertainty spikes you receive a failure report (and, if available, what it sees). "
    "Reply with ONE short corrective instruction (<= 20 words) telling the policy what to do next. "
    "No preamble, no explanation — just the instruction."
)
DEFAULT_TASK = "pick up the cube and place it in the box"


class RolloutClient:
    """Thin sync client of the drishti rollout server (called off the event loop via to_thread)."""

    def __init__(self, base_url: str, timeout: float = 180.0):
        self._c = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def start(self, seed, cube_xy, task) -> dict:
        return self._c.post("/episode/start",
                            json={"seed": seed, "cube_xy": cube_xy, "task": task}).json()

    def run(self, episode_id, max_frames) -> dict:
        return self._c.post("/episode/run",
                            json={"episode_id": episode_id, "max_frames": max_frames}).json()

    def correct(self, episode_id, prompt) -> dict:
        return self._c.post("/episode/correct",
                            json={"episode_id": episode_id, "prompt": prompt}).json()

    def end(self, episode_id) -> dict:
        return self._c.post("/episode/end", json={"episode_id": episode_id}).json()


def _last_assistant_text(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            c = m.get("content", "")
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            return str(c)
    return ""


def _terminal_message(report: dict) -> list:
    """verifiers >=0.1.15 requires state['final_env_response'] to be a list of Message-like objects
    (not the raw report dict). The reward is read from state['success'] (set by _absorb), not from here."""
    return [{"role": "user",
             "content": f"Episode ended ({report['status']}): success={report['success']}."}]


def _report_message(report: dict, include_image: bool) -> dict:
    """Build the user turn the corrector sees: the text failure report (+ spike frames as image parts)."""
    text = report["failure_report"]
    if not include_image or not report.get("frames_b64"):
        return {"role": "user", "content": text}
    content: list[Any] = [{"type": "text", "text": text}]
    for _cam, b64 in report["frames_b64"].items():
        if b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return {"role": "user", "content": content}


class VLACorrectorEnv(vf.MultiTurnEnv):
    def __init__(self, *, client: RolloutClient, max_frames: int = 110,
                 include_image: bool = True, **kwargs):
        super().__init__(**kwargs)                       # dataset, rubric, max_turns via kwargs
        self.client = client
        self.max_frames = max_frames
        self.include_image = include_image

    def _absorb(self, state: dict, report: dict) -> None:
        state["last_report"] = report
        state["status"] = report["status"]
        state["success"] = bool(report["success"])
        state["done"] = bool(report["done"])
        state["progress"] = max(state.get("progress", 0.0), float(report.get("progress", 0.0)))
        state.setdefault("uq_trace", []).append(report["uncertainty"])

    async def setup_state(self, state: vf.State) -> None:
        info = state.get("info") or {}
        ep = await asyncio.to_thread(self.client.start, info.get("seed"),
                                     info.get("cube_xy"), info.get("vla_task", DEFAULT_TASK))
        state["episode_id"] = ep["episode_id"]
        state["n_interventions"] = 0
        state["uq_trace"] = []
        report = await asyncio.to_thread(self.client.run, ep["episode_id"], self.max_frames)
        self._absorb(state, report)
        if report["status"] == "spike":                  # ask the model to fix it
            state["prompt"] = list(state["prompt"]) + [_report_message(report, self.include_image)]
        else:                                            # VLA already finished (success/timeout) -> stop
            state["final_env_response"] = _terminal_message(report)

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        correction = _last_assistant_text(messages)
        await asyncio.to_thread(self.client.correct, state["episode_id"], correction)
        state["n_interventions"] += 1
        report = await asyncio.to_thread(self.client.run, state["episode_id"], self.max_frames)
        self._absorb(state, report)
        if report["status"] == "spike":
            return [_report_message(report, self.include_image)]
        state["final_env_response"] = _terminal_message(report)   # terminal -> ends the rollout
        return []


# ------------------------------------------------------------------ reward / metric functions
async def reward_success(*, state=None, **kwargs) -> float:
    """Verifiable reward: did the VLA place the cube in the tray?"""
    return 1.0 if (state and state.get("success")) else 0.0


async def penalty_interventions(*, state=None, **kwargs) -> float:
    """Small shaping penalty: discourage over-correcting (weight negative in the Rubric)."""
    return float((state or {}).get("n_interventions", 0))


async def reward_progress(*, state=None, **kwargs) -> float:
    """Dense staged task-progress in [0,1] (reached<grasped<lifted<placed), max over the episode.
    Gives GRPO reward variance from *real progress* on failing episodes — the main fix for the flat-reward
    problem. Group-relative advantage isolates the corrector's effect (same start pose across the group)."""
    return float((state or {}).get("progress", 0.0))


async def reward_uq_reduction(*, state=None, **kwargs) -> float:
    """Optional shaping: did the corrections calm the uncertainty over the episode? (>= 0)."""
    tr = (state or {}).get("uq_trace", [])
    if len(tr) < 2:
        return 0.0
    return max(0.0, float(tr[0]["combined"]) - float(tr[-1]["combined"]))


def load_environment(rollout_url: str = "http://localhost:8100", n_episodes: int = 64, seed0: int = 0,
                     max_turns: int = 4, max_frames: int = 110, cube_x=(0.24, 0.28),
                     cube_y=(-0.03, 0.03), curriculum=None, curriculum_jitter: float = 0.008,
                     shaping: bool = True, dense: bool = True, include_image: bool = True,
                     task: str = DEFAULT_TASK, client: Optional[RolloutClient] = None,
                     **kwargs) -> vf.Environment:
    import numpy as np
    rng = np.random.default_rng(seed0)

    def _cube_xy():
        # curriculum: sample cube positions from the frontier band (where the VLA succeeds SOMETIMES, so a
        # correction can flip the outcome -> real reward variance) with small jitter. Else uniform box.
        if curriculum:
            base = curriculum[int(rng.integers(len(curriculum)))]
            j = curriculum_jitter
            return [float(base[0] + rng.uniform(-j, j)), float(base[1] + rng.uniform(-j, j))]
        return [float(rng.uniform(*cube_x)), float(rng.uniform(*cube_y))]

    rows = [{
        "prompt": [{"role": "system", "content": SYSTEM}],
        "answer": "",
        # verifiers >=0.1.15 reserves info['task'] as a JSON task-payload + routes by info['env_id'];
        # keep our per-episode instruction under a non-reserved key.
        "info": {"env_id": "vla-corrector",
                 "seed": int(seed0 + i),
                 "cube_xy": _cube_xy(),
                 "vla_task": task},
    } for i in range(n_episodes)]
    dataset = Dataset.from_list(rows)

    funcs = [reward_success, penalty_interventions]
    weights = [1.0, -0.05]
    if dense:                                            # dense staged progress — the main reward-variance fix
        funcs.append(reward_progress)
        weights.append(0.3)
    if shaping:                                          # uncertainty-reduction shaping (secondary, noisy)
        funcs.append(reward_uq_reduction)
        weights.append(0.1)
    rubric = vf.Rubric(funcs=funcs, weights=weights)

    return VLACorrectorEnv(client=client or RolloutClient(rollout_url), dataset=dataset, rubric=rubric,
                           max_turns=max_turns, max_frames=max_frames, include_image=include_image)
