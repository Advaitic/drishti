"""The verifiers env drives spike -> correct -> success against a fake rollout client (no server, no GPU).

Exercises setup_state seeding the first report, env_response injecting a correction and absorbing the
terminal outcome, the `final_env_response` stop sentinel, and the three reward funcs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parents[1] / "environments" / "vla-corrector"
sys.path.insert(0, str(ENV_DIR))
import vla_corrector as vc  # noqa: E402


def _report(status, comb, *, success, done, frame):
    return {
        "episode_id": "ep_fake", "status": status, "frame": frame, "phase": "grasp",
        "task": "pick up the cube and place it in the box", "n_interventions": 0,
        "done": done, "success": success, "reward": 1.0 if success else 0.0,
        "geometry": {"gripper_tcp": [0, 0, 0], "cube_pos": [0, 0, 0], "tray_xy": [0.34, 0.02],
                     "cube_to_tray_dx": 0.0, "cube_to_tray_dy": 0.0, "gripper_to_cube": 0.0,
                     "grasping": False},
        "uncertainty": {"smd": 0.0, "rnd": 0.0, "ace": comb, "combined": comb,
                        "threshold": 0.4, "dominant": "ace", "margin": 0.0},
        "uncertainty_trace": [], "vla_reasoning": "x", "frames_b64": {"up": "", "side": ""},
        "failure_report": f"[UNCERTAINTY SPIKE @ frame {frame} | phase=grasp] fix it",
    }


class FakeClient:
    """Returns a spike on the first /run, a success on the second — i.e. the correction fixes it."""

    def __init__(self):
        self.corrections = []
        self.n_runs = 0
        self.started_with = None

    def start(self, seed, cube_xy, task):
        self.started_with = (seed, tuple(cube_xy) if cube_xy else None, task)
        return {"episode_id": "ep_fake", "task": task, "phase": "approach"}

    def run(self, episode_id, max_frames):
        self.n_runs += 1
        return (_report("spike", 0.60, success=False, done=False, frame=5) if self.n_runs == 1
                else _report("success", 0.10, success=True, done=True, frame=9))

    def correct(self, episode_id, prompt):
        self.corrections.append(prompt)
        return {"episode_id": episode_id, "accepted": True, "task": prompt}

    def end(self, episode_id):
        return {"ended": True}


def test_spike_correct_success_and_rewards():
    fake = FakeClient()
    env = vc.load_environment(n_episodes=1, client=fake, include_image=False, shaping=True)
    state = {"prompt": [{"role": "system", "content": vc.SYSTEM}],
             "info": {"seed": 3, "cube_xy": [0.26, 0.0], "task": "pick up the cube and place it in the box"}}

    asyncio.run(env.setup_state(state))
    assert state["episode_id"] == "ep_fake"
    assert fake.started_with[0] == 3
    assert state["status"] == "spike"
    assert len(state["prompt"]) == 2 and state["prompt"][-1]["role"] == "user"
    assert "UNCERTAINTY SPIKE" in state["prompt"][-1]["content"]

    messages = list(state["prompt"]) + [{"role": "assistant", "content": "lower onto the cube and close"}]
    out = asyncio.run(env.env_response(messages, state))
    assert out == []                                      # terminal turn appends nothing
    assert fake.corrections == ["lower onto the cube and close"]
    assert state["success"] is True and state["status"] == "success"
    assert state["n_interventions"] == 1
    assert state.get("final_env_response") is not None    # @vf.stop sentinel set

    assert asyncio.run(vc.reward_success(state=state)) == 1.0
    assert asyncio.run(vc.penalty_interventions(state=state)) == 1.0
    assert asyncio.run(vc.reward_uq_reduction(state=state)) == 0.5     # 0.60 -> 0.10


def test_reward_success_zero_on_failure():
    assert asyncio.run(vc.reward_success(state={"success": False})) == 0.0
    assert asyncio.run(vc.reward_success(state=None)) == 0.0


def test_load_environment_builds():
    env = vc.load_environment(n_episodes=5, client=FakeClient(), shaping=False)
    assert isinstance(env, vc.VLACorrectorEnv)
    assert len(env.dataset) == 5
