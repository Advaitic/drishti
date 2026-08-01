"""A CPU-only rollout server, so the RL loop is runnable before you have a policy.

`drishti.sim.rollout_server` speaks the HTTP contract the `vla-corrector` environment expects, but
its default factory loads a multi-GB VLA. That makes the whole RL half of this repo unrunnable for
anyone who just cloned it. This module wires the same server to `StubVLA` — a deterministic,
dependency-light stand-in that moves the arm on a fixed schedule — so the full
start -> run -> spike -> correct -> resume -> reward loop can be exercised on a laptop.

    uvicorn drishti.sim.stub_server:app --port 8100

    # then, in the environment:
    prime eval run vla-corrector -a '{"rollout_url": "http://127.0.0.1:8100", "max_turns": 2}'

What this is good for: checking the contract, the env plumbing, and your own client code. What it
is not: a policy. The stub does not learn, and its "uncertainty" is the same ACE spread a real
policy would produce but computed over made-up actions — so the rewards are meaningless as science.
Swap in a real VLA (set `DRISHTI_LEROBOT_CKPT` and use `rollout_server:build_default_app`) once the
plumbing works.

**macOS caveat.** MuJoCo rendering inside the async event loop stalls across successive HTTP
requests on macOS; a Linux box with `MUJOCO_GL=egl` is unaffected. On a Mac, drive `EpisodeManager`
directly instead of going over HTTP — that is what `tests/test_rollout_server.py` does, and it
exercises the same state machine.
"""

from __future__ import annotations

import os

from drishti.sim.rollout_server import AceOnlyScorer, create_app
from drishti.sim.stub_vla import StubVLA

# Low enough that a spike fires within the first few frames, so a smoke run finishes fast.
THRESHOLD = float(os.environ.get("ACE_THRESHOLD", "2.0"))


def build_app():
    return create_app(
        vla_factory=lambda env: StubVLA(env, seed=0),
        scorer=AceOnlyScorer(threshold=THRESHOLD),
    )


app = build_app()
