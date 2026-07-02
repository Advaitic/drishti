"""Stage-0 plumbing test: the rollout logic drives StubVLA through spike -> correct -> success.

CPU only, ace-only uncertainty, no MolmoAct2 / no calibration artifacts. The episode state machine
lives in `EpisodeManager` (decoupled from HTTP) and is driven directly here; a light TestClient check
covers the FastAPI wiring. (On macOS, rendering inside the async loop stalls across successive requests
— a Linux+EGL pod is unaffected — so the multi-step flow is tested via EpisodeManager, not over HTTP.)
"""

from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from drishti.sim.rollout_server import EpisodeManager, AceOnlyScorer, create_app
from drishti.sim.stub_vla import StubVLA

REPORT_KEYS = ("episode_id", "status", "frame", "phase", "geometry", "uncertainty",
               "uncertainty_trace", "frames_b64", "failure_report", "vla_reasoning", "reward")


def _mgr():
    return EpisodeManager(vla_factory=lambda env: StubVLA(env, seed=0),
                          scorer=AceOnlyScorer(threshold=2.0), ace_n=6)


def test_spike_then_correct_then_success():
    m = _mgr()
    eid = m.start(seed=1, task="pick up the cube and place it in the box")["episode_id"]

    # 1) uncorrected -> StubVLA flails -> ACE spikes -> status "spike" with a well-formed report
    rep = m.run(eid, max_frames=40)
    assert rep["status"] == "spike"
    assert rep["success"] is False and rep["done"] is False
    for k in REPORT_KEYS:
        assert k in rep, f"missing report key {k}"
    assert rep["uncertainty"]["combined"] > rep["uncertainty"]["threshold"]
    assert rep["uncertainty"]["dominant"] == "ace"
    assert rep["failure_report"].startswith("[UNCERTAINTY SPIKE")
    img = Image.open(io.BytesIO(base64.b64decode(rep["frames_b64"]["up"])))
    assert img.size == (320, 240)

    # 2) inject a corrective prompt -> intervention count bumps, task changes
    cor = m.correct(eid, "move right and lower onto the cube, then close")
    assert cor["accepted"] and "lower onto the cube" in cor["task"]

    # 3) corrected -> StubVLA solves -> cube reaches the tray -> success, reward 1.0
    rep2 = m.run(eid, max_frames=40)
    assert rep2["status"] == "success"
    assert rep2["success"] is True and rep2["done"] is True and rep2["reward"] == 1.0
    assert rep2["n_interventions"] == 1

    m.end(eid)


def test_baseline_no_correction_fails():
    """Without a correction the flailing stub never places the cube -> not a success."""
    m = _mgr()
    eid = m.start(seed=2)["episode_id"]
    last = None
    for _ in range(6):
        last = m.run(eid, max_frames=20)
        if last["done"]:
            break
    assert last["success"] is False
    assert last["status"] in ("spike", "timeout")


def test_http_wiring():
    """FastAPI wiring: healthz + start + one run (a single request cycle is safe on macOS)."""
    c = TestClient(create_app(vla_factory=lambda env: StubVLA(env, seed=0),
                              scorer=AceOnlyScorer(threshold=2.0)))
    h = c.get("/healthz").json()
    assert h["ok"] and h["scorer"] == "AceOnlyScorer" and h["threshold"] == 2.0
    ep = c.post("/episode/start", json={"seed": 1}).json()
    assert ep["episode_id"].startswith("ep_")
    rep = c.post("/episode/run", json={"episode_id": ep["episode_id"], "max_frames": 20}).json()
    assert rep["status"] == "spike" and rep["uncertainty"]["dominant"] == "ace"
    assert c.post("/episode/run", json={"episode_id": "nope"}).status_code == 404
