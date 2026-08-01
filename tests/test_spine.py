"""Stage-A spine tests: store roundtrip, SDK→emitter→store, ingest endpoints.

No model, no data — proves the telemetry plumbing end to end.
"""

import importlib
import os
import tempfile

from drishti import schema, start_run
from drishti.store import Store


def _store() -> Store:
    d = tempfile.mkdtemp()
    return Store(f"sqlite:///{d}/t.db")


def test_store_roundtrip():
    s = _store()
    rid = "run_test"
    s.apply_event(schema.run_start(rid, model_id="smolvla", task="pick"))
    for i in range(3):
        s.apply_event(schema.step(
            rid, ep_idx=0, step_idx=i, t=float(i),
            observation_state=[0.1 * i] * 8, action=[0.0] * 7, latency_ms=12.0,
            vision_embed_norm=1.0 + i, smd=0.5 * i, rnd=0.1 * i, ace=0.2 * i))
    s.apply_event(schema.run_end(rid, n_steps=3, success_rate=1.0))

    runs = s.list_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "done"
    assert runs[0]["n_steps"] == 3

    steps = s.get_steps(rid)
    assert [st["step_idx"] for st in steps] == [0, 1, 2]
    assert steps[1]["smd"] is not None
    assert steps[2]["vision_embed_norm"] == 3.0


def test_sdk_emitter_to_store():
    s = _store()
    run = start_run("smolvla", "pick", store=s)
    for i in range(4):
        with run.step(ep_idx=0, step_idx=i, t=float(i)) as st:
            st.log_observation([float(i)] * 8).log_action([0.0] * 7)
            st.log_uncertainty(smd=float(i), rnd=0.1, ace=0.2)
            st.internals["vision_embed_norm"] = 2.0  # what a forward hook would write
    run.finish(success_rate=0.75)  # close() flushes + joins the emitter thread

    runs = s.list_runs()
    assert len(runs) == 1 and runs[0]["n_steps"] == 4
    steps = s.get_steps(runs[0]["run_id"])
    assert len(steps) == 4
    assert steps[0]["vision_embed_norm"] == 2.0
    assert steps[3]["smd"] == 3.0


def test_ingest_endpoints():
    from fastapi.testclient import TestClient

    d = tempfile.mkdtemp()
    os.environ["DRISHTI_OBS_DB_URL"] = f"sqlite:///{d}/i.db"
    import drishti.server.app as A
    importlib.reload(A)
    c = TestClient(A.app)

    rid = "run_ing"
    evs = [schema.run_start(rid, model_id="smolvla", task="pick")]
    for i in range(3):
        evs.append(schema.step(rid, ep_idx=0, step_idx=i, t=float(i), action=[0.0] * 7, smd=float(i)))
    evs.append(schema.run_end(rid, n_steps=3))

    r = c.post("/v1/ingest", json={"events": evs})
    assert r.json()["n"] == len(evs)
    assert len(c.get("/v1/runs").json()["runs"]) == 1
    steps = c.get(f"/v1/runs/{rid}/steps").json()["steps"]
    assert len(steps) == 3 and steps[2]["smd"] == 2.0
