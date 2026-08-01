"""Ingest + query backend (FastAPI, port 8010).

Receives batched telemetry, persists it via the Store, exposes query endpoints
for the dashboard, and rebroadcasts every event to live-tail WebSocket clients.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from drishti.scoring import CAP, capped_z
from drishti.store import Store

app = FastAPI(title="drishti")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = Store(os.environ.get("DRISHTI_OBS_DB_URL"))

_ROOT = Path(__file__).resolve().parents[2]   # <repo>/drishti/server/app.py -> <repo>
WEB_DIR = _ROOT / "web"
BLOB_DIR = Path(os.environ.get("DRISHTI_BLOB_DIR") or (_ROOT / "data" / "obs_blobs"))
BLOB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/blobs", StaticFiles(directory=str(BLOB_DIR)), name="blobs")
CALIB_DIR = Path(os.environ.get("DRISHTI_CALIB_DIR") or (_ROOT / "data" / "calib"))
SWEEP_WARMUP = 3   # ignore the first few frames (start transient) — matches the live Monitor


@app.get("/")
def index():
    page = WEB_DIR / "dashboard.html"
    if page.exists():
        return FileResponse(str(page), headers={"Cache-Control": "no-store"})
    return {"ok": True, "hint": "dashboard.html not built yet"}


class Hub:
    """Fan-out of ingested events to connected live-tail clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def drop(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, events: list[dict]) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json({"events": events})
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.drop(ws)


hub = Hub()


@app.post("/v1/ingest")
async def ingest(body: dict):
    events = body.get("events", []) or []
    for ev in events:
        store.apply_event(ev)
    await hub.broadcast(events)
    return {"ok": True, "n": len(events)}


@app.get("/v1/runs")
def runs():
    return {"runs": store.list_runs()}


@app.get("/v1/runs/{run_id}")
def run_detail(run_id: str):
    run = store.get_run(run_id)
    if not run:
        return {"run": None}
    return {"run": run, "frames": store.get_frames(run_id)}


@app.get("/v1/runs/{run_id}/steps")
def run_steps(run_id: str, t_from: Optional[float] = None, t_to: Optional[float] = None):
    return {"steps": store.get_steps(run_id, t_from, t_to)}


@app.get("/v1/events")
def events(run_id: Optional[str] = None, type: Optional[str] = None, limit: int = 500):
    return {"events": store.get_events(run_id, type, limit)}


@app.post("/v1/calibrate")
def calibrate_route(body: dict):
    """Fit the detectors on the selected (successful) rollouts → combined conformal threshold.
    Runs drishti/calibrate.py as a subprocess (it fits on the STORED embeddings — no policy reload,
    so it's quick) and returns the validation report."""
    run_ids = body.get("run_ids", []) or []
    if not run_ids:
        return {"ok": False, "error": "no run_ids"}
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    proc = subprocess.run(
        [sys.executable, "-m", "drishti.calibrate", "--runs", *run_ids],
        cwd=str(_ROOT), env=env, capture_output=True, text=True, timeout=900)
    rep = CALIB_DIR / "report.json"
    if proc.returncode != 0 or not rep.exists():
        return {"ok": False, "error": (proc.stderr or proc.stdout)[-2000:]}
    return {"ok": True, "report": json.loads(rep.read_text())}


@app.get("/v1/calib")
def calib_get():
    """Current calibration (validation report + per-signal normalization) for the UI, or null."""
    rep, cal = CALIB_DIR / "report.json", CALIB_DIR / "calib.json"
    if not rep.exists():
        return {"report": None}
    out = {"report": json.loads(rep.read_text())}
    if cal.exists():
        cj = json.loads(cal.read_text())
        out["norm"] = cj.get("norm")
        out["cap"] = cj.get("cap", CAP)
    return out


@app.post("/v1/fleet/run")
def fleet_run(body: dict):
    """Launch ONE live rollout (detached subprocess) with the calibrated Monitor at the given
    threshold. It streams steps/frames here over HTTP → broadcast to /v1/tail. Returns immediately."""
    if not (CALIB_DIR / "calib.json").exists():
        return {"ok": False, "error": "not calibrated yet — calibrate on the successful rollouts in Logs first"}
    args = [sys.executable, "-m", "drishti.drivers.live_fleet", "--ingest", "http://127.0.0.1:8010"]
    if body.get("threshold") is not None:
        args += ["--threshold", str(float(body["threshold"]))]
    if body.get("cube_xy"):
        args += ["--cube-xy", str(body["cube_xy"][0]), str(body["cube_xy"][1])]
    if body.get("task"):
        args += ["--task", str(body["task"])]
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    subprocess.Popen(args, cwd=str(_ROOT), env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"ok": True}


@app.get("/v1/sweep")
def sweep():
    """Per-run PEAK combined uncertainty (max-z over the stored detector scores) for every scored
    rollout, so the dashboard can sweep the threshold over the WHOLE log set and show the live
    catches / false-alarms / missed trade-off as you drag the Monitor slider — tuning against all
    rollouts at once, not just the one on screen."""
    cal = CALIB_DIR / "calib.json"
    if not cal.exists():
        return {"runs": []}
    caljson = json.loads(cal.read_text())
    norm = caljson.get("norm") or {}
    cap = float(caljson.get("cap", CAP))
    if not all(k in norm for k in ("smd", "rnd", "act")):
        return {"runs": []}

    def cz(x, k):   # the shared rule, so this can never drift from calibrate's
        return float(capped_z(x, norm[k][0], norm[k][1], cap))

    out = []
    for r in store.list_runs(limit=2000):
        peak = None
        for s in store.get_steps(r["run_id"]):
            if (s.get("step_idx") or 0) < SWEEP_WARMUP or s.get("smd") is None:
                continue
            c = (cz(s["smd"], "smd") + cz(s.get("rnd") or 0.0, "rnd") + cz(s.get("ace") or 0.0, "act")) / 3.0
            peak = c if peak is None else max(peak, c)
        if peak is None:
            continue
        m = r.get("meta") or {}
        out.append({
            "run_id": r["run_id"],
            "success": (r.get("success_rate") or 0.0) >= 0.5,
            "live": bool(m.get("live")),
            "max_combined": max(0.0, float(peak)),   # floor at 0 — matches the UI's combined chart
        })
    return {"runs": out}


@app.get("/health")
def health():
    return {"ok": True, "clients": len(hub.clients)}


@app.websocket("/v1/tail")
async def tail(ws: WebSocket):
    await hub.connect(ws)
    try:
        while True:
            # Clients only listen; this keeps the socket open until they leave.
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.drop(ws)
    except Exception:
        hub.drop(ws)
