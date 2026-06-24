"""Ingest + query backend (FastAPI, port 8010).

Receives batched telemetry, persists it via the Store, exposes query endpoints
for the dashboard, and rebroadcasts every event to live-tail WebSocket clients.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from drishti.store import Store

app = FastAPI(title="drishti ingest")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = Store(os.environ.get("DRISHTI_OBS_DB_URL"))


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
