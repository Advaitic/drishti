"""Telemetry store: SQLAlchemy core over SQLite, Postgres-ready.

Single thin layer so the dashboard queries stay identical when we swap the
`DRISHTI_OBS_DB_URL` from sqlite to postgres later. JSON columns hold the small
vectors (state, action, per-dim stats); frames/raw tensors live in a blob dir
and are referenced by path through the `frames` table.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import sqlalchemy as sa

DEFAULT_URL = "sqlite:///data/obs.db"

_STEP_KEYS = [
    "run_id", "ep_idx", "step_idx", "t", "observation_state", "action",
    "latency_ms", "actor", "vision_embed_norm", "vision_embed_std",
    "action_head_mean", "action_head_std", "smd", "rnd", "ace",
]
_FRAME_KEYS = ["run_id", "ep_idx", "step_idx", "t", "camera", "blob_path"]


class Store:
    def __init__(self, url: Optional[str] = None):
        url = url or os.environ.get("DRISHTI_OBS_DB_URL") or DEFAULT_URL
        if url.startswith("sqlite:///"):
            Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the async emitter thread and the request
        # thread both touch this engine; sqlite serializes writes anyway.
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = sa.create_engine(url, future=True, connect_args=connect_args)
        self.md = sa.MetaData()
        self._define()
        self.md.create_all(self.engine)

    def _define(self) -> None:
        self.runs = sa.Table(
            "runs", self.md,
            sa.Column("run_id", sa.Text, primary_key=True),
            sa.Column("model_id", sa.Text),
            sa.Column("task", sa.Text),
            sa.Column("backend", sa.Text),
            sa.Column("capture_mode", sa.Text),
            sa.Column("status", sa.Text),
            sa.Column("started_at", sa.Float),
            sa.Column("ended_at", sa.Float),
            sa.Column("n_steps", sa.Integer),
            sa.Column("n_episodes", sa.Integer),
            sa.Column("success_rate", sa.Float),
            sa.Column("meta", sa.JSON),
        )
        self.steps = sa.Table(
            "steps", self.md,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.Text, index=True),
            sa.Column("ep_idx", sa.Integer),
            sa.Column("step_idx", sa.Integer),
            sa.Column("t", sa.Float),
            sa.Column("observation_state", sa.JSON),
            sa.Column("action", sa.JSON),
            sa.Column("latency_ms", sa.Float),
            sa.Column("actor", sa.Integer),
            sa.Column("vision_embed_norm", sa.Float),
            sa.Column("vision_embed_std", sa.Float),
            sa.Column("action_head_mean", sa.JSON),
            sa.Column("action_head_std", sa.JSON),
            sa.Column("smd", sa.Float),
            sa.Column("rnd", sa.Float),
            sa.Column("ace", sa.Float),
        )
        sa.Index("ix_steps_order", self.steps.c.run_id, self.steps.c.ep_idx, self.steps.c.step_idx)
        self.frames = sa.Table(
            "frames", self.md,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.Text, index=True),
            sa.Column("ep_idx", sa.Integer),
            sa.Column("step_idx", sa.Integer),
            sa.Column("t", sa.Float),
            sa.Column("camera", sa.Text),
            sa.Column("blob_path", sa.Text),
        )
        self.events = sa.Table(
            "events", self.md,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.Text, index=True),
            sa.Column("type", sa.Text),
            sa.Column("ts", sa.Float),
            sa.Column("payload", sa.JSON),
        )

    # ---- writes ----
    def apply_event(self, ev: dict[str, Any]) -> None:
        t = ev.get("type")
        p = ev.get("payload", {}) or {}
        with self.engine.begin() as cx:
            cx.execute(sa.insert(self.events).values(
                run_id=p.get("run_id"), type=t, ts=ev.get("ts"), payload=p))
            if t == "run_start":
                self._run_start(cx, p)
            elif t == "run_end":
                self._run_end(cx, p)
            elif t == "step":
                cx.execute(sa.insert(self.steps).values(**{k: p.get(k) for k in _STEP_KEYS}))
            elif t == "frame":
                cx.execute(sa.insert(self.frames).values(**{k: p.get(k) for k in _FRAME_KEYS}))

    def _run_start(self, cx, p: dict) -> None:
        vals = dict(
            run_id=p["run_id"], model_id=p.get("model_id"), task=p.get("task"),
            backend=p.get("backend"), capture_mode=p.get("capture_mode"),
            status="running", started_at=p.get("started_at"),
            n_steps=0, n_episodes=p.get("n_episodes", 1), meta=p.get("meta") or {},
        )
        exists = cx.execute(
            sa.select(self.runs.c.run_id).where(self.runs.c.run_id == p["run_id"])).first()
        if exists:
            cx.execute(sa.update(self.runs).where(self.runs.c.run_id == p["run_id"]).values(**vals))
        else:
            cx.execute(sa.insert(self.runs).values(**vals))

    def _run_end(self, cx, p: dict) -> None:
        n = cx.execute(sa.select(sa.func.count()).select_from(self.steps).where(
            self.steps.c.run_id == p["run_id"])).scalar() or 0
        cx.execute(sa.update(self.runs).where(self.runs.c.run_id == p["run_id"]).values(
            status="done", ended_at=p.get("ended_at"),
            n_steps=p.get("n_steps") or n, n_episodes=p.get("n_episodes", 1),
            success_rate=p.get("success_rate")))

    # ---- reads ----
    def list_runs(self, limit: int = 200) -> list[dict]:
        with self.engine.begin() as cx:
            rows = cx.execute(sa.select(self.runs).order_by(
                self.runs.c.started_at.desc()).limit(limit)).mappings().all()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[dict]:
        with self.engine.begin() as cx:
            r = cx.execute(sa.select(self.runs).where(
                self.runs.c.run_id == run_id)).mappings().first()
        return dict(r) if r else None

    def get_steps(self, run_id: str, t_from: Optional[float] = None,
                  t_to: Optional[float] = None) -> list[dict]:
        q = sa.select(self.steps).where(self.steps.c.run_id == run_id)
        if t_from is not None:
            q = q.where(self.steps.c.t >= t_from)
        if t_to is not None:
            q = q.where(self.steps.c.t <= t_to)
        q = q.order_by(self.steps.c.ep_idx, self.steps.c.step_idx)
        with self.engine.begin() as cx:
            rows = cx.execute(q).mappings().all()
        return [dict(r) for r in rows]

    def get_frames(self, run_id: str) -> list[dict]:
        with self.engine.begin() as cx:
            rows = cx.execute(sa.select(self.frames).where(
                self.frames.c.run_id == run_id).order_by(
                self.frames.c.ep_idx, self.frames.c.step_idx)).mappings().all()
        return [dict(r) for r in rows]

    def get_events(self, run_id: Optional[str] = None, type: Optional[str] = None,
                   limit: int = 500) -> list[dict]:
        q = sa.select(self.events)
        if run_id:
            q = q.where(self.events.c.run_id == run_id)
        if type:
            q = q.where(self.events.c.type == type)
        q = q.order_by(self.events.c.id.desc()).limit(limit)
        with self.engine.begin() as cx:
            rows = cx.execute(q).mappings().all()
        return [dict(r) for r in rows]
