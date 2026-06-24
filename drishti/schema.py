"""Telemetry event schema.

Events are JSON-serializable dicts shaped like:
    {"type": "<kind>", "ts": <wall_seconds_float>, "payload": {...}}

Kinds and payloads:
    run_start : {run_id, model_id, task, backend, capture_mode, started_at, meta}
    run_end   : {run_id, n_steps, n_episodes, duration_sec, success_rate, ended_at}
    step      : {run_id, ep_idx, step_idx, t,
                 observation_state, action, latency_ms, actor,
                 vision_embed_norm, vision_embed_std,       # from forward hooks
                 action_head_mean, action_head_std,         # from forward hooks
                 smd, rnd, ace}                             # uncertainty scores
    frame     : {run_id, ep_idx, step_idx, t, camera, blob_path}

The `step` event folds the per-tick observation, action, model internals, and
uncertainty onto one record so the dashboard fetches a run's whole time series
in one query. Heavy payloads (frames, raw tensors) go to a blob dir and are
referenced by path via `frame` events.
"""

from __future__ import annotations

import time
from typing import Any, Optional


def _vec(x: Any) -> Optional[list[float]]:
    """Coerce a scalar / list / numpy array to a plain list[float] (or None)."""
    if x is None:
        return None
    try:
        return [float(v) for v in x]
    except TypeError:
        return [float(x)]


def _f(x: Any) -> Optional[float]:
    return None if x is None else float(x)


def _ev(type_: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": type_, "ts": time.time(), "payload": payload}


def run_start(
    run_id: str,
    *,
    model_id: str,
    task: str,
    backend: str = "in_process",
    capture_mode: str = "summary",
    meta: Optional[dict] = None,
) -> dict[str, Any]:
    return _ev(
        "run_start",
        {
            "run_id": run_id,
            "model_id": model_id,
            "task": task,
            "backend": backend,
            "capture_mode": capture_mode,
            "started_at": time.time(),
            "meta": meta or {},
        },
    )


def run_end(
    run_id: str,
    *,
    n_steps: int,
    n_episodes: int = 1,
    duration_sec: Optional[float] = None,
    success_rate: Optional[float] = None,
) -> dict[str, Any]:
    return _ev(
        "run_end",
        {
            "run_id": run_id,
            "n_steps": int(n_steps),
            "n_episodes": int(n_episodes),
            "duration_sec": _f(duration_sec),
            "success_rate": _f(success_rate),
            "ended_at": time.time(),
        },
    )


def step(
    run_id: str,
    *,
    ep_idx: int,
    step_idx: int,
    t: Optional[float],
    observation_state: Any = None,
    action: Any = None,
    latency_ms: Optional[float] = None,
    actor: int = 0,
    vision_embed_norm: Optional[float] = None,
    vision_embed_std: Optional[float] = None,
    action_head_mean: Any = None,
    action_head_std: Any = None,
    smd: Optional[float] = None,
    rnd: Optional[float] = None,
    ace: Optional[float] = None,
) -> dict[str, Any]:
    return _ev(
        "step",
        {
            "run_id": run_id,
            "ep_idx": int(ep_idx),
            "step_idx": int(step_idx),
            "t": _f(t),
            "observation_state": _vec(observation_state),
            "action": _vec(action),
            "latency_ms": _f(latency_ms),
            "actor": int(actor),
            "vision_embed_norm": _f(vision_embed_norm),
            "vision_embed_std": _f(vision_embed_std),
            "action_head_mean": _vec(action_head_mean),
            "action_head_std": _vec(action_head_std),
            "smd": _f(smd),
            "rnd": _f(rnd),
            "ace": _f(ace),
        },
    )


def frame(
    run_id: str,
    *,
    ep_idx: int,
    step_idx: int,
    t: Optional[float],
    camera: str,
    blob_path: str,
) -> dict[str, Any]:
    return _ev(
        "frame",
        {
            "run_id": run_id,
            "ep_idx": int(ep_idx),
            "step_idx": int(step_idx),
            "t": _f(t),
            "camera": camera,
            "blob_path": blob_path,
        },
    )
