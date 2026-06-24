"""The customer-facing SDK surface.

    run = drishti.start_run("smolvla", "pick book", ingest_url="http://localhost:8010")
    policy = drishti.instrument(policy)            # registers forward hooks
    for t, obs in enumerate(observations):
        with run.step(step_idx=t) as s:
            action = policy.select_action(obs)     # hooks fill s.internals
            s.log_observation(state).log_action(action)
            s.log_uncertainty(smd=..., rnd=..., ace=...)
    run.finish(success_rate=...)

`run.step()` is the unit a forward hook attributes to: it sets a thread-local
"current step" so a hook firing anywhere in the model knows which tick to tag.
The step context manager returns in microseconds; all I/O is on the emitter
thread.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from typing import Any, Optional

from . import schema
from .emit import Emitter

_CTX = threading.local()


def current_step() -> Optional["Step"]:
    """The Step being recorded on this thread, or None. Used by forward hooks."""
    return getattr(_CTX, "step", None)


class Step:
    def __init__(self, run: "Run", ep_idx: int, step_idx: int, t: Optional[float]):
        self.run = run
        self.ep_idx = ep_idx
        self.step_idx = step_idx
        self.t = t if t is not None else float(step_idx)
        self._start = time.time()
        self.fields: dict[str, Any] = {}      # user-logged: observation_state, action, smd, rnd, ace
        self.internals: dict[str, Any] = {}   # hook-written: vision_embed_*, action_head_*

    def log_observation(self, state: Any) -> "Step":
        self.fields["observation_state"] = state
        return self

    def log_action(self, action: Any) -> "Step":
        self.fields["action"] = action
        return self

    def log_uncertainty(self, *, smd: Any = None, rnd: Any = None, ace: Any = None) -> "Step":
        if smd is not None:
            self.fields["smd"] = smd
        if rnd is not None:
            self.fields["rnd"] = rnd
        if ace is not None:
            self.fields["ace"] = ace
        return self

    def _finish(self) -> None:
        latency = (time.time() - self._start) * 1000.0
        payload = {**self.internals, **self.fields}
        self.run.emitter.emit(schema.step(
            self.run.run_id, ep_idx=self.ep_idx, step_idx=self.step_idx,
            t=self.t, latency_ms=latency, **payload))


class Run:
    def __init__(self, run_id: str, emitter: Emitter, model_id: str = "", task: str = ""):
        self.run_id = run_id
        self.emitter = emitter
        self.model_id = model_id
        self.task = task
        self._n = 0

    @contextlib.contextmanager
    def step(self, ep_idx: int = 0, step_idx: Optional[int] = None, t: Optional[float] = None):
        if step_idx is None:
            step_idx = self._n
        s = Step(self, ep_idx, step_idx, t)
        _CTX.step = s
        try:
            yield s
        finally:
            _CTX.step = None
            s._finish()
            self._n += 1

    def log_frame(self, *, camera: str, blob_path: str, ep_idx: int = 0,
                  step_idx: Optional[int] = None, t: Optional[float] = None) -> None:
        self.emitter.emit(schema.frame(
            self.run_id, ep_idx=ep_idx,
            step_idx=step_idx if step_idx is not None else self._n,
            t=t, camera=camera, blob_path=blob_path))

    def finish(self, *, success_rate: Optional[float] = None, n_episodes: int = 1) -> None:
        self.emitter.emit(schema.run_end(
            self.run_id, n_steps=self._n, n_episodes=n_episodes, success_rate=success_rate))
        self.emitter.close()


def start_run(
    model_id: str,
    task: str,
    *,
    backend: str = "in_process",
    capture_mode: str = "summary",
    ingest_url: Optional[str] = None,
    store: Any = None,
    run_id: Optional[str] = None,
    meta: Optional[dict] = None,
) -> Run:
    run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
    emitter = Emitter(ingest_url=ingest_url, store=store)
    emitter.emit(schema.run_start(
        run_id, model_id=model_id, task=task, backend=backend,
        capture_mode=capture_mode, meta=meta))
    return Run(run_id, emitter, model_id=model_id, task=task)
