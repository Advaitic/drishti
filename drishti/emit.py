"""Batched async telemetry emitter.

A daemon thread drains a bounded queue and either writes events straight to a
local Store (in-process dev) or POSTs them in batches to the ingest server.
`emit()` is non-blocking and drops on a full queue, so it never stalls the
policy control loop (the whole point — xVLA-class inference is ~hundreds of ms
and we must not add a synchronous network hop per tick).
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request
from typing import Any, Optional


class Emitter:
    def __init__(
        self,
        ingest_url: Optional[str] = None,
        store: Any = None,
        batch: int = 64,
        flush_interval: float = 0.15,
        maxq: int = 20000,
    ):
        self.ingest_url = ingest_url.rstrip("/") if ingest_url else None
        self.store = store
        self.batch = batch
        self.flush_interval = flush_interval
        self.q: queue.Queue = queue.Queue(maxsize=maxq)
        self.dropped = 0
        self.sent = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="drishti-emit", daemon=True)
        self._t.start()

    def emit(self, ev: dict[str, Any]) -> None:
        try:
            self.q.put_nowait(ev)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._drain()
            if batch:
                self._send(batch)
        batch = self._drain()
        while batch:
            self._send(batch)
            batch = self._drain()

    def _drain(self) -> list[dict]:
        out: list[dict] = []
        try:
            out.append(self.q.get(timeout=self.flush_interval))
        except queue.Empty:
            return out
        while len(out) < self.batch:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out

    def _send(self, batch: list[dict]) -> None:
        if self.store is not None:
            for ev in batch:
                try:
                    self.store.apply_event(ev)
                except Exception:
                    self.dropped += 1
            self.sent += len(batch)
            return
        if self.ingest_url:
            data = json.dumps({"events": batch}).encode()
            req = urllib.request.Request(
                self.ingest_url + "/v1/ingest", data=data,
                headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=5).read()
                self.sent += len(batch)
            except Exception:
                self.dropped += len(batch)

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._t.join(timeout=timeout)
