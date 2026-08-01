"""advaitic platform — observability + uncertainty SDK for robot policies."""

from __future__ import annotations

from . import schema
from .sdk import Run, Step, current_step, start_run
from .store import Store

__all__ = ["start_run", "Run", "Step", "current_step", "schema", "Store",
           "instrument", "uninstrument"]


def __getattr__(name: str):
    # Lazy so `import drishti` doesn't pull torch/lerobot for the telemetry path.
    if name in ("instrument", "uninstrument"):
        from . import hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'drishti' has no attribute {name!r}")
