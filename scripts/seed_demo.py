"""Seed a synthetic run so the dashboard can be validated without the model.

Writes to data/obs.db (what ingest serves by default) + fake frames under
data/obs_blobs/<run_id>/. The uncertainty curves rise into a 'failure' near the
end so the traces read like a real degrading rollout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from drishti import start_run
from drishti.store import Store

BLOB = Path("data/obs_blobs")


def _frame(path: Path, i: int, n: int) -> None:
    img = np.zeros((180, 240, 3), np.uint8)
    img[:] = (12, 16, 22)
    x = int(10 + (240 - 50) * i / max(n - 1, 1))
    img[70:130, x:x + 40] = (90, 200, 160)
    Image.fromarray(img).save(path)


def main() -> None:
    s = Store("sqlite:///data/obs.db")
    run = start_run("smolvla_base (seed)", "pick up the cube", store=s, backend="in_process")
    n = 40
    d = BLOB / run.run_id
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        with run.step(step_idx=i, t=float(i)) as st:
            st.log_observation(rng.normal(size=8).tolist())
            st.log_action((rng.normal(size=7) * 0.1).tolist())
            spike = max(0, (i - 28)) / 10.0  # degrade near the end
            st.log_uncertainty(
                smd=0.4 + 0.3 * rng.random() + spike,
                rnd=0.2 + 0.2 * rng.random() + 0.8 * spike,
                ace=1.0 + 0.5 * rng.random() + 1.5 * spike,
            )
            st.internals["vision_embed_norm"] = 8.0 + rng.random()
            if i % 2 == 0:
                rel = f"{run.run_id}/0_{i}_exterior.jpg"
                _frame(BLOB / rel, i, n)
                run.log_frame(camera="exterior", blob_path=rel, step_idx=i, t=float(i))
    run.finish(success_rate=0.0)
    print("seeded run", run.run_id)


if __name__ == "__main__":
    main()
