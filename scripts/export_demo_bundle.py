"""Export a small, committable slice of a telemetry store as the bundled demo dataset.

The dashboard, the calibration and every published number here rest on recorded rollouts, and the
full store is ~100 MB of JPEGs — too big to ship, and without it a fresh clone shows an empty
dashboard and cannot reproduce anything. So this carves out a bundle that fits in the repo and
still supports the real workflow end to end:

  * **every run's vision embeddings**, in float16. These are what SMD/RND actually score, so keeping
    all of them is what makes `drishti calibrate --auto` reproduce the committed report exactly.
    float16 halves the size; calibrate casts back to float32 on load, and the quantization does not
    move the reported AUC (checked, see docs/results/calibration/).
  * **camera frames for a few representative rollouts only**, subsampled. Frames are just for
    looking at — the filmstrip in the dashboard — so a handful of rollouts is enough to show what
    the thing does, and they are 90% of the bytes.
  * **the telemetry rows** (runs / steps / frames), minus the raw `events` table, which is a
    write-ahead duplicate of the other three and doubles the file for nothing.

    PYTHONPATH=. .venv/bin/python -m scripts.export_demo_bundle --out data/demo

Then anyone can point the tools at the bundle:

    DRISHTI_OBS_DB_URL=sqlite:///data/demo/obs.db DRISHTI_BLOB_DIR=data/demo/blobs drishti calibrate --auto
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np

# Rollouts that keep their camera frames: enough to show a clean success, a clear failure, and the
# in-between. Chosen by outcome + length rather than by hand-picking, so re-running is deterministic.
N_FRAME_RUNS = 3          # per outcome (success / failure)
FRAME_STRIDE = 3          # keep every Nth logged frame of those runs


def pick_frame_runs(rows, n=N_FRAME_RUNS):
    """The longest n successes and n failures — the most legible rollouts to page through."""
    succ = sorted([r for r in rows if r["fail"] == 0], key=lambda r: -r["n_steps"])[:n]
    fail = sorted([r for r in rows if r["fail"] == 1], key=lambda r: -r["n_steps"])[:n]
    return {r["run_id"] for r in succ + fail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/obs.db")
    ap.add_argument("--blobs", default="data/obs_blobs")
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--calib", default="data/calib", help="calibration bundle to copy alongside")
    ap.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    args = ap.parse_args()

    src_db, src_blobs = Path(args.db), Path(args.blobs)
    out = Path(args.out)
    out_blobs = out / "blobs"
    if out.exists():
        shutil.rmtree(out)
    out_blobs.mkdir(parents=True)

    # --- telemetry rows ---
    shutil.copy(src_db, out / "obs.db")
    db = sqlite3.connect(out / "obs.db")
    db.execute("DELETE FROM events")           # write-ahead duplicate of runs/steps/frames
    db.commit()

    db.row_factory = sqlite3.Row
    rows = [{"run_id": r["run_id"], "n_steps": r["n_steps"] or 0,
             "fail": int((r["success_rate"] or 0.0) < 0.5)}
            for r in db.execute("SELECT run_id, n_steps, success_rate FROM runs")]
    keep_frames = pick_frame_runs(rows)

    # --- blobs ---
    n_emb = n_jpg = 0
    for r in rows:
        rid = r["run_id"]
        src = src_blobs / rid
        if not src.exists():
            continue
        dst = out_blobs / rid
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("embed_*.npy")):
            np.save(dst / f.name, np.load(f).astype(np.float16))
            n_emb += 1
        if rid in keep_frames:
            for i, f in enumerate(sorted(src.glob("*.jpg"))):
                if i % args.frame_stride:
                    continue
                shutil.copy(f, dst / f.name)
                n_jpg += 1

    # Drop frame rows whose blob did not come along, so the dashboard never shows a broken image.
    kept = {str(p.relative_to(out_blobs)) for p in out_blobs.rglob("*.jpg")}
    dropped = 0
    for fid, path in list(db.execute("SELECT id, blob_path FROM frames")):
        if path not in kept:
            db.execute("DELETE FROM frames WHERE id = ?", (fid,))
            dropped += 1
    db.commit()
    db.execute("VACUUM")
    db.close()

    # --- calibration bundle, so the dashboard renders scored traces out of the box ---
    calib_src = Path(args.calib)
    if calib_src.exists():
        calib_dst = out / "calib"
        calib_dst.mkdir(exist_ok=True)
        for name in ("calib.json", "report.json", "smd.npz", "rnd_innorm.npz",
                     "rnd_predictor.safetensors", "rnd_target.safetensors"):
            if (calib_src / name).exists():
                shutil.copy(calib_src / name, calib_dst / name)

    (out / "MANIFEST.json").write_text(json.dumps({
        "source_db": str(src_db),
        "n_runs": len(rows),
        "n_success": sum(1 for r in rows if not r["fail"]),
        "n_failure": sum(1 for r in rows if r["fail"]),
        "runs_with_frames": sorted(keep_frames),
        "frame_stride": args.frame_stride,
        "n_embeddings": n_emb,
        "n_frames": n_jpg,
        "embedding_dtype": "float16",
        "frame_rows_dropped": dropped,
    }, indent=2) + "\n")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{len(rows)} runs · {n_emb} embeddings · {n_jpg} frames · {size/1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
