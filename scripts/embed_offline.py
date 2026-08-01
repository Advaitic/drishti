"""Re-embed the raw frames saved by ingest_aha (saved by default; suppressed with --no-frames) with any encoder, offline.

Reads  data/obs_blobs/<run>/<cam>_<f>.jpg  (saved at the embed-stride steps) and writes
       data/obs_blobs/<run>/embed_<f>_<encoder>.npy
so an encoder ablation (SigLIP vs DINOv2-cls vs DINOv2-mean vs ...) is a downstream
choice — never a re-generation. GPU-fast when CUDA is present (run it on the pod right
after ingest for the whole encoder sweep), CPU-fine otherwise.

    PYTHONPATH=. python scripts/embed_offline.py --encoder dinov2-mean --cams front wrist
    PYTHONPATH=. python scripts/embed_offline.py --encoder siglip                # all encoders one by one
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from drishti.uq.embed_frames import ENCODERS, load_embedder

BLOB_DIR = Path("data/obs_blobs")


def frame_steps(run_dir: Path, cams):
    """Step indices where EVERY camera has a saved frame — the steps we can embed."""
    steps = None
    for c in cams:
        s = {int(p.stem.split("_")[-1]) for p in run_dir.glob(f"{c}_*.jpg")}
        steps = s if steps is None else (steps & s)
    return sorted(steps or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    ap.add_argument("--cams", nargs="+", default=["front", "wrist"])
    ap.add_argument("--blob-dir", default=str(BLOB_DIR))
    ap.add_argument("--runs-prefix", default="run_aha_", help="only re-embed run dirs with this prefix")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from PIL import Image
    root = Path(args.blob_dir)
    run_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith(args.runs_prefix))
    if not run_dirs:
        raise SystemExit(f"no run dirs under {root} with prefix {args.runs_prefix!r}")
    embed = load_embedder(args.encoder)

    n_runs = n_frames = n_skip = 0
    for rd in run_dirs:
        steps = frame_steps(rd, args.cams)
        if not steps:
            continue
        n_runs += 1
        for t in steps:
            out = rd / f"embed_{t:03d}_{args.encoder}.npy"
            if out.exists() and not args.overwrite:
                n_skip += 1
                continue
            imgs = [np.asarray(Image.open(rd / f"{c}_{t:03d}.jpg").convert("RGB"), dtype=np.uint8)
                    for c in args.cams]
            np.save(out, embed(*imgs).astype(np.float32))
            n_frames += 1
        print(f"{rd.name}: {len(steps)} frames -> {args.encoder}", flush=True)

    print(f"\n{args.encoder}: embedded {n_frames} frames across {n_runs} runs "
          f"({n_skip} already present)", flush=True)


if __name__ == "__main__":
    main()
