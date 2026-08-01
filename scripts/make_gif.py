"""Export animated GIFs from recorded scripted demos (data/demos/ep_*.npz).

Whole episode:
    PYTHONPATH=. .venv/bin/python scripts/make_gif.py --episode 0 --cams up wrist

A frame-range segment (e.g. one phase/macro-phase):
    PYTHONPATH=. .venv/bin/python scripts/make_gif.py --episode 0 --cams up wrist \
        --start 0 --end 27 --tag seg1_reach-grasp

Writes one GIF per camera to --out (default data/gifs/).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_gif(frames: np.ndarray, out_path: Path, fps: int = 10) -> None:
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        out_path, save_all=True, append_images=imgs[1:],
        duration=int(1000 / fps), loop=0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True, help="episode index (data/demos/ep_NNN.npz)")
    ap.add_argument("--demos-dir", type=str, default="data/demos")
    ap.add_argument("--cams", nargs="+", default=["up", "wrist"], choices=["up", "side", "wrist"])
    ap.add_argument("--start", type=int, default=None, help="first frame index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last frame index (inclusive)")
    ap.add_argument("--tag", type=str, default=None, help="suffix for output filenames, e.g. a phase name")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", type=str, default="data/gifs")
    args = ap.parse_args()

    npz_path = Path(args.demos_dir) / f"ep_{args.episode:03d}.npz"
    d = np.load(npz_path)
    n = d["up"].shape[0]
    start = 0 if args.start is None else args.start
    end = n - 1 if args.end is None else args.end
    assert 0 <= start <= end < n, f"bad range [{start},{end}] for {n} frames"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    for cam in args.cams:
        frames = d[cam][start:end + 1]
        out_path = out_dir / f"ep{args.episode:03d}_{cam}{tag}.gif"
        make_gif(frames, out_path, fps=args.fps)
        print(f"wrote {out_path} ({len(frames)} frames, {frames.shape[1]}x{frames.shape[2]})")


if __name__ == "__main__":
    main()
