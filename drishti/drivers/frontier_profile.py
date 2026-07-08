"""Frontier profile — data-pick the GRPO curriculum band.

Sweep a grid of cube positions; at each, run the finetuned VLA **baseline (no correction)** N times and
record success rate + mean staged progress. Emit the *frontier band*: positions where the VLA succeeds
SOMETIMES (0 < rate < 1). Those are the only positions where a corrector can flip the outcome, so they're
the only ones that give GRPO non-zero within-group reward variance. Always-succeed / always-fail positions
are degenerate (uniform reward -> zero advantage) and are dropped.

Runs on the pod (needs CUDA + DRISHTI_LEROBOT_CKPT + MUJOCO_GL=egl). Reuses the real-VLA manager from
`build_default_app` and drives it directly (no HTTP). Output JSON's `curriculum` list pastes straight into
the GRPO config's `[[orchestrator.train.env]] args.curriculum`.

  DRISHTI_LEROBOT_CKPT=/ephemeral/ft_ckpt UNCERTAINTY=calib MUJOCO_GL=egl \
    <lerobot-venv>/bin/python -m drishti.drivers.frontier_profile --nx 5 --ny 4 --trials 5 --out data/frontier.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-min", type=float, default=0.215)
    ap.add_argument("--x-max", type=float, default=0.305)
    ap.add_argument("--y-min", type=float, default=-0.055)
    ap.add_argument("--y-max", type=float, default=0.055)
    ap.add_argument("--nx", type=int, default=5, help="grid points along x")
    ap.add_argument("--ny", type=int, default=4, help="grid points along y")
    ap.add_argument("--trials", type=int, default=5, help="baseline rollouts per cube position")
    ap.add_argument("--max-frames", type=int, default=110)
    ap.add_argument("--lo", type=float, default=0.20, help="frontier band lower success-rate bound")
    ap.add_argument("--hi", type=float, default=0.80, help="frontier band upper success-rate bound")
    ap.add_argument("--out", type=str, default="data/frontier.json")
    args = ap.parse_args()

    # real-VLA manager (loads the finetuned MolmoAct2 + calibrated scorer) — no HTTP, drive it directly.
    from drishti.sim.rollout_server import build_default_app
    mgr = build_default_app().state.manager

    xs = np.linspace(args.x_min, args.x_max, args.nx)
    ys = np.linspace(args.y_min, args.y_max, args.ny)
    grid = [(round(float(x), 4), round(float(y), 4)) for x in xs for y in ys]
    print(f">>> profiling {len(grid)} positions x {args.trials} baseline trials "
          f"= {len(grid) * args.trials} episodes", flush=True)

    results = []
    t0 = time.time()
    for gi, (x, y) in enumerate(grid):
        succ, progs = 0, []
        for k in range(args.trials):
            r = mgr.start(seed=1000 * gi + k, cube_xy=[x, y])
            rep = mgr.run(r["episode_id"], max_frames=args.max_frames, stop_on_spike=False)  # baseline
            succ += int(rep["success"])
            progs.append(float(rep.get("progress", 0.0)))
            mgr.end(r["episode_id"])
        rate = succ / args.trials
        results.append({"cube_xy": [x, y], "success_rate": rate,
                        "mean_progress": round(float(np.mean(progs)), 3), "n": args.trials})
        print(f"  [{gi + 1}/{len(grid)}] ({x:+.3f},{y:+.3f})  success {succ}/{args.trials}"
              f"  progress~{np.mean(progs):.2f}  ({time.time() - t0:.0f}s)", flush=True)

    frontier = [r["cube_xy"] for r in results if args.lo <= r["success_rate"] <= args.hi]
    # fallback: if the band is empty (VLA too binary), take the positions closest to 50% by mean progress
    if not frontier:
        ranked = sorted(results, key=lambda r: -r["mean_progress"] if r["success_rate"] == 0
                        else abs(r["success_rate"] - 0.5))
        frontier = [r["cube_xy"] for r in ranked[:max(3, len(results) // 4)]]
        note = "band empty; fell back to mid-progress positions"
    else:
        note = f"{len(frontier)} positions in [{args.lo},{args.hi}] success band"

    out = {"grid": results, "curriculum": frontier, "note": note,
           "params": {k: getattr(args, k) for k in ("nx", "ny", "trials", "lo", "hi")}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    always = sum(r["success_rate"] == 1 for r in results)
    never = sum(r["success_rate"] == 0 for r in results)
    print(f">>> {note}. always-succeed={always} never={never} frontier={len(frontier)}. wrote {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
