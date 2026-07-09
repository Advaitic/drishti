"""Test #1 — DOES A GOOD CORRECTION ACTUALLY HELP?

For each cube position, compare the VLA baseline (no correction, run to completion) against injecting a
hand-crafted correction at the first uncertainty spike. This isolates the two things we've been conflating:

  a hand-correction LIFTS success  -> correction *works*; our RL just never learned a good one.
  a hand-correction does NOTHING   -> the failure isn't correction-recoverable (text or otherwise) ->
                                      no reward/curriculum/RL would have helped; the VLA is the ceiling.

Needs the calibrated rollout server (VLA + UQ) for the spike. No training. 1 GPU.

  DRISHTI_LEROBOT_CKPT=~/ft_ckpt UNCERTAINTY=calib MUJOCO_GL=egl PYTHONPATH=. \
    <lerobot-venv>/bin/python -m drishti.drivers.correction_ab --positions 8 --trials 3
"""
from __future__ import annotations

import argparse
import numpy as np

# hand-crafted corrections aimed at the observed failure mode (gripper stays open / doesn't secure the cube),
# plus one weak/neutral control. If NONE of these beat baseline, correction is not the lever.
CORRECTIONS = [
    "lower the gripper straight down onto the cube and close it to grasp",
    "close the gripper now to grasp the cube",
    "move down and forward until the gripper surrounds the cube, then close",
    "keep going",                                             # neutral control
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--trials", type=int, default=3, help="rollouts per (position, condition) to average VLA noise")
    ap.add_argument("--max-frames", type=int, default=110)
    args = ap.parse_args()

    from drishti.sim.rollout_server import build_default_app
    mgr = build_default_app().state.manager

    rng = np.random.default_rng(0)
    cubes = [(round(float(rng.uniform(0.24, 0.28)), 4), round(float(rng.uniform(-0.03, 0.03)), 4))
             for _ in range(args.positions)]

    def baseline(cxy, seed):
        r = mgr.start(seed=seed, cube_xy=list(cxy))
        rep = mgr.run(r["episode_id"], max_frames=args.max_frames, stop_on_spike=False)
        mgr.end(r["episode_id"]); return int(rep["success"])

    def corrected(cxy, seed, correction):
        r = mgr.start(seed=seed, cube_xy=list(cxy))
        rep = mgr.run(r["episode_id"], max_frames=args.max_frames, stop_on_spike=True)   # to first spike
        if rep["status"] == "spike":
            mgr.correct(r["episode_id"], correction)
            rep = mgr.run(r["episode_id"], max_frames=args.max_frames, stop_on_spike=False)  # resume
        mgr.end(r["episode_id"]); return int(rep["success"])

    tot = args.positions * args.trials
    base = 0
    per = {c: 0 for c in CORRECTIONS}
    for ci, cxy in enumerate(cubes):
        for t in range(args.trials):
            seed = 1000 * ci + t
            base += baseline(cxy, seed)
            for c in CORRECTIONS:
                per[c] += corrected(cxy, seed, c)
        print(f"  [{ci+1}/{args.positions}] {cxy} done", flush=True)

    print("\n=== TEST #1 — does a good correction help? ===")
    b = base / tot
    print(f"baseline (no correction): {base}/{tot} = {b:.2f}")
    for c in CORRECTIONS:
        s = per[c] / tot
        print(f"  + '{c[:44]:44}' : {per[c]}/{tot} = {s:.2f}   (Δ vs baseline {s - b:+.2f})")
    best = max(per.values()) / tot
    print(f"\nBest correction Δ = {best - b:+.2f}. "
          + ("A correction meaningfully helps -> the lever is real; our RL/reward is the gap."
             if best - b > 0.1 else
             "No correction beats baseline -> the failure is NOT correction-recoverable; the VLA is the ceiling."))


if __name__ == "__main__":
    main()
