"""Targeted follow-up to steerability_probe: does the model respond to its OWN trained vocabulary?

steerability_probe's FINE/TASK pairs are generic corrective phrasing that was never in this
finetune's training data (only the 3 phase-caption sentences in drishti.sim.phase_captions
were). A near-zero delta there could mean "doesn't generalize to novel phrasing" rather than
"learned nothing about language at all." This probes the narrower, fairer question: at a frozen
state, does swapping between the model's OWN trained macro-phase captions change its action?

  MUJOCO_GL=egl PYTHONPATH=. <lerobot-venv>/bin/python -m drishti.drivers.phase_caption_probe \
    --lerobot-ckpt <hf-repo-or-path> --positions 4 --phase-steps 25 --nsamp 16
"""
from __future__ import annotations

import argparse
import itertools
import numpy as np

from drishti.sim.so101_env import SO101Env
from drishti.sim.rollout_server import N_SUBSTEPS, _render_pair
from drishti.sim.scripted_pick import _grasp_config
from drishti.sim.phase_captions import MACRO_PHASES
from drishti.drivers.steerability_probe import mean_action, DEFAULT_TASK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-ckpt", required=True)
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--phase-steps", type=int, default=25)
    ap.add_argument("--nsamp", type=int, default=16)
    args = ap.parse_args()

    from drishti.policy_molmoact2_lerobot import load_molmoact2_lerobot
    vla = load_molmoact2_lerobot(args.lerobot_ckpt)

    captions = [paraphrases[0] for (_, _, paraphrases) in MACRO_PHASES]
    pairs = list(itertools.combinations(range(len(captions)), 2))
    print(f">>> probing OWN trained vocabulary: {captions}", flush=True)

    rng = np.random.default_rng(0)
    cubes = [(float(rng.uniform(0.24, 0.28)), float(rng.uniform(-0.03, 0.03))) for _ in range(args.positions)]
    agg = []

    for ci, cxy in enumerate(cubes):
        env = SO101Env(); _grasp_config(env, cxy)
        if hasattr(vla, "reset"):
            vla.reset()
        for _ in range(args.phase_steps):
            f0, f1 = _render_pair(env)
            a = np.asarray(vla.predict([f0, f1], DEFAULT_TASK, env.state()), float).reshape(-1, 6)[0]
            env.step(a.astype(np.float64), n_substeps=N_SUBSTEPS)
        f0, f1 = _render_pair(env); state = env.state()
        print(f"--- position {ci} {tuple(round(v,3) for v in cxy)} (frozen @ frame {args.phase_steps}) ---", flush=True)
        for i, j in pairs:
            ma = mean_action(vla, [f0, f1], state, captions[i], args.nsamp)
            mb = mean_action(vla, [f0, f1], state, captions[j], args.nsamp)
            d = float(np.abs(ma - mb).mean())
            print(f"  phase{i} '{captions[i][:30]}' vs phase{j} '{captions[j][:30]}' -> mean |Δ| {d:.2f}", flush=True)
            agg.append(d)

    print(f"\n=== SUMMARY (own trained vocabulary) — mean |action delta| (~deg/joint) ===")
    print(f"  GRAND MEAN: {np.mean(agg):.2f}")
    print("Read: <~2 => still deaf even to its own training vocabulary; >~2 => did learn SOMETHING, just doesn't"
          " generalize to novel phrasing.")


if __name__ == "__main__":
    main()
