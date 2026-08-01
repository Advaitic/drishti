"""Test #2 — IS THE FINETUNED VLA LANGUAGE-STEERABLE?

At a frozen mid-episode state, sample the VLA's action under CONTRASTING instructions and measure how much
the *mean* action changes (averaging over flow-matching noise so the delta reflects the INSTRUCTION, not the
sampling noise). This isolates the one question everything hinges on:

  small delta (< ~2 deg/joint)  -> the VLA barely responds to language -> RL/reward CAN'T fix that.
  large delta (> ~10 deg/joint) -> it IS steerable -> the bottleneck is our RL / reward / curriculum.

Uses `sample()` (fresh chunk via predict_action_chunk — does NOT touch the execution buffer). No calibration,
no training. 1 GPU. ~a few minutes.

  MUJOCO_GL=egl PYTHONPATH=. <lerobot-venv>/bin/python -m drishti.drivers.steerability_probe \
    --lerobot-ckpt ~/ft_ckpt --positions 4 --phase-steps 25 --nsamp 16
"""
from __future__ import annotations

import argparse
import numpy as np

from drishti.sim.so101_env import SO101Env
from drishti.sim.rollout_server import N_SUBSTEPS, _render_pair
from drishti.sim.scripted_pick import _grasp_config

from drishti.sim.task import DEFAULT_TASK

# FINE-corrective pairs — the kind our corrector needs (small nudges). Steerable => action moves a lot.
PAIRS = [
    ("move the gripper to the left", "move the gripper to the right"),
    ("lower the gripper straight down onto the cube", "lift the gripper straight up, away from the cube"),
    ("close the gripper now to grasp the cube", "keep the gripper fully open, do not close it"),
    ("move toward the cube", "move away from the cube"),
]

# TASK-level pairs — the kind MolmoAct is *built* for (whole different goals). If the base is language-
# conditioned at all, THESE should move the action even if the fine corrections don't.
TASK_PAIRS = [
    ("pick up the cube and place it in the box", "push the cube off the table"),
    ("grasp the cube", "raise your arm straight up and hold still"),
    ("pick up the cube and place it in the box", "go back to the home position"),
]


def mean_action(vla, images, state, instr: str, n: int) -> np.ndarray:
    """Mean first-action over n fresh samples (averages out flow noise -> isolates the instruction effect)."""
    chunk = np.asarray(vla.sample(images, instr, state, n=n), dtype=float).reshape(n, -1)[:, :6]
    return chunk.mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["base", "finetuned"], default="finetuned",
                    help="base = zero-shot allenai/MolmoAct2-SO100_101; finetuned = our lerobot ckpt")
    ap.add_argument("--lerobot-ckpt", default=None, help="required for --policy finetuned")
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--phase-steps", type=int, default=25, help="frames to advance (VLA-driven) before probing")
    ap.add_argument("--nsamp", type=int, default=16, help="samples per instruction to average out flow noise")
    args = ap.parse_args()

    if args.policy == "base":
        from drishti.policy_molmoact2 import load_molmoact2
        vla = load_molmoact2()                               # zero-shot Ai2 base (HF venv, transformers 4.57)
    else:
        from drishti.policy_molmoact2_lerobot import load_molmoact2_lerobot
        assert args.lerobot_ckpt, "--lerobot-ckpt required for --policy finetuned"
        vla = load_molmoact2_lerobot(args.lerobot_ckpt)
    print(f">>> probing policy={args.policy}", flush=True)

    rng = np.random.default_rng(0)
    cubes = [(float(rng.uniform(0.24, 0.28)), float(rng.uniform(-0.03, 0.03))) for _ in range(args.positions)]
    groups = {"FINE (corrections)": PAIRS, "TASK (whole goals)": TASK_PAIRS}
    agg = {name: [] for name in groups}

    for ci, cxy in enumerate(cubes):
        env = SO101Env(); _grasp_config(env, cxy)
        if hasattr(vla, "reset"):
            vla.reset()
        for _ in range(args.phase_steps):                       # let the VLA drive to a mid-episode state
            f0, f1 = _render_pair(env)
            a = np.asarray(vla.predict([f0, f1], DEFAULT_TASK, env.state()), float).reshape(-1, 6)[0]
            env.step(a.astype(np.float64), n_substeps=N_SUBSTEPS)
        f0, f1 = _render_pair(env); state = env.state()
        print(f"--- position {ci} {tuple(round(v,3) for v in cxy)} (frozen @ frame {args.phase_steps}) ---", flush=True)
        for gname, pairs in groups.items():
            for a_instr, b_instr in pairs:
                ma = mean_action(vla, [f0, f1], state, a_instr, args.nsamp)
                mb = mean_action(vla, [f0, f1], state, b_instr, args.nsamp)
                d = float(np.abs(ma - mb).mean())
                print(f"  [{gname[:4]}] '{a_instr[:22]:22}' vs '{b_instr[:22]:22}' -> mean |Δ| {d:.2f}", flush=True)
                agg[gname].append(d)

    print(f"\n=== SUMMARY ({args.policy}) — mean |action delta| under OPPOSITE instructions (~deg/joint) ===")
    for gname in groups:
        print(f"  {gname:22} GRAND MEAN: {np.mean(agg[gname]):.2f}")
    print("\nRead: <~2 => near-deaf; >~10 => clearly steerable. If BASE's TASK pairs are large but our"
          " finetuned's were ~0 -> our finetune collapsed a real capability -> refinetune WILL help.")


if __name__ == "__main__":
    main()
