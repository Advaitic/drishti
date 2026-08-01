"""Measure TASK SUCCESS of a (phase-captioned) MolmoAct2 checkpoint, closed-loop.

The steerability probe only measures action delta at a frozen state under instruction swaps.
It says nothing about whether the policy still grasps, or whether feeding the phase captions
at inference changes rollout behavior/success. This driver runs full closed-loop rollouts and
reports success + grasp rate, so we can compare annotated vs non-annotated and (crucially)
phase-caption inference vs static-string inference on the SAME checkpoint.

Caption modes:
  --caption-mode static : feed the one training string every step (how the ORIGINAL ckpt ran)
  --caption-mode phase  : feed the geometry-appropriate macro-phase caption every step (how the
                          phasecap ckpt was TRAINED to run) — approach/descend->phase0,
                          lift/carry->phase1, place->phase2, via rollout_server._phase geometry.

  MUJOCO_GL=egl PYTHONPATH=. <lerobot-venv>/bin/python -m drishti.drivers.success_eval \
    --lerobot-ckpt <hf-repo-or-path> --caption-mode phase --positions 12 --trials 2
"""
from __future__ import annotations

import argparse
import numpy as np

from drishti.sim.so101_env import SO101Env
from drishti.sim.rollout_server import N_SUBSTEPS, _render_pair, _phase
from drishti.sim.scripted_pick import TRAY_XY, _cube_in_tray, _grasp_config, _is_grasping
from drishti.sim.phase_captions import MACRO_PHASES

STATIC_TASK = "Pick up the cube and place it in the box"   # matches the dataset's static string (capital P)

# geometry phase (rollout_server._phase) -> macro-phase index into MACRO_PHASES
_PHASE_TO_MACRO = {"approach": 0, "descend": 0, "lift": 1, "carry": 1, "place": 2}


def caption_for_geometry(env: SO101Env, grasping: bool) -> str:
    macro = _PHASE_TO_MACRO[_phase(env, grasping)]
    return MACRO_PHASES[macro][2][0]   # first (canonical) paraphrase of that macro-phase


def run_episode(vla, env, cube_xy, caption_mode: str, max_frames: int) -> dict:
    _grasp_config(env, cube_xy)
    if hasattr(vla, "reset"):
        vla.reset()
    ever_grasped = False
    for _ in range(max_frames):
        grasping = _is_grasping(env)
        ever_grasped = ever_grasped or grasping
        task = STATIC_TASK if caption_mode == "static" else caption_for_geometry(env, grasping)
        f0, f1 = _render_pair(env)
        a = np.asarray(vla.predict([f0, f1], task, env.state()), float).reshape(-1, 6)[0]
        env.step(a.astype(np.float64), n_substeps=N_SUBSTEPS)
        if _cube_in_tray(env.cube_pos(), TRAY_XY):
            return {"success": True, "grasped": True}
    return {"success": bool(_cube_in_tray(env.cube_pos(), TRAY_XY)), "grasped": ever_grasped}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-ckpt", required=True)
    ap.add_argument("--caption-mode", choices=["static", "phase"], default="phase")
    ap.add_argument("--positions", type=int, default=12)
    ap.add_argument("--trials", type=int, default=2, help="rollouts per position (averages VLA flow noise)")
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    # eval cube distribution: matches the tightened training range (drishti.sim.record X_RANGE 0.24-0.26)
    ap.add_argument("--x-range", type=float, nargs=2, default=[0.24, 0.26])
    ap.add_argument("--y-range", type=float, nargs=2, default=[-0.03, 0.03])
    args = ap.parse_args()

    from drishti.policy_molmoact2_lerobot import load_molmoact2_lerobot
    vla = load_molmoact2_lerobot(args.lerobot_ckpt)
    print(f">>> success eval: ckpt={args.lerobot_ckpt} caption_mode={args.caption_mode}", flush=True)

    rng = np.random.default_rng(args.seed)
    cubes = [(float(rng.uniform(*args.x_range)), float(rng.uniform(*args.y_range)))
             for _ in range(args.positions)]
    env = SO101Env()

    n_succ = n_grasp = n_tot = 0
    for ci, cxy in enumerate(cubes):
        for t in range(args.trials):
            r = run_episode(vla, env, cxy, args.caption_mode, args.max_frames)
            n_tot += 1
            n_succ += int(r["success"]); n_grasp += int(r["grasped"])
            print(f"  [{ci+1}/{args.positions} t{t}] cube=({cxy[0]:.3f},{cxy[1]:+.3f}) "
                  f"success={r['success']} grasped={r['grasped']}", flush=True)

    print(f"\n=== SUCCESS EVAL ({args.caption_mode}) — {args.lerobot_ckpt} ===")
    print(f"  success {n_succ}/{n_tot} ({n_succ/n_tot:.0%})  ·  grasped {n_grasp}/{n_tot} ({n_grasp/n_tot:.0%})")


if __name__ == "__main__":
    main()
