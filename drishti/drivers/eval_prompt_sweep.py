"""Stage 2, test 1 — does a DIFFERENT/corrective instruction change the finetuned VLA's success?

The VLA was finetuned on ONE fixed string. Before wiring a VLM corrector, the cheap gate is: run the
whole episode under each of several instruction variants (color, procedural, localize, firm-grasp, spatial)
and see if any beats the base prompt. Same cube positions across prompts (seed 0 = what calibration used,
so the `base` column should reproduce the ~10/16 baseline — a sanity check on this driver).

Buffered receding horizon (n_action_steps=4). No ACE, no uncertainty scoring — just run + success oracle,
so it's fast (1 VLM forward/frame).

    MUJOCO_GL=egl PYTHONPATH=. ~/lerobot-ft/.venv/bin/python -m drishti.drivers.eval_prompt_sweep \
        --lerobot-ckpt <ckpt> --episodes 8
"""

from __future__ import annotations

import argparse

import numpy as np

from drishti.sim.so101_env import SO101Env
from drishti.sim.rollout_server import N_SUBSTEPS, _render_pair
from drishti.sim.scripted_pick import _grasp_config, _cube_in_tray, TRAY_XY

PROMPTS = {
    "base":          "Pick up the cube and place it in the box",
    "red":           "Pick up the red cube and place it in the box",
    "procedural":    "Lower the gripper straight down onto the cube, close it to grasp, lift, and place it in the box",
    "locate":        "Find the red cube, move directly above it, grasp it, then place it in the box",
    "firm":          "Grasp the cube firmly so the gripper is fully closed on it, then place it in the box",
    "descend_close": "Reach down to the cube and close the gripper on it, then lift and place it in the box",
}


def run_ep(vla, cube_xy, task, max_frames):
    env = SO101Env()
    _grasp_config(env, cube_xy)
    vla.reset()
    for f in range(max_frames):
        f0, f1 = _render_pair(env)
        state = env.state()
        action = np.asarray(vla.predict([f0, f1], task, state), float).reshape(-1, 6)[0]
        env.step(action.astype(np.float64), n_substeps=N_SUBSTEPS)
        if _cube_in_tray(env.cube_pos(), TRAY_XY):
            return True, f + 1
    return False, max_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=130)
    ap.add_argument("--seed", type=int, default=0)      # 0 = same positions calibration drew
    args = ap.parse_args()

    from drishti.policy_molmoact2_lerobot import load_molmoact2_lerobot
    vla = load_molmoact2_lerobot(args.lerobot_ckpt)

    rng = np.random.default_rng(args.seed)
    positions = [(float(rng.uniform(0.24, 0.28)), float(rng.uniform(-0.03, 0.03)))
                 for _ in range(args.episodes)]

    results = {name: [] for name in PROMPTS}
    for pi, cxy in enumerate(positions):
        line = [f"pos{pi} {np.round(cxy,3)}"]
        for name, prompt in PROMPTS.items():
            ok, fr = run_ep(vla, cxy, prompt, args.max_frames)
            results[name].append(ok)
            line.append(f"{name}={'S' if ok else '.'}")
        print("  ".join(line), flush=True)

    print("\n=== PROMPT SWEEP (success / {} positions) ===".format(args.episodes))
    base_rate = np.mean(results["base"])
    for name in PROMPTS:
        r = np.mean(results[name])
        tag = "  <-- base" if name == "base" else (f"  Δ={r-base_rate:+.2f}" if r != base_rate else "  (=base)")
        print(f"  {name:14s} {int(np.sum(results[name]))}/{args.episodes}  ({r:.2f}){tag}")


if __name__ == "__main__":
    main()
