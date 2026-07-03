"""Receding-horizon A/B — success rate at different n_action_steps over the same cube positions.

The finetuned VLA commits to `n_action_steps` of a planned chunk before re-planning. 4 worked (69% calib);
the user has had 10 work before. This loads the model ONCE and toggles `n_action_steps` between sweeps so
the comparison is on identical positions (seed 0 = what calibration draws).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. MUJOCO_GL=egl /root/lerobot/.venv/bin/python \
        -m drishti.drivers.nas_ab --lerobot-ckpt /root/ft_ckpt --episodes 8 --nas 4 10
"""

from __future__ import annotations

import argparse

import numpy as np

from drishti.sim.so101_env import SO101Env
from drishti.sim.rollout_server import N_SUBSTEPS, _render_pair
from drishti.sim.scripted_pick import _grasp_config, _cube_in_tray, TRAY_XY

TASK = "Pick up the cube and place it in the box"


def run_ep(vla, cxy, max_frames):
    env = SO101Env()
    _grasp_config(env, cxy)
    vla.reset()
    for f in range(max_frames):
        f0, f1 = _render_pair(env)
        a = np.asarray(vla.predict([f0, f1], TASK, env.state()), float).reshape(-1, 6)[0]
        env.step(a.astype(np.float64), n_substeps=N_SUBSTEPS)
        if _cube_in_tray(env.cube_pos(), TRAY_XY):
            return True, f + 1
    return False, max_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--nas", type=int, nargs="+", default=[4, 10])
    args = ap.parse_args()

    from drishti.policy_molmoact2_lerobot import load_molmoact2_lerobot
    vla = load_molmoact2_lerobot(args.lerobot_ckpt)

    rng = np.random.default_rng(0)
    positions = [(float(rng.uniform(0.24, 0.28)), float(rng.uniform(-0.03, 0.03)))
                 for _ in range(args.episodes)]

    print(f"positions (seed 0): {[tuple(np.round(p,3)) for p in positions]}", flush=True)
    results = {}
    for nas in args.nas:
        vla.policy.config.n_action_steps = int(nas)
        succ, steps = 0, []
        for i, cxy in enumerate(positions):
            ok, fr = run_ep(vla, cxy, args.max_frames)
            succ += int(ok); steps.append(fr)
            print(f"  nas={nas:2d} pos{i} {np.round(cxy,3)} -> {'S' if ok else '.'} ({fr})", flush=True)
        results[nas] = succ
        print(f"=> nas={nas:2d}: {succ}/{args.episodes} success ({succ/args.episodes:.2f}) "
              f"avg_steps={np.mean(steps):.0f}", flush=True)
    best = max(results, key=results.get)
    print(f"\n=== A/B RESULT: {results} -> best nas={best} ===")


if __name__ == "__main__":
    main()
