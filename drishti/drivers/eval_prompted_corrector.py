"""Stage 2 — does a corrective instruction help? Baseline vs corrected Δsuccess for the finetuned VLA.

The loop: run the finetuned MolmoAct2 closed-loop; score RND+SMD uncertainty every frame (calibrated);
when the combined score crosses the threshold (a "spike") call a **corrector** that returns a NEW task
string, swap it in, and continue. Compare success WITH vs WITHOUT corrections over the same cube positions.

Two speed/correctness choices from the calibration finding (RND AUC 1.0, ACE AUC 0.43 = dead weight):
- **Skip ACE at serving** — don't sample action chunks; feed ACE at its calibrated success-mean so its
  capped-z ≈ 0. RND still triggers, threshold stays valid, and we drop ~75% of the per-frame compute.
- The **steerability probe**: at each spike, also predict the action under the ORIGINAL task and log the
  joint-space delta vs the corrected task. If ≈0, the VLA ignores language (the gate fails) — no point
  serving a VLM corrector. This is the cheap first test (`--corrector scripted`).

Run on the pod (CUDA + EGL):
    MUJOCO_GL=egl PYTHONPATH=. ~/lerobot-ft/.venv/bin/python -m drishti.drivers.eval_prompted_corrector \
        --lerobot-ckpt <ckpt> --corrector scripted --episodes 10
"""

from __future__ import annotations

import argparse

import numpy as np

from drishti.sim.so101_env import SO101Env
from drishti.sim.rollout_server import CAMS_VLA, DEFAULT_TASK, N_SUBSTEPS, env_deg, _render_pair
from drishti.sim.scripted_pick import _grasp_config, _cube_in_tray, TRAY_XY

TASK = "Pick up the cube and place it in the box"       # the fixed string the VLA was finetuned on


def scripted_correction(env) -> str:
    """A geometry-aware re-instruction (stand-in for the VLM corrector) — tests VLA steerability."""
    g, c = env.gripper_tcp(), env.cube_pos()
    dxy = float(np.linalg.norm(g[:2] - c[:2]))
    dz = float(g[2] - c[2])
    if dxy > 0.03:
        return "The gripper is off to the side. Move it directly above the red cube, then lower and close to grasp it, and place it in the box."
    if dz > 0.03:
        return "The gripper is too high. Lower it straight down onto the red cube and close the gripper to grasp it, then place it in the box."
    return "Close the gripper firmly on the red cube now, lift it, and place it in the box."


def run_episode(vla, scorer, embedder, cube_xy, *, correct_fn, max_frames, warmup, cooldown, probe):
    """One rollout. correct_fn=None → baseline. Returns dict(success, interventions, spikes, probe_deltas)."""
    env = SO101Env()
    _grasp_config(env, cube_xy)
    vla.reset()
    ace_mean = float(scorer.norm["act"][0])                 # skip-ACE: feed the success-mean → cz_act≈0
    task = TASK
    interventions = spikes = 0
    since_correct = 10**9
    probe_deltas = []
    for f in range(max_frames):
        f0, f1 = _render_pair(env)
        state = env.state()
        action = np.asarray(vla.predict([f0, f1], task, state), float).reshape(-1, 6)[0]
        smd_raw = float(scorer.smd.score(env_deg(state)))
        emb = np.asarray(embedder(f0, f1), dtype=np.float32)
        rnd_raw = float(scorer.rnd.score(emb))
        comb, _, _ = scorer.score(smd_raw=smd_raw, rnd_raw=rnd_raw, ace_raw=ace_mean)
        env.step(action.astype(np.float64), n_substeps=N_SUBSTEPS)
        since_correct += 1
        if _cube_in_tray(env.cube_pos(), TRAY_XY):
            return {"success": True, "interventions": interventions, "spikes": spikes,
                    "probe_deltas": probe_deltas, "frames": f + 1}
        if f >= warmup and comb > scorer.threshold:
            spikes += 1
            if correct_fn is not None and since_correct >= cooldown:
                new_task = correct_fn(env)
                if probe:                                    # steerability probe: does the string move the action?
                    a_old = np.asarray(vla.predict([f0, f1], task, state), float).reshape(-1, 6)[0]
                    a_new = np.asarray(vla.predict([f0, f1], new_task, state), float).reshape(-1, 6)[0]
                    probe_deltas.append(float(np.linalg.norm((a_new - a_old)) * 180.0 / np.pi))  # deg
                task = new_task
                interventions += 1
                since_correct = 0
    return {"success": False, "interventions": interventions, "spikes": spikes,
            "probe_deltas": probe_deltas, "frames": max_frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-ckpt", required=True)
    ap.add_argument("--corrector", choices=["scripted"], default="scripted")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=130)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--cooldown", type=int, default=15, help="min frames between corrections")
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    from drishti.policy_molmoact2_lerobot import load_molmoact2_lerobot
    from drishti.uq.embed_frames import load_rnd_embedder
    from drishti.sim.rollout_server import CalibScorer
    vla = load_molmoact2_lerobot(args.lerobot_ckpt)
    embedder = load_rnd_embedder()
    scorer = CalibScorer(calib_dir=args.calib_dir, embedder=embedder)
    print(f"threshold={scorer.threshold:.4f}  (RND-driven; ACE skipped at serving)", flush=True)

    correct_fn = scripted_correction
    rng = np.random.default_rng(args.seed)
    base_succ = corr_succ = 0
    all_probe = []
    for i in range(args.episodes):
        cxy = (float(rng.uniform(0.24, 0.28)), float(rng.uniform(-0.03, 0.03)))
        b = run_episode(vla, scorer, embedder, cxy, correct_fn=None,
                        max_frames=args.max_frames, warmup=args.warmup, cooldown=args.cooldown, probe=False)
        c = run_episode(vla, scorer, embedder, cxy, correct_fn=correct_fn,
                        max_frames=args.max_frames, warmup=args.warmup, cooldown=args.cooldown, probe=True)
        base_succ += int(b["success"]); corr_succ += int(c["success"])
        all_probe += c["probe_deltas"]
        print(f"[{i+1}/{args.episodes}] cube={np.round(cxy,3)} "
              f"BASE succ={b['success']} spikes={b['spikes']} | "
              f"CORR succ={c['success']} interv={c['interventions']} "
              f"probeΔ={np.round(np.mean(c['probe_deltas']),1) if c['probe_deltas'] else '-'}deg", flush=True)
    pd = np.array(all_probe) if all_probe else np.array([0.0])
    print(f"\n=== STAGE 2: baseline {base_succ}/{args.episodes} → corrected {corr_succ}/{args.episodes} "
          f"(Δ={corr_succ-base_succ:+d}) ===")
    print(f"steerability probe: mean action-delta from re-instruction = {pd.mean():.1f}° "
          f"(max {pd.max():.1f}°, n={len(all_probe)}) — ~0 means the VLA ignores language")


if __name__ == "__main__":
    main()
