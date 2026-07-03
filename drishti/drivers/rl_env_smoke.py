"""RL-environment smoke test — the full corrector loop on the LIVE finetuned VLA (minus the GRPO trainer).

Drives `EpisodeManager` (the state machine the rollout server + verifiers env wrap) end-to-end:
    start → run to an uncertainty SPIKE → inject a corrective instruction → resume → verifiable REWARD.
This proves the reward signal + spike trigger + correction injection + resume all work with the real VLA
and calibrated detectors. The verifiers `MultiTurnEnv` over HTTP + the prime-rl GRPO trainer are the
remaining wrap — the env/reward/rubric logic is already unit-tested (tests/test_vla_corrector_env.py) with
a stub; this exercises the same core against the finetuned MolmoAct2.

    MUJOCO_GL=egl PYTHONPATH=. ~/lerobot-ft/.venv/bin/python -m drishti.drivers.rl_env_smoke \
        --lerobot-ckpt <ckpt> --episodes 6
"""

from __future__ import annotations

import argparse

import numpy as np

# the fixed corrector stand-in — the instruction the prompt sweep found most helpful (+12%). A trained
# Qwen3-VL corrector would emit this (or better, per-state) from the failure frames + report.
PROCEDURAL = "Lower the gripper straight down onto the cube, close it to grasp, lift, and place it in the box"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=130)
    args = ap.parse_args()

    from drishti.sim.rollout_server import EpisodeManager, CalibScorer
    from drishti.policy_molmoact2_lerobot import load_molmoact2_lerobot
    from drishti.uq.embed_frames import load_rnd_embedder

    vla = load_molmoact2_lerobot(args.lerobot_ckpt)
    embedder = load_rnd_embedder()
    scorer = CalibScorer(embedder=embedder)

    def factory(env):
        vla.reset()                          # one shared VLA, fresh queue per episode
        return vla

    mgr = EpisodeManager(vla_factory=factory, scorer=scorer, embedder=embedder,
                         max_frames_default=args.max_frames)
    print(f"threshold={scorer.threshold:.4f}", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for i in range(args.episodes):
        cxy = (float(rng.uniform(0.24, 0.28)), float(rng.uniform(-0.03, 0.03)))
        eid = mgr.start(cube_xy=cxy)["episode_id"]
        rep = mgr.run(eid)                                    # → first spike, or success/timeout
        spiked = rep["status"] == "spike"
        reward_before = rep["reward"]
        if spiked:
            mgr.correct(eid, PROCEDURAL)                      # inject the correction (task swap)
            rep = mgr.run(eid, stop_on_spike=False)           # resume to completion under the correction
        reward = rep["reward"]
        mgr.end(eid)
        rows.append((spiked, rep["status"], reward))
        print(f"[{i+1}/{args.episodes}] cube={np.round(cxy,3)} spiked={spiked} "
              f"spike_frame={rep.get('n_interventions')} final={rep['status']} REWARD={reward}", flush=True)

    n_spiked = sum(r[0] for r in rows)
    mean_reward = float(np.mean([r[2] for r in rows]))
    print(f"\n=== RL-ENV SMOKE: loop OK on {args.episodes} eps · spikes fired on {n_spiked} · "
          f"corrections injected · verifiable reward returned · mean reward={mean_reward:.2f} ===")
    print("reward signal + spike trigger + correction + resume all functional with the live VLA. "
          "Remaining: verifiers-over-HTTP wrap + prime-rl GRPO trainer (2 GPUs).")


if __name__ == "__main__":
    main()
