"""Record finetuned-policy rollouts into the telemetry store (for the log section).

Runs the finetuned SmolVLA policy closed-loop in the SO-101 sim for N episodes and
logs EACH episode as a `run` (a "rollout" in the log), capturing everything needed to
later (a) REPLAY it and (b) CALIBRATE the UQ detectors on the successful ones:
  - per step: observation.state, action, and up/side/wrist frames (blobs)
  - per rollout: auto success label (cube_in_tray) + cube-start in meta

Receding horizon (n_action_steps small) is required — open-loop chunks drift and fail.
UQ scores are NOT computed here (detectors aren't calibrated yet); calibration replays
the selected successful rollouts' frames through the policy afterwards.

    PYTHONPATH=. .venv/bin/python -m drishti.drivers.rollout_record --episodes 50

Then start the ingest server and open the dashboard to see the log section.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from drishti import start_run
from drishti.store import Store
from drishti.policy_smolvla import load_smolvla
from drishti.sim.so101_env import SO101Env
from drishti.sim.scripted_pick import _grasp_config, _cube_in_tray, TRAY_XY

DEG2RAD = np.pi / 180.0
from drishti.sim.task import DEFAULT_TASK as TASK
CKPT = "data/policy/pretrained_model"
CAMS = ("up", "side", "wrist")
LANES = {"left": (0.012, 0.030), "center": (-0.012, 0.012), "right": (-0.030, -0.012)}
LANE_ORDER = ["left", "center", "right"]


def build_obs(env, dev, task=TASK):
    def chw(cam):
        img = env.render(cam, 320, 240)
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()).float().div(255).unsqueeze(0).to(dev)
    state_deg = env.state().astype(np.float32) / DEG2RAD
    obs = {f"observation.images.camera{i+1}": chw(c) for i, c in enumerate(CAMS)}
    obs["observation.state"] = torch.from_numpy(state_deg).unsqueeze(0).to(dev)
    obs["task"] = task
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--nas", type=int, default=1, help="n_action_steps (receding horizon)")
    ap.add_argument("--max-frames", type=int, default=110)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--task", default=TASK,
                    help="language instruction sent to the policy (keep matched to finetuning for reliable behavior)")
    args = ap.parse_args()

    store = Store()
    blob_root = Path("data/obs_blobs")
    env = SO101Env()
    # instrumented load: hooks capture the vision embedding + action-head stats per forward
    policy, pre, post, dev = load_smolvla(args.ckpt, device=args.device)
    policy.config.n_action_steps = args.nas
    rng = np.random.default_rng(args.seed)
    model_id = f"smolvla_ft@4500 (nas={args.nas})"
    n_ok = 0

    print(f"recording {args.episodes} rollouts on {dev}  ->  store=data/obs.db", flush=True)
    for ep in range(args.episodes):
        lane = LANE_ORDER[ep % 3]
        ylo, yhi = LANES[lane]
        # x capped at 0.28: above ~0.295 the cube spawns inside the tray zone (overlap + instant
        # trivial success that pollutes calibration). See live_fleet for the same fix.
        cx, cy = float(rng.uniform(0.24, 0.28)), float(rng.uniform(ylo, yhi))
        _grasp_config(env, (cx, cy))
        policy.reset()

        run = start_run(model_id, args.task, backend="in_process", store=store,
                        meta={"sim": "so101", "rollout": ep, "lane": lane,
                              "cube_xy": [round(cx, 4), round(cy, 4)], "nas": args.nas,
                              "checkpoint": "4500"})
        bdir = blob_root / run.run_id
        bdir.mkdir(parents=True, exist_ok=True)

        success = False
        f = 0
        for f in range(args.max_frames):
            obs = build_obs(env, dev, args.task)
            with run.step(step_idx=f, t=float(f)) as st:
                with torch.no_grad():
                    action = post(policy.select_action(pre(obs)))
                a_deg = action.squeeze(0).detach().float().cpu().numpy()[:6]
                st.log_observation((env.state() / DEG2RAD).tolist()).log_action(a_deg.tolist())
            # the vision hook stashed the multi-camera vision embedding (up+side+wrist concat)
            # on the step during select_action; persist it (calibration fits RND on these)
            emb = getattr(st, "_vision_embed", None)
            if emb is not None:
                np.save(bdir / f"embed_{f:03d}.npy", emb.numpy().astype(np.float32))
            for cam in CAMS:
                img = env.render(cam, 320, 240)
                rel = f"{run.run_id}/{cam}_{f:03d}.jpg"
                Image.fromarray(img).save(blob_root / rel, quality=80)
                run.log_frame(camera=cam, blob_path=rel, step_idx=f, t=float(f))
            env.step(a_deg.astype(np.float64) * DEG2RAD, n_substeps=18)
            if _cube_in_tray(env.cube_pos(), TRAY_XY):
                success = True
                break

        run.finish(success_rate=float(success), n_episodes=1)
        n_ok += int(success)
        print(f"[{ep+1}/{args.episodes}] lane={lane:6s} cube=({cx:.3f},{cy:+.3f}) "
              f"frames={f+1} success={success}  [{n_ok}/{ep+1}]", flush=True)

    print(f">>> done. {n_ok}/{args.episodes} successful rollouts in the log.", flush=True)


if __name__ == "__main__":
    main()
