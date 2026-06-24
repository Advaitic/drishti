"""Closed-loop driver — SmolVLA drives the SO-101 in MuJoCo, live uncertainty → dashboard.

The Stage-A wedge, now closed-loop: each control step renders the sim cameras, runs
instrumented SmolVLA (hooks fire → SMD/RND/ACE), executes the predicted action on the
SO-101, and streams telemetry to the dashboard.

    .venv/bin/python -m drishti.drivers.sim_loop --steps 24 --ace-b 4

Action chain (verified): model outputs normalized actions → unnormalize with the
svla_so101_pickplace action mean/std → degrees → ×π/180 → radian actuator targets.
State to the model: env joints (radians) → ÷(π/180) → degrees (the model's input units).

Caveat: SmolVLA was trained on REAL SO-101 camera frames; MuJoCo renders are synthetic,
so the policy is out-of-distribution and will likely wander (RND should spike on the OOD
visuals — itself a real signal). We're proving the closed-loop FLOW + live uncertainty,
not task success. Finetuning on sim frames is the later step to make it actually work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from drishti import start_run
from drishti.drivers.replay import _capture_embed, _obs_batch as _dataset_obs_batch
from drishti.policy_smolvla import load_smolvla
from drishti.sim.so101_env import SO101Env
from drishti.uq import RND, SMD, ace_score, sample_action_chunks

DATASET = "lerobot/svla_so101_pickplace"
BLOB_DIR = Path("data/obs_blobs")
DEG2RAD = np.pi / 180.0
TASK = "grab the red cube and put it in the box"


def _sim_obs_batch(up_img, side_img, state_deg, task, device) -> dict:
    """Build SmolVLA obs from sim renders (HWC uint8) + 6-d state (degrees).
    up→camera1, side→camera2 (camera3 is auto-padded by SmolVLA.prepare_images)."""
    def chw(img):
        t = torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()).float() / 255.0
        return t.unsqueeze(0).to(device)
    return {
        "observation.images.camera1": chw(up_img),
        "observation.images.camera2": chw(side_img),
        "observation.state": torch.from_numpy(
            np.asarray(state_deg, dtype=np.float32)).unsqueeze(0).to(device),
        "task": task,
    }


def run(
    dataset: str = DATASET,
    model_id: str = "lerobot/smolvla_base",
    ingest_url: str = "http://localhost:8010",
    steps: int = 24,
    ace_b: int = 4,
    calib_frames: int = 24,
    action_repeat: int = 8,
    task: str = TASK,
) -> str:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(dataset)
    a_mean = np.asarray(ds.meta.stats["action"]["mean"], dtype=np.float64)
    a_std = np.asarray(ds.meta.stats["action"]["std"], dtype=np.float64)
    policy, preprocess, postprocess, device = load_smolvla(model_id)

    # --- calibrate SMD on dataset states, RND on dataset vision embeddings ---
    n_states = min(len(ds), 512)
    states = np.stack([np.asarray(ds[i]["observation.state"], float) for i in range(n_states)])
    smd = SMD.fit(states)
    print(f"[calib] SMD fit on {n_states} states", flush=True)
    embs = []
    for i in range(min(calib_frames, len(ds))):
        emb = _capture_embed(policy, preprocess(_dataset_obs_batch(ds[i], device)))
        if emb is not None:
            embs.append(emb.numpy())
    rnd = RND(np.stack(embs).shape[1], device="cpu").fit(np.stack(embs), epochs=10) if embs else None
    print(f"[calib] RND fit on {len(embs)} embeddings", flush=True)

    # --- closed loop ---
    env = SO101Env(n_substeps=action_repeat)
    env.reset()
    run = start_run(model_id, task, ingest_url=ingest_url, backend="in_process",
                    meta={"sim": "so101", "mode": "closed_loop", "dataset": dataset})
    print(f"[run] {run.run_id} closed-loop SO-101, {steps} steps", flush=True)
    for k in range(steps):
        up = env.render("up", 640, 480)
        side = env.render("side", 640, 480)
        state_deg = (env.state() / DEG2RAD).astype(np.float32)
        proc = preprocess(_sim_obs_batch(up, side, state_deg, task, device))
        with run.step(step_idx=k, t=float(k)) as st:
            chunks = sample_action_chunks(policy, proc, n=ace_b)  # hooks fire → st.internals
            emb = getattr(st, "_vision_embed", None)
            action_norm = np.asarray(chunks[0, 0], dtype=np.float64).reshape(-1)[:6]
            action_deg = action_norm * a_std + a_mean
            st.log_observation(state_deg.tolist())
            st.log_action(action_deg.tolist())
            st.log_uncertainty(
                smd=smd.score(state_deg),
                rnd=(rnd.score(emb.numpy()) if (rnd is not None and emb is not None) else None),
                ace=ace_score(chunks),
            )
        env.step(action_deg * DEG2RAD)  # degrees → radian ctrl (env clamps to ctrlrange)
        # save all three views for the dashboard (model still only consumed up+side)
        wrist = env.render("wrist", 640, 480)
        (BLOB_DIR / run.run_id).mkdir(parents=True, exist_ok=True)
        for cam, img in (("up", up), ("side", side), ("wrist", wrist)):
            rel = f"{run.run_id}/0_{k}_{cam}.jpg"
            Image.fromarray(img).save(BLOB_DIR / rel)
            run.log_frame(camera=cam, blob_path=rel, step_idx=k, t=float(k))
        if (k + 1) % 4 == 0:
            print(f"[run] step {k+1}/{steps}", flush=True)
    run.finish()
    print(f"[run] done -> {run.run_id}", flush=True)
    return run.run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ace-b", type=int, default=4)
    ap.add_argument("--calib-frames", type=int, default=24)
    ap.add_argument("--action-repeat", type=int, default=8)
    ap.add_argument("--ingest-url", default="http://localhost:8010")
    ap.add_argument("--model-id", default="lerobot/smolvla_base")
    args = ap.parse_args()
    run(model_id=args.model_id, ingest_url=args.ingest_url, steps=args.steps,
        ace_b=args.ace_b, calib_frames=args.calib_frames, action_repeat=args.action_repeat)


if __name__ == "__main__":
    main()
