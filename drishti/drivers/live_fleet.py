"""Live Fleet driver — run ONE rollout live, with the calibrated Monitor watching.

Runs the finetuned policy in the sim and STREAMS every step + frame to the ingest server over
HTTP, so the dashboard's `/v1/tail` WebSocket shows it live. Each step is scored against the
calibrated detectors (SMD/RND/ACE → combined), and the rollout HALTS the moment the combined
uncertainty crosses the (tunable) threshold — the Monitor catching an impending failure before
it happens. The dashboard sees the scores climb, cross, and the run end = halted.

    PYTHONPATH=. .venv/bin/python -m drishti.drivers.live_fleet --threshold 5 [--cube-xy x y]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from drishti import start_run
from drishti.policy_smolvla import load_smolvla
from drishti.calibrate import load_calib, combined_score, _act_spread
from drishti.sim.so101_env import SO101Env
from drishti.sim.scripted_pick import _grasp_config, _cube_in_tray, TRAY_XY

DEG2RAD = np.pi / 180.0
from drishti.sim.task import DEFAULT_TASK as TASK
CKPT = "data/policy/pretrained_model"
CAMS = ("up", "side", "wrist")
WARMUP = 3   # don't halt on the first few frames (start transient)


def build_obs(env, dev, task=TASK):
    def chw(c):
        img = env.render(c, 320, 240)
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()).float().div(255).unsqueeze(0).to(dev)
    obs = {f"observation.images.camera{i+1}": chw(c) for i, c in enumerate(CAMS)}
    obs["observation.state"] = torch.from_numpy((env.state() / DEG2RAD).astype(np.float32)).unsqueeze(0).to(dev)
    obs["task"] = task
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None, help="halt threshold (defaults to calibrated)")
    ap.add_argument("--cube-xy", type=float, nargs=2, default=None)
    ap.add_argument("--nas", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=110)
    ap.add_argument("--pace", type=float, default=0.0, help="extra sleep per frame (watchability)")
    ap.add_argument("--ingest", default="http://localhost:8010")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--task", default=TASK,
                    help="language instruction (keep matched to finetuning for reliable behavior)")
    args = ap.parse_args()

    smd, rnd, norm, thr_cal = load_calib()
    threshold = args.threshold if args.threshold is not None else thr_cal
    policy, pre, post, dev = load_smolvla(CKPT)
    policy.config.n_action_steps = args.nas
    env = SO101Env()
    rng = np.random.default_rng(args.seed)
    # x capped at 0.28 (not 0.31): the tray "in-zone" starts at x=0.295, so a higher spawn would
    # overlap the tray — and a cube starting in-zone counts as an instant trivial success.
    cx, cy = (args.cube_xy if args.cube_xy
              else (float(rng.uniform(0.24, 0.28)), float(rng.uniform(-0.03, 0.03))))
    _grasp_config(env, (cx, cy))
    policy.reset()

    run = start_run("smolvla_ft@4500 · Fleet 1", args.task, ingest_url=args.ingest, backend="http",
                    meta={"fleet": "Fleet 1", "cube_xy": [round(cx, 4), round(cy, 4)],
                          "threshold": threshold, "live": True})
    blob_root = Path("data/obs_blobs")
    bdir = blob_root / run.run_id
    bdir.mkdir(parents=True, exist_ok=True)
    print(f"LIVE {run.run_id} threshold={threshold:.2f} cube=({cx:.3f},{cy:+.3f})", flush=True)

    last_emb, last_act, predicted_at, comb = None, 0.0, None, 0.0
    for f in range(args.max_frames):
        obs = build_obs(env, dev, args.task)
        with run.step(step_idx=f, t=float(f)) as st:
            with torch.no_grad():
                action = post(policy.select_action(pre(obs)))
            a_deg = action.squeeze(0).detach().float().cpu().numpy()[:6]
            emb = getattr(st, "_vision_embed", None)      # set only on inference (every nas) frames
            if emb is not None:
                last_emb = emb.numpy().astype(np.float32)
                last_act = _act_spread(st.internals.get("action_head_std"))
                # persist the embedding too (not just frames) so a successful LIVE rollout can feed
                # re-calibration — calibrate skips any run without embed blobs (the flywheel).
                np.save(bdir / f"embed_{f:03d}.npy", last_emb)
            state_deg = (env.state() / DEG2RAD).astype(np.float32)
            smd_s = float(smd.score(state_deg))
            rnd_s = float(rnd.score(last_emb)) if last_emb is not None else 0.0
            comb = combined_score(smd_s, rnd_s, last_act, norm) if last_emb is not None else 0.0
            st.log_observation(state_deg.tolist()).log_action(a_deg.tolist())
            st.log_uncertainty(smd=smd_s, rnd=rnd_s, ace=last_act)
        for cam in CAMS:
            img = env.render(cam, 320, 240)
            rel = f"{run.run_id}/{cam}_{f:03d}.jpg"
            Image.fromarray(img).save(blob_root / rel, quality=80)
            run.log_frame(camera=cam, blob_path=rel, step_idx=f, t=float(f))
        env.step(a_deg.astype(np.float64) * DEG2RAD, n_substeps=18)
        if args.pace > 0:
            time.sleep(args.pace)
        if _cube_in_tray(env.cube_pos(), TRAY_XY):
            break   # placed it — let success end the rollout
        # MONITOR: flag the predicted failure but DON'T halt — run to the real end so we can
        # compare the prediction against ground truth (detection accuracy).
        if predicted_at is None and f >= WARMUP and last_emb is not None and comb > threshold:
            predicted_at = f
            print(f"⚠ failure predicted @ frame {f}: combined {comb:.1f} > threshold {threshold:.1f}", flush=True)

    success = _cube_in_tray(env.cube_pos(), TRAY_XY)
    run.finish(success_rate=float(success))
    verdict = ("correct catch (true positive)" if (predicted_at is not None and not success)
               else "false alarm (false positive)" if predicted_at is not None
               else "missed (false negative)" if not success else "nominal (true negative)")
    print(f">>> actual={'success' if success else 'FAIL'} · "
          f"predicted={'fail@'+str(predicted_at) if predicted_at is not None else 'none'} · {verdict}", flush=True)


if __name__ == "__main__":
    main()
