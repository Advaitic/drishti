"""Checkpoint shootout: evaluate finetuned SmolVLA checkpoints on the SO-101 pick-place sim.

Closed-loop rollout — the policy drives the arm from the SAME locked-wrist top-down init the
training demos began from (via _grasp_config), reading 3 cameras (up/side/wrist -> camera1/2/3)
+ proprioceptive state. Action chain mirrors training: state radians ->deg (model units);
model action ->deg, then xpi/180 -> radian actuator targets. Success = cube ends in the tray.

Runs the SAME randomized cube positions across every checkpoint for a fair comparison, so we
can pick the best GENERALIZER (often not the final step).

    PYTHONPATH=. .venv/bin/python scripts/eval_policy.py --episodes 12 \
        --checkpoints data/checkpoints/00{1500,3000,4500,6000}/pretrained_model --render
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from drishti.sim.so101_env import SO101Env
from drishti.sim.scripted_pick import _grasp_config, _cube_in_tray, TRAY_XY

DEG2RAD = np.pi / 180.0
from drishti.sim.task import DEFAULT_TASK as TASK


def load_policy(ckpt, dev):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    policy = SmolVLAPolicy.from_pretrained(ckpt).to(dev).eval()
    pre, post = make_pre_post_processors(
        policy.config, ckpt,
        preprocessor_overrides={"device_processor": {"device": str(dev)}})
    return policy, pre, post


def build_obs(env, dev):
    def chw(cam):
        img = env.render(cam, 320, 240)  # HWC uint8
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()).float().div(255).unsqueeze(0).to(dev)
    state_deg = env.state().astype(np.float32) / DEG2RAD  # radians -> degrees (model units)
    return {
        "observation.images.camera1": chw("up"),
        "observation.images.camera2": chw("side"),
        "observation.images.camera3": chw("wrist"),
        "observation.state": torch.from_numpy(state_deg).unsqueeze(0).to(dev),
        "task": TASK,
    }


def rollout(env, policy, pre, post, dev, cube_xy, max_frames=130, n_sub=18, record=False):
    _grasp_config(env, cube_xy)   # locked-wrist top-down init == training episode start
    policy.reset()
    frames = []
    for f in range(max_frames):
        obs = build_obs(env, dev)
        with torch.no_grad():
            action = post(policy.select_action(pre(obs)))
        a = action.squeeze(0).detach().float().cpu().numpy().astype(np.float64)[:6] * DEG2RAD
        env.step(a, n_substeps=n_sub)
        if record and f % 4 == 0:
            frames.append(env.render("up", 320, 240))
        if _cube_in_tray(env.cube_pos(), TRAY_XY):
            return True, env.cube_pos().copy(), frames
    return _cube_in_tray(env.cube_pos(), TRAY_XY), env.cube_pos().copy(), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--render", action="store_true", help="save an up-cam montage of ep0 per checkpoint")
    ap.add_argument("--nas", type=int, default=1,
                    help="n_action_steps for receding horizon: 1 = re-plan every step "
                         "(open-loop 50-step chunks drift and fail on this BC policy)")
    args = ap.parse_args()

    dev = (torch.device(args.device) if args.device else
           torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    env = SO101Env()
    rng = np.random.default_rng(args.seed)
    # x capped at 0.28 (not 0.31): above ~0.295 the cube spawns inside the tray zone → instant
    # trivial success that inflates the eval score. Matches live_fleet / rollout_record.
    positions = [(float(rng.uniform(0.24, 0.28)), float(rng.uniform(-0.03, 0.03)))
                 for _ in range(args.episodes)]
    print(f"device={dev}  episodes={args.episodes}  eval-seed={args.seed}\n", flush=True)

    out = Path("data/eval"); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for ckpt in args.checkpoints:
        tag = Path(ckpt).parent.name
        policy, pre, post = load_policy(ckpt, dev)
        policy.config.n_action_steps = args.nas   # receding horizon
        ok = 0
        for i, (cx, cy) in enumerate(positions):
            success, fc, frames = rollout(env, policy, pre, post, dev, (cx, cy),
                                          record=(args.render and i == 0))
            ok += int(success)
            print(f"  [{tag}] ep{i:02d} cube=({cx:.3f},{cy:+.3f}) -> {'OK ' if success else 'miss'} "
                  f"final={np.round(fc, 3)}", flush=True)
            if frames:
                n = len(frames); idxs = np.linspace(0, n - 1, min(8, n)).astype(int)
                sheet = Image.new("RGB", (320 * len(idxs), 240), (15, 15, 17))
                for j, idx in enumerate(idxs):
                    sheet.paste(Image.fromarray(frames[idx]), (j * 320, 0))
                sheet.save(out / f"{tag}_ep0.png")
        results[tag] = ok / args.episodes
        print(f"  ==> {tag}: {ok}/{args.episodes} = {ok / args.episodes * 100:.0f}%\n", flush=True)
        del policy
        if dev.type == "mps":
            torch.mps.empty_cache()

    print("=== CHECKPOINT SHOOTOUT (success on fresh positions) ===")
    for tag, sr in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {sr * 100:5.0f}%   {tag}")
    import sys
    sys.stdout.flush()
    os._exit(0)   # skip mujoco's noisy EGL __del__ teardown on headless boxes


if __name__ == "__main__":
    import os
    main()
