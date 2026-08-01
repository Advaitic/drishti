"""Convert recorded scripted demos (data/demos/ep_*.npz) into a LeRobotDataset.

The npz demos store joint state/action in RADIANS (MuJoCo units); the LeRobot / SmolVLA
convention is DEGREES with joint-position names, so we convert (x 180/pi). Images (up/side/
wrist) are encoded as video. The result is a self-consistent dataset SmolVLA can finetune on.

Language: each frame gets a sub-goal caption from `drishti.sim.phase_captions` keyed to its
frame index within the episode (reach+grasp / lift+carry / lower+release+retreat), instead of
one static sentence for the whole dataset -- see `phase_captions.py` for why the captions are
grouped into 3 macro-phases. Set --static-task to reproduce the old single-sentence behavior
(useful for an apples-to-apples finetune comparison).

    PYTHONPATH=. .venv/bin/python scripts/build_lerobot_dataset.py
    PYTHONPATH=. .venv/bin/python scripts/build_lerobot_dataset.py --static-task
    PYTHONPATH=. .venv/bin/python scripts/build_lerobot_dataset.py --root data/lerobot/so101_sim_pickplace_static --static-task

Verifies it loads as LeRobotDataset[0] at the end.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from drishti.sim.phase_captions import caption_for

REPO_ID = "drishti/so101_sim_pickplace"
ROOT = Path("data/lerobot/so101_sim_pickplace")
DEMOS = Path("data/demos")
STATIC_TASK = "Pick up the cube and place it in the box"
FPS = 30
RAD2DEG = 180.0 / np.pi

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
CAMS = ("up", "side", "wrist")
# shapes MUST be tuples — LeRobot's add_frame validator compares value.shape (tuple) with
# `!=` against the declared shape, so a list [6] never equals (6,) and validation fails.
IMG_HWC = (240, 320, 3)   # our recorded frame size (H, W, C)


def build(root: Path, static_task: bool, seed: int = 0):
    if root.exists():
        shutil.rmtree(root)

    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    }
    for c in CAMS:
        features[f"observation.images.{c}"] = {
            "dtype": "video", "shape": IMG_HWC, "names": ["height", "width", "channels"]}

    ds = LeRobotDataset.create(
        repo_id=REPO_ID, fps=FPS, features=features, root=root,
        robot_type="so101", use_videos=True,
    )

    rng = np.random.default_rng(seed)
    eps = sorted(DEMOS.glob("ep_*.npz"))
    print(f"converting {len(eps)} episodes -> {root}  (static_task={static_task})")
    for ep in eps:
        d = np.load(ep, allow_pickle=True)
        state = d["state"].astype(np.float32) * RAD2DEG       # (N,6) deg
        action = d["action"].astype(np.float32) * RAD2DEG     # (N,6) deg
        imgs = {c: d[c] for c in CAMS}                        # (N,H,W,3) uint8
        n = len(state)
        for i in range(n):
            task = STATIC_TASK if static_task else caption_for(i, rng)
            frame = {
                "action": action[i],
                "observation.state": state[i],
                "task": task,
            }
            for c in CAMS:
                frame[f"observation.images.{c}"] = imgs[c][i]
            ds.add_frame(frame)
        ds.save_episode()
        print(f"  {ep.name}: {n} frames")

    print("done writing. verifying load...")
    return ds


def verify(root: Path):
    ds = LeRobotDataset(REPO_ID, root=root)
    print(f"\n=== LeRobotDataset loaded ===")
    print(f"episodes: {ds.num_episodes}  frames: {ds.num_frames}  fps: {ds.fps}")
    sample = ds[0]
    print("sample[0] keys + shapes:")
    for k, v in sample.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else v
        print(f"  {k}: {shape}")
    # stats sanity
    st = ds.meta.stats
    print("\naction mean(deg):", np.round(np.asarray(st['action']['mean']), 2))
    print("action std (deg):", np.round(np.asarray(st['action']['std']), 2))
    print("state  mean(deg):", np.round(np.asarray(st['observation.state']['mean']), 2))
    print(f"\ntasks: {ds.meta.tasks}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(ROOT))
    ap.add_argument("--static-task", action="store_true",
                     help="write the old single fixed sentence instead of per-frame phase captions")
    ap.add_argument("--seed", type=int, default=0, help="paraphrase-sampling RNG seed")
    args = ap.parse_args()

    root = Path(args.root)
    build(root, static_task=args.static_task, seed=args.seed)
    verify(root)
