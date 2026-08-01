"""Record scripted SO-101 pick-place demos, optionally watching live in the viewer.

    drishti record --episodes 150                      # headless, fast
    mjpython -m drishti.sim.record --episodes 50       # live MuJoCo window (macOS)

Runs the scripted top-down pick-place repeatedly, with the cube placed in three lanes —
left / centre / right (round-robin, equal counts) — to capture varied approach modes.
Each successful episode is saved to data/demos/ep_XXX.npz (state, action, and up/side/wrist
camera frames). A manifest.json indexes them.

Live mode opens a MuJoCo window and paces in real time (needs the viewer venv's `mjpython`
on macOS). --headless skips the window and real-time pacing entirely — much faster for bulk
generation, and works in the regular .venv since no GUI is involved. The LeRobot-dataset
conversion is a separate pass either way.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import mujoco

from drishti.sim.scripted_pick import run_episode
from drishti.sim.so101_env import SO101Env

# Three lanes across the reachable band (y = left/right axis, x = reach). Equal counts.
LANES = {
    "left":   (0.012, 0.030),
    "center": (-0.012, 0.012),
    "right":  (-0.030, -0.012),
}
LANE_ORDER = ["left", "center", "right"]
X_RANGE = (0.24, 0.26)   # capped: tray near-wall renders at x~0.287-0.293 (visual only, contype=0) and the
                          # success-oracle box starts at x>0.295 (dx<0.045 from tray center 0.34) -- 0.26 keeps
                          # the cube visibly clear of the tray AND >3cm outside the success box at spawn
CAMS = ("up", "side", "wrist")
RENDER_SIZE = (320, 240)


def _record_loop(env, args, is_running) -> list[dict]:
    """Shared recording loop; `is_running()` lets the live viewer stop early on window close."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(getattr(args, "seed", 7))
    manifest = []

    saved = 0
    while saved < args.episodes and is_running():
        lane = LANE_ORDER[saved % 3]
        ylo, yhi = LANES[lane]
        success = False
        for _ in range(args.retries):
            if not is_running():
                break
            cx = float(rng.uniform(*X_RANGE))
            cy = float(rng.uniform(ylo, yhi))
            traj, success = run_episode(
                env, record=True, render_cams=CAMS, render_size=RENDER_SIZE,
                on_step=args.on_step, cube_xy=(cx, cy))
            if success:
                break
            print(f"   retry (lane={lane}, cube=({cx:.3f},{cy:+.3f}) missed)", flush=True)

        if not success:
            print(f"!! lane {lane} slot {saved} failed after {args.retries} tries", flush=True)
            continue

        fname = out / f"ep_{saved:03d}.npz"
        np.savez_compressed(
            fname,
            state=np.asarray(traj["state"], np.float32),
            action=np.asarray(traj["action"], np.float32),
            up=np.asarray(traj["images"]["up"], np.uint8),
            side=np.asarray(traj["images"]["side"], np.uint8),
            wrist=np.asarray(traj["images"]["wrist"], np.uint8),
            cube_xy=np.asarray([cx, cy], np.float32),
            lane=lane,
            success=True,
        )
        n = len(traj["state"])
        manifest.append({"idx": saved, "file": fname.name, "lane": lane,
                         "cube_xy": [round(cx, 4), round(cy, 4)], "frames": n})
        saved += 1
        print(f"[{saved}/{args.episodes}] lane={lane:6s} cube=({cx:.3f},{cy:+.3f}) "
              f"frames={n} -> {fname.name}", flush=True)

    (out / "manifest.json").write_text(json.dumps(
        {"episodes": manifest, "cams": list(CAMS), "render_size": list(RENDER_SIZE),
         "lanes": {k: list(v) for k, v in LANES.items()}}, indent=2))
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", type=str, default="data/demos")
    ap.add_argument("--pace", type=float, default=0.002, help="sleep per control tick (live mode only)")
    ap.add_argument("--retries", type=int, default=3, help="re-tries per lane slot on failure")
    ap.add_argument("--seed", type=int, default=7, help="cube-placement seed")
    ap.add_argument("--headless", action="store_true",
                     help="skip the live viewer window and real-time pacing (fast bulk generation)")
    args = ap.parse_args(argv)

    env = SO101Env()

    if args.headless:
        print(f">>> recording {args.episodes} demos headless (lanes: left/center/right).", flush=True)
        args.on_step = None
        manifest = _record_loop(env, args, is_running=lambda: True)
    else:
        import mujoco.viewer

        print(f">>> recording {args.episodes} demos (lanes: left/center/right). "
              f"Watch the window; close it to stop early.", flush=True)
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            def on_step(_env):
                v.sync()
                time.sleep(args.pace)
            args.on_step = on_step
            manifest = _record_loop(env, args, is_running=v.is_running)

    counts = {l: sum(1 for m in manifest if m["lane"] == l) for l in LANE_ORDER}
    print(f">>> done. saved {len(manifest)} demos to {args.out}/  lanes={counts}", flush=True)


if __name__ == "__main__":
    main()
