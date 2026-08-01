"""Watch the scripted SO-101 pick-place live in the MuJoCo viewer.

    PYTHONPATH=. .venv-viewer/bin/mjpython scripts/watch_demos.py [--episodes N]

Opens a MuJoCo window and runs the scripted controller repeatedly with the brick
at randomized start positions, so you can see the episode creation in play. Watch
mode does NOT render the offscreen cameras (keeps the viewer smooth); recording the
dataset is a separate headless pass. Close the window to exit.
"""

import argparse
import time

import numpy as np

import mujoco
import mujoco.viewer

from drishti.sim.scripted_pick import IK, run_episode
from drishti.sim.so101_env import SO101Env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--pace", type=float, default=0.015, help="sleep per control step (watchable speed)")
    args = ap.parse_args()

    env = SO101Env(n_substeps=20)
    ik = IK()
    rng = np.random.default_rng(1)
    print(">>> opening viewer — watch the SO-101 pick the brick into the tray. Close to exit.", flush=True)
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        def on_step(_env):
            v.sync()
            time.sleep(args.pace)

        ep = 0
        ok = 0
        while v.is_running() and ep < args.episodes:
            cx = float(rng.uniform(0.24, 0.28))   # capped to clear the tray zone (starts at x=0.295)
            cy = float(rng.uniform(-0.03, 0.03))
            _, success = run_episode(env, ik, record=False, render_cams=(),
                                     on_step=on_step, cube_xy=(cx, cy))
            ok += int(success)
            print(f"episode {ep}: brick=({cx:.2f},{cy:.2f}) success={success}  [{ok}/{ep+1}]", flush=True)
            ep += 1
            time.sleep(0.4)
    print(">>> done", flush=True)


if __name__ == "__main__":
    main()
