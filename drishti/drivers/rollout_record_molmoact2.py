"""Stage-1 recorder: run MolmoAct2-SO100_101 closed-loop in the SO-101 sim and log the telemetry that
`drishti/calibrate.py` needs to fit the uncertainty threshold — per-frame proprio state (degrees), the
SigLIP RND embedding (`embed_###.npy`), and the intrinsic **ACE** 'act' signal (stored as a 1-vector
`action_head_std=[ace]` so `calibrate._act_spread` reads it, exactly matching what the live server scores).

Also the **R4 joint-convention gate** via `--smoke`: step MolmoAct2 a few frames and save the overhead
view so you can eyeball that the arm moves TOWARD the cube. If it moves the wrong way, flip
`JOINT_SIGNS` / `JOINT_OFFSETS` in `drishti/policy_molmoact2.py`.

Runs on the pod (CUDA + `MUJOCO_GL=egl`):
    PYTHONPATH=. MUJOCO_GL=egl python -m drishti.drivers.rollout_record_molmoact2 --smoke
    PYTHONPATH=. MUJOCO_GL=egl python -m drishti.drivers.rollout_record_molmoact2 --episodes 30
    # then:  python -m drishti.calibrate --auto     (fits SMD+RND, sets the 90th-pct threshold)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from drishti import start_run
from drishti.sim.so101_env import SO101Env
from drishti.sim.rollout_server import ACE_RANGE, CAMS_VLA, DEFAULT_TASK, N_SUBSTEPS, env_deg
from drishti.sim.scripted_pick import _grasp_config, _cube_in_tray, TRAY_XY
from drishti.uq import ace_score

BLOB_DIR = Path("data/obs_blobs")
CAMS = ("up", "side", "wrist")


def _first_action(chunk) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=float)
    return chunk.reshape(-1, chunk.shape[-1])[0][:6]


def record_episode(vla, embedder, env, task, cube_xy, *, ingest, ace_b, max_frames):
    _grasp_config(env, cube_xy)
    run = start_run("MolmoAct2-SO100_101", task, ingest_url=ingest, backend="in_process",
                    meta={"sim": "so101", "policy": "MolmoAct2-SO100_101",
                          "cube_xy": [round(cube_xy[0], 4), round(cube_xy[1], 4)]})
    bdir = BLOB_DIR / run.run_id
    bdir.mkdir(parents=True, exist_ok=True)
    for f in range(max_frames):
        f0, f1 = env.render(CAMS_VLA[0], 320, 240), env.render(CAMS_VLA[1], 320, 240)
        state_rad = env.state()
        with run.step(step_idx=f, t=float(f)) as st:
            action = _first_action(vla.predict([f0, f1], task, state_rad))
            samples = vla.sample([f0, f1], task, state_rad, ace_b)
            ace = float(ace_score(samples, value_range=ACE_RANGE))
            emb = np.asarray(embedder(f0, f1), dtype=np.float32)
            st.log_observation(env_deg(state_rad).tolist()).log_action(action.tolist())
            st.internals["action_head_std"] = [ace]      # the intrinsic 'act' signal calibrate.py fits on
            st.log_uncertainty(ace=ace)
            np.save(bdir / f"embed_{f:03d}.npy", emb)     # RND source (SigLIP), where calibrate looks
        env.step(action.astype(np.float64), n_substeps=N_SUBSTEPS)
        for cam in CAMS:
            rel = f"{run.run_id}/{cam}_{f:03d}.jpg"
            Image.fromarray(env.render(cam, 320, 240)).save(BLOB_DIR / rel, quality=80)
            run.log_frame(camera=cam, blob_path=rel, step_idx=f, t=float(f))
        if _cube_in_tray(env.cube_pos(), TRAY_XY):
            break
    success = bool(_cube_in_tray(env.cube_pos(), TRAY_XY))
    run.finish(success_rate=float(success))
    return success, run.run_id


def _smoke(vla, task):
    """R4 gate: step MolmoAct2 a few frames, save the overhead view, report motion + success."""
    env = SO101Env()
    _grasp_config(env, (0.26, 0.0))
    out = Path("data/molmoact2_smoke")
    out.mkdir(parents=True, exist_ok=True)
    start = env.state().copy()
    for f in range(20):
        Image.fromarray(env.render("up", 640, 480)).save(out / f"f{f:02d}.png")
        action = _first_action(vla.predict([env.render(CAMS_VLA[0], 320, 240),
                                            env.render(CAMS_VLA[1], 320, 240)], task, env.state()))
        env.step(action.astype(np.float64), n_substeps=N_SUBSTEPS)
    moved = float(np.linalg.norm(env.state() - start))
    print(f"smoke: cube={np.round(env.cube_pos(), 3)} success={_cube_in_tray(env.cube_pos(), TRAY_XY)} "
          f"joint_motion={moved:.3f} rad")
    print(f"saved 20 frames to {out} — verify the arm moves TOWARD the cube. If it moves AWAY, "
          f"flip JOINT_SIGNS/JOINT_OFFSETS in drishti/policy_molmoact2.py (R4).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=110)
    ap.add_argument("--ace-b", type=int, default=6)
    ap.add_argument("--ingest", default="http://localhost:8010")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 short rollout + saved frames (R4 gate)")
    args = ap.parse_args()

    from drishti.policy_molmoact2 import load_molmoact2
    vla = load_molmoact2()
    if args.smoke:
        _smoke(vla, args.task)
        return

    from drishti.uq.embed_frames import load_rnd_embedder
    embedder = load_rnd_embedder()
    rng = np.random.default_rng(args.seed)
    n_succ = 0
    for i in range(args.episodes):
        cxy = (float(rng.uniform(0.24, 0.28)), float(rng.uniform(-0.03, 0.03)))
        succ, rid = record_episode(vla, embedder, SO101Env(), args.task, cxy,
                                   ingest=args.ingest, ace_b=args.ace_b, max_frames=args.max_frames)
        n_succ += int(succ)
        print(f"[{i+1}/{args.episodes}] {rid} success={succ} rate={n_succ}/{i+1}", flush=True)
    print(f"done: {n_succ}/{args.episodes} success. Next: python -m drishti.calibrate --auto")


if __name__ == "__main__":
    main()
