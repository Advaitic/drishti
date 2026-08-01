"""Ingest a generated AHA / rlbench-failgen dataset into the drishti Store.

FailGen writes RLBench-canonical demos:

    <root>/<task>/<label>/episodes/episode<N>/
        low_dim_obs.pkl            # pickled Demo == list[Observation]
        <cam>_rgb/0.png, 1.png ...  # per-camera RGB frames (e.g. front_rgb/, wrist_rgb/)

where <label> is `success` for nominal demos and `<fail_type>[_wp<k>]` for injected
failures (grasp, slip, translation_x, rotation_z, wrong_object, ...). We map each episode
to one Store run:

    * success_rate = 1.0 (success) | 0.0 (failure)   -> the binary label the AUC keys off
    * meta = {source:"aha", task, failure_type, fail_waypoint}
    * per step: observation_state = a FIXED proprioceptive vector (task-agnostic dim), so
      SMD sees the same features across tasks and does NOT get task_low_dim_state (which
      leaks ground-truth object pose — that would let the detector cheat)
    * per inference frame: a SigLIP embedding over the chosen cameras -> obs_blobs/<run>/embed_<f>.npy
      (load-bearing for RND; calibrate/eval silently skip runs without it)

Runs where `rlbench` is importable (the generation pod) since low_dim_obs.pkl unpickles
into rlbench.backend.observation.Observation. After ingest, pull data/obs.db + data/obs_blobs/
to the Mac for eval_aha.py / the dashboard.

    PYTHONPATH=. python scripts/ingest_aha.py --data /path/to/failgen_out --cams front wrist
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import time
from pathlib import Path

import numpy as np

from drishti.store import Store

# the 11 FailGen failure types (first token of a <label> dir marks a failure)
FAILURE_TYPES = {
    "grasp", "slip", "no_rotation", "wrong_sequence", "wrong_object",
    "rotation_x", "rotation_y", "rotation_z",
    "translation_x", "translation_y", "translation_z",
}
# Store the FULL proprioceptive superset (incl. force channels) so downstream ablations
# — e.g. SMD with vs without forces — are an offline column-slice, not a re-generation. The
# per-run field->offset map is written to meta.field_slices. task_low_dim_state stays EXCLUDED
# (it leaks ground-truth object pose). Pass --fields to override.
DEFAULT_FIELDS = ("joint_positions", "joint_velocities", "joint_forces",
                  "gripper_open", "gripper_joint_positions", "gripper_touch_forces")
BLOB_DIR = Path("data/obs_blobs")


# --------------------------------------------------------------------------- pure helpers
def parse_episode_path(root: Path, episode_dir: Path):
    """(task, failure_type|None, fail_wp|None, is_success) from <root>/<task>/<label>/episodes/episodeN."""
    rel = episode_dir.relative_to(root).parts        # (task, label, "episodes", "episodeN")
    task = rel[0]
    label = rel[1] if len(rel) > 1 else "success"
    tok = label.split("_wp")
    head = tok[0]                                     # label minus the _wp<k> suffix
    wp = int(tok[1]) if len(tok) > 1 and tok[1].isdigit() else None
    # head is the full failure type ("grasp", "translation_x", ...) for failures; anything
    # else ("success", "variation0", "nominal") is a success demo.
    if head in FAILURE_TYPES:
        return task, head, wp, False
    return task, None, None, True


def obs_to_state(obs, fields=DEFAULT_FIELDS):
    """Concatenate the selected fields of one rlbench Observation into a 1-D vector.

    Returns (vector, slices) where slices maps each present field -> [start, end] into the
    vector, so an ablation can select any subset offline by column-slicing.
    """
    parts, slices, i = [], {}, 0
    for f in fields:
        v = getattr(obs, f, None)
        if v is None:
            continue
        a = np.atleast_1d(np.asarray(v, dtype=float)).reshape(-1)
        slices[f] = [i, i + len(a)]
        i += len(a)
        parts.append(a)
    return (np.concatenate(parts) if parts else np.array([], dtype=float)), slices


def run_id_for(root: Path, episode_dir: Path) -> str:
    h = hashlib.sha1(str(episode_dir.relative_to(root)).encode()).hexdigest()[:12]
    return f"run_aha_{h}"


def find_episodes(root: Path):
    return sorted(p.parent for p in root.glob("*/*/episodes/episode*/low_dim_obs.pkl"))


# --------------------------------------------------------------------------- ingest one episode
def load_rgb(episode_dir: Path, cam: str, t: int):
    from PIL import Image
    p = episode_dir / f"{cam}_rgb" / f"{t}.png"
    if not p.exists():
        return None
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def ingest_episode(store, embed, root, episode_dir, cams, fields, save_frames, embed_stride=1):
    task, ftype, wp, is_success = parse_episode_path(root, episode_dir)
    rid = run_id_for(root, episode_dir)
    if store.get_run(rid):
        return rid, 0, True   # already ingested — skip
    with open(episode_dir / "low_dim_obs.pkl", "rb") as fh:
        demo = pickle.load(fh)          # list[Observation]

    _, slices = obs_to_state(demo[0], fields)     # field->offset map for offline ablation slicing
    now = time.time()
    store.apply_event({"type": "run_start", "ts": now, "payload": {
        "run_id": rid, "model_id": "aha-failgen", "task": task,
        "backend": "rlbench", "capture_mode": "offline", "started_at": now,
        "n_episodes": 1,
        "meta": {"source": "aha", "task": task, "failure_type": ftype, "fail_waypoint": wp,
                 "cams": list(cams), "state_fields": list(fields), "field_slices": slices},
    }})

    bdir = BLOB_DIR / rid
    bdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for t, obs in enumerate(demo):
        state, _ = obs_to_state(obs, fields)
        store.apply_event({"type": "step", "ts": now, "payload": {
            "run_id": rid, "ep_idx": 0, "step_idx": t, "t": float(t),
            "observation_state": state.tolist(),
            "action": state.tolist(),   # placeholder; smd/rnd don't consume action
        }})
        # embed only every `embed_stride` frames — RND scores carry forward from the last
        # embed frame (score_run_live_cadence), so sparse embeddings are fully supported and
        # cut the CPU SigLIP cost by `embed_stride`x.
        imgs = [load_rgb(episode_dir, c, t) for c in cams] if (t % embed_stride == 0) else [None]
        if all(im is not None for im in imgs):
            np.save(bdir / f"embed_{t:03d}.npy", embed(*imgs).astype(np.float32))
            if save_frames:   # keep raw frames at embed steps -> encoder swaps become offline re-embeds
                from PIL import Image
                for c, im in zip(cams, imgs):
                    Image.fromarray(im).save(bdir / f"{c}_{t:03d}.jpg", quality=90)
        n += 1

    store.apply_event({"type": "run_end", "ts": now, "payload": {
        "run_id": rid, "n_steps": n, "n_episodes": 1,
        "success_rate": 1.0 if is_success else 0.0, "ended_at": now}})
    return rid, n, False


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="FailGen output root (contains <task>/<label>/episodes/...)")
    ap.add_argument("--cams", nargs="+", default=["front", "wrist"], help="camera names -> <cam>_rgb/")
    ap.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS), help="Observation attrs for the state vector")
    ap.add_argument("--no-frames", action="store_true", help="skip saving RGB JPEGs + frame events (embeds only)")
    ap.add_argument("--embed-stride", type=int, default=1, help="embed every Nth frame (RND carries forward)")
    ap.add_argument("--limit", type=int, default=0, help="cap episodes (0 = all)")
    args = ap.parse_args()

    root = Path(args.data)
    episodes = find_episodes(root)
    if args.limit:
        episodes = episodes[:args.limit]
    if not episodes:
        raise SystemExit(f"no episodes under {root} (expected <task>/<label>/episodes/episode*/low_dim_obs.pkl)")

    from drishti.uq.embed_frames import load_rnd_embedder
    embed = load_rnd_embedder()
    store = Store()

    n_succ = n_fail = n_skip = 0
    for ep in episodes:
        rid, n, skipped = ingest_episode(store, embed, root, ep, args.cams, tuple(args.fields),
                                         not args.no_frames, args.embed_stride)
        _, ftype, _, is_success = parse_episode_path(root, ep)
        if skipped:
            n_skip += 1
        elif is_success:
            n_succ += 1
        else:
            n_fail += 1
        print(f"{'skip' if skipped else 'ok  '} {rid}  {ep.relative_to(root)}  steps={n}", flush=True)

    print(f"\ningested {n_succ} success + {n_fail} failure runs ({n_skip} already present)", flush=True)


if __name__ == "__main__":
    main()
