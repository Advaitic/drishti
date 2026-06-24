"""Open-loop replay driver — the Stage-A end-to-end proof.

Streams a LeRobotDataset episode through an instrumented SmolVLA: each frame is a
real forward pass, so the hooks capture real vision/action internals, and we
compute the SMD / RND-OE / ACE uncertainty cascade per step and emit it as
telemetry (frames to a blob dir). Open-loop (we don't execute the actions), so it
needs no sim — it proves "we see inside a running policy and turn its internals
into a live uncertainty signal" on real robot frames.

    .venv/bin/python -m drishti.drivers.replay --episode 0 --steps 40 --ace-b 6

Calibration: SMD is fit on the dataset's state vectors (free); RND is fit on
vision embeddings gathered over a short calibration pass of real forwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from drishti import start_run
from drishti.policy_smolvla import load_smolvla
from drishti.sdk import current_step
from drishti.uq import RND, SMD, ace_score, sample_action_chunks

DATASET = "lerobot/svla_so101_pickplace"
BLOB_DIR = Path("data/obs_blobs")

# SmolVLA base expects observation.images.camera1/2/3; this dataset ships up/side.
# Map them onto the model's expected camera slots (the 3rd is auto-padded/masked
# by SmolVLA.prepare_images).
CAMERA_MAP = {
    "observation.images.up": "observation.images.camera1",
    "observation.images.side": "observation.images.camera2",
}


def _episode_range(ds, ep: int) -> range:
    try:
        lo = int(ds.episode_data_index["from"][ep])
        hi = int(ds.episode_data_index["to"][ep])
        return range(lo, hi)
    except Exception:
        # fallback: scan for matching episode_index
        idx = [i for i in range(len(ds)) if int(ds[i]["episode_index"]) == ep]
        return range(idx[0], idx[-1] + 1) if idx else range(0, min(len(ds), 50))


def _obs_batch(item: dict, device) -> dict:
    batch = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            batch[CAMERA_MAP.get(k, k)] = v.unsqueeze(0).to(device)
    batch["task"] = item.get("task", "")
    return batch


def _save_frame(item: dict, cam_key: str, path: Path) -> bool:
    t = item.get(cam_key)
    if not isinstance(t, torch.Tensor):
        return False
    arr = t.detach().cpu().numpy()  # (3, H, W) float[0,1]
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.max() <= 1.0 + 1e-3 else arr.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return True


@torch.inference_mode()
def _capture_embed(policy, proc):
    """Run one full forward and return the pooled vision embedding.

    Uses predict_action_chunk, NOT select_action: select_action pops a cached
    50-action queue and only runs the model when the queue is empty, so it fires
    the hooks once per ~50 calls. predict_action_chunk always runs the full model,
    so the hooks fire every call (what we need for per-step internals).
    """
    from drishti.sdk import _CTX

    class _Stub:
        def __init__(self):
            self.internals: dict = {}
    _CTX.step = _Stub()
    try:
        policy.predict_action_chunk(proc, noise=None)
        return getattr(_CTX.step, "_vision_embed", None)
    finally:
        _CTX.step = None


def run(
    dataset: str = DATASET,
    episode: int = 0,
    steps: int = 40,
    model_id: str = "lerobot/smolvla_base",
    ingest_url: str = "http://localhost:8010",
    ace_b: int = 6,
    calib_frames: int = 24,
    save_cam: str = "observation.images.up",
) -> str:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(dataset)
    policy, preprocess, postprocess, device = load_smolvla(model_id)
    rng = np.random.default_rng(0)

    # --- SMD: fit on state vectors (free, no forward) ---
    n_states = min(len(ds), 512)
    states = np.stack([np.asarray(ds[i]["observation.state"], dtype=float) for i in range(n_states)])
    smd = SMD.fit(states)
    print(f"[calib] SMD fit on {n_states} states (dim={states.shape[1]})", flush=True)

    ep = _episode_range(ds, episode)
    frame_ids = list(ep)[:steps]

    # --- RND: fit on vision embeddings from a short calibration pass ---
    calib_ids = list(range(0, min(calib_frames, len(ds))))
    embs = []
    for j, i in enumerate(calib_ids):
        emb = _capture_embed(policy, preprocess(_obs_batch(ds[i], device)))
        if emb is not None:
            embs.append(emb.numpy())
        if (j + 1) % 8 == 0:
            print(f"[calib] RND embeddings {j+1}/{len(calib_ids)}", flush=True)
    rnd = None
    if embs:
        E = np.stack(embs)
        rnd = RND(E.shape[1], device=str(device) if device.type != "mps" else "cpu").fit(E, epochs=10)
        print(f"[calib] RND fit on {len(embs)} embeddings (dim={E.shape[1]})", flush=True)

    # --- scored replay ---
    run = start_run(model_id, ds[frame_ids[0]].get("task", "replay"),
                    ingest_url=ingest_url, backend="in_process",
                    meta={"dataset": dataset, "episode": episode})
    print(f"[run] {run.run_id} replaying {len(frame_ids)} frames", flush=True)
    for k, i in enumerate(frame_ids):
        item = ds[i]
        proc = preprocess(_obs_batch(item, device))
        state = np.asarray(item["observation.state"], dtype=float)
        with run.step(step_idx=k, t=float(k)) as st:
            # ace_b full forwards: hooks fire each time (filling st.internals +
            # st._vision_embed), and the chunks feed ACE. First chunk = the action.
            chunks = sample_action_chunks(policy, proc, n=max(ace_b, 2))
            emb = getattr(st, "_vision_embed", None)
            st.log_observation(state.tolist())
            st.log_action(np.asarray(chunks[0, 0]).reshape(-1).tolist())
            st.log_uncertainty(
                smd=smd.score(state),
                rnd=(rnd.score(emb.numpy()) if (rnd is not None and emb is not None) else None),
                ace=ace_score(chunks),
            )
        rel = f"{run.run_id}/0_{k}_{save_cam.split('.')[-1]}.jpg"
        if _save_frame(item, save_cam, BLOB_DIR / rel):
            run.log_frame(camera=save_cam.split(".")[-1], blob_path=rel, step_idx=k, t=float(k))
        if (k + 1) % 5 == 0:
            print(f"[run] step {k+1}/{len(frame_ids)}", flush=True)
    run.finish()
    print(f"[run] done -> {run.run_id}", flush=True)
    return run.run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--ace-b", type=int, default=6)
    ap.add_argument("--calib-frames", type=int, default=24)
    ap.add_argument("--ingest-url", default="http://localhost:8010")
    ap.add_argument("--model-id", default="lerobot/smolvla_base")
    args = ap.parse_args()
    run(dataset=args.dataset, episode=args.episode, steps=args.steps, ace_b=args.ace_b,
        calib_frames=args.calib_frames, ingest_url=args.ingest_url, model_id=args.model_id)


if __name__ == "__main__":
    main()
