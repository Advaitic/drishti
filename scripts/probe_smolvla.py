"""Introspect SmolVLA + a LeRobot dataset so the replay driver builds the right batch.

Prints: model input/output features, dataset features, a sample item's shapes,
then runs ONE instrumented forward (checks hooks captured real internals) and one
ACE sampling pass. Run after the background download finishes:

    .venv/bin/python scripts/probe_smolvla.py
"""

from __future__ import annotations

import tempfile

import torch

from drishti import start_run
from drishti.policy_smolvla import load_smolvla
from drishti.store import Store
from drishti.uq.ace import ace_score, sample_action_chunks

DATASET = "lerobot/svla_so101_pickplace"


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("== loading dataset ==", flush=True)
    ds = LeRobotDataset(DATASET)
    print("dataset features:")
    for k, v in ds.meta.features.items():
        print(f"  {k:32} {v.get('dtype'):10} shape={v.get('shape')}")
    item = ds[0]
    print("\nsample item ds[0]:")
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:32} tensor {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:32} {type(v).__name__}: {str(v)[:48]}")

    print("\n== loading SmolVLA ==", flush=True)
    policy, preprocess, postprocess, device = load_smolvla()
    cfg = policy.config
    print(f"device={device}")
    print("input_features:")
    for k, ft in cfg.input_features.items():
        print(f"  {k:32} {ft}")
    print("output_features:")
    for k, ft in cfg.output_features.items():
        print(f"  {k:32} {ft}")
    print(f"max_state_dim={getattr(cfg,'max_state_dim',None)} "
          f"max_action_dim={getattr(cfg,'max_action_dim',None)} "
          f"chunk_size={getattr(cfg,'chunk_size',None)} "
          f"n_action_steps={getattr(cfg,'n_action_steps',None)}")

    # build an observation batch from the dataset frame (add batch dim, to device)
    batch = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
    batch["task"] = item.get("task", "pick up the object")
    print("\n== one instrumented forward ==", flush=True)
    s = Store(f"sqlite:///{tempfile.mkdtemp()}/probe.db")
    run = start_run("smolvla_base", "probe", store=s)
    with run.step(step_idx=0) as st:
        proc = preprocess(batch)
        with torch.inference_mode():
            action = policy.select_action(proc)
        print("action:", tuple(action.shape), action.dtype)
        print("captured internals keys:", sorted(st.internals.keys()))
        print("  vision_embed_norm:", st.internals.get("vision_embed_norm"))
        ah = st.internals.get("action_head_mean")
        print("  action_head_mean dim:", None if ah is None else len(ah))
        print("  _vision_embed:", getattr(st, "_vision_embed", None) is not None and
              tuple(st._vision_embed.shape))
    run.finish()

    print("\n== ACE sampling (B=4) ==", flush=True)
    chunks = sample_action_chunks(policy, preprocess(batch), n=4)
    print("chunks shape:", chunks.shape, "ace_score:", round(ace_score(chunks), 3))
    print("\nPROBE OK")


if __name__ == "__main__":
    main()
