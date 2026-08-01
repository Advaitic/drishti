"""Standalone frozen vision embedder for RND — decoupled from the VLA.

The SmolVLA RND source hooked `vlm_with_expert.embed_image` (see `drishti/hooks.py`); MolmoAct2 is a
different architecture, so instead of hooking its internals we run a small frozen SigLIP over the two
sim frames and concatenate the pooled features. This keeps RND fully model-agnostic — the same detector
works for any VLA — at the cost of not reusing the VLA's own perception. If separation is poor (measured
via `calibrate.py`'s per-signal AUC), the fallback is to hook MolmoAct2's vision tower instead.

Returns an `embed(*imgs) -> np.ndarray` callable producing a fixed-length vector that RND fits
on (per-dim whitening inside `RND.fit` handles the raw feature scales). The number of cameras is
variadic: `embed(up, side)` yields (2*768,) as before, `embed(front, wrist, overhead)` yields
(3*768,) — the output dim is `n_cams * 768` and RND's `d_in` follows it automatically.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import numpy as np

DEFAULT_EMBED_MODEL = "google/siglip-base-patch16-224"

# Frozen encoders selectable for RND. Each entry says how to turn one image into a vector.
# All are used head-free (no classifier) — we pull raw features, never logits. Adding a new
# encoder here (e.g. VC-1 via its own package) is the single extension point.
ENCODERS = {
    "siglip":      {"model_id": "google/siglip-base-patch16-224", "kind": "siglip"},
    "dinov2-cls":  {"model_id": "facebook/dinov2-base",           "kind": "dino", "pool": "cls"},
    "dinov2-mean": {"model_id": "facebook/dinov2-base",           "kind": "dino", "pool": "mean"},
}


def _device(pref: Optional[str] = None):
    import torch
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@functools.lru_cache(maxsize=4)
def load_embedder(encoder: str = "siglip", device: Optional[str] = None) -> Callable:
    """Frozen vision encoder -> `embed(*imgs) -> np.ndarray` (per-image feature, concatenated over cameras).

    encoder in ENCODERS: 'siglip' (get_image_features), 'dinov2-cls' (CLS token),
    'dinov2-mean' (mean over patch tokens — better for small-object/spatial cues). Auto-selects
    CUDA when available, so the same call is GPU-fast on a pod and CPU-fine on a laptop.
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder {encoder!r}; choices: {sorted(ENCODERS)}")
    spec = ENCODERS[encoder]
    dev = _device(device)
    # image processor only (no tokenizer). use_fast=False -> PIL path, no sentencepiece needed.
    processor = AutoImageProcessor.from_pretrained(spec["model_id"], use_fast=False)
    model = AutoModel.from_pretrained(spec["model_id"]).to(dev).eval()
    kind, pool = spec["kind"], spec.get("pool")

    @torch.no_grad()
    def _feat(pil):
        inputs = processor(images=pil, return_tensors="pt").to(dev)
        if kind == "siglip":
            f = model.get_image_features(**inputs)               # (1, D) or output object (tf>=4.57)
            f = getattr(f, "pooler_output", f)
        else:  # dinov2: raw hidden states, no head
            h = model(**inputs).last_hidden_state                # (1, 1+num_patches, D)
            f = h[:, 0, :] if pool == "cls" else h[:, 1:, :].mean(dim=1)
        return f.float().cpu().numpy().reshape(-1)

    @torch.no_grad()
    def embed(*imgs) -> np.ndarray:
        if not imgs:
            raise ValueError("embed() needs at least one camera frame")
        parts = [_feat(Image.fromarray(np.asarray(im, dtype=np.uint8))) for im in imgs]
        return np.concatenate(parts).astype(np.float32)          # (n_cams * D,)

    return embed


def load_rnd_embedder(encoder: str = "siglip", device: Optional[str] = None) -> Callable:
    """The SigLIP embedder RND is fitted on. Pass another key from ENCODERS to swap it.

    Changing the encoder changes the embedding dimension, so a calibration fitted with one encoder
    cannot score with another — refit after switching.
    """
    return load_embedder(encoder, device)
