"""Standalone frozen vision embedder for RND — decoupled from the VLA.

The SmolVLA RND source hooked `vlm_with_expert.embed_image` (see `drishti/hooks.py`); MolmoAct2 is a
different architecture, so instead of hooking its internals we run a small frozen SigLIP over the two
sim frames and concatenate the pooled features. This keeps RND fully model-agnostic — the same detector
works for any VLA — at the cost of not reusing the VLA's own perception. If separation is poor (measured
via `calibrate.py`'s per-signal AUC), the fallback is to hook MolmoAct2's vision tower instead.

Returns an `embed(up_img, side_img) -> np.ndarray` callable producing a fixed-length vector that RND fits
on (per-dim whitening inside `RND.fit` handles the raw feature scales).
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import numpy as np

DEFAULT_EMBED_MODEL = "google/siglip-base-patch16-224"
CAMS = ("up", "side")


def _device(pref: Optional[str] = None):
    import torch
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@functools.lru_cache(maxsize=1)
def load_rnd_embedder(model_id: str = DEFAULT_EMBED_MODEL, device: Optional[str] = None) -> Callable:
    """Load a frozen SigLIP vision tower -> `embed(up_img, side_img) -> np.ndarray` (concat pooled feats)."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    dev = _device(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()

    @torch.no_grad()
    def embed(up_img, side_img) -> np.ndarray:
        parts = []
        for img in (up_img, side_img):
            pil = Image.fromarray(np.asarray(img, dtype=np.uint8))
            inputs = processor(images=pil, return_tensors="pt").to(dev)
            feats = model.get_image_features(**inputs)          # (1, D) tensor, or an output object (tf>=4.57)
            feats = getattr(feats, "pooler_output", feats)      # unwrap BaseModelOutputWithPooling if returned
            parts.append(feats.float().cpu().numpy().reshape(-1))
        return np.concatenate(parts).astype(np.float32)         # (2D,)

    return embed
