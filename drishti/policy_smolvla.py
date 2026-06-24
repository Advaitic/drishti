"""Load + instrument a SmolVLA policy (lerobot 0.4.4).

Mirrors the verified load pattern (from_pretrained + make_pre_post_processors).
The returned policy is already instrumented, so `select_action` /
`predict_action_chunk` fire the vision + action-head hooks. Lazy + cached so
swapping checkpoints later is one call.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Optional

import torch

from .hooks import instrument

DEFAULT_MODEL_ID = "lerobot/smolvla_base"


def _device(pref: Optional[str] = None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _activate_offline_if_cached(model_id: str) -> None:
    repo = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    if repo.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@functools.lru_cache(maxsize=2)
def load_smolvla(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    do_instrument: bool = True,
) -> tuple[Any, Any, Any, torch.device]:
    _activate_offline_if_cached(model_id)
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dev = _device(device)
    policy = SmolVLAPolicy.from_pretrained(model_id).to(dev).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, model_id,
        preprocessor_overrides={"device_processor": {"device": str(dev)}},
    )
    if do_instrument:
        instrument(policy)
    return policy, preprocess, postprocess, dev
