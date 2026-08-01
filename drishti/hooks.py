"""Forward hooks — the differentiated capability.

`instrument(policy)` registers PyTorch hooks that capture, per forward pass:
  - the **vision embedding** (SigLIP image features) — norm/std summaries for the
    dashboard, plus a pooled vector stashed on the current step for RND-OE scoring
  - the **action head** output (pre-normalization action / flow-matching velocity)
    — per-dim mean/std summaries

Targets for SmolVLA (verified against lerobot 0.4.4):
  vision = model.vlm_with_expert.embed_image (a method → wrapped)
  head   = model.action_out_proj            (nn.Linear → forward hook)

Falls back to a generic locator (named action-head attrs, then the last Linear
projecting to the action dim) so the same call works on ACT / other policies.
Hooks attribute to `drishti.current_step()`; they never raise into the model.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .sdk import current_step

# common action-head attribute names across VLA families
_HEAD_ATTRS = ("action_out_proj", "action_head", "action_out")


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
    """Walk a hook output (Tensor / tuple / dict / HF ModelOutput) for the first tensor."""
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _action_dim(policy: Any) -> Optional[int]:
    cfg = getattr(policy, "config", None)
    for attr in ("max_action_dim", "action_dim"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int):
            return v
    try:
        return policy.config.output_features["action"].shape[0]
    except Exception:
        return None


def _find_action_head(policy: Any, model: nn.Module) -> Optional[nn.Module]:
    for attr in _HEAD_ATTRS:
        mod = getattr(model, attr, None)
        if isinstance(mod, nn.Module):
            return mod
    adim = _action_dim(policy)
    linears = [m for m in model.modules()
               if isinstance(m, nn.Linear) and (adim is None or m.out_features == adim)]
    return linears[-1] if linears else None


def _record_vision(output: Any) -> None:
    s = current_step()
    if s is None:
        return
    try:
        emb = _first_tensor(output)
        if emb is None:
            return
        with torch.no_grad():
            e = emb.detach().float()
            s.internals["vision_embed_norm"] = float(e.norm(dim=-1).mean().item())
            s.internals["vision_embed_std"] = float(e.std().item())
            # pooled (D,) over all but the feature axis — ONE vector per camera.
            # embed_image fires once per camera within a forward pass; accumulate EVERY
            # camera and concat, so RND-OE scoring sees the whole scene (up + side + wrist).
            # An overwrite would keep only the last camera (the wrist close-up) and leave
            # RND blind to the workspace views that actually show the cube/tray.
            pooled = e.reshape(-1, e.shape[-1]).mean(0).cpu()
            parts = getattr(s, "_vision_embed_parts", None)
            if parts is None:
                parts = []
                s._vision_embed_parts = parts  # type: ignore[attr-defined]
            parts.append(pooled)
            s._vision_embed = torch.cat(parts)  # type: ignore[attr-defined]
    except Exception:
        pass


def _action_head_hook(module, inputs, output) -> None:
    s = current_step()
    if s is None:
        return
    try:
        out = _first_tensor(output)
        if out is None:
            return
        with torch.no_grad():
            flat = out.detach().float().reshape(-1, out.shape[-1])
            # flow-matching fires this hook once per denoise step; last write wins
            s.internals["action_head_mean"] = flat.mean(0).cpu().tolist()
            # std is undefined for a single row → report zeros rather than NaN
            std = flat.std(0) if flat.shape[0] > 1 else torch.zeros(flat.shape[-1])
            s.internals["action_head_std"] = std.cpu().tolist()
    except Exception:
        pass


def _wrap_embed_image(vmod: nn.Module) -> None:
    if getattr(vmod, "_drishti_wrapped", False):
        return
    orig = vmod.embed_image

    def wrapped(*a, **k):
        out = orig(*a, **k)
        _record_vision(out)
        return out

    vmod.embed_image = wrapped  # type: ignore[assignment]
    vmod._drishti_orig_embed_image = orig  # type: ignore[attr-defined]
    vmod._drishti_wrapped = True  # type: ignore[attr-defined]


def instrument(policy: Any) -> Any:
    """Register vision + action-head hooks on a policy. Idempotent. Returns the policy."""
    model = getattr(policy, "model", policy)
    handles = []

    head = _find_action_head(policy, model)
    if head is not None:
        handles.append(head.register_forward_hook(_action_head_hook))

    vmod = getattr(model, "vlm_with_expert", None)
    if vmod is not None and hasattr(vmod, "embed_image"):
        _wrap_embed_image(vmod)
    else:
        venc = _find_vision_module(model)
        if venc is not None:
            handles.append(venc.register_forward_hook(lambda m, i, o: _record_vision(o)))

    policy._drishti_handles = handles  # type: ignore[attr-defined]
    return policy


def _find_vision_module(model: nn.Module) -> Optional[nn.Module]:
    import re

    pat = re.compile(r"(vision|backbone|image|visual|vit|resnet|siglip|dino)", re.I)
    for name, mod in model.named_modules():
        if pat.search(type(mod).__name__) or pat.search(name.split(".")[-1]):
            return mod
    return None


def uninstrument(policy: Any) -> None:
    for h in getattr(policy, "_drishti_handles", []):
        h.remove()
    policy._drishti_handles = []  # type: ignore[attr-defined]
    model = getattr(policy, "model", policy)
    vmod = getattr(model, "vlm_with_expert", None)
    if vmod is not None and getattr(vmod, "_drishti_wrapped", False):
        vmod.embed_image = vmod._drishti_orig_embed_image
        vmod._drishti_wrapped = False
