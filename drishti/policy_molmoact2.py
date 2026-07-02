"""Load + adapt the MolmoAct2-SO100_101 VLA (Ai2) for the SO-101 MuJoCo sim.

MolmoAct2 is an HF `AutoModelForImageTextToText` (not a lerobot policy), so this does NOT reuse
`make_pre_post_processors`. It exposes `model.predict_action(...).actions` -> a (T, 6) chunk of **6-D
absolute joint pose** in the checkpoint's SO-100/101 (model) frame. We wrap it behind the small `VLA`
surface (`predict` / `sample` / `reasoning`) that `rollout_server` drives — the same surface
`drishti.sim.stub_vla.StubVLA` implements, so the server code is identical for stub vs. real model.

**Joint frame (R4 — confirmed).** The SO-101 arm reports joint angles in one convention; the public
LeRobot SO-100/101 data MolmoAct2 trained on uses another: a sign flip on shoulder_lift and a +90° offset
on shoulder_lift & elbow_flex (LeRobot's v3.0->v2.1 backwardcomp conversion). We hold that in exactly two
functions — `env_state_to_vla` / `vla_action_to_env`:
    state_model_frame = signs * arm_state_deg + offsets       (env rad -> deg -> model frame)
    action_env_frame  = ((model_action - offsets) * signs)    (-> deg -> env rad)
NO IK adapter — the 6-D output feeds `SO101Env.step(action6_rad)` directly.

**bf16 on a 24GB 4090 (required).** The published `modeling_molmoact2.py` hardcodes the action-flow
trajectory as float32 and calls `.numpy()` on a bf16 tensor, so bf16 inference crashes unpatched (and
float32 is ~26 GiB -> OOM). We snapshot the repo and patch those two sites in place (idempotent).

Frame convention, dtype patch, and the predict_action kwargs follow the community SO-101 implementation
by Irene Grace (github.com/irenegracekp/molmoact2-so101), adapted to our env units. Needs CUDA — the
Stage-0 tests use StubVLA instead.
"""

from __future__ import annotations

import functools
import os
import shutil
from typing import Any, Optional

import numpy as np

DEFAULT_MODEL_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
CAMS_VLA = ("up", "wrist")           # MolmoAct2 SO-100/101 wants [scene, wrist]; see rollout_server
DEG2RAD = np.pi / 180.0

# --- Joint-frame correction (R4). LeRobot v3.0->v2.1 backwardcomp for the SO-100/101 checkpoint. ---
JOINT_SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=float)      # flip shoulder_lift
JOINT_OFFSETS = np.array([0, 90, 90, 0, 0, 0], dtype=float)   # degrees; shift shoulder_lift & elbow_flex


def env_state_to_vla(state_rad) -> np.ndarray:
    """env joints (radians, our `so101_new_calib` frame) -> MolmoAct2 state (degrees, model frame)."""
    deg = np.asarray(state_rad, dtype=float).reshape(-1)[:6] / DEG2RAD
    return JOINT_SIGNS * deg + JOINT_OFFSETS


def vla_action_to_env(action_vla) -> np.ndarray:
    """MolmoAct2 6-D joint pose (degrees, model frame) -> env ctrl (radians). Inverse of the above."""
    a = np.asarray(action_vla, dtype=float).reshape(-1)[:6]
    return ((a - JOINT_OFFSETS) * JOINT_SIGNS) * DEG2RAD          # signs in {±1} so *signs == /signs


def _device(pref: Optional[str] = None):
    import torch
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# bf16/fp16 fixes for the published modeling_molmoact2.py (from the community SO-101 impl).
_DTYPE_PATCHES = [
    ("""            dtype=torch.float32,
            generator=generator,
        )""",
     """            dtype=next(action_expert.parameters()).dtype,  # patched
            generator=generator,
        )"""),
    ("        return value.detach().cpu().numpy().astype(np.float32, copy=False)",
     "        return value.detach().to(torch.float32).cpu().numpy()  # patched"),
]


def _patch_model_for_mixed_dtype(local_dir: str) -> None:
    """Edit the snapshot's modeling_molmoact2.py in place for bf16/fp16. Idempotent."""
    target = os.path.join(local_dir, "modeling_molmoact2.py")
    with open(target) as f:
        src = f.read()
    changed, missing = False, []
    for needle, repl in _DTYPE_PATCHES:
        if repl in src:
            continue
        if needle not in src:
            missing.append(needle.splitlines()[0].strip())
            continue
        src, changed = src.replace(needle, repl), True
    if missing:
        print(f"[MolmoAct] WARNING: dtype-patch needles not found: {missing!r}; bf16 may fail there.")
    if not changed:
        return
    with open(target, "w") as f:
        f.write(src)
    dyn = os.path.expanduser(
        f"~/.cache/huggingface/modules/transformers_modules/{os.path.basename(local_dir.rstrip('/'))}")
    if os.path.isdir(dyn):
        shutil.rmtree(dyn)
    print(f"[MolmoAct] Patched {target} for bf16/fp16 inference.")


def _patch_tokenizer_config(local_dir: str) -> None:
    """MolmoAct2 ships `extra_special_tokens` as a LIST, but transformers 4.57's
    `_set_model_specific_special_tokens` expects a name->token DICT (it calls `.keys()`), so the fast
    tokenizer crashes. Convert the list to a dict (sanitized names) in place. Idempotent."""
    import json
    import re
    p = os.path.join(local_dir, "tokenizer_config.json")
    cfg = json.load(open(p))
    est = cfg.get("extra_special_tokens")
    if not isinstance(est, list):
        return
    d: dict = {}
    for tok in est:
        name = re.sub(r"[^0-9a-zA-Z]+", "_", str(tok)).strip("_").lower() or "extra"
        while name in d:
            name += "_x"
        d[name] = tok
    cfg["extra_special_tokens"] = d
    if os.path.islink(p):
        os.remove(p)                                     # replace the snapshot symlink with a real file
    json.dump(cfg, open(p, "w"))
    print(f"[MolmoAct] Patched tokenizer_config.json extra_special_tokens list->dict ({len(d)} tokens).")


class MolmoAct2VLA:
    """Adapter exposing the `VLA` surface (predict / sample / reasoning) over the HF MolmoAct2 model.

    - `predict(images, task, state_rad) -> (T, 6)` env-unit joint-pose chunk (one flow solve).
    - `sample(images, task, state_rad, n) -> (n, T, 6)` env-unit chunks (n stochastic solves, for ACE).
    - `reasoning() -> str` last decoded reasoning trace (often '' with depth reasoning disabled).
    """

    def __init__(self, model: Any, processor: Any, device: Any, *, num_steps: int = 10,
                 enable_cuda_graph: bool = False):
        self.model = model
        self.processor = processor
        self.device = device
        self.num_steps = num_steps
        self.enable_cuda_graph = enable_cuda_graph
        self._reasoning = ""

    def _raw_predict(self, images, task, state_rad) -> np.ndarray:
        from PIL import Image
        pil = [im if hasattr(im, "size") else Image.fromarray(np.asarray(im, np.uint8)) for im in images]
        out = self.model.predict_action(
            processor=self.processor, images=pil, task=task,
            state=env_state_to_vla(state_rad), norm_tag=NORM_TAG,
            inference_action_mode="continuous", enable_depth_reasoning=False,
            num_steps=self.num_steps, normalize_language=True,
            enable_cuda_graph=self.enable_cuda_graph,
        )
        actions = out.actions if hasattr(out, "actions") else (out["actions"] if isinstance(out, dict) else out)
        if hasattr(actions, "detach"):
            import torch
            actions = actions.detach().to("cpu", dtype=torch.float32).numpy()
        actions = np.asarray(actions, dtype=float)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        actions = actions[:, :6]                                    # (T, 6) model-frame degrees
        self._reasoning = self._decode_reasoning(out)
        return np.stack([vla_action_to_env(a) for a in actions], axis=0)   # (T, 6) env rad

    @staticmethod
    def _decode_reasoning(out) -> str:
        for attr in ("reasoning", "trace", "text", "thinking"):
            v = getattr(out, attr, None) if not isinstance(out, dict) else out.get(attr)
            if isinstance(v, str) and v:
                return v
        return ""

    def predict(self, images, task, state_rad) -> np.ndarray:
        import torch
        with torch.inference_mode():
            return self._raw_predict(images, task, state_rad)

    def sample(self, images, task, state_rad, n: int = 6) -> np.ndarray:
        """n independent flow solves at the same obs (fresh noise each) -> (n, T, 6) for ACE."""
        import torch
        with torch.inference_mode():
            outs = [self._raw_predict(images, task, state_rad) for _ in range(int(n))]
        T = min(c.shape[0] for c in outs)
        return np.stack([c[:T] for c in outs], axis=0)

    def reasoning(self) -> str:
        return self._reasoning


@functools.lru_cache(maxsize=1)
def load_molmoact2(model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None,
                   dtype: Optional[str] = None, num_steps: int = 10) -> MolmoAct2VLA:
    """Load MolmoAct2-SO100_101. Snapshots the repo (for norm_stats.json), patches modeling for bf16,
    loads .to(cuda).eval(). `dtype` from arg or $DRISHTI_VLA_DTYPE (default bfloat16 ~12 GiB fits a 24GB
    4090; use float32 ~26 GiB on a >=40GB card to skip the bf16 patch). LRU-cached."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = dtype or os.environ.get("DRISHTI_VLA_DTYPE", "bfloat16")
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    local_dir = snapshot_download(model_id)
    _patch_tokenizer_config(local_dir)
    if dtype != "float32":
        _patch_model_for_mixed_dtype(local_dir)
    dev = _device(device)
    processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
    model = (AutoModelForImageTextToText
             .from_pretrained(local_dir, trust_remote_code=True, dtype=torch_dtype)
             .to(dev).eval())
    return MolmoAct2VLA(model, processor, dev, num_steps=num_steps)
