"""Compact, policy-agnostic uncertainty detection for the vla-corrector loop.

Fit on your policy's SUCCESSFUL rollouts only (no labels), then score any step to decide when the policy is
off-distribution and the corrector should intervene. Two detectors, combined:

  SMD  - Mahalanobis distance of the proprioceptive state from the mean successful pose (cheap, no model).
  RND  - Random Network Distillation residual on a frozen HuggingFace SigLIP embedding of the frame(s):
         a predictor MLP trained to match a frozen random target on successful views -> low residual on
         familiar views, high on novel ones. Model-agnostic (works for any policy; no hooks).

Usage:
    from uq import Calibrator
    cal = Calibrator().fit(success_states, success_frames)   # states (N,D); frames: list of (img, ...) per step
    u   = cal.score(state, frames)                           # {smd, rnd, combined, threshold, spike}

Deps (the `[uq]` extra): numpy, torch, transformers, pillow.
"""
from __future__ import annotations

import functools
from typing import Optional, Sequence

import numpy as np

SIGLIP = "google/siglip-base-patch16-224"


# --------------------------------------------------------------------------- SMD (proprio Mahalanobis)
class SMD:
    def __init__(self, mean, inv):
        self.mean, self.inv = np.asarray(mean, float), np.asarray(inv, float)

    @classmethod
    def fit(cls, states, eps: float = 1e-6) -> "SMD":
        X = np.asarray(states, float)
        if X.ndim == 1:
            X = X[:, None]
        cov = np.cov(X, rowvar=False)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        return cls(X.mean(0), np.linalg.inv(cov + eps * np.eye(cov.shape[0])))

    def score(self, x) -> float:
        d = np.asarray(x, float).reshape(-1) - self.mean
        return float(np.sqrt(max(d @ self.inv @ d, 0.0)))


# --------------------------------------------------------------------------- RND (vision novelty)
class RND:
    def __init__(self, d_in: int, d_out: int = 64, h: int = 128, device: str = "cpu"):
        import torch.nn as nn
        mk = lambda: nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, d_out)).to(device)
        self.device = device
        self.target = mk().eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = mk()
        self.mean = self.std = None

    def _white(self, x):
        return x if self.mean is None else (x - self.mean) / self.std

    def fit(self, embeddings, epochs: int = 8, lr: float = 1e-3, bs: int = 128) -> "RND":
        import torch
        X = torch.as_tensor(np.asarray(embeddings, np.float32), device=self.device)
        self.mean, self.std = X.mean(0), X.std(0) + 1e-6         # per-dim whitening (biggest RND lever)
        X = self._white(X)
        opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.predictor.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), bs):
                xb = X[perm[i:i + bs]]
                with torch.no_grad():
                    tb = self.target(xb)
                loss = ((self.predictor(xb) - tb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self.predictor.eval()
        return self

    def score(self, emb) -> float:
        import torch
        with torch.no_grad():
            x = self._white(torch.as_tensor(np.asarray(emb, np.float32), device=self.device).reshape(1, -1))
            return float(((self.predictor(x) - self.target(x)) ** 2).mean())


# --------------------------------------------------------------------------- frozen SigLIP embedder
@functools.lru_cache(maxsize=2)
def load_embedder(model_id: str = SIGLIP, device: Optional[str] = None):
    """Frozen SigLIP -> embed(*frames) -> np.ndarray (per-frame pooled feature, concatenated over cameras)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()

    @torch.no_grad()
    def embed(*frames) -> np.ndarray:
        parts = []
        for im in frames:
            inp = proc(images=Image.fromarray(np.asarray(im, np.uint8)), return_tensors="pt").to(dev)
            f = model.get_image_features(**inp)
            parts.append(getattr(f, "pooler_output", f).float().cpu().numpy().reshape(-1))
        return np.concatenate(parts).astype(np.float32)
    return embed


# --------------------------------------------------------------------------- combined calibrator
class Calibrator:
    """Fit SMD + RND on successful trajectories; combine as max-of-z; threshold at a success percentile."""

    def __init__(self, pctl: float = 95.0, embedder=None):
        self.pctl, self.embed = pctl, embedder

    def fit(self, states, frames: Sequence[Sequence]) -> "Calibrator":
        states = np.asarray(states, float)
        self.smd = SMD.fit(states)
        self.embed = self.embed or load_embedder()
        embs = np.stack([self.embed(*f) for f in frames])
        self.rnd = RND(d_in=embs.shape[1]).fit(embs)
        s = np.array([self.smd.score(x) for x in states])
        r = np.array([self.rnd.score(e) for e in embs])
        self._sm, self._ss = s.mean(), s.std() + 1e-6           # z-norm on successes -> comparable scales
        self._rm, self._rs = r.mean(), r.std() + 1e-6
        comb = np.maximum((s - self._sm) / self._ss, (r - self._rm) / self._rs)
        self.threshold = float(np.percentile(comb, self.pctl))
        return self

    def score(self, state, frames) -> dict:
        s = (self.smd.score(state) - self._sm) / self._ss
        r = (self.rnd.score(self.embed(*frames)) - self._rm) / self._rs
        comb = float(max(s, r))
        return {"smd": float(s), "rnd": float(r), "combined": comb,
                "threshold": self.threshold, "spike": comb > self.threshold}
