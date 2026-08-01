"""RND-OE — Random Network Distillation on the vision embedding.

A frozen random `target` MLP and a `predictor` MLP trained to match it on
in-distribution embeddings. On a familiar embedding the predictor matches the
target (low residual); on an off-manifold embedding it diverges (high residual).
Runs on the SigLIP image embedding (sharper than raw pixels — the encoder has
already squashed action-irrelevant axes). The first detector that actually needs
a hook into the model, so it's what proves the internals wedge.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int = 64, h: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


class RND:
    """Seeded by default. The target network is random and the predictor's training shuffles, so an
    unseeded fit moves the reported AUC by ~0.005 between identical runs — enough to make a
    published number unreproducible. Pass `seed=None` for a fresh draw."""

    def __init__(self, d_in: int, d_out: int = 64, h: int = 128, device: str = "cpu",
                 seed: int | None = 0):
        self.device = device
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
        self.target = _MLP(d_in, d_out, h).to(device).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = _MLP(d_in, d_out, h).to(device)
        self.mean = None   # per-dim input standardization, fit on the in-distribution embeddings
        self.std = None

    def _white(self, x):
        return x if self.mean is None else (x - self.mean) / self.std

    def fit(self, embeddings, epochs: int = 8, lr: float = 1e-3, bs: int = 128) -> "RND":
        X = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=self.device)
        # Per-dim whitening. The raw SigLIP embedding dims have wildly different scales (and after
        # concatenating up/side/wrist, the camera blocks differ too); without standardizing, the
        # RND residual is dominated by a handful of high-variance dims and the detector barely
        # separates. Whitening is the single biggest lever for RND quality (AUC 0.58 -> ~1.0 in
        # our ablation). mean/std are fit here on the success embeddings and reused at score time.
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6
        X = self._white(X)
        opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.predictor.train()
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(epochs):
            perm = torch.randperm(len(X), generator=gen).to(X.device)
            for i in range(0, len(X), bs):
                xb = X[perm[i:i + bs]]
                with torch.no_grad():
                    tb = self.target(xb)
                loss = ((self.predictor(xb) - tb) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.predictor.eval()
        return self

    @torch.no_grad()
    def score(self, emb) -> float:
        x = torch.as_tensor(np.asarray(emb, dtype=np.float32), device=self.device).reshape(1, -1)
        x = self._white(x)
        return float(((self.predictor(x) - self.target(x)) ** 2).mean().item())
