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
    def __init__(self, d_in: int, d_out: int = 64, h: int = 128, device: str = "cpu"):
        self.device = device
        self.target = _MLP(d_in, d_out, h).to(device).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = _MLP(d_in, d_out, h).to(device)

    def fit(self, embeddings, epochs: int = 8, lr: float = 1e-3, bs: int = 128) -> "RND":
        X = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=self.device)
        opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.predictor.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X))
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
        return float(((self.predictor(x) - self.target(x)) ** 2).mean().item())
