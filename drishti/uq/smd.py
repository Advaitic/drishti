"""SMD — Mahalanobis distance on the proprioceptive state vector.

The cheapest observation-side detector: fit a single Gaussian (mean, covariance)
to the state vectors over successful/training rollouts, then score a fresh state
by its covariance-corrected distance from that mean. Zero model access, zero
training; ReconVLA reports AUC ~0.91 with this alone. Lives on the low-dim state
(8-20 floats) where covariance estimation is well-posed.
"""

from __future__ import annotations

import numpy as np


class SMD:
    def __init__(self, mean: np.ndarray, cov: np.ndarray, eps: float = 1e-6):
        self.mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        self.inv = np.linalg.inv(cov + eps * np.eye(cov.shape[0]))

    @classmethod
    def fit(cls, states) -> "SMD":
        X = np.asarray(states, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        return cls(mean, cov)

    def score(self, x) -> float:
        d = np.asarray(x, dtype=float).reshape(-1) - self.mean
        return float(np.sqrt(max(d @ self.inv @ d, 0.0)))
