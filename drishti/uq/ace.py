"""ACE — Action-Chunk Entropy on the stochastic action head.

Sample B action chunks at the same observation (fresh flow-matching noise each
time), then for every (timestep, action-dim) cell bin the B values into a
histogram and take its Shannon entropy; sum over cells. Confident → all B land
in one bin → ~0; multimodal → a few bins → moderate; confused → flat → high.
Entropy (not L2 variance) so legitimate multimodality reads moderate, not as
alarming as scatter.

Binning uses a FIXED action range (default normalized [-1, 1]) so "spread" is
measured on an absolute scale, not auto-rescaled to the samples. Pass the
dataset's per-dim action range for a tighter signal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def ace_score(chunks, n_bins: int = 20, value_range: tuple[float, float] = (-1.0, 1.0)) -> float:
    """chunks: array (B, H, D) of B sampled action chunks. Returns summed cell entropy."""
    A = np.asarray(chunks, dtype=float)
    if A.ndim == 2:  # (B, D) → single-step chunk
        A = A[:, None, :]
    B, H, D = A.shape
    lo, hi = value_range
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    total = 0.0
    for h in range(H):
        for d in range(D):
            counts, _ = np.histogram(A[:, h, d], bins=edges)
            n = counts.sum()
            if n == 0:
                continue
            p = counts / n
            nz = p[p > 0]
            total += float(-(nz * np.log(nz)).sum())
    return total


def sample_action_chunks(policy, batch, n: int = 8) -> np.ndarray:
    """Sample n action chunks from a stochastic lerobot policy at the same observation.

    Relies on `predict_action_chunk(batch, noise=None)` drawing fresh flow-matching
    noise per call (SmolVLA). Returns (n, H, D).
    """
    import torch

    outs = []
    for _ in range(n):
        with torch.inference_mode():
            ch = policy.predict_action_chunk(batch, noise=None)
        arr = ch.detach().float().cpu().numpy()
        outs.append(arr[0] if arr.ndim == 3 else arr)
    return np.stack(outs, axis=0)
