"""Hook mechanism + UQ primitive tests — synthetic torch module, no model download.

De-risks the hardest unknown (do forward hooks fire and attribute to the right
step, capturing vision + action-head internals) before touching real SmolVLA.
"""

import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from drishti import instrument, start_run  # noqa: E402
from drishti.store import Store  # noqa: E402


class _FakeExpert(nn.Module):
    """Stands in for SmolVLA's vlm_with_expert: exposes embed_image()."""

    def __init__(self, d=32):
        super().__init__()
        self.lin = nn.Linear(3 * 8 * 8, d)

    def embed_image(self, img):
        return self.lin(img.flatten(1)).unsqueeze(1)  # (B, 1, d) token embedding


class _FakeModel(nn.Module):
    def __init__(self, d=32, adim=7):
        super().__init__()
        self.vlm_with_expert = _FakeExpert(d)
        self.action_out_proj = nn.Linear(d, adim)

    def forward(self, img):
        return self.action_out_proj(self.vlm_with_expert.embed_image(img))


class _FakePolicy:
    def __init__(self):
        self.model = _FakeModel()

    def select_action(self, img):
        return self.model(img)


def test_hooks_fire_and_attribute_to_step():
    s = Store(f"sqlite:///{tempfile.mkdtemp()}/h.db")
    p = _FakePolicy()
    instrument(p)
    run = start_run("fake", "t", store=s)
    img = torch.randn(1, 3, 8, 8)
    with run.step(step_idx=0) as st:
        p.select_action(img)
        assert "vision_embed_norm" in st.internals
        assert len(st.internals["action_head_mean"]) == 7
        assert hasattr(st, "_vision_embed") and st._vision_embed.shape[0] == 32
    run.finish()
    rows = s.get_steps(run.run_id)
    assert rows[0]["vision_embed_norm"] is not None
    assert rows[0]["action_head_mean"] is not None


def test_smd_separates():
    from drishti.uq.smd import SMD

    rng = np.random.default_rng(0)
    smd = SMD.fit(rng.normal(size=(500, 8)))
    assert smd.score(np.ones(8) * 6) > smd.score(np.zeros(8))


def test_rnd_separates_in_vs_out_of_dist():
    from drishti.uq.rnd import RND

    rng = np.random.default_rng(0)
    rnd = RND(16).fit(rng.normal(size=(400, 16)), epochs=8)
    id_score = np.mean([rnd.score(rng.normal(size=16)) for _ in range(30)])
    ood_score = np.mean([rnd.score(rng.normal(size=16) + 8) for _ in range(30)])
    assert ood_score > id_score


def test_ace_confident_vs_scattered():
    from drishti.uq.ace import ace_score

    rng = np.random.default_rng(0)
    confident = rng.normal(scale=1e-3, size=(16, 4, 7))
    scattered = rng.normal(scale=1.0, size=(16, 4, 7))
    assert ace_score(scattered) > ace_score(confident)
