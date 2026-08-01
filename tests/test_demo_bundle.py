"""The bundled demo dataset must reproduce the published calibration numbers.

Every headline result in the README comes out of `docs/results/calibration/report.json`, and that
file is only trustworthy if a stranger can regenerate it. This test does exactly what the quickstart
tells them to do — calibrate on `data/demo/` — and asserts the report it gets back matches the one
in the repo. If a change to the detectors, the combine rule or the demo bundle moves the number, the
test fails here rather than quietly making the documentation wrong.

Skipped when the bundle is absent (a source checkout without the data), and when torch is missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "demo"
PUBLISHED = ROOT / "docs" / "results" / "calibration" / "report.json"

pytest.importorskip("torch", reason="calibration needs torch")
pytestmark = pytest.mark.skipif(
    not (BUNDLE / "obs.db").exists() or not PUBLISHED.exists(),
    reason="demo bundle or published report not present",
)


@pytest.fixture(scope="module")
def recomputed(tmp_path_factory):
    """Run the real calibration against the bundle, into a throwaway calib dir."""
    out = tmp_path_factory.mktemp("calib")
    env = {
        "DRISHTI_OBS_DB_URL": f"sqlite:///{BUNDLE / 'obs.db'}",
        "DRISHTI_BLOB_DIR": str(BUNDLE / "blobs"),
        "DRISHTI_CALIB_DIR": str(out),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        # Imported here, after the env vars are set — the module reads them at import time.
        import importlib

        import drishti.calibrate as calibrate
        importlib.reload(calibrate)
        return calibrate.calibrate(auto=True)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_matches_published_report(recomputed):
    published = json.loads(PUBLISHED.read_text())

    # The headline claims. Exact, because calibration is seeded.
    assert recomputed["auc_fail_vs_success"] == pytest.approx(
        published["auc_fail_vs_success"], abs=1e-9)
    assert recomputed["auc_per_signal"]["rnd"] == pytest.approx(
        published["auc_per_signal"]["rnd"], abs=1e-9)
    assert recomputed["fail_caught_frac"] == pytest.approx(published["fail_caught_frac"], abs=1e-9)
    assert recomputed["false_alarm_frac"] == pytest.approx(published["false_alarm_frac"], abs=1e-9)
    assert recomputed["threshold"] == pytest.approx(published["threshold"], rel=1e-6)

    # The set it was measured on, so "n=" in the docs can't drift either.
    for k in ("n_runs_scored", "n_fail_runs", "n_succ_runs", "n_success_runs_calib", "embed_dim"):
        assert recomputed[k] == published[k], k


def test_separates_success_from_failure(recomputed):
    """The claim, stated as a claim: uncertainty ranks failures above successes, and the calibrated
    threshold catches all of them."""
    assert recomputed["auc_fail_vs_success"] > 0.95
    assert recomputed["fail_caught_frac"] == 1.0
    assert recomputed["false_alarm_frac"] < 0.25


def test_bundle_manifest_matches_contents():
    manifest = json.loads((BUNDLE / "MANIFEST.json").read_text())
    embeds = list((BUNDLE / "blobs").rglob("embed_*.npy"))
    frames = list((BUNDLE / "blobs").rglob("*.jpg"))
    assert len(embeds) == manifest["n_embeddings"]
    assert len(frames) == manifest["n_frames"]
