"""Figures for the calibration result — regenerated from the telemetry store + data/calib/.

Re-scores every rollout in the store with the saved detectors, at the same live cadence
`drishti.calibrate` validates at (SMD per frame, RND/ACE carried from the last inference frame),
takes each run's PEAK combined score, and draws:

  1. calibration_separation.png — per-rollout peak score, success vs failure, vs the threshold
  2. calibration_roc.png        — ROC of the combined score and of each detector alone
  3. calibration_sweep.png      — catches / false alarms as the threshold slides

    drishti figures --demo --out assets/figures

Scoring goes through `calibrate.score_run_live_cadence` rather than the smd/rnd/ace columns cached
on the steps, so the figures reproduce report.json by construction and cannot drift from it if an
older calibration left stale columns behind.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drishti.calibrate import load_calib, score_run_live_cadence  # noqa: E402
from drishti.store import Store  # noqa: E402

CALIB_DIR = Path(os.environ.get("DRISHTI_CALIB_DIR", "data/calib"))
# Warm-up frames to skip. calibrate.py's validation report scores every frame (warmup 0), so that is
# the default here and the figures reproduce report.json exactly. The live Monitor and /v1/sweep skip
# the first 3 frames (start transient); pass --warmup 3 to see the serve-time numbers instead.
WARMUP = 0

INK, DIM, OK, BAD, LINE = "#e6edf3", "#8b95a1", "#38d39f", "#ff6b6b", "#2a3441"
BG, PANEL = "#0a0c0f", "#0f1318"


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.grid(color=LINE, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=9)


def _fig(w=7.2, h=4.0):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(BG)
    return fig, ax


def peaks(store: Store, smd_det, rnd_det, norm: dict, cap: float, warmup: int = WARMUP):
    """Per-rollout peak combined score + peak per-detector raw score, labelled by the oracle."""
    def cz(v, k):
        mu, sd = norm[k]
        return np.clip((np.asarray(v) - mu) / sd, 0.0, cap) / cap

    out = []
    for r in store.list_runs(limit=5000):
        sm, rn, ac, idxs = score_run_live_cadence(store, r["run_id"], smd_det, rnd_det)
        if len(idxs) <= warmup:
            continue
        sm, rn, ac = sm[warmup:], rn[warmup:], ac[warmup:]
        comb = (cz(sm, "smd") + cz(rn, "rnd") + cz(ac, "act")) / 3.0
        out.append({"run_id": r["run_id"], "peak": float(max(comb.max(), 0.0)),
                    "per": {"smd": float(sm.max()), "rnd": float(rn.max()), "act": float(ac.max())},
                    "fail": (r.get("success_rate") or 0.0) < 0.5})
    return out


def _roc(scores: np.ndarray, fail: np.ndarray):
    """(fpr, tpr, auc) for 'high score => failure'."""
    order = np.argsort(-scores)
    f = fail[order]
    tpr = np.concatenate([[0.0], np.cumsum(f) / max(f.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~f) / max((~f).sum(), 1)])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def separation(runs, thr, out: Path):
    fig, ax = _fig()
    fails = [r["peak"] for r in runs if r["fail"]]
    succs = [r["peak"] for r in runs if not r["fail"]]
    rng = np.random.default_rng(0)
    for xs, x0, color, label in ((succs, 0, OK, f"success (n={len(succs)})"),
                                 (fails, 1, BAD, f"failure (n={len(fails)})")):
        ax.scatter(x0 + rng.uniform(-0.13, 0.13, len(xs)), xs, s=26, c=color,
                   alpha=0.8, edgecolors="none", label=label)
        ax.plot([x0 - 0.26, x0 + 0.26], [np.median(xs)] * 2, color=color, lw=2, alpha=0.9)
    ax.axhline(thr, color="#f5b54a", ls="--", lw=1.2)
    ax.text(1.42, thr, f" calibrated threshold {thr:.3f}", color="#f5b54a", fontsize=8, va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cube placed", "cube not placed"], color=DIM, fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    _style(ax, "peak combined uncertainty, per rollout", ylabel="combined score (0-1)")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for t in leg.get_texts():
        t.set_color(DIM)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def roc(runs, out: Path):
    fig, ax = _fig(5.6, 4.6)
    fail = np.array([r["fail"] for r in runs])
    series = [("combined", np.array([r["peak"] for r in runs]), "#b083f0", 2.2)]
    for k, label, color in (("rnd", "RND (vision)", "#f5b54a"),
                            ("smd", "SMD (state)", "#5b9cf6"),
                            ("act", "ACE (action)", BAD)):
        series.append((label, np.array([r["per"][k] for r in runs]), color, 1.2))
    for label, sc, color, lw in series:
        fpr, tpr, auc = _roc(sc, fail)
        ax.plot(fpr, tpr, color=color, lw=lw, label=f"{label} · AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=LINE, lw=1, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _style(ax, "failure detection ROC (per rollout)",
           xlabel="false-alarm rate", ylabel="failures caught")
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for t in leg.get_texts():
        t.set_color(DIM)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def sweep(runs, thr, out: Path):
    fig, ax = _fig()
    grid = np.linspace(0, max(r["peak"] for r in runs) * 1.05, 300)
    fails = np.array([r["peak"] for r in runs if r["fail"]])
    succs = np.array([r["peak"] for r in runs if not r["fail"]])
    caught = [(fails > t).mean() * 100 for t in grid]
    alarms = [(succs > t).mean() * 100 for t in grid]
    ax.plot(grid, caught, color=OK, lw=2, label="failures caught")
    ax.plot(grid, alarms, color=BAD, lw=2, label="false alarms")
    ax.axvline(thr, color="#f5b54a", ls="--", lw=1.2)
    ax.text(thr, 52, f" calibrated {thr:.3f}", color="#f5b54a", fontsize=8, rotation=90, va="bottom")
    ax.set_ylim(-3, 103)
    _style(ax, "threshold sweep over every logged rollout",
           xlabel="threshold (combined score)", ylabel="% of rollouts")
    leg = ax.legend(frameon=False, fontsize=8, loc="center right")
    for t in leg.get_texts():
        t.set_color(DIM)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets", help="output directory for the figures")
    ap.add_argument("--calib", default=None, help="calibration dir (default $DRISHTI_CALIB_DIR or data/calib)")
    ap.add_argument("--warmup", type=int, default=WARMUP,
                    help="skip this many opening frames (0 = match report.json, 3 = match the live Monitor)")
    args = ap.parse_args()

    cdir = Path(args.calib) if args.calib else CALIB_DIR
    cal = json.loads((cdir / "calib.json").read_text())
    rep = json.loads((cdir / "report.json").read_text())
    smd_det, rnd_det, norm, _ = load_calib(cdir)
    runs = peaks(Store(), smd_det, rnd_det, norm, float(cal.get("cap", 32.0)), warmup=args.warmup)
    if not runs:
        raise SystemExit("no scored rollouts in the store — run drishti.calibrate first")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    separation(runs, float(cal["threshold"]), out / "calibration_separation.png")
    roc(runs, out / "calibration_roc.png")
    sweep(runs, float(cal["threshold"]), out / "calibration_sweep.png")
    print(f"{len(runs)} rollouts scored · report AUC {rep['auc_fail_vs_success']:.3f} → {out}/")


if __name__ == "__main__":
    main()
