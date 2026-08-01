"""Seed-average the RND-dependent AUCs to expose (and average out) RND's init variance.

RND's random target + predictor init are not seeded, so at small n the per-mode AUCs swing
run-to-run. This fixes the train/eval SPLIT once and refits RND under K torch seeds, reporting
mean ± std per signal and per failure mode. SMD is deterministic (its column is a sanity check).

    DRISHTI_OBS_DB_URL="sqlite:///data/aha_pod/obs.db" PYTHONPATH=. \
      .venv/bin/python scripts/eval_aha_seeds.py --source aha --split episode --k 6
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import numpy as np
import torch

from drishti.store import Store
from scripts.eval_aha import (
    select_runs, split_ids, fit_detectors, norm_and_threshold, score_runs, build_report,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="aha")
    ap.add_argument("--split", choices=["episode", "task"], default="episode")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=6, help="number of RND seeds to average")
    ap.add_argument("--embedder", default=None, help="encoder suffix (default = embed_<f>.npy)")
    ap.add_argument("--state-fields", nargs="+", default=None, help="proprio fields to slice (default = all)")
    ap.add_argument("--state-window", type=int, default=1, help="stack last W states for SMD (>1 = temporal)")
    ap.add_argument("--embed-window", type=int, default=1, help="stack last W embed frames for RND (temporal vision)")
    ap.add_argument("--task", default=None, help="restrict to one task (per-deployment eval)")
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()
    enc, sf, w, ew = args.embedder, args.state_fields, args.state_window, args.embed_window

    store = Store()
    runs = select_runs(store, args.source, args.limit, enc, args.task)
    runs_by_id = {r["run_id"]: r for r in runs}
    # fix the split ONCE so only RND init varies
    rng = np.random.default_rng(args.split_seed)
    train_success, eval_ids = split_ids(runs, args.split, args.train_frac, rng)

    overall = defaultdict(list)          # metric -> [per-seed values]
    per_mode = defaultdict(lambda: defaultdict(list))   # mode -> signal -> [values]
    for s in range(args.k):
        torch.manual_seed(s)             # <- the fix: reproducible RND init per seed
        smd, rnd, embed_dim, n_states, n_embeds = fit_detectors(store, train_success, runs_by_id, enc, sf, w, ew)
        norm, thr = norm_and_threshold(store, train_success, runs_by_id, smd, rnd, enc, sf, w, ew)
        scored = score_runs(store, eval_ids, runs_by_id, smd, rnd, norm, enc, sf, w, ew)
        rep = build_report(runs_by_id, scored, thr, {})
        overall["combined"].append(rep["auc_fail_vs_success"])
        overall["smd"].append(rep["auc_per_signal"]["smd"])
        overall["rnd"].append(rep["auc_per_signal"]["rnd"])
        for m, d in rep["per_failure_mode"].items():
            per_mode[m]["combined"].append(d["auc_combined"])
            per_mode[m]["smd"].append(d["auc_smd"])
            per_mode[m]["rnd"].append(d["auc_rnd"])

    def ms(v):
        a = np.array([x for x in v if x is not None], float)
        return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), 0.0)

    print(f"\n=== {args.k} RND seeds · split={args.split} · fixed split-seed={args.split_seed} ===")
    print(f"{'':<16}{'combined':>16}{'smd':>14}{'rnd':>14}")
    om, os_ = {k: ms(overall[k]) for k in ("combined", "smd", "rnd")}, None
    print(f"{'OVERALL':<16}" + "".join(f"{om[k][0]:>9.2f}±{om[k][1]:<5.2f}" for k in ("combined","smd","rnd")))
    print("-" * 60)
    n_of = {m: max((d for d in per_mode[m]["combined"]), default=None) for m in per_mode}
    for m in per_mode:
        row = {k: ms(per_mode[m][k]) for k in ("combined", "smd", "rnd")}
        print(f"{m:<16}" + "".join(f"{row[k][0]:>9.2f}±{row[k][1]:<5.2f}" for k in ("combined","smd","rnd")))


if __name__ == "__main__":
    main()
