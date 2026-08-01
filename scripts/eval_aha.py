"""Rigorous held-out benchmark of the SMD + RND failure detectors on labeled runs.

Unlike `drishti.calibrate` (which fits AND validates on the same successful runs), this
harness enforces a real train/eval split so the reported AUC is honest:

    * fit SMD (state) + RND (vision-embedding) on TRAIN successes only
    * set the conformal threshold on TRAIN successes only
    * score a HELD-OUT set = eval successes + ALL failures, and measure separation

Built for the AHA dataset (binary success/failure labels + a `failure_type` per failure,
stored in each run's `meta`), but source-agnostic — any run in the Store with an
`observation_state` per frame and `embed_*.npy` blobs works. Signals are SMD + RND only:
AHA has no live policy, so the ACE / action-head-spread channel is dropped.

    # smoke-test the harness on whatever labeled runs are already in the Store:
    PYTHONPATH=. .venv/bin/python scripts/eval_aha.py --split episode --seed 0

    # AHA (only runs tagged source=aha in meta), cross-task generalization split:
    PYTHONPATH=. .venv/bin/python scripts/eval_aha.py --source aha --split task

Writes data/calib/report_aha.json (overall + per-failure-mode + per-signal AUC).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from drishti.store import Store
from drishti.uq import SMD, RND
from drishti.calibrate import CAP, _auc   # pure helpers reused unchanged

SIGNALS = ("smd", "rnd")   # no "act" — AHA has no policy action head
PCTL_RUN = 90.0            # threshold = this pctl of TRAIN-success run-peak combined scores
OUT = Path("data/calib/report_aha.json")
BLOB_DIR = Path(os.environ.get("DRISHTI_BLOB_DIR", "data/obs_blobs"))   # embeddings/frames location


def _capped_z(x, mu, sd, cap=CAP):
    return np.clip((np.asarray(x, dtype=float) - mu) / sd, 0.0, cap) / cap


def combined(sm, rn, norm, cap=CAP):
    """Mean of capped-z over SMD + RND — bounded so neither detector dominates by raw scale."""
    return (_capped_z(sm, *norm["smd"], cap) + _capped_z(rn, *norm["rnd"], cap)) / 2.0


# --------------------------------------------------------------------------- selection
def _label(run) -> int:
    """1 = failure, 0 = success (from success_rate, the same key calibrate keys off)."""
    return 0 if (run.get("success_rate") or 0.0) >= 0.5 else 1


def _mode(run) -> str:
    m = run.get("meta") or {}
    return m.get("failure_type") or ("success" if _label(run) == 0 else "unknown")


def _group(run, split: str) -> str:
    if split == "task":
        return run.get("task") or (run.get("meta") or {}).get("task") or "_notask"
    return run["run_id"]   # by-episode: each run is its own group


# ------------------------------------------- encoder / field-aware data access (ablation grid)
# Local replacements for calibrate's fixed-path loaders so an experiment can pick:
#   encoder      -> which embed_<f>[_<encoder>].npy to read  (None = the default embed_<f>.npy)
#   state_fields -> which proprio fields to slice out of the stored superset (None = whole vector)
# Defaults (None, None) reproduce the original behaviour exactly.
def _embed_file(rid: str, f: int, encoder):
    suf = "" if not encoder else f"_{encoder}"
    return BLOB_DIR / rid / f"embed_{f:03d}{suf}.npy"


def _sel_idx(run, state_fields):
    if not state_fields:
        return None
    fs = (run.get("meta") or {}).get("field_slices")
    if not fs:
        raise SystemExit(f"run {run['run_id']} lacks meta.field_slices — --state-fields needs "
                         "data ingested with the updated ingester")
    idx = []
    for f in state_fields:
        if f not in fs:
            raise SystemExit(f"field {f!r} not in field_slices {sorted(fs)}")
        a, b = fs[f]
        idx.extend(range(a, b))
    return np.array(idx, dtype=int)


def _slice(vec, sel):
    v = np.asarray(vec, dtype=float)
    return v if sel is None else v[sel]


def _window_stack(states, window):
    """Stack the last `window` sliced states at each step (front-padded with states[0]).

    window=1 -> unchanged (the memoryless per-frame SMD). window>1 gives SMD a short-horizon
    view so it can see *transitions/order*, not just single poses — the lever for temporal
    failures like wrong_sequence. Output dim = window * state_dim.
    """
    if window <= 1 or not states:
        return states
    out = []
    for i in range(len(states)):
        chunk = [states[max(j, 0)] for j in range(i - window + 1, i + 1)]
        out.append(np.concatenate(chunk))
    return out


def _run_states(store, run, sel, window=1):
    """Every-frame states (state is logged each frame), sliced to the selected fields, optionally windowed."""
    states = [_slice(s["observation_state"], sel) for s in store.get_steps(run["run_id"])
              if s.get("observation_state") is not None]
    return _window_stack(states, window)


def _run_embeds(store, run, encoder, embed_window=1):
    """Embed-frame embeddings for `encoder`; optionally stack the last `embed_window` embed frames
    (temporal vision — note embed frames are `--embed-stride` apart, a coarser timescale than state)."""
    rid = run["run_id"]
    embs = []
    for s in store.get_steps(rid):
        ep = _embed_file(rid, int(s["step_idx"]), encoder)
        if s.get("observation_state") is None or not ep.exists():
            continue
        embs.append(np.load(ep).astype(np.float32))
    if not embs:
        return None
    return np.stack(_window_stack(embs, embed_window))


def _score_run(store, run, smd, rnd, encoder, sel, window=1, embed_window=1):
    """Live-cadence scoring: SMD on windowed state, RND on the last `embed_window` embed frames (carried)."""
    rid = run["run_id"]
    steps = [s for s in store.get_steps(rid) if s.get("observation_state") is not None]
    wstates = _window_stack([_slice(s["observation_state"], sel) for s in steps], window)
    ebuf, last = [], None
    sm, rn, idxs = [], [], []
    for k, s in enumerate(steps):
        f = int(s.get("step_idx") or 0)
        ep = _embed_file(rid, f, encoder)
        if ep.exists():
            ebuf.append(np.load(ep).astype(np.float32))
            chunk = [ebuf[max(j, 0)] for j in range(len(ebuf) - embed_window, len(ebuf))]
            last = np.concatenate(chunk)     # last `embed_window` distinct embeds, front-padded
        if last is None:        # before the first embed frame — nothing to carry yet
            continue
        sm.append(float(smd.score(wstates[k])))
        rn.append(float(rnd.score(last)))
        idxs.append(f)
    return np.array(sm), np.array(rn), idxs


def select_runs(store: Store, source, limit, encoder=None, task=None):
    runs = store.list_runs(limit=limit)
    if source:
        runs = [r for r in runs if (r.get("meta") or {}).get("source") == source]
    if task:   # per-task ("per-deployment") eval — fit + score within ONE task
        runs = [r for r in runs if (r.get("task") or (r.get("meta") or {}).get("task")) == task]
    # keep only runs that have at least one embedding for the chosen encoder
    return [r for r in runs if _run_embeds(store, r, encoder) is not None]


# --------------------------------------------------------------------------- fit / threshold
def fit_detectors(store: Store, train_success_ids, runs_by_id, encoder=None, state_fields=None,
                  window=1, embed_window=1):
    S, E = [], []
    for rid in train_success_ids:
        run = runs_by_id[rid]
        S.extend(_run_states(store, run, _sel_idx(run, state_fields), window))
        e = _run_embeds(store, run, encoder, embed_window)
        if e is not None:
            E.append(e)
    if not S or not E:
        raise SystemExit("no train-success states/embeddings to fit on")
    S, E = np.array(S), np.concatenate(E)
    smd = SMD.fit(S)
    rnd = RND(d_in=E.shape[1]).fit(E, epochs=12)
    return smd, rnd, E.shape[1], len(S), len(E)


def norm_and_threshold(store: Store, train_success_ids, runs_by_id, smd, rnd,
                       encoder=None, state_fields=None, window=1, embed_window=1, pctl=PCTL_RUN):
    """Per-signal (mean,std) + conformal threshold, both from TRAIN successes at live cadence.
    pctl = the percentile of train-success run-peaks used as the threshold (higher = fewer false
    alarms, ~(100-pctl)% target per-rollout false-alarm rate)."""
    rs = {k: [] for k in SIGNALS}
    peaks = []
    per_run_raw = {}
    for rid in train_success_ids:
        run = runs_by_id[rid]
        sm, rn, _idx = _score_run(store, run, smd, rnd, encoder, _sel_idx(run, state_fields),
                                  window, embed_window)
        if len(sm) == 0:
            continue
        rs["smd"].append(sm); rs["rnd"].append(rn)
        per_run_raw[rid] = (sm, rn)
    raw = {k: np.concatenate(v) for k, v in rs.items() if v}
    norm = {k: (float(raw[k].mean()), float(raw[k].std() + 1e-9)) for k in SIGNALS}
    for sm, rn in per_run_raw.values():
        peaks.append(float(combined(sm, rn, norm).max()))
    threshold = float(np.percentile(peaks, pctl)) if peaks else 0.0
    return norm, threshold


# --------------------------------------------------------------------------- score / report
def score_runs(store: Store, run_ids, runs_by_id, smd, rnd, norm, encoder=None, state_fields=None,
               window=1, embed_window=1):
    """Per run: combined peak, per-signal peaks, first-cross fraction, peak-step fraction."""
    out = {}
    for rid in run_ids:
        run = runs_by_id[rid]
        sm, rn, idxs = _score_run(store, run, smd, rnd, encoder, _sel_idx(run, state_fields),
                                  window, embed_window)
        if len(sm) == 0:
            continue
        cb = combined(sm, rn, norm)
        out[rid] = {
            "combined_peak": float(cb.max()),
            "smd_peak": float(sm.max()),
            "rnd_peak": float(rn.max()),
            "peak_step_frac": float(np.argmax(cb) / max(len(cb) - 1, 1)),
            "n_steps": int(len(cb)),
            "cb": cb,           # kept in-memory for threshold-based caught/first-cross
        }
    return out


def build_report(runs_by_id, scored, threshold, extra):
    rids = [r for r in scored]
    labels = np.array([_label(runs_by_id[r]) for r in rids])
    comb = np.array([scored[r]["combined_peak"] for r in rids])
    smd_p = np.array([scored[r]["smd_peak"] for r in rids])
    rnd_p = np.array([scored[r]["rnd_peak"] for r in rids])

    fails = [r for r in rids if _label(runs_by_id[r]) == 1]
    succs = [r for r in rids if _label(runs_by_id[r]) == 0]
    caught = {r: bool((scored[r]["cb"] > threshold).any()) for r in rids}

    # per-failure-mode AUC: successes vs failures of ONE mode (the honest per-mode number).
    # We break out EACH signal (combined / smd / rnd) so we can see WHICH detector catches
    # WHICH mode — the internal attribution the overall per-signal numbers can't give.
    per_mode = {}
    modes = sorted({_mode(runs_by_id[r]) for r in fails})
    succ_comb, succ_smd, succ_rnd = comb[labels == 0], smd_p[labels == 0], rnd_p[labels == 0]
    for mode in modes:
        mode_ids = [r for r in fails if _mode(runs_by_id[r]) == mode]

        def _mode_auc(peak_key, succ_vec):
            m = np.array([scored[r][peak_key] for r in mode_ids])
            s = np.concatenate([succ_vec, m])
            y = np.concatenate([np.zeros(len(succ_vec)), np.ones(len(m))])
            return _auc(s, y)

        per_mode[mode] = {
            "n_fail": len(mode_ids),
            "auc_combined": _mode_auc("combined_peak", succ_comb),
            "auc_smd": _mode_auc("smd_peak", succ_smd),
            "auc_rnd": _mode_auc("rnd_peak", succ_rnd),
            "caught_frac": float(np.mean([caught[r] for r in mode_ids])) if mode_ids else None,
            "mean_peak_step_frac": float(np.mean([scored[r]["peak_step_frac"] for r in mode_ids]))
            if mode_ids else None,
        }

    report = {
        **extra,
        "threshold": threshold, "pctl": PCTL_RUN,
        "signals": list(SIGNALS), "combine": "mean_capped_z(smd,rnd)", "cap": CAP,
        "n_eval_runs": len(rids), "n_fail": len(fails), "n_succ": len(succs),
        "auc_fail_vs_success": _auc(comb, labels),
        "auc_per_signal": {
            "smd": _auc(smd_p, labels),
            "rnd": _auc(rnd_p, labels),
        },
        "fail_caught_frac": float(np.mean([caught[r] for r in fails])) if fails else None,
        "false_alarm_frac": float(np.mean([caught[r] for r in succs])) if succs else None,
        "per_failure_mode": per_mode,
    }
    return report


# --------------------------------------------------------------------------- split + main
def split_ids(runs, split: str, train_frac: float, rng):
    """Return (train_success_ids, eval_ids). Failures always land in eval."""
    succ = [r for r in runs if _label(r) == 0]
    fail = [r for r in runs if _label(r) == 1]
    if split == "task":
        groups = sorted({_group(r, "task") for r in runs})
        rng.shuffle(groups)
        n_train = max(1, int(round(len(groups) * train_frac)))
        train_g = set(groups[:n_train])
        train_success = [r["run_id"] for r in succ if _group(r, "task") in train_g]
        eval_ids = [r["run_id"] for r in runs if _group(r, "task") not in train_g]
    else:  # episode
        ids = [r["run_id"] for r in succ]
        rng.shuffle(ids)
        n_train = max(1, int(round(len(ids) * train_frac)))
        train_success = ids[:n_train]
        held_success = ids[n_train:]
        eval_ids = held_success + [r["run_id"] for r in fail]
    return train_success, eval_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="only runs with meta.source == this (e.g. 'aha')")
    ap.add_argument("--split", choices=["episode", "task"], default="episode")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedder", default=None,
                    help="encoder suffix: reads embed_<f>_<enc>.npy (default None = embed_<f>.npy)")
    ap.add_argument("--state-fields", nargs="+", default=None,
                    help="proprio fields to slice from the stored superset (default = whole vector)")
    ap.add_argument("--state-window", type=int, default=1,
                    help="stack the last W states for SMD (>1 = short-horizon/temporal; default 1 = per-frame)")
    ap.add_argument("--embed-window", type=int, default=1,
                    help="stack the last W embed frames for RND (temporal vision; default 1 = single frame)")
    ap.add_argument("--task", default=None, help="restrict to one task (per-deployment eval)")
    ap.add_argument("--threshold-pctl", type=float, default=PCTL_RUN,
                    help="percentile of train-success run-peaks for the threshold (higher = fewer false alarms)")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    enc, sf, w, ew = args.embedder, args.state_fields, args.state_window, args.embed_window
    store = Store()
    runs = select_runs(store, args.source, args.limit, enc, args.task)
    if not runs:
        raise SystemExit(f"no eligible runs (need embed_*{('_'+enc) if enc else ''}.npy + a label)")
    runs_by_id = {r["run_id"]: r for r in runs}
    rng = np.random.default_rng(args.seed)

    train_success, eval_ids = split_ids(runs, args.split, args.train_frac, rng)
    if not train_success:
        raise SystemExit("no train successes after split")

    smd, rnd, embed_dim, n_states, n_embeds = fit_detectors(store, train_success, runs_by_id, enc, sf, w, ew)
    norm, threshold = norm_and_threshold(store, train_success, runs_by_id, smd, rnd, enc, sf, w, ew,
                                         args.threshold_pctl)
    scored = score_runs(store, eval_ids, runs_by_id, smd, rnd, norm, enc, sf, w, ew)
    if not scored:
        raise SystemExit("no eval runs scored (missing embeddings?)")

    extra = {
        "source": args.source, "split": args.split, "seed": args.seed,
        "embedder": enc or "default", "state_fields": sf or "all",
        "state_window": w, "embed_window": ew,
        "train_frac": args.train_frac,
        "n_train_success_runs": len(train_success),
        "n_calib_state_frames": n_states, "n_calib_embed_frames": n_embeds,
        "embed_dim": embed_dim, "norm": norm,
    }
    report = build_report(runs_by_id, scored, threshold, extra)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
