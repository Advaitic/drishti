"""Calibrate the UQ detectors on selected successful rollouts → one combined conformal threshold.

Reads the selected (successful) runs' per-step states + vision-embedding blobs, fits SMD (on
states) and RND (on embeddings), z-normalizes each raw signal (SMD / RND / action-head spread) on
the success steps, combines them as max-of-z into one uncertainty score, and sets the conformal
threshold at the 95th percentile of that score over the successful steps. Then it RE-SCORES every
rollout (writes smd/rnd/ace back to the steps so replays show the traces) and writes a VALIDATION
report — does the combined score separate success from failure? — which is the honest "aha" test.

    PYTHONPATH=. .venv/bin/python -m drishti.calibrate --auto        # all successful runs
    PYTHONPATH=. .venv/bin/python -m drishti.calibrate --runs r1 r2  # explicit selection

Persists to data/calib/ (detectors + threshold + report) for the Stage-2 live Monitor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sqlalchemy as sa
import torch
from safetensors.torch import save_file

from drishti.store import Store
from drishti.uq import SMD, RND

CALIB_DIR = Path("data/calib")
BLOB_DIR = Path("data/obs_blobs")
PCTL = 95.0
PCTL_RUN = 90.0   # threshold = this percentile of success RUN-PEAKS → ~(100-PCTL_RUN)% per-rollout false alarms
SIGNALS = ("smd", "rnd", "act")
CAP = 32.0  # capped-z ceiling: clip each signal's z at CAP sigmas above normal, then scale to [0,1].
            # Must stay well above the out-of-sample z range — cap=8 saturated held-out successes AND
            # failures to 1.0 and destroyed separation (held-out AUC 0.78 vs 0.99 at cap>=32).


def _act_spread(action_head_std) -> float:
    if not action_head_std:
        return 0.0
    return float(np.mean(np.abs(np.asarray(action_head_std, dtype=float))))


def load_calib(calib_dir: Path = CALIB_DIR):
    """Reconstruct the calibrated detectors + (norm, threshold) from data/calib/ for live scoring."""
    from safetensors.torch import load_file
    cal = json.loads((calib_dir / "calib.json").read_text())
    sd = np.load(calib_dir / "smd.npz")
    smd = SMD.__new__(SMD)            # bypass __init__ (it would re-invert); we saved mean + inv
    smd.mean, smd.inv = sd["mean"], sd["inv"]
    rnd = RND(d_in=int(cal["embed_dim"]))
    rnd.predictor.load_state_dict(load_file(str(calib_dir / "rnd_predictor.safetensors")))
    rnd.target.load_state_dict(load_file(str(calib_dir / "rnd_target.safetensors")))
    inn = np.load(calib_dir / "rnd_innorm.npz")   # per-dim input whitening fit at calibration
    rnd.mean = torch.tensor(inn["mean"], dtype=torch.float32)
    rnd.std = torch.tensor(inn["std"], dtype=torch.float32)
    rnd.predictor.eval()
    return smd, rnd, cal["norm"], float(cal["threshold"])


def _capped_z(x, mu, sd, cap=CAP) -> float:
    return min(max((x - mu) / sd, 0.0), cap) / cap


def combined_score(smd_raw, rnd_raw, act_raw, norm, cap=CAP) -> float:
    """Mean of capped-z over the three signals (matches calibrate's combine).
    Each signal -> clip((x - mu_success)/sd_success, 0, cap)/cap in [0,1] — BOUNDED, so no single
    detector can dominate by raw magnitude (the whitened RND's tiny success-std used to blow its
    z ~100x past the others). Averaging the three bounded scores is agreement-aware: a lone detector
    grazing can't max-out the combined, which is what cut false alarms in the ablation."""
    vals = [_capped_z(r, norm[k][0], norm[k][1], cap)
            for r, k in ((smd_raw, "smd"), (rnd_raw, "rnd"), (act_raw, "act"))]
    return float(sum(vals) / len(vals))


def load_run_data(store: Store, run_id: str):
    """Per-step (state, embedding, action-spread, step_idx) for one run, where the embedding blob
    exists. Returns arrays aligned by step."""
    states, embs, acts, idxs = [], [], [], []
    for st in store.get_steps(run_id):
        s = st.get("observation_state")
        ep = BLOB_DIR / run_id / f"embed_{int(st['step_idx']):03d}.npy"
        if s is None or not ep.exists():
            continue
        states.append(np.asarray(s, dtype=float))
        embs.append(np.load(ep).astype(np.float32))
        acts.append(_act_spread(st.get("action_head_std")))
        idxs.append(int(st["step_idx"]))
    if not states:
        return None
    return np.array(states), np.stack(embs), np.array(acts), idxs


def _all_states(store: Store, run_id: str):
    """Every-frame proprioceptive states for a run (state is logged each frame, unlike embeddings
    which only exist on inference frames). Used to fit SMD at the SAME cadence the Monitor scores."""
    return [np.asarray(s["observation_state"], dtype=float)
            for s in store.get_steps(run_id) if s.get("observation_state") is not None]


def score_run_live_cadence(store: Store, run_id: str, smd, rnd):
    """Score EVERY frame the way the live Monitor does: SMD on each frame's state, RND/ACE carried
    forward from the last inference (embed) frame. Returns (smd[], rnd[], act[], step_idx[]) over the
    frames from the first inference on. Calibrating at this cadence (not just embed frames) is what
    keeps the threshold valid at serve time — otherwise mid-chunk poses the calibration never saw
    spike SMD and every live rollout false-alarms."""
    last_emb, last_act = None, 0.0
    sm, rn, ac, idxs = [], [], [], []
    for s in store.get_steps(run_id):
        st = s.get("observation_state")
        if st is None:
            continue
        f = int(s.get("step_idx") or 0)
        ep = BLOB_DIR / run_id / f"embed_{f:03d}.npy"
        if ep.exists():
            last_emb = np.load(ep).astype(np.float32)
            last_act = _act_spread(s.get("action_head_std"))
        if last_emb is None:        # before the first inference — nothing to carry yet
            continue
        sm.append(float(smd.score(np.asarray(st, dtype=float))))
        rn.append(float(rnd.score(last_emb)))
        ac.append(last_act)
        idxs.append(f)
    return np.array(sm), np.array(rn), np.array(ac), idxs


def _auc(scores: np.ndarray, labels: np.ndarray):
    """AUC that a higher score flags label==1 (failure). Mann-Whitney U, average ranks for ties."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibrate(run_ids=None, auto=True):
    store = Store()
    runs = store.list_runs(limit=2000)
    succ = {r["run_id"]: (r.get("success_rate") or 0.0) >= 0.5 for r in runs}
    sel = [r for r in succ if succ[r]] if (auto or not run_ids) else list(run_ids)
    sel = [r for r in sel if succ.get(r)]   # only ever calibrate on successes
    if not sel:
        raise SystemExit("no successful runs to calibrate on")

    # --- fit detectors: SMD on EVERY-frame success states, RND on embed-frame success embeddings ---
    S, E = [], []
    for rid in sel:
        S.extend(_all_states(store, rid))
        d = load_run_data(store, rid)
        if d:
            E.append(d[1])
    S, E = np.array(S), np.concatenate(E)
    print(f"calibrating on {len(sel)} successful runs · {len(S)} state-frames · "
          f"{len(E)} embed-frames · embed_dim {E.shape[1]}", flush=True)
    smd = SMD.fit(S)
    rnd = RND(d_in=E.shape[1]).fit(E, epochs=12)

    # --- raw signals on success steps AT LIVE CADENCE (every frame, RND carried) → norm + threshold ---
    rs = {"smd": [], "rnd": [], "act": []}
    for rid in sel:
        sm, rn, ac, _ = score_run_live_cadence(store, rid, smd, rnd)
        if len(sm) == 0:    # success run with no embeddings (e.g. a pre-flywheel-fix live run) — skip
            continue
        rs["smd"].append(sm); rs["rnd"].append(rn); rs["act"].append(ac)
    raw_s = {k: np.concatenate(v) for k, v in rs.items()}
    norm = {k: (float(v.mean()), float(v.std() + 1e-9)) for k, v in raw_s.items()}

    def _comb_vec(sm, rn, a):   # mean of capped-z, vectorized — matches combined_score()
        cz = lambda v, k: np.clip((np.asarray(v) - norm[k][0]) / norm[k][1], 0, CAP) / CAP
        return (cz(sm, "smd") + cz(rn, "rnd") + cz(a, "act")) / 3.0

    # threshold = conformal on success RUN-PEAKS (per-ROLLOUT control): a new successful rollout
    # exceeds it with prob ~<=(100-PCTL_RUN)%. The old per-STEP 95th pct made ~73% of multi-step
    # runs graze at least once (5% of steps × ~50 steps/run), so it was the wrong object to bound.
    run_peaks = np.array([_comb_vec(rs["smd"][i], rs["rnd"][i], rs["act"][i]).max()
                          for i in range(len(rs["smd"]))])
    threshold = float(np.percentile(run_peaks, PCTL_RUN))

    # --- re-score ALL runs + per-run stats for validation ---
    # PHASE 1: score every run (reads + compute only — NO open write transaction). Holding a write
    # transaction while calling get_steps deadlocks against its own reads and the live ingest server
    # ("database is locked"); do all the slow reads/scoring first, collect the updates.
    per_run = {}  # rid -> (max_combined, crossed_bool, success_bool, first_cross_from_end)
    per_sig = {}  # rid -> {signal: max raw score}  (for per-detector AUC)
    writes = []   # (rid, step_idx, smd, rnd, ace)
    for r in runs:
        rid = r["run_id"]
        sm, rn, a, idxs = score_run_live_cadence(store, rid, smd, rnd)   # every frame, RND carried
        if not idxs:
            continue
        cb = _comb_vec(sm, rn, a)
        for i, f in enumerate(idxs):
            writes.append((rid, f, float(sm[i]), float(rn[i]), float(a[i])))
        crossed = bool((cb > threshold).any())
        first = int(np.argmax(cb > threshold)) if crossed else -1
        lead = (len(cb) - first) if crossed else 0   # steps from first cross to end
        per_run[rid] = (float(cb.max()), crossed, succ[rid], lead)
        per_sig[rid] = {"smd": float(sm.max()), "rnd": float(rn.max()), "act": float(a.max())}

    # PHASE 2: write all the re-scored steps in one short transaction (reads are already done)
    with store.engine.begin() as cx:
        for rid, f, smv, rnv, acv in writes:
            cx.execute(sa.update(store.steps)
                       .where((store.steps.c.run_id == rid) & (store.steps.c.step_idx == f))
                       .values(smd=smv, rnd=rnv, ace=acv))

    # --- validation: does the combined score separate success from failure? ---
    rids = list(per_run)
    scores = np.array([per_run[r][0] for r in rids])
    labels = np.array([0 if per_run[r][2] else 1 for r in rids])   # 1 = failure
    fails = [r for r in rids if not per_run[r][2]]
    succs = [r for r in rids if per_run[r][2]]
    report = {
        "threshold": threshold, "pctl": PCTL_RUN,
        "n_success_runs_calib": len(sel), "n_calib_steps": int(len(S)),
        "embed_dim": int(E.shape[1]), "combine": "mean_capped_z(smd,rnd,act)", "cap": CAP, "norm": norm,
        "n_runs_scored": len(rids), "n_fail_runs": len(fails), "n_succ_runs": len(succs),
        "auc_fail_vs_success": _auc(scores, labels),
        "auc_per_signal": {k: _auc(np.array([per_sig[r][k] for r in rids]), labels) for k in SIGNALS},
        "fail_caught_frac": (float(np.mean([per_run[r][1] for r in fails])) if fails else None),
        "false_alarm_frac": (float(np.mean([per_run[r][1] for r in succs])) if succs else None),
        "mean_lead_steps_on_caught_fails": (
            float(np.mean([per_run[r][3] for r in fails if per_run[r][1]]))
            if any(per_run[r][1] for r in fails) else None),
    }

    # --- persist (detectors + threshold + report) for the Stage-2 live Monitor ---
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CALIB_DIR / "smd.npz", mean=smd.mean, inv=smd.inv)
    save_file({k: v.cpu().contiguous() for k, v in rnd.predictor.state_dict().items()},
              str(CALIB_DIR / "rnd_predictor.safetensors"))
    save_file({k: v.cpu().contiguous() for k, v in rnd.target.state_dict().items()},
              str(CALIB_DIR / "rnd_target.safetensors"))
    np.savez(CALIB_DIR / "rnd_innorm.npz",
             mean=rnd.mean.detach().cpu().numpy(), std=rnd.std.detach().cpu().numpy())
    (CALIB_DIR / "calib.json").write_text(json.dumps(
        {"embed_dim": int(E.shape[1]), "norm": norm, "threshold": threshold,
         "combine": "capped_z_mean", "cap": CAP, "signals": list(SIGNALS), "pctl": PCTL_RUN}, indent=2))
    (CALIB_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--auto", action="store_true", help="calibrate on ALL successful runs")
    args = ap.parse_args()
    calibrate(run_ids=args.runs, auto=args.auto or not args.runs)


if __name__ == "__main__":
    main()
