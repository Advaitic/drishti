"""The combine rule, in one dependency-free place.

Three things need to turn raw detector scores into one number the same way: `drishti.calibrate`
(which fits the normalization and sets the threshold), the dashboard backend (which sweeps the
threshold over every logged rollout), and the AHA benchmark (which runs the same detectors on a
different robot). When each carried its own copy they drifted — the dashboard defaulted to a
capped-z ceiling of 8 that calibration had already measured as harmful. So the rule lives here and
everything imports it.

Pure stdlib + numpy on purpose: the server imports this and must not pull torch in with it.

The rule itself:

    capped-z   clip((x - mu_success) / sigma_success, 0, CAP) / CAP  ->  each signal in [0, 1]
    combine    the MEAN of the capped-z's
    threshold  the PCTL_RUN-th percentile of the per-rollout PEAK of that mean, over successes

Why each choice, since they were all arrived at the hard way:

- **Capped, not raw, z.** After whitening, RND's spread over successful rollouts is razor-thin, so
  its raw z runs ~100x past SMD's and ACE's and single-handedly decides the combined score. Bounding
  every signal to [0, 1] puts them back on speaking terms.
- **CAP = 32, not 8.** The cap has to sit above the z-range real out-of-distribution data reaches.
  At 8 both held-out successes and failures saturate to 1.0 and the separation disappears
  (held-out AUC 0.78 at cap 8 vs 0.99 at cap >= 32).
- **Mean, not max.** `max` fires on whichever detector is loudest, so one detector grazing its
  threshold is a false alarm. The mean is agreement-aware: a lone graze gets averaged down. In our
  ablation it won on both AUC (0.986 vs 0.979) and false-alarms-at-full-recall (1 vs 2).
- **Percentile of run peaks, not of steps.** The thing you want to bound is "how often does a good
  rollout trip the alarm", which is a per-rollout question. Bounding steps at the 95th percentile
  sounds tight until you notice that over a ~50-step rollout, 5% of steps compounds to ~73% of
  rollouts grazing at least once.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

# Signals in the production cascade, in the order the reports list them.
SIGNALS: tuple[str, ...] = ("smd", "rnd", "act")

CAP = 32.0
PCTL_RUN = 90.0

Norm = Mapping[str, Sequence[float]]   # signal -> (mean, std) over successful steps


def capped_z(x, mu: float, sd: float, cap: float = CAP):
    """Scale a raw signal to [0, 1] against the success distribution. Works on scalars or arrays."""
    return np.clip((np.asarray(x, dtype=float) - mu) / sd, 0.0, cap) / cap


def combined_score(values: Mapping[str, float], norm: Norm, cap: float = CAP,
                   signals: Iterable[str] = SIGNALS) -> float:
    """Mean of capped-z over `signals`, for one step. Missing signals count as 0 (fully nominal)."""
    keys = [k for k in signals]
    return float(np.mean([capped_z(values.get(k) or 0.0, norm[k][0], norm[k][1], cap) for k in keys]))


def combined_series(series: Mapping[str, Sequence[float]], norm: Norm, cap: float = CAP,
                    signals: Iterable[str] = SIGNALS) -> np.ndarray:
    """Vectorized `combined_score` over aligned per-step arrays. Returns one score per step."""
    keys = [k for k in signals]
    stacked = np.stack([capped_z(series[k], norm[k][0], norm[k][1], cap) for k in keys])
    return stacked.mean(axis=0)


def conformal_threshold(run_peaks: Sequence[float], pctl: float = PCTL_RUN) -> float:
    """Threshold a new successful rollout exceeds with probability ~<= (100 - pctl)%."""
    return float(np.percentile(np.asarray(run_peaks, dtype=float), pctl))
