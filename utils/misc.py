# utils/misc.py
import numpy as np


def logsumexp(a):
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m) + 1e-300))


def weighted_quantile(values, weights, quantile):
    values = np.asarray(values).astype(float)
    weights = np.asarray(weights).astype(float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    v = values[mask]; w = weights[mask]
    if v.size == 0 or w.sum() <= 1e-12:
        return np.nan
    idx = np.argsort(v)
    v = v[idx]; w = w[idx]
    cw = np.cumsum(w)
    cutoff = quantile * cw[-1]
    j = np.searchsorted(cw, cutoff, side="left")
    if j == 0:
        return v[0]
    w_lo = cw[j-1]; w_hi = cw[j]
    frac = (cutoff - w_lo) / max(w_hi - w_lo, 1e-12)
    return v[j-1]*(1-frac) + v[j]*frac


def tau_post_to_time_responsibilities(q, tau_candidates, T):
    q_at = np.zeros(T)
    valid = (tau_candidates >= 0) & (tau_candidates < T)
    q_at[tau_candidates[valid]] = q[valid]
    cumsum_q = np.cumsum(q_at)
    r2 = cumsum_q
    r1 = 1.0 - np.concatenate([[0.0], cumsum_q[:-1]])
    return r1, r2
