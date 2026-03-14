from __future__ import annotations

import math
import numpy as np


def _unit(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / max(n, 1e-12)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def vmf_logC_d(kappa: float, dim: int) -> float:
    # Lightweight approximation sufficient for relative scoring and plotting.
    kappa = float(max(kappa, 1e-12))
    dim = int(dim)
    return (dim / 2.0 - 1.0) * math.log(kappa + 1e-12) - (dim / 2.0) * math.log(2.0 * math.pi) - kappa


def vmf_grad_wrt_g(X: np.ndarray, g: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    X = np.asarray(X, float)
    g = np.asarray(g, float)
    if len(X) < 2:
        return np.zeros_like(g)
    v = X[1] - X[0]
    v_n = np.linalg.norm(v)
    if v_n < 1e-12:
        return np.zeros_like(g)
    v_hat = v / v_n

    u = g - X[0]
    r = np.linalg.norm(u)
    if r < 1e-12:
        return np.zeros_like(g)

    return float(kappa) * (v_hat / r - (u * np.dot(u, v_hat)) / (r ** 3 + 1e-12))
