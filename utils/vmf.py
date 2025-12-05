# utils/vmf.py
import numpy as np
from scipy.special import iv  # modified Bessel I_v


def _unit(x, eps=1e-12):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def vmf_logC_d(kappa: float, d: int) -> float:
    """
    log C_d(kappa) for vMF on S^{d-1} embedded in R^d:
    C_d(kappa)=kappa^{d/2-1}/((2π)^{d/2} I_{d/2-1}(kappa))
    """
    kappa = float(kappa)
    if kappa <= 0:
        return - (d/2) * np.log(2*np.pi)
    nu = d/2 - 1.0
    return (nu)*np.log(kappa + 1e-12) - (d/2)*np.log(2*np.pi) - np.log(iv(nu, kappa) + 1e-300)


def vmf_segment_loglike(Xseg: np.ndarray, g: np.ndarray, kappa: float) -> float:
    D = Xseg.shape[1]
    logC = vmf_logC_d(kappa, D)
    Vs = _unit(Xseg[1:] - Xseg[:-1])
    Us = _unit(g[None, :] - Xseg[:-1])
    cos = np.sum(Vs * Us, axis=1)
    return float((kappa * cos).sum() + len(cos) * logC)


def vmf_grad_wrt_g(Xseg: np.ndarray, g: np.ndarray, kappa: float) -> np.ndarray:
    """
    ∂/∂g [kappa * <v_hat, u_hat>] =
       kappa * (I/||u|| - uu^T/||u||^3) @ v_hat
    """
    D = Xseg.shape[1]
    grad = np.zeros(D, dtype=float)
    for t in range(len(Xseg)-1):
        v = Xseg[t+1] - Xseg[t]
        v_n = np.linalg.norm(v)
        if v_n < 1e-12:
            continue
        v_hat = v / v_n

        u = g - Xseg[t]
        r = np.linalg.norm(u)
        if r < 1e-12:
            continue

        Jv = v_hat / r - (u * (np.dot(u, v_hat))) / (r**3 + 1e-12)
        grad += kappa * Jv
    return grad
