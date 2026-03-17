from __future__ import annotations

import math
import numpy as np


class BaseEmissionModel:
    model_type = "base"

    def interval(self, q_low=None, q_high=None):
        if q_low is None:
            q_low = getattr(self, "q_low", 0.1)
        if q_high is None:
            q_high = getattr(self, "q_high", 0.9)
        if float(q_low) == float(getattr(self, "q_low", q_low)) and float(q_high) == float(getattr(self, "q_high", q_high)):
            return float(self.L), float(self.U)
        raise NotImplementedError("Custom interval queries are not implemented for this model.")

    def init_from_data(self, xs, ws=None):
        self.m_step_update(xs, ws)

    def get_summary(self):
        raise NotImplementedError


class GaussianModel(BaseEmissionModel):
    model_type = "gauss"

    def __init__(self, mu=None, sigma=None, fixed_sigma=None, q_low=0.1, q_high=0.9):
        self.mu = 0.0 if mu is None else float(mu)
        self.sigma = 1.0 if sigma is None else float(max(float(sigma), 1e-6))
        self.fixed_sigma = None if fixed_sigma is None else float(max(float(fixed_sigma), 1e-6))
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self._update_interval()

    def _quantile_z(self, q):
        # Common symmetric quantiles used by this project.
        table = {
            0.1: -1.2815515655446004,
            0.9: 1.2815515655446004,
            0.05: -1.6448536269514729,
            0.95: 1.6448536269514722,
        }
        q = float(q)
        if q in table:
            return table[q]
        # Fallback probit approximation via erfinv approximation.
        a = 0.147
        y = 2.0 * q - 1.0
        ln = math.log(max(1.0 - y * y, 1e-12))
        term = 2.0 / (math.pi * a) + ln / 2.0
        erfinv = math.copysign(math.sqrt(max(math.sqrt(term * term - ln / a) - term, 0.0)), y)
        return math.sqrt(2.0) * erfinv

    def _update_interval(self):
        z_low = self._quantile_z(self.q_low)
        z_high = self._quantile_z(self.q_high)
        self.L = float(self.mu + z_low * self.sigma)
        self.U = float(self.mu + z_high * self.sigma)

    def logpdf(self, x):
        x = np.asarray(x, float)
        sigma = self.fixed_sigma if self.fixed_sigma is not None else self.sigma
        sigma = max(float(sigma), 1e-6)
        return -0.5 * np.log(2.0 * np.pi * sigma ** 2) - 0.5 * ((x - self.mu) ** 2) / (sigma ** 2)

    def m_step_update(self, xs, ws=None):
        vals = np.concatenate([np.asarray(x, float).reshape(-1) for x in xs], axis=0)
        if ws is None:
            weights = np.ones_like(vals)
        else:
            weights = np.concatenate([np.asarray(w, float).reshape(-1) for w in ws], axis=0)
        total_w = float(np.sum(weights))
        if total_w <= 1e-12:
            return
        mu = float(np.sum(weights * vals) / total_w)
        self.mu = mu
        if self.fixed_sigma is None:
            var = float(np.sum(weights * (vals - mu) ** 2) / total_w)
            self.sigma = max(math.sqrt(max(var, 1e-12)), 1e-6)
        else:
            self.sigma = float(self.fixed_sigma)
        self._update_interval()

    def get_summary(self):
        return {
            "type": self.model_type,
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "L": float(self.L),
            "U": float(self.U),
        }


class ZeroMeanGaussianModel(GaussianModel):
    model_type = "gauss_zero"

    def __init__(self, sigma=None, fixed_sigma=None, q_low=0.1, q_high=0.9):
        super().__init__(mu=0.0, sigma=sigma, fixed_sigma=fixed_sigma, q_low=q_low, q_high=q_high)

    def m_step_update(self, xs, ws=None):
        vals = np.concatenate([np.asarray(x, float).reshape(-1) for x in xs], axis=0)
        if ws is None:
            weights = np.ones_like(vals)
        else:
            weights = np.concatenate([np.asarray(w, float).reshape(-1) for w in ws], axis=0)
        total_w = float(np.sum(weights))
        if total_w <= 1e-12:
            return
        self.mu = 0.0
        if self.fixed_sigma is None:
            var = float(np.sum(weights * (vals ** 2)) / total_w)
            self.sigma = max(math.sqrt(max(var, 1e-12)), 1e-6)
        else:
            self.sigma = float(self.fixed_sigma)
        self._update_interval()

    def get_summary(self):
        return {
            "type": self.model_type,
            "mu": 0.0,
            "sigma": float(self.sigma),
            "L": float(self.L),
            "U": float(self.U),
        }


class MarginExpLowerEmission(BaseEmissionModel):
    model_type = "margin_exp_lower"

    def __init__(
        self,
        b_init=0.0,
        lam_init=1.0,
        q_low=0.1,
        q_high=0.9,
        violation_tau=0.25,
        violation_scale=2.0,
    ):
        self.b = float(b_init)
        self.lam = max(float(lam_init), 1e-6)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.violation_tau = max(float(violation_tau), 1e-6)
        self.violation_scale = max(float(violation_scale), 0.0)
        self._update_interval()

    def _update_interval(self):
        def q_to_margin(q):
            return -self.lam * math.log(max(1.0 - float(q), 1e-12))
        self.L = float(self.b + q_to_margin(self.q_low))
        self.U = float(self.b + q_to_margin(self.q_high))

    def logpdf(self, x):
        x = np.asarray(x, float)
        residual = x - self.b
        positive_cost = np.maximum(residual, 0.0) / self.lam
        violation_barrier = self.violation_scale * np.log1p(np.exp((-residual) / self.violation_tau))
        return -math.log(self.lam) - positive_cost - violation_barrier

    def m_step_update(self, xs, ws=None):
        vals = np.concatenate([np.asarray(x, float).reshape(-1) for x in xs], axis=0)
        if ws is None:
            weights = np.ones_like(vals)
        else:
            weights = np.concatenate([np.asarray(w, float).reshape(-1) for w in ws], axis=0)
        total_w = float(np.sum(weights))
        if total_w <= 1e-12:
            return
        # Robust lower-bound estimate with weighted lower quantile.
        order = np.argsort(vals)
        vals_sorted = vals[order]
        w_sorted = weights[order]
        cdf = np.cumsum(w_sorted) / total_w
        idx = int(np.searchsorted(cdf, 0.1, side="left"))
        self.b = float(vals_sorted[min(idx, len(vals_sorted) - 1)])
        residual = np.maximum(vals - self.b, 0.0)
        mean_residual = float(np.sum(weights * residual) / total_w)
        self.lam = max(mean_residual, 1e-6)
        self._update_interval()

    def get_summary(self):
        return {
            "type": self.model_type,
            "b": float(self.b),
            "lam": float(self.lam),
            "L": float(self.L),
            "U": float(self.U),
            "violation_tau": float(self.violation_tau),
            "violation_scale": float(self.violation_scale),
        }
