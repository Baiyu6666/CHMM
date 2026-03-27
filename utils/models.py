from __future__ import annotations

import math
import numpy as np


def _weighted_quantile(values, weights, q):
    vals = np.asarray(values, dtype=float).reshape(-1)
    ws = np.asarray(weights, dtype=float).reshape(-1)
    if vals.size == 0:
        return np.nan
    if ws.size != vals.size:
        raise ValueError("weights must match values for weighted quantile.")
    total_w = float(np.sum(ws))
    if total_w <= 1e-12:
        return float(np.quantile(vals, q))
    order = np.argsort(vals)
    vals = vals[order]
    ws = ws[order]
    cdf = np.cumsum(ws) / total_w
    idx = int(np.searchsorted(cdf, float(np.clip(q, 0.0, 1.0)), side="left"))
    return float(vals[min(idx, len(vals) - 1)])


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


class StudentTModel(BaseEmissionModel):
    model_type = "student_t"

    def __init__(self, mu=None, sigma=None, nu=3.0, q_low=0.1, q_high=0.9):
        self.mu = 0.0 if mu is None else float(mu)
        self.sigma = 1.0 if sigma is None else float(max(float(sigma), 1e-6))
        self.nu = max(float(nu), 1e-3)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self._update_interval()

    def _t_quantile_approx(self, q):
        # Keep interval support simple; for plotting we use a Gaussian-like
        # approximation widened slightly by heavy tails.
        base = GaussianModel(mu=0.0, sigma=1.0, q_low=q, q_high=1.0 - q)._quantile_z(q)
        widen = math.sqrt(max(self.nu / max(self.nu - 2.0, 1e-6), 1.0)) if self.nu > 2.0 else 2.0
        return float(base * widen)

    def _update_interval(self):
        z_low = self._t_quantile_approx(self.q_low)
        z_high = self._t_quantile_approx(self.q_high)
        self.L = float(self.mu + z_low * self.sigma)
        self.U = float(self.mu + z_high * self.sigma)

    def logpdf(self, x):
        x = np.asarray(x, float)
        sigma = max(float(self.sigma), 1e-6)
        nu = max(float(self.nu), 1e-6)
        z2 = ((x - self.mu) / sigma) ** 2
        log_norm = (
            math.lgamma((nu + 1.0) / 2.0)
            - math.lgamma(nu / 2.0)
            - 0.5 * (math.log(nu) + math.log(math.pi))
            - math.log(sigma)
        )
        return log_norm - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)

    def m_step_update(self, xs, ws=None):
        vals = np.concatenate([np.asarray(x, float).reshape(-1) for x in xs], axis=0)
        if ws is None:
            weights = np.ones_like(vals)
        else:
            weights = np.concatenate([np.asarray(w, float).reshape(-1) for w in ws], axis=0)
        total_w = float(np.sum(weights))
        if total_w <= 1e-12:
            return
        # Fixed-nu EM / IRLS update for a proper Student-t fit.
        order = np.argsort(vals)
        vals_sorted = vals[order]
        w_sorted = weights[order]
        cdf = np.cumsum(w_sorted) / total_w
        idx = int(np.searchsorted(cdf, 0.5, side="left"))
        mu = float(vals_sorted[min(idx, len(vals_sorted) - 1)])
        centered0 = vals - mu
        sigma = max(float(np.sqrt(np.sum(weights * centered0 * centered0) / total_w)), 1e-6)
        nu = max(float(self.nu), 1e-6)
        for _ in range(25):
            z2 = ((vals - mu) / max(sigma, 1e-6)) ** 2
            latent_w = (nu + 1.0) / (nu + z2)
            eff_w = weights * latent_w
            eff_total = float(np.sum(eff_w))
            if eff_total <= 1e-12:
                break
            next_mu = float(np.sum(eff_w * vals) / eff_total)
            centered = vals - next_mu
            next_sigma = max(float(np.sqrt(np.sum(weights * latent_w * centered * centered) / total_w)), 1e-6)
            if abs(next_mu - mu) < 1e-6 and abs(next_sigma - sigma) < 1e-6:
                mu = next_mu
                sigma = next_sigma
                break
            mu = next_mu
            sigma = next_sigma
        self.mu = float(mu)
        self.sigma = max(float(sigma), 1e-6)
        self._update_interval()

    def get_summary(self):
        return {
            "type": self.model_type,
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "nu": float(self.nu),
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
        tail_nu=3.0,
    ):
        self.b = float(b_init)
        self.lam = max(float(lam_init), 1e-6)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.tail_nu = max(float(tail_nu), 1e-3)
        self._update_interval()

    def _update_interval(self):
        def q_to_margin(q):
            alpha = max(self.tail_nu, 1e-6)
            return self.lam * ((max(1.0 - float(q), 1e-12)) ** (-1.0 / alpha) - 1.0)
        self.L = float(self.b + q_to_margin(self.q_low))
        self.U = float(self.b + q_to_margin(self.q_high))

    def logpdf(self, x):
        x = np.asarray(x, float)
        residual = x - self.b
        alpha = max(self.tail_nu, 1e-6)
        logp = np.full_like(residual, -1e9, dtype=float)
        mask = residual >= 0.0
        if np.any(mask):
            z = residual[mask] / max(self.lam, 1e-6)
            logp[mask] = math.log(alpha) - math.log(max(self.lam, 1e-6)) - (alpha + 1.0) * np.log1p(z)
        return logp

    def m_step_update(self, xs, ws=None):
        vals = np.concatenate([np.asarray(x, float).reshape(-1) for x in xs], axis=0)
        if ws is None:
            weights = np.ones_like(vals)
        else:
            weights = np.concatenate([np.asarray(w, float).reshape(-1) for w in ws], axis=0)
        total_w = float(np.sum(weights))
        if total_w <= 1e-12:
            return
        alpha = max(self.tail_nu, 1.0 + 1e-6)
        x_min = float(np.min(vals))
        x_max = float(np.max(vals))
        span = max(float(x_max - x_min), 1e-4)
        b_grid = np.linspace(x_min - 0.08 * span - 0.01, x_min, 48)
        best_total = -np.inf
        best_b = float(self.b)
        best_lam = max(float(self.lam), 1e-6)
        for b_hat in b_grid:
            residual = vals - float(b_hat)
            logp = np.full_like(residual, -1e9, dtype=float)
            mask = residual >= 0.0
            if not np.any(mask):
                continue
            slack = residual[mask]
            mean_slack = float(np.sum(weights[mask] * slack) / max(float(np.sum(weights[mask])), 1e-12))
            lam = max(mean_slack * (alpha - 1.0), 1e-6)
            z = slack / lam
            logp[mask] = math.log(alpha) - math.log(lam) - (alpha + 1.0) * np.log1p(z)
            total = float(np.sum(weights * logp))
            if total > best_total:
                best_total = total
                best_b = float(b_hat)
                best_lam = float(lam)
        self.b = float(best_b)
        self.lam = max(float(best_lam), 1e-6)
        self._update_interval()

    def get_summary(self):
        return {
            "type": self.model_type,
            "b": float(self.b),
            "lam": float(self.lam),
            "L": float(self.L),
            "U": float(self.U),
            "tail_nu": float(self.tail_nu),
        }


class MarginExpLowerLeftHNEmission(BaseEmissionModel):
    model_type = "margin_exp_lower_left_hn"

    def __init__(
        self,
        b_init=0.0,
        lam_init=1.0,
        sigma_left_init=0.1,
        pi_left_init=0.08,
        q_low=0.1,
        q_high=0.9,
        tail_nu=3.0,
        pi_left_max=0.1,
    ):
        self.b = float(b_init)
        self.lam = max(float(lam_init), 1e-6)
        self.sigma_left = max(float(sigma_left_init), 1e-6)
        self.pi_left_max = float(np.clip(float(pi_left_max), 1e-3, 0.49))
        self.pi_left = float(np.clip(float(pi_left_init), 1e-4, self.pi_left_max))
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.tail_nu = max(float(tail_nu), 1e-3)
        self._update_interval()

    def _update_interval(self):
        def q_to_margin(q):
            alpha = max(self.tail_nu, 1e-6)
            return self.lam * ((max(1.0 - float(q), 1e-12)) ** (-1.0 / alpha) - 1.0)
        self.L = float(self.b + q_to_margin(self.q_low))
        self.U = float(self.b + q_to_margin(self.q_high))

    def logpdf(self, x):
        x = np.asarray(x, float)
        residual = x - self.b
        alpha = max(self.tail_nu, 1e-6)
        lam = max(float(self.lam), 1e-6)
        sigma_left = max(float(self.sigma_left), 1e-6)
        pi_left = float(np.clip(self.pi_left, 1e-8, 1.0 - 1e-8))
        logp = np.empty_like(residual, dtype=float)

        right_mask = residual >= 0.0
        if np.any(right_mask):
            z = residual[right_mask] / lam
            logp[right_mask] = (
                math.log(max(1.0 - pi_left, 1e-12))
                + math.log(alpha)
                - math.log(lam)
                - (alpha + 1.0) * np.log1p(z)
            )
        if np.any(~right_mask):
            left_slack = -residual[~right_mask]
            logp[~right_mask] = (
                math.log(pi_left)
                + 0.5 * math.log(2.0 / math.pi)
                - math.log(sigma_left)
                - 0.5 * (left_slack / sigma_left) ** 2
            )
        return logp

    def m_step_update(self, xs, ws=None):
        vals = np.concatenate([np.asarray(x, float).reshape(-1) for x in xs], axis=0)
        if ws is None:
            weights = np.ones_like(vals)
        else:
            weights = np.concatenate([np.asarray(w, float).reshape(-1) for w in ws], axis=0)
        total_w = float(np.sum(weights))
        if total_w <= 1e-12:
            return

        alpha = max(self.tail_nu, 1.0 + 1e-6)
        x_min = float(np.min(vals))
        x_max = float(np.max(vals))
        span = max(float(x_max - x_min), 1e-4)
        anchor_q = float(np.clip(self.pi_left_max, 0.05, 0.2))
        b_anchor = _weighted_quantile(vals, weights, anchor_q)
        left_width = max(0.05 * span, 0.01)
        right_width = max(0.015 * span, 0.003)
        grid_lo = max(x_min - 0.01, b_anchor - left_width)
        grid_hi = min(x_max, b_anchor + right_width)
        if grid_hi <= grid_lo + 1e-8:
            grid_lo = min(grid_lo, b_anchor - 0.01)
            grid_hi = max(grid_hi, b_anchor + 0.01)
        b_grid = np.linspace(grid_lo, grid_hi, 56)
        best_total = -np.inf
        best_b = float(self.b)
        best_lam = max(float(self.lam), 1e-6)
        best_sigma_left = max(float(self.sigma_left), 1e-6)
        best_pi_left = float(np.clip(self.pi_left, 1e-4, self.pi_left_max))

        for b_hat in b_grid:
            residual = vals - float(b_hat)
            right_mask = residual >= 0.0
            if not np.any(right_mask):
                continue

            right_w = weights[right_mask]
            right_slack = residual[right_mask]
            right_total = float(np.sum(right_w))
            if right_total <= 1e-12:
                continue
            mean_slack = float(np.sum(right_w * right_slack) / right_total)
            lam = max(mean_slack * (alpha - 1.0), 1e-6)

            left_mask = ~right_mask
            left_total = float(np.sum(weights[left_mask]))
            raw_pi_left = left_total / total_w
            pi_left = float(np.clip(raw_pi_left, 1e-4, self.pi_left_max))
            pi_left = min(pi_left, 1.0 - 1e-4)

            if np.any(left_mask) and left_total > 1e-12:
                left_slack = -residual[left_mask]
                sigma_left = max(float(np.sqrt(np.sum(weights[left_mask] * left_slack * left_slack) / left_total)), 1e-6)
            else:
                sigma_left = max(0.05 * span, 1e-4)

            logp = np.empty_like(residual, dtype=float)
            z = right_slack / lam
            logp[right_mask] = (
                math.log(max(1.0 - pi_left, 1e-12))
                + math.log(alpha)
                - math.log(lam)
                - (alpha + 1.0) * np.log1p(z)
            )
            if np.any(left_mask):
                left_slack = -residual[left_mask]
                logp[left_mask] = (
                    math.log(pi_left)
                    + 0.5 * math.log(2.0 / math.pi)
                    - math.log(sigma_left)
                    - 0.5 * (left_slack / sigma_left) ** 2
                )

            total = float(np.sum(weights * logp))
            if total > best_total:
                best_total = total
                best_b = float(b_hat)
                best_lam = float(lam)
                best_sigma_left = float(sigma_left)
                best_pi_left = float(pi_left)

        self.b = float(best_b)
        self.lam = max(float(best_lam), 1e-6)
        self.sigma_left = max(float(best_sigma_left), 1e-6)
        self.pi_left = float(np.clip(best_pi_left, 1e-4, self.pi_left_max))
        self._update_interval()

    def get_summary(self):
        return {
            "type": self.model_type,
            "b": float(self.b),
            "lam": float(self.lam),
            "sigma_left": float(self.sigma_left),
            "pi_left": float(self.pi_left),
            "L": float(self.L),
            "U": float(self.U),
            "tail_nu": float(self.tail_nu),
        }


class MarginExpUpperEmission(BaseEmissionModel):
    model_type = "margin_exp_upper"

    def __init__(
        self,
        b_init=0.0,
        lam_init=1.0,
        q_low=0.1,
        q_high=0.9,
        tail_nu=3.0,
    ):
        self.b = float(b_init)
        self.lam = max(float(lam_init), 1e-6)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.tail_nu = max(float(tail_nu), 1e-3)
        self._update_interval()

    def _update_interval(self):
        def q_to_margin(q):
            alpha = max(self.tail_nu, 1e-6)
            return self.lam * ((max(1.0 - float(q), 1e-12)) ** (-1.0 / alpha) - 1.0)
        self.L = float(self.b - q_to_margin(self.q_high))
        self.U = float(self.b - q_to_margin(self.q_low))

    def logpdf(self, x):
        x = np.asarray(x, float)
        residual = self.b - x
        alpha = max(self.tail_nu, 1e-6)
        logp = np.full_like(residual, -1e9, dtype=float)
        mask = residual >= 0.0
        if np.any(mask):
            z = residual[mask] / max(self.lam, 1e-6)
            logp[mask] = math.log(alpha) - math.log(max(self.lam, 1e-6)) - (alpha + 1.0) * np.log1p(z)
        return logp

    def m_step_update(self, xs, ws=None):
        vals = np.concatenate([np.asarray(x, float).reshape(-1) for x in xs], axis=0)
        if ws is None:
            weights = np.ones_like(vals)
        else:
            weights = np.concatenate([np.asarray(w, float).reshape(-1) for w in ws], axis=0)
        total_w = float(np.sum(weights))
        if total_w <= 1e-12:
            return
        alpha = max(self.tail_nu, 1.0 + 1e-6)
        x_min = float(np.min(vals))
        x_max = float(np.max(vals))
        span = max(float(x_max - x_min), 1e-4)
        b_grid = np.linspace(x_max, x_max + 0.08 * span + 0.01, 48)
        best_total = -np.inf
        best_b = float(self.b)
        best_lam = max(float(self.lam), 1e-6)
        for b_hat in b_grid:
            residual = float(b_hat) - vals
            logp = np.full_like(residual, -1e9, dtype=float)
            mask = residual >= 0.0
            if not np.any(mask):
                continue
            slack = residual[mask]
            mean_slack = float(np.sum(weights[mask] * slack) / max(float(np.sum(weights[mask])), 1e-12))
            lam = max(mean_slack * (alpha - 1.0), 1e-6)
            z = slack / lam
            logp[mask] = math.log(alpha) - math.log(lam) - (alpha + 1.0) * np.log1p(z)
            total = float(np.sum(weights * logp))
            if total > best_total:
                best_total = total
                best_b = float(b_hat)
                best_lam = float(lam)
        self.b = float(best_b)
        self.lam = max(float(best_lam), 1e-6)
        self._update_interval()

    def get_summary(self):
        return {
            "type": self.model_type,
            "b": float(self.b),
            "lam": float(self.lam),
            "L": float(self.L),
            "U": float(self.U),
            "tail_nu": float(self.tail_nu),
        }


class MarginExpUpperRightHNEmission(BaseEmissionModel):
    model_type = "margin_exp_upper_right_hn"

    def __init__(
        self,
        b_init=0.0,
        lam_init=1.0,
        sigma_right_init=0.1,
        pi_right_init=0.08,
        q_low=0.1,
        q_high=0.9,
        tail_nu=3.0,
        pi_right_max=0.1,
    ):
        self.b = float(b_init)
        self.lam = max(float(lam_init), 1e-6)
        self.sigma_right = max(float(sigma_right_init), 1e-6)
        self.pi_right_max = float(np.clip(float(pi_right_max), 1e-3, 0.49))
        self.pi_right = float(np.clip(float(pi_right_init), 1e-4, self.pi_right_max))
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.tail_nu = max(float(tail_nu), 1e-3)
        self._update_interval()

    def _update_interval(self):
        def q_to_margin(q):
            alpha = max(self.tail_nu, 1e-6)
            return self.lam * ((max(1.0 - float(q), 1e-12)) ** (-1.0 / alpha) - 1.0)
        self.L = float(self.b - q_to_margin(self.q_high))
        self.U = float(self.b - q_to_margin(self.q_low))

    def logpdf(self, x):
        x = np.asarray(x, float)
        residual = self.b - x
        alpha = max(self.tail_nu, 1e-6)
        lam = max(float(self.lam), 1e-6)
        sigma_right = max(float(self.sigma_right), 1e-6)
        pi_right = float(np.clip(self.pi_right, 1e-8, 1.0 - 1e-8))
        logp = np.empty_like(residual, dtype=float)

        left_mask = residual >= 0.0
        if np.any(left_mask):
            z = residual[left_mask] / lam
            logp[left_mask] = (
                math.log(max(1.0 - pi_right, 1e-12))
                + math.log(alpha)
                - math.log(lam)
                - (alpha + 1.0) * np.log1p(z)
            )
        if np.any(~left_mask):
            right_slack = -residual[~left_mask]
            logp[~left_mask] = (
                math.log(pi_right)
                + 0.5 * math.log(2.0 / math.pi)
                - math.log(sigma_right)
                - 0.5 * (right_slack / sigma_right) ** 2
            )
        return logp

    def m_step_update(self, xs, ws=None):
        vals = np.concatenate([np.asarray(x, float).reshape(-1) for x in xs], axis=0)
        if ws is None:
            weights = np.ones_like(vals)
        else:
            weights = np.concatenate([np.asarray(w, float).reshape(-1) for w in ws], axis=0)
        total_w = float(np.sum(weights))
        if total_w <= 1e-12:
            return

        alpha = max(self.tail_nu, 1.0 + 1e-6)
        x_min = float(np.min(vals))
        x_max = float(np.max(vals))
        span = max(float(x_max - x_min), 1e-4)
        anchor_q = float(np.clip(1.0 - self.pi_right_max, 0.8, 0.95))
        b_anchor = _weighted_quantile(vals, weights, anchor_q)
        left_width = max(0.015 * span, 0.003)
        right_width = max(0.05 * span, 0.01)
        grid_lo = max(x_min, b_anchor - left_width)
        grid_hi = min(x_max + 0.01, b_anchor + right_width)
        if grid_hi <= grid_lo + 1e-8:
            grid_lo = min(grid_lo, b_anchor - 0.01)
            grid_hi = max(grid_hi, b_anchor + 0.01)
        b_grid = np.linspace(grid_lo, grid_hi, 56)
        best_total = -np.inf
        best_b = float(self.b)
        best_lam = max(float(self.lam), 1e-6)
        best_sigma_right = max(float(self.sigma_right), 1e-6)
        best_pi_right = float(np.clip(self.pi_right, 1e-4, self.pi_right_max))

        for b_hat in b_grid:
            residual = float(b_hat) - vals
            left_mask = residual >= 0.0
            if not np.any(left_mask):
                continue

            left_w = weights[left_mask]
            left_slack = residual[left_mask]
            left_total = float(np.sum(left_w))
            if left_total <= 1e-12:
                continue
            mean_slack = float(np.sum(left_w * left_slack) / left_total)
            lam = max(mean_slack * (alpha - 1.0), 1e-6)

            right_mask = ~left_mask
            right_total = float(np.sum(weights[right_mask]))
            raw_pi_right = right_total / total_w
            pi_right = float(np.clip(raw_pi_right, 1e-4, self.pi_right_max))
            pi_right = min(pi_right, 1.0 - 1e-4)

            if np.any(right_mask) and right_total > 1e-12:
                right_slack = -residual[right_mask]
                sigma_right = max(float(np.sqrt(np.sum(weights[right_mask] * right_slack * right_slack) / right_total)), 1e-6)
            else:
                sigma_right = max(0.05 * span, 1e-4)

            logp = np.empty_like(residual, dtype=float)
            z = left_slack / lam
            logp[left_mask] = (
                math.log(max(1.0 - pi_right, 1e-12))
                + math.log(alpha)
                - math.log(lam)
                - (alpha + 1.0) * np.log1p(z)
            )
            if np.any(right_mask):
                right_slack = -residual[right_mask]
                logp[right_mask] = (
                    math.log(pi_right)
                    + 0.5 * math.log(2.0 / math.pi)
                    - math.log(sigma_right)
                    - 0.5 * (right_slack / sigma_right) ** 2
                )

            total = float(np.sum(weights * logp))
            if total > best_total:
                best_total = total
                best_b = float(b_hat)
                best_lam = float(lam)
                best_sigma_right = float(sigma_right)
                best_pi_right = float(pi_right)

        self.b = float(best_b)
        self.lam = max(float(best_lam), 1e-6)
        self.sigma_right = max(float(best_sigma_right), 1e-6)
        self.pi_right = float(np.clip(best_pi_right, 1e-4, self.pi_right_max))
        self._update_interval()

    def get_summary(self):
        return {
            "type": self.model_type,
            "b": float(self.b),
            "lam": float(self.lam),
            "sigma_right": float(self.sigma_right),
            "pi_right": float(self.pi_right),
            "L": float(self.L),
            "U": float(self.U),
            "tail_nu": float(self.tail_nu),
        }
