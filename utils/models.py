# utils/models.py
# ------------------------------------------------------------
# Unified emission models for 1D features:
#   - BaseEmission: common interface
#   - GaussianModel: 1D Gaussian
#   - MarginExpLowerEmission: one-sided exponential (lower-bound constraint)
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np


def _norm_ppf(p: float) -> float:
    """
    Rational approximation of the inverse standard normal CDF.
    Reference: Peter J. Acklam's approximation.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
        )
    if p > phigh:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0
        )

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r) + 1.0
    )


class BaseEmission:
    """
    所有发射模型的统一基类。

    约定接口：
      - logpdf(x: np.ndarray) -> np.ndarray
      - init_from_data(xs, ws=None): 用初始化数据给一个合理初值
      - m_step_update(xs, ws, q_low=None, q_high=None): EM 的 M-step 更新参数，
            并在需要时更新 z-space 下的 [L, U] 区间
      - interval(q_low, q_high) -> (L, U): 基于当前参数给出 z-space 区间
      - get_summary() -> dict: 给 plotting / eval 用的摘要信息

    其中：
      xs: list[np.ndarray]，每条轨迹在某个维度上的 z 值
      ws: list[np.ndarray]，对应的 gamma 权重（可以为 None）
    """

    def __init__(self):
        # z-space 下的“约束带”区间，供外部（plot/eval）直接读取
        self.L: float = 0.0
        self.U: float = 0.0

    # ---------------- 必须实现的接口 ----------------
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def init_from_data(
        self,
        xs: list[np.ndarray],
        ws: list[np.ndarray] | None = None,
    ) -> None:
        """
        初始化阶段用的数据拟合。
        典型使用：SegCons 的 init_taus 之后，用初次分段的数据给出 mu/b 等初值。
        """
        pass

    def m_step_update(
        self,
        xs: list[np.ndarray],
        ws: list[np.ndarray],
        q_low: float | None = None,
        q_high: float | None = None,
    ) -> None:
        """
        EM 的 M-step。与 init_from_data 的区别主要是“语义”——这里是迭代更新。

        q_low/q_high:
          若给出，则模型应在更新参数后，同时更新自身的
              self.L, self.U
          用于可视化 / 约束估计。
        """
        raise NotImplementedError

    # ---------------- 可选接口：区间 & summary ----------------
    # def interval(self, q_low: float, q_high: float) -> tuple[float, float]:
    #     """
    #     返回 z-space 的 [L, U] 区间。默认直接返回当前 self.L / self.U。
    #     子类可重载给出解析分位数。
    #     """
    #     return float(self.L), float(self.U)

    def get_summary(self) -> dict:
        """
        给 plotting / constraint extraction 用的统一 summary。
        子类应至少带上:
          {"type": "...", "L": L, "U": U, ...其他参数}
        """
        return {"type": "base", "L": float(self.L), "U": float(self.U)}


# ============================================================
# 1D Gaussian emission
# ============================================================

class GaussianModel(BaseEmission):
    """
    一维高斯发射模型。

    统一接口：
      - logpdf(x)
      - init_from_data(xs, ws=None)
      - m_step_update(xs, ws, q_low, q_high)
      - interval(q_low, q_high)
      - get_summary()
    """

    def __init__(
        self,
        mu: float | None = None,
        sigma: float | None = None,
        fixed_sigma: float | None = None,
        min_sigma: float = 1e-3,
    ):
        super().__init__()
        self.mu: float = float(mu) if mu is not None else 0.0
        self.sigma: float = float(sigma) if sigma is not None else 1.0
        self.fixed_sigma: float | None = fixed_sigma  # 若不为 None，则 sigma 恒定
        self.min_sigma: float = float(min_sigma)

    # ---------------- 核心：logpdf ----------------
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        sig = float(self.sigma)
        sig2 = sig * sig + 1e-12
        c = -0.5 * np.log(2.0 * np.pi * sig2)
        return c - 0.5 * (x - self.mu) ** 2 / sig2

    # ---------------- 内部辅助：带权拟合 mu/sigma ----------------
    def _fit_weighted(
        self,
        xs: list[np.ndarray],
        ws: list[np.ndarray] | None = None,
    ) -> None:
        """
        带权 MLE 拟合 mu/sigma。供 init_from_data 和 m_step_update 共用。
        """
        xs_cat = np.concatenate(xs, axis=0).astype(float)
        if xs_cat.size == 0:
            xs_cat = np.zeros(1, dtype=float)

        if ws is None:
            w_cat = np.ones_like(xs_cat)
        else:
            w_cat = np.concatenate(ws, axis=0).astype(float)
            if w_cat.size == 0:
                w_cat = np.ones_like(xs_cat)

        w_cat = np.maximum(w_cat, 0.0)
        w_sum = float(np.sum(w_cat)) + 1e-12

        mu = float(np.sum(w_cat * xs_cat) / w_sum)
        var = float(np.sum(w_cat * (xs_cat - mu) ** 2) / w_sum)

        self.mu = mu
        if self.fixed_sigma is None:
            sigma = np.sqrt(max(var, 0.0))
            self.sigma = float(max(sigma, self.min_sigma))
        else:
            self.sigma = float(self.fixed_sigma)

    # ---------------- init / m_step_update ----------------
    def init_from_data(
            self,
            xs: list[np.ndarray],
            ws: list[np.ndarray] | None = None,
    ) -> None:
        """
        初始化阶段使用的数据；SegCons 在根据 init_taus 做初次分段后会调用这里。
        """
        self._fit_weighted(xs, ws)
        self.L, self.U = self.interval(0.05, 0.95)

    def m_step_update(
            self,
            xs: list[np.ndarray],
            ws: list[np.ndarray],
            q_low: float | None = 0.05,
            q_high: float | None = 0.95,
    ) -> None:
        """
        EM 的 M-step：更新 mu/sigma，并在需要时更新 L/U。
        """
        self._fit_weighted(xs, ws)
        self.L, self.U = self.interval(q_low, q_high)

    def m_update(self, x: np.ndarray, w: np.ndarray) -> None:
        self.m_step_update([np.asarray(x, dtype=float)], [np.asarray(w, dtype=float)])

    # ---------------- 区间 & summary ----------------
    def interval(self, q_low: float, q_high: float) -> tuple[float, float]:
        """
        使用当前高斯 N(mu, sigma^2) 的分位点定义区间 [L, U]。
        完全去掉 k_sigma 逻辑，不再从数据重算分位数。
        """
        assert 0.0 <= q_low < q_high <= 1.0

        sigma = max(float(self.sigma), 1e-8)

        z_low = _norm_ppf(q_low)
        z_high = _norm_ppf(q_high)

        L = self.mu + z_low * sigma
        U = self.mu + z_high * sigma
        return float(L), float(U)

    def get_summary(self) -> dict:
        return {
            "type": "gauss",
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "fixed_sigma": self.fixed_sigma,
            "min_sigma": float(self.min_sigma),
            "L": float(self.L),
            "U": float(self.U),
        }

class ZeroMeanGaussianModel(BaseEmission):
    """
    Zero-mean 1D Gaussian emission for equality-style residuals:
        z ~ N(0, sigma^2)

    Key semantics:
      - mu is fixed to 0 (not learned).
      - sigma can be learned from data (weighted MLE) OR fixed via fixed_sigma.
      - interval(q_low, q_high) is defined by Gaussian quantiles around mu=0.

    This is meant for learned constraint residuals h(x) where the constraint is h(x)=0
    with tolerance controlled by sigma (ideally fixed and small for hard-ish equality).
    """

    def __init__(
        self,
        sigma: float | None = None,
        fixed_sigma: float | None = None,
        min_sigma: float = 1e-3,
    ):
        super().__init__()
        self.mu: float = 0.0
        self.fixed_sigma: float | None = fixed_sigma
        if fixed_sigma is not None:
            self.sigma = float(fixed_sigma)
        else:
            self.sigma: float = float(sigma) if sigma is not None else 1.0

        self.min_sigma: float = float(min_sigma)

    # ---------------- core: logpdf ----------------
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        sig = float(self.sigma)
        sig2 = sig * sig + 1e-12
        c = -0.5 * np.log(2.0 * np.pi * sig2)
        # mu is fixed to 0
        return c - 0.5 * (x ** 2) / sig2

    # ---------------- internal helper: weighted fit sigma ----------------
    def _fit_weighted(
        self,
        xs: list[np.ndarray],
        ws: list[np.ndarray] | None = None,
    ) -> None:
        if self.fixed_sigma is not None:
            self.sigma = float(self.fixed_sigma)
            return
        xs_cat = np.concatenate(xs, axis=0).astype(float)
        if xs_cat.size == 0:
            xs_cat = np.zeros(1, dtype=float)

        if ws is None:
            w_cat = np.ones_like(xs_cat)
        else:
            w_cat = np.concatenate(ws, axis=0).astype(float)
            if w_cat.size == 0:
                w_cat = np.ones_like(xs_cat)

        w_cat = np.maximum(w_cat, 0.0)
        w_sum = float(np.sum(w_cat)) + 1e-12

        # mu fixed to 0 => var = E[x^2]
        var = float(np.sum(w_cat * (xs_cat ** 2)) / w_sum)

        self.mu = 0.0
        sigma = float(np.sqrt(max(var, 0.0)))
        self.sigma = float(max(sigma, self.min_sigma))


    # ---------------- init / m_step_update ----------------
    def init_from_data(
        self,
        xs: list[np.ndarray],
        ws: list[np.ndarray] | None = None,
    ) -> None:
        self._fit_weighted(xs, ws)
        # follow GaussianModel convention
        self.L, self.U = self.interval(0.05, 0.95)

    def m_step_update(
        self,
        xs: list[np.ndarray],
        ws: list[np.ndarray],
        q_low: float | None = 0.05,
        q_high: float | None = 0.95,
    ) -> None:
        self._fit_weighted(xs, ws)
        if q_low is not None and q_high is not None:
            self.L, self.U = self.interval(q_low, q_high)
            print(self.L, self.U)

    def m_update(self, x: np.ndarray, w: np.ndarray) -> None:
        self.m_step_update([np.asarray(x, dtype=float)], [np.asarray(w, dtype=float)])

    # ---------------- interval & summary ----------------
    def interval(self, q_low: float, q_high: float) -> tuple[float, float]:
        assert 0.0 <= q_low < q_high <= 1.0
        sigma = max(float(self.sigma), 1e-8)

        z_low = _norm_ppf(q_low)
        z_high = _norm_ppf(q_high)

        # mu == 0
        L = z_low * sigma
        U = z_high * sigma
        return float(L), float(U)

    def get_summary(self) -> dict:
        return {
            "type": "gauss_zero",
            "mu": 0.0,
            "sigma": float(self.sigma),
            "fixed_sigma": self.fixed_sigma,
            "min_sigma": float(self.min_sigma),
            "L": float(self.L),
            "U": float(self.U),
        }
# ============================================================
# One-sided exponential (lower-bound) emission
# ============================================================

class MarginExpLowerEmission(BaseEmission):
    """
    一侧指数分布：对 “下界约束” 的简单模型。
      - Z = b + Exp(λ), 支持集为 [b, +∞)
      - logpdf(d) 在 d < b 时给一个很小的常数 big_neg，相当于 hard-ish violation

    典型使用：distance-like 约束，d >= d_safe
    """

    def __init__(
        self,
        b_init: float = 0.0,
        lam_init: float = 1.0,
        min_lam: float = 1e-3,
        big_neg: float = -1e6,
    ):
        super().__init__()
        self.b: float = float(b_init)
        self.lam: float = float(lam_init)
        self.min_lam: float = float(min_lam)
        self.big_neg: float = float(big_neg)

    # ---------------- logpdf ----------------
    def logpdf(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=np.float64)
        m = d - self.b

        ll = np.full_like(d, self.big_neg, dtype=np.float64)
        mask = m >= 0.0
        if np.any(mask):
            lam = self.lam + 1e-12
            ll[mask] = -np.log(lam) - m[mask] / lam
        return ll

    # ---------------- m_step_update ----------------
    def m_step_update(
        self,
        xs_list: list[np.ndarray],
        ws_list: list[np.ndarray],
        q_low: float | None = None,
        q_high: float | None = None,
    ) -> None:
        """
        xs_list: list of 1D arrays (z-values from different demos)
        ws_list: list of 1D arrays (gamma weights for this state)

        实现的是：带权 one-sided exponential 的 MLE：
          - b* = min(z_eff) （在有效样本子集上）
          - λ* = Σ w_i (z_i - b*) / Σ w_i
        """
        z_all = np.concatenate([np.asarray(x, float) for x in xs_list], axis=0)
        w_all = np.concatenate([np.asarray(w, float) for w in ws_list], axis=0)

        w_all = np.maximum(w_all, 0.0)
        w_sum = float(np.sum(w_all))
        if w_sum <= 1e-8:
            return

        # 丢掉权重极小的点，避免某个 outlier 卡死 b
        w_norm = w_all / w_sum
        mask_eff = w_norm > 1e-4
        if np.sum(mask_eff) < 3:
            mask_eff = w_all > 0

        z_eff = z_all[mask_eff]
        w_eff = w_all[mask_eff]
        if z_eff.size == 0:
            return

        # --- 更新 b：带权情形下，最大化 Q(b,λ) 的解仍然是 b = min(z_eff) ---
        b_new = float(np.min(z_eff))

        # --- 更新 λ：λ = Σ w_i (z_i - b) / Σ w_i ---
        deltas = z_eff - b_new
        deltas = np.maximum(deltas, 0.0)
        lam_num = float(np.sum(w_eff * deltas))
        lam_den = float(np.sum(w_eff))

        if lam_den <= 1e-8:
            return

        lam_new = lam_num / lam_den
        lam_new = float(np.clip(lam_new, self.min_lam, np.inf))

        self.b = b_new
        self.lam = lam_new

        # 更新 L/U：若给了分位数则用分位数，否则用一个默认区间
        if q_low is not None and q_high is not None:
            self.L, self.U = self.interval(q_low, q_high)
        else:
            self.L, self.U = self.interval(0., 0.9)

    # ---------------- init_from_data ----------------
    def init_from_data(
        self,
        xs: list[np.ndarray],
        ws: list[np.ndarray] | None = None,
    ) -> None:
        """
        初始化时给一个比较保守的基线：
          - b: 5% 分位数
          - λ: E[z - b]
        """
        d_all = np.concatenate(xs, axis=0).astype(float)
        if d_all.size == 0:
            return

        self.b = float(np.percentile(d_all, 5))
        mean_delta = float(np.maximum((d_all - self.b).mean(), self.min_lam))
        self.lam = mean_delta

        # 初始给一个区间，之后 M-step 会覆盖
        self.L, self.U = self.interval(0., 0.9)

    # ---------------- 区间 & summary ----------------
    def interval(self, q_low: float, q_high: float) -> tuple[float, float]:
        """
        Z = b + Exp(λ)，则分位数：
            z_q = b - λ log(1 - q)
        这里用它来给出 [L, U] 区间。
        """
        q_low = np.clip(q_low, 0.0, 0.999999)
        q_high = np.clip(q_high, 0.0, 0.999999)
        lam = float(self.lam) + 1e-12

        m_low = -lam * np.log(max(1.0 - q_low, 1e-8))
        m_high = -lam * np.log(max(1.0 - q_high, 1e-8))

        L = self.b + m_low
        U = self.b + m_high
        return float(L), float(U)

    def get_summary(self) -> dict:
        return {
            "type": "margin_exp_lower",
            "b": float(self.b),
            "lam": float(self.lam),
            "L": float(self.L),
            "U": float(self.U),
        }
