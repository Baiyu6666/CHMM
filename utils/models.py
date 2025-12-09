# utils/models.py
import numpy as np
from scipy.stats import norm, t as student_t
from scipy.special import digamma, polygamma


class BaseEmission:
    """
    所有发射模型的统一接口
    """
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def m_step_update(self,
                      xs: list[np.ndarray],
                      ws: list[np.ndarray]) -> None:
        """
        xs: [demo1_x, demo2_x, ...], 每个 shape (T_d,)
        ws: [demo1_w, demo2_w, ...], 每个 shape (T_d,)
        """
        raise NotImplementedError

    def init_from_data(self,
                       xs: list[np.ndarray],
                       ws: list[np.ndarray] | None = None) -> None:
        """
        可选：用初始化数据给个比较合理的初值（mu/sigma 或 b/λ）。
        不想实现可以空着。
        """
        pass

    def get_summary(self) -> dict:
        """
        给 plotting / constraint extraction 用的统一 summary。
        例如：
          Gaussian: {"type": "gauss", "mu": mu, "sigma": sigma}
          MarginExp: {"type": "margin_exp_lower", "b": b, "lam": lam}
        """
        return {"type": "base"}


import numpy as np


class GaussianModel:
    """
    一维高斯发射模型，接口统一成：
      - logpdf(x)
      - init_from_data(xs, ws=None)
      - m_step_update(xs, ws)
      - interval(q_low, q_high)
      - get_summary()

    其中 xs 是 list[np.ndarray]，ws 是 list[np.ndarray] 或 None。
    """

    def __init__(self, mu=None, sigma=None, fixed_sigma=None, min_sigma=1e-3):
        self.mu = float(mu) if mu is not None else 0.0
        self.sigma = float(sigma) if sigma is not None else 1.0
        self.fixed_sigma = fixed_sigma  # 若不为 None，则 sigma 恒定
        self.min_sigma = float(min_sigma)

    # ---------------- 基本接口 ----------------
    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        sig = float(self.sigma)
        sig2 = sig * sig + 1e-12
        c = -0.5 * np.log(2.0 * np.pi * sig2)
        return c - 0.5 * (x - self.mu) ** 2 / sig2

    # 老接口（兼容）：单批数据 + 权重
    def m_update(self, y, w=None):
        y = np.asarray(y, dtype=float)
        if w is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(w, dtype=float)

        w_sum = float(np.sum(w)) + 1e-12
        mu = float(np.sum(w * y) / w_sum)
        var = float(np.sum(w * (y - mu) ** 2) / w_sum)

        self.mu = mu
        if self.fixed_sigma is None:
            sigma = np.sqrt(max(var, 0.0))
            self.sigma = float(max(sigma, self.min_sigma))
        # 若 fixed_sigma 不为 None，则保留当前 sigma，不改

    # ---------------- 统一的新接口 ----------------
    def init_from_data(self, xs, ws=None):
        """
        xs: list of 1D arrays
        ws: None 或 list of 1D arrays（与 xs 对应）
        """
        # 防止旧对象没有 min_sigma 属性
        if not hasattr(self, "min_sigma"):
            self.min_sigma = 1e-3

        xs_cat = np.concatenate(xs, axis=0).astype(float)
        if xs_cat.size == 0:
            xs_cat = np.zeros(1, dtype=float)

        if ws is None:
            w_cat = np.ones_like(xs_cat)
        else:
            w_cat = np.concatenate(ws, axis=0).astype(float)
            if w_cat.size == 0:
                w_cat = np.ones_like(xs_cat)

        w_sum = float(np.sum(w_cat)) + 1e-12
        mu = float(np.sum(w_cat * xs_cat) / w_sum)
        var = float(np.sum(w_cat * (xs_cat - mu) ** 2) / w_sum)

        self.mu = mu
        if self.fixed_sigma is None:
            sigma = np.sqrt(max(var, 0.0))
            self.sigma = float(max(sigma, self.min_sigma))
        else:
            self.sigma = float(self.fixed_sigma)

    def m_step_update(self, xs, ws):
        """
        EM 的 M-step：与 init_from_data 相同逻辑，只是语义上是“更新”。
        """
        self.init_from_data(xs, ws)

    # ---------------- 可视化/诊断用接口 ----------------
    def interval(self, q_low, q_high):
        """
        根据分位数近似给出 [L, U] 区间。
        这里简单用 μ ± kσ，其中 k 由 q_low/q_high 决定或直接取常数。
        """
        # 如果你真想严格按 q_low/q_high 算，可以用 scipy;
        # 这里保持简单：固定 k=2，对应大致 0.025~0.975。
        k = 2.0
        L = self.mu - k * self.sigma
        U = self.mu + k * self.sigma
        return float(L), float(U)

    def get_summary(self):
        return {
            "type": "gauss",
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "fixed_sigma": self.fixed_sigma,
            "min_sigma": float(self.min_sigma),
        }


class MarginExpLowerEmission(BaseEmission):
    def __init__(self, b_init=0.0, lam_init=1.0, min_lam=1e-3, big_neg=-1e6):
        self.b = float(b_init)
        self.lam = float(lam_init)
        self.min_lam = float(min_lam)
        self.big_neg = float(big_neg)

    def logpdf(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=np.float64)
        m = d - self.b
        ll = np.full_like(d, self.big_neg, dtype=np.float64)
        mask = m >= 0.0
        if np.any(mask):
            lam = self.lam + 1e-12
            ll[mask] = -np.log(lam) - m[mask] / lam
        return ll

    def m_step_update(self, xs_list, ws_list):
        """
        xs_list: list of 1D arrays (z values from different demos)
        ws_list: list of 1D arrays (same shape, gamma weights for this state)

        实现的是：带权 one-sided exponential 的 MLE：
          - 约束 b <= min(z_i)；
          - 在该约束下，Q(b,λ) 对 b 单调递增 → b* = min(z_i)；
          - λ* = Σ w_i (z_i - b*) / Σ w_i 。
        """
        import numpy as np

        # 拼接所有 demo
        z_all = np.concatenate([np.asarray(x, float) for x in xs_list], axis=0)
        w_all = np.concatenate([np.asarray(w, float) for w in ws_list], axis=0)

        # 丢掉权重太小的点，防止某个极端 outlier 权重几乎 0 却把 b 卡死
        w_all = np.maximum(w_all, 0.0)
        w_sum = float(np.sum(w_all))
        if w_sum <= 1e-8:
            # 没有有效数据，就不更新
            return

        # 可选：只用权重>某阈值的点来估 b，避免极小 gamma 的 outlier
        w_norm = w_all / w_sum
        mask_eff = w_norm > 1e-4
        if np.sum(mask_eff) < 3:
            # 有效点太少，就用全部点
            mask_eff = w_all > 0

        z_eff = z_all[mask_eff]
        w_eff = w_all[mask_eff]
        if z_eff.size == 0:
            return

        # --- 更新 b：带权情形下，最大化 Q 对 b 的解仍然是 b = min(z_eff) ---
        b_new = float(np.min(z_eff))

        # --- 更新 λ：λ = Σ w_i (z_i - b) / Σ w_i ---
        deltas = z_eff - b_new
        deltas = np.maximum(deltas, 0.0)  # 理论上应该非负，数值上保险一下
        lam_num = float(np.sum(w_eff * deltas))
        lam_den = float(np.sum(w_eff))

        if lam_den <= 1e-8:
            return

        lam_new = lam_num / lam_den
        # 数值裁剪
        lam_new = float(np.clip(lam_new, self.min_lam, np.inf))

        self.b = b_new
        self.lam = lam_new

    def init_from_data(self, xs, ws=None):
        d_all = np.concatenate(xs, axis=0)
        if d_all.size == 0:
            return
        self.b = float(np.percentile(d_all, 5))      # 随便给个保守一点的下界初值
        self.lam = float(max((d_all - self.b).mean(), self.min_lam))

    def get_summary(self) -> dict:
        return {"type": "margin_exp_lower", "b": self.b, "lam": self.lam}
