# utils/learned_feature.py
import numpy as np
import torch
from torch import nn


class LearnedFeatureBase(nn.Module):
    """
    可学习 feature 的抽象基类。

    新增：
      - 内部维护 mean_ / std_，只用于对本 feature 做标准化；
      - eval_raw_numpy: 返回未标准化的 g_raw(X)
      - eval_numpy: 返回标准化后的 z(X) = (g_raw - mean_) / std_
    """
    def __init__(self, name: str, state_index: int = 0):
        super().__init__()
        self.name = name
        self.state_index = int(state_index)

        # 内部标准化的统计量（注册成 buffer，方便保存 / to(device)）
        self.register_buffer("mean_", torch.zeros(1))
        self.register_buffer("std_", torch.ones(1))

    # --------- 子类仍然只需要实现 forward ---------
    def forward(self, X_torch: torch.Tensor) -> torch.Tensor:
        """
        X_torch: (T, D)
        return : (T,) or (T,1) 的 原始 g_raw(X)
        """
        raise NotImplementedError

    # --------- 原始输出（不做标准化） ---------
    @torch.no_grad()
    def eval_raw_numpy(self, X_np: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(np.asarray(X_np, dtype=np.float32))
        g = self.forward(X_t)          # (T,) 或 (T,1)
        g = g.squeeze(-1)
        return g.detach().cpu().numpy()

    # --------- 标准化后的输出（给 GoalHMM 用） ---------
    @torch.no_grad()
    def eval_numpy(self, X_np: np.ndarray) -> np.ndarray:
        """
        返回已经标准化后的 z(X) = (g_raw - mean_) / std_
        """
        g_raw = self.eval_raw_numpy(X_np)
        m = float(self.mean_.item())
        s = float(self.std_.item())
        return (g_raw - m) / (s + 1e-8)

    # --------- 每轮 EM 后更新内部 mean/std ---------
    @torch.no_grad()
    def update_stats(self, all_X_list):
        """
        all_X_list: list[np.ndarray (T_d, D)]
        用当前参数下的 g_raw(X) 重新估计 mean/std。
        """
        vals = []
        for X in all_X_list:
            vals.append(self.eval_raw_numpy(X))   # 注意是 raw
        vals = np.concatenate(vals, axis=0)
        m = float(np.mean(vals))
        s = float(np.std(vals) + 1e-8)

        self.mean_[0] = m
        self.std_[0] = s

    # --------- 可选：根据 env / demos 初始化参数
    @torch.no_grad()
    def init_from_env(self, env, demos=None, taus=None):
        """
        缺省什么都不做；子类可以 override。
        典型用法：
          - 从 env 里读 true A/ω/φ/bias
          - 或者根据 demos 做一次粗略拟合
        """
        return


class SineCorridorResidual(LearnedFeatureBase):
    def __init__(self, A_init=1.0, w_init=1.0, phi_init=0.0, bias=0.0, state_index=0):
        super().__init__(name="sine_residual", state_index=state_index)
        self.A = nn.Parameter(torch.tensor(float(A_init)))
        self.w = nn.Parameter(torch.tensor(float(w_init)))
        self.phi = nn.Parameter(torch.tensor(float(phi_init)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, X_torch: torch.Tensor) -> torch.Tensor:
        x = X_torch[:, 0]
        y = X_torch[:, 1]
        y_hat = self.A * torch.sin(self.w * x + self.phi) + self.bias
        g = y - y_hat
        return g
