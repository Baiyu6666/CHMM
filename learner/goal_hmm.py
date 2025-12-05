# learner/goal_hmm.py
# ------------------------------------------------------------
# GoalHMM3D with:
#   - Two-state left-to-right HMM
#   - Emission = feature Gaussians + vMF progress
#   - Transition = Gaussian bump near g1 with learnable delta
#   - g1/g2 updated by vMF + transition gradient
#   - Oracle constraint metrics (d_safe, v2_max)
#   - Optional automatic feature selection via binary mask r
#     r: shape (K_states=2, M_features=2), each entry in {0,1}
#     Each stage can select any subset of features as "relevant"
# ------------------------------------------------------------

import numpy as np
from scipy.stats import norm
from plots.plot4panel import plot_results_4panel

from utils.vmf import _unit, vmf_logC_d, vmf_grad_wrt_g
from utils.models import GaussianModel


class GoalHMM3D:
    def __init__(
        self,
        demos,
        env,
        true_taus=None,

        # init
        g1_init=None,
        g2_init=None,
        tau_init=None,

        # weights
        feat_weight=1.0,
        prog_weight=1.0,
        trans_weight=1.0,

        # vMF kappas
        prog_kappa1=8.0,
        prog_kappa2=6.0,

        # irrelevant feature width in z-space
        fixed_sigma_irrelevant=5.0,

        # transition bump
        trans_eps=1e-6,
        delta_init=0.15,
        learn_delta=True,
        lr_delta=5e-4,

        # g update
        vmf_steps=3,
        vmf_lr=5e-4,
        g_step=0.1,
        g_grad_clip=None,
        g1_vmf_weight=1.0,
        g1_trans_weight=1.0,

        # plotting bands
        q_low=0.1,
        q_high=0.9,
        width_reg=0.0,

        plot_every=200,

        auto_feature_select=False,     # 是否启用自动 feature 选择
        r_sparse_lambda=1.7,          # 对 r 的稀疏正则强度（越大越鼓励少选 feature）
    ):
        self.demos = list(demos)
        self.env = env
        self.true_taus = (
            list(true_taus) if true_taus is not None else [None] * len(self.demos)
        )

        self.tau_init = None
        if tau_init is not None:
            tau_init = np.asarray(tau_init, dtype=int)
            assert len(tau_init) == len(self.demos), "tau_init 长度必须等于 demos 数量"
            self.tau_init = tau_init

        # HMM / emission 参数
        self.feat_weight = float(feat_weight)
        self.prog_weight = float(prog_weight)
        self.trans_weight = float(trans_weight)

        self.prog_kappa1 = float(prog_kappa1)
        self.prog_kappa2 = float(prog_kappa2)
        self.sigma_irrel = float(fixed_sigma_irrelevant)

        self.trans_eps = float(trans_eps)
        self.delta = float(delta_init)
        self.learn_delta = bool(learn_delta)
        self.lr_delta = float(lr_delta)

        self.vmf_steps = int(vmf_steps)
        self.vmf_lr = float(vmf_lr)
        self.g_step = float(g_step)
        self.g_grad_clip = g_grad_clip
        self.g1_vmf_weight = float(g1_vmf_weight)
        self.g1_trans_weight = float(g1_trans_weight)

        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.width_reg = float(width_reg)
        self.plot_every = plot_every

        # temperature on progress (kept for compatibility)
        self.tau = 1.0

        # ---- feature 结构: 当前有 2 个 feature（distance, speed）----
        self.num_states = 2
        self.num_features = 2  # 0: distance, 1: speed

        # 自动 feature 选择 & 稀疏先验
        self.auto_feature_select = bool(auto_feature_select)
        self.r_sparse_lambda = float(r_sparse_lambda)

        # r[k, m] ∈ {0,1}，初始化为 [[1,0],[0,1]]：stage1 用 distance，stage2 用 speed
        self.r = np.array([[1, 0],
                           [0, 1]], dtype=int)

        # ---- feature preprocessing (full series) ----
        self._init_feature_preprocessing()

        # ---- feature Gaussians in z-space ----
        # 为每个 state × feature 建一个 GaussianModel
        self.feature_models = [
            [GaussianModel(mu=None, sigma=None, fixed_sigma=None)
             for _ in range(self.num_features)]
            for _ in range(self.num_states)
        ]

        # 兼容原有接口：model_stage1 / model_stage2 用于约束可视化：
        #   - model_stage1 绑定到 state0 的 distance feature（m=0）
        #   - model_stage2 绑定到 state1 的 speed    feature（m=1）
        self.model_stage1 = self.feature_models[0][0]
        self.model_stage2 = self.feature_models[1][1]

        # 用于记录初始化时的 cutpoints（用于 feature init）
        self.init_taus = None

        # ---- 初始化 g1/g2 + 根据分段初始化 feature Gaussians ----
        self._init_goals_and_features(g1_init, g2_init)

        self.L1, self.U1 = self._interval_from_model(self.model_stage1)
        self.L2, self.U2 = self._interval_from_model(self.model_stage2)

        # logs
        self.g1_hist = [self.g1.copy()]
        self.g2_hist = [self.g2.copy()]
        self.loss_loglik = []
        self.loss_feat = []
        self.loss_prog = []
        self.loss_trans = []

        # ---------------- Metrics logs ----------------
        self.metric_tau_mae = []    # mean |tau_hat - tau_true|
        self.metric_tau_nmae = []   # mean |tau_hat - tau_true| / T

        self.metric_g1_err = []     # ||g1 - g1_true||
        self.metric_g2_err = []     # ||g2 - g2_true||

        self.metric_d_relerr = []   # |d_safe - d_safe_oracle| / d_safe_oracle
        self.metric_v_relerr = []   # |v2_max - v2_max_oracle| / v2_max_oracle

        # 为了约束阈值，先预计算 oracle thresholds（基于 true_taus）
        self._init_oracle_constraints()

    # ------------------------------------------------------------------
    # Oracle constraints (used only for metrics)
    # ------------------------------------------------------------------
    def _init_oracle_constraints(self):
        """
        使用 true_taus 把 stage1 / stage2 的特征分开，
        在 z 空间拟合 Gaussian，然后用 inv1/inv2 得到
        oracle 的 d_safe / v2_max（raw 单位），只用于评估。
        """
        self.oracle_d_safe_raw = None
        self.oracle_v2_max_raw = None

        if self.true_taus is None:
            return
        if all(t is None for t in self.true_taus):
            return

        stage1_dists = []
        stage2_speeds = []

        for X, tau_true in zip(self.demos, self.true_taus):
            if tau_true is None:
                continue
            tau_true = int(tau_true)
            d_raw, s_raw = self.env.compute_features_all(X)  # raw distance, raw speed
            T = len(X)
            tau_true = np.clip(tau_true, 0, T - 1)

            # stage1: 0..tau_true 的距离
            stage1_dists.append(d_raw[: tau_true + 1])

            # stage2: 从 tau_true 开始的速度
            s_idx_start = min(max(tau_true, 0), len(s_raw) - 1)
            stage2_speeds.append(s_raw[s_idx_start:])

        if len(stage1_dists) == 0 or len(stage2_speeds) == 0:
            return

        stage1_dists = np.concatenate(stage1_dists).astype(float)
        stage2_speeds = np.concatenate(stage2_speeds).astype(float)

        # 使用与模型相同的 z 变换
        z_d = self._z1(stage1_dists)
        z_s = self._z2(stage2_speeds)

        mu1_or = float(np.mean(z_d))
        sig1_or = float(np.std(z_d) + 1e-8)
        mu2_or = float(np.mean(z_s))
        sig2_or = float(np.std(z_s) + 1e-8)

        k1 = 1.0
        k2 = 1.0
        z1_low_or = mu1_or - k1 * sig1_or
        z2_high_or = mu2_or + k2 * sig2_or

        self.oracle_d_safe_raw = float(self._inv1(z1_low_or))
        self.oracle_v2_max_raw = float(self._inv2(z2_high_or))

    # ------------------------------------------------------------------
    # Feature preprocessing
    # ------------------------------------------------------------------
    def _init_feature_preprocessing(self):
        d_all, s_all = [], []
        for X in self.demos:
            d, s = self.env.compute_features_all(X)
            d_all.append(d)
            s_all.append(s)
        d_all = np.concatenate(d_all).astype(float)
        s_all = np.concatenate(s_all).astype(float)
        self._all_dists = d_all
        self._all_speeds = s_all

        self.f1_mean = float(np.nanmean(d_all))
        self.f1_std = float(np.nanstd(d_all) + 1e-8)
        self.f2_mean = float(np.nanmean(s_all))
        self.f2_std = float(np.nanstd(s_all) + 1e-8)

        self._z1 = lambda v: (np.asarray(v, float) - self.f1_mean) / self.f1_std
        self._z2 = lambda v: (np.asarray(v, float) - self.f2_mean) / self.f2_std
        self._inv1 = lambda v: np.asarray(v, float) * self.f1_std + self.f1_mean
        self._inv2 = lambda v: np.asarray(v, float) * self.f2_std + self.f2_mean

    def _features_for_demo(self, X):
        """
        返回 (phi1, phi2)：
          phi1: distance 的 z 值
          phi2: speed   的 z 值
        """
        d, s = self.env.compute_features_all(X)
        T = len(X)
        if len(s) < T:
            s = np.concatenate([s, [s[-1]]])  # pad to length T
        return self._z1(d), self._z2(s)

    def _features_for_demo_matrix(self, X):
        """
        返回 shape=(T, M=2) 的 feature matrix：
          [:,0] = distance (z)
          [:,1] = speed    (z)
        """
        phi1, phi2 = self._features_for_demo(X)
        return np.stack([phi1, phi2], axis=1)

    # ------------------------------------------------------------------
    # Irrelevant feature log-pdf（宽高斯）
    # ------------------------------------------------------------------
    def _log_irrelevant(self, phi):
        """
        phi: 1D array of feature values (already in z-space)
        使用固定宽度 sigma_irrel 的高斯，作为 "irrelevant" 背景模型。
        """
        sig = self.sigma_irrel
        c = -0.5 * np.log(2 * np.pi * sig ** 2)
        phi = np.asarray(phi, float)
        return c - 0.5 * (phi ** 2) / (sig ** 2 + 1e-12)

    # ------------------------------------------------------------------
    # 初始化 g1/g2 + 根据初始分段初始化 feature Gaussians
    # ------------------------------------------------------------------
    def _init_goals_and_features(self, g1_init, g2_init):
        """
        g1_init / g2_init 支持：
          - np.ndarray / list(3,): 直接视为 workspace 坐标
          - 字符串：
              "true_tau"  : 使用 true_taus 作为切点，取对应点平均
              "random"    : 每条轨迹采一个随机归一化 tau，取平均
              "heuristic" : 寻找使不同轨迹位置最聚集的 tau（共享 subgoal 假设）
              "from_tau"  : 使用 self.tau_init 中给定的 cutpoints（外部指定）

        初始化步骤：
          1) 先得到 g1（以及每条轨迹的 init_taus）
          2) 初始化 feature_models（利用 init_taus 分段）
          3) 初始化 g2（终点平均等）
        """
        demos = self.demos
        true_taus = self.true_taus

        # ---------------- g1: subgoal ----------------
        taus_init = None

        if g1_init is None:
            g1_init = "heuristic"

        # 0) 特殊模式："from_tau" 且存在 self.tau_init
        if isinstance(g1_init, str) and g1_init.lower() == "from_tau" and (
            self.tau_init is not None
        ):
            taus = []
            sub_pts = []
            for X, t in zip(demos, self.tau_init):
                T = len(X)
                t = int(np.clip(int(t), 1, T - 2))
                taus.append(t)
                sub_pts.append(X[t])
            taus_init = np.asarray(taus, dtype=int)
            self.g1 = np.mean(np.stack(sub_pts, axis=0), axis=0)

        else:
            # 1) 显式坐标
            if isinstance(g1_init, (list, tuple, np.ndarray)) and not isinstance(
                g1_init, str
            ):
                self.g1 = np.array(g1_init, dtype=float)

            # 2) 字符串模式
            elif isinstance(g1_init, str):
                mode = g1_init.lower()

                # 2.1 true_tau
                if (
                    mode == "true_tau"
                    and true_taus is not None
                    and true_taus[0] is not None
                ):
                    taus_init = np.array(true_taus, dtype=int)
                    pts = [X[t] for X, t in zip(demos, taus_init)]
                    self.g1 = np.mean(np.stack(pts, axis=0), axis=0)

                # 2.2 random (shared lambda)
                elif mode == "random":
                    taus = []
                    pts = []
                    lam = np.clip(np.random.rand(), 0.0, 1.0)
                    for X in demos:
                        T = len(X)
                        t = int(round(lam * (T - 1)))
                        t = np.clip(t, 1, T - 2)
                        taus.append(t)
                        pts.append(X[t])
                    taus_init = np.array(taus, dtype=int)
                    self.g1 = np.mean(np.stack(pts, axis=0), axis=0)

                # 2.3 heuristic: 找一个共享的 subgoal，使不同轨迹位置最聚集
                elif mode == "heuristic":
                    all_pts = np.concatenate(demos, axis=0)

                    center_all = all_pts.mean(axis=0)
                    dists_all = np.linalg.norm(all_pts - center_all[None, :], axis=1)
                    d_scale = np.median(dists_all) + 1e-8

                    n_cand = min(200, len(all_pts))
                    cand_idx = np.random.choice(len(all_pts), size=n_cand, replace=False)
                    cand_pts = all_pts[cand_idx]

                    candidates = []  # (mean_dist_over_trajs, center, taus_for_center)

                    for c in cand_pts:
                        d_list = []
                        taus_for_c = []
                        for X in demos:
                            dists_i = np.linalg.norm(X - c[None, :], axis=1)
                            t_i = int(np.argmin(dists_i))
                            t_i = np.clip(t_i, 1, len(X) - 2)
                            d_i = float(dists_i[t_i])
                            taus_for_c.append(t_i)
                            d_list.append(d_i)

                        if len(d_list) == 0:
                            continue

                        mean_d = float(np.mean(d_list))
                        candidates.append(
                            (mean_d, c.copy(), np.array(taus_for_c, dtype=int))
                        )

                    if len(candidates) == 0:
                        # fallback: 用全局中心
                        self.g1 = center_all
                        taus = []
                        for X in demos:
                            dists = np.linalg.norm(X - self.g1[None, :], axis=1)
                            t = int(np.argmin(dists))
                            t = np.clip(t, 1, len(X) - 2)
                            taus.append(t)
                        taus_init = np.array(taus, dtype=int)
                    else:
                        temp = 0.02  # temperature
                        mean_ds = np.array([c[0] for c in candidates], dtype=float)
                        scores = -mean_ds / max(temp, 1e-6)
                        scores -= scores.max()
                        probs = np.exp(scores)
                        probs /= probs.sum() + 1e-12

                        idx = np.random.choice(len(candidates), p=probs)
                        _, best_center, best_taus = candidates[idx]
                        self.g1 = best_center
                        taus_init = best_taus

        # 如果上面还没得到 taus_init（例如用户直接给了 g1 坐标）
        if taus_init is None:
            taus = []
            for X in demos:
                dists = np.linalg.norm(X - self.g1[None, :], axis=1)
                t = int(np.argmin(dists))
                t = np.clip(t, 1, len(X) - 2)
                taus.append(t)
            taus_init = np.array(taus, dtype=int)

        self.init_taus = taus_init

        # 记录初始化时 subgoal 的方差（workspace）
        sub_pts = np.stack([X[t] for X, t in zip(demos, taus_init)], axis=0)
        self.g1_init_var = float(
            np.mean(np.sum((sub_pts - self.g1[None, :]) ** 2, axis=1))
        )

        # ---------------- 初始化 feature_models ----------------
        # 使用 init_taus 把每条 demo 分成 [0..tau] / [tau+1..T-1]
        # 对每个 state k、每个 feature m 拟合一个 GaussianModel。
        feat_state0 = [[] for _ in range(self.num_features)]
        feat_state1 = [[] for _ in range(self.num_features)]

        for X, tau in zip(demos, self.init_taus):
            F = self._features_for_demo_matrix(X)  # shape=(T,2)
            tau = int(tau)
            tau = np.clip(tau, 1, len(F) - 2)

            F0 = F[: tau + 1]  # state 0
            F1 = F[tau + 1 :]  # state 1

            if F0.shape[0] > 0:
                for m in range(self.num_features):
                    feat_state0[m].append(F0[:, m])
            if F1.shape[0] > 0:
                for m in range(self.num_features):
                    feat_state1[m].append(F1[:, m])

        # 拼成数组并用 median + MAD 初始化
        for m in range(self.num_features):
            # state 0
            if len(feat_state0[m]) > 0:
                vals0 = np.concatenate(feat_state0[m]).astype(float)
            else:
                vals0 = self._z1(self._all_dists) if m == 0 else self._z2(self._all_speeds)
            med0 = np.median(vals0)
            mad0 = np.median(np.abs(vals0 - med0)) + 1e-3
            self.feature_models[0][m].mu = float(med0)
            self.feature_models[0][m].sigma = float(mad0)

            # state 1
            if len(feat_state1[m]) > 0:
                vals1 = np.concatenate(feat_state1[m]).astype(float)
            else:
                vals1 = self._z1(self._all_dists) if m == 0 else self._z2(self._all_speeds)
            med1 = np.median(vals1)
            mad1 = np.median(np.abs(vals1 - med1)) + 1e-3
            self.feature_models[1][m].mu = float(med1)
            self.feature_models[1][m].sigma = float(mad1)

        # ---------------- g2: final goal ----------------
        if isinstance(g2_init, (list, tuple, np.ndarray)) and not isinstance(
            g2_init, str
        ):
            self.g2 = np.array(g2_init, dtype=float)
        elif isinstance(g2_init, str):
            mode = g2_init.lower()
            if mode == "true_tau":
                self.g2 = np.mean([X[-1] for X in demos], axis=0)
            elif mode == "random":
                pts = []
                for X in demos:
                    T = len(X)
                    lam = 0.5 + 0.5 * np.random.rand()
                    t = int(round(lam * (T - 1)))
                    t = np.clip(t, 1, T - 1)
                    pts.append(X[t])
                self.g2 = np.mean(np.stack(pts, axis=0), axis=0)
            elif mode == "heuristic":
                self.g2 = np.mean([X[-1] for X in demos], axis=0)
            elif mode == "from_tau" and (self.tau_init is not None):
                self.g2 = np.mean([X[-1] for X in demos], axis=0)
            else:
                self.g2 = np.mean([X[-1] for X in demos], axis=0)
        else:
            self.g2 = np.mean([X[-1] for X in demos], axis=0)

    # ------------------------------------------------------------------
    def _interval_from_model(self, model):
        L, U = model.interval(self.q_low, self.q_high)
        if self.width_reg > 0 and np.isfinite(L) and np.isfinite(U):
            c = 0.5 * (L + U)
            h = 0.5 * (U - L) / (1.0 + self.width_reg)
            L, U = c - h, c + h
        return float(L), float(U)

    def get_bounds_for_plot(self, k_sigma=2):
        L1_raw, U1_raw = self._inv1(self.L1), self._inv1(self.U1)
        L2_raw, U2_raw = self._inv2(self.L2), self._inv2(self.U2)
        return float(L1_raw), float(U1_raw), float(L2_raw), float(U2_raw)

    # ------------------------------------------------------------------
    # Emission log-likelihood (使用 r 矩阵做 feature 选择)
    # ------------------------------------------------------------------
    def _emission_loglik(self, X):
        T = len(X)
        F = self._features_for_demo_matrix(X)  # shape = (T, 2)
        phi1 = F[:, 0]
        phi2 = F[:, 1]

        # 各种基础 logpdf
        # relevant：用对应 state, feature 的 GaussianModel
        ll_state_feat = np.zeros((self.num_states, self.num_features, T))
        for k in range(self.num_states):
            for m in range(self.num_features):
                ll_state_feat[k, m] = self.feature_models[k][m].logpdf(F[:, m])

        # irrelevant：统一用宽高斯
        ll_irrel = np.zeros((self.num_features, T))
        for m in range(self.num_features):
            ll_irrel[m] = self._log_irrelevant(F[:, m])

        # 根据 r 拼出每个 state 的 feature log-likelihood
        ll_feat_state = np.zeros((T, self.num_states))

        for k in range(self.num_states):
            ll_k = np.zeros(T)
            for m in range(self.num_features):
                if not self.auto_feature_select:
                    # 保持原来的 “stage1=dist, stage2=speed” 逻辑
                    # k=0: distance relevant, speed irrelevant
                    # k=1: speed    relevant, distance irrelevant
                    if k == 0:
                        # m=0: dist relevant; m=1: speed irrelevant
                        if m == 0:
                            ll_k += ll_state_feat[0, 0]
                        else:
                            ll_k += ll_irrel[1]
                    else:
                        # k=1: m=1 relevant, m=0 irrelevant
                        if m == 1:
                            ll_k += ll_state_feat[1, 1]
                        else:
                            ll_k += ll_irrel[0]
                    break  # k=0,1 时，这个分支只需要处理一次即可
                else:
                    # auto_feature_select: 按 r[k,m] 组合
                    if self.r[k, m] == 1:
                        ll_k += ll_state_feat[k, m]
                    else:
                        ll_k += ll_irrel[m]
            ll_feat_state[:, k] = ll_k

        # ---- progress (vMF) 部分 ----
        ll_prog1 = np.zeros(T)
        ll_prog2 = np.zeros(T)

        if self.prog_weight > 0.0 and T > 1:
            D = X.shape[1]
            logC1 = vmf_logC_d(self.prog_kappa1, D)
            logC2 = vmf_logC_d(self.prog_kappa2, D)

            Vs = _unit(X[1:] - X[:-1])
            U1 = _unit(self.g1[None, :] - X[:-1])
            U2 = _unit(self.g2[None, :] - X[:-1])

            cos1 = np.sum(Vs * U1, axis=1)
            cos2 = np.sum(Vs * U2, axis=1)
            ll_prog1[:-1] = self.prog_weight * (logC1 + self.prog_kappa1 * cos1)
            ll_prog2[:-1] = self.prog_weight * (logC2 + self.prog_kappa2 * cos2)

        ll_emit = np.zeros((T, 2))
        ll_emit[:, 0] = self.feat_weight * ll_feat_state[:, 0] + self.tau * ll_prog1
        ll_emit[:, 1] = self.feat_weight * ll_feat_state[:, 1] + self.tau * ll_prog2
        return ll_emit

    # ------------------------------------------------------------------
    # Transition logprob with Gaussian bump around g1
    # ------------------------------------------------------------------
    def _transition_logprob(self, X, return_aux=False):
        T = len(X)
        if T <= 1:
            logA = np.zeros((0, 2, 2))
            if return_aux:
                return logA, {"dists": np.zeros(T), "p12": np.zeros(T)}
            return logA

        dists = np.linalg.norm(X - self.g1[None, :], axis=1)
        eps = self.trans_eps
        delta = max(self.delta, 1e-6)
        a = 1.0 - 2.0 * eps

        exp_term = np.exp(-0.5 * (dists**2) / (delta**2))
        p12 = eps + a * exp_term
        p12 = np.clip(p12, eps, 1.0 - eps)

        logA = np.full((T - 1, 2, 2), -np.inf)
        logA[:, 0, 1] = np.log(p12[:-1])
        logA[:, 0, 0] = np.log(1.0 - p12[:-1])
        logA[:, 1, 1] = 0.0
        logA[:, 1, 0] = -np.inf

        if return_aux:
            return logA, {"p12": p12, "dists": dists}
        return logA

    # ------------------------------------------------------------------
    # Forward-backward
    # ------------------------------------------------------------------
    def _forward_backward(self, ll_emit, logA):
        T, K = ll_emit.shape
        log_pi = np.array([0.0, -np.inf])

        alpha = np.full((T, K), -np.inf)
        alpha[0] = log_pi + ll_emit[0]
        for t in range(T - 1):
            for j in range(K):
                prev = alpha[t] + logA[t, :, j]
                m = np.max(prev)
                alpha[t + 1, j] = ll_emit[t + 1, j] + (
                    m + np.log(np.sum(np.exp(prev - m)) + 1e-300)
                )

        beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            for i in range(K):
                vals = logA[t, i, :] + ll_emit[t + 1, :] + beta[t + 1, :]
                m = np.max(vals)
                beta[t, i] = m + np.log(np.sum(np.exp(vals - m)) + 1e-300)

        m = np.max(alpha[-1])
        loglik = m + np.log(np.sum(np.exp(alpha[-1] - m)) + 1e-300)

        gamma = alpha + beta
        m = np.max(gamma, axis=1, keepdims=True)
        gamma = np.exp(gamma - m)
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            log_xi = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    log_xi[i, j] = (
                        alpha[t, i]
                        + logA[t, i, j]
                        + ll_emit[t + 1, j]
                        + beta[t + 1, j]
                    )
            mm = np.max(log_xi)
            tmp = np.exp(log_xi - mm)
            tmp /= tmp.sum()
            xi[t] = tmp

        return gamma, xi, loglik, alpha, beta

    # ------------------------------------------------------------------
    # M-step: feature Gaussians
    # ------------------------------------------------------------------
    def _mstep_update_features(self, gammas):
        """
        更新 feature_models[k][m] 的参数（mu, sigma）。
        如果 auto_feature_select=False：
            - 保持原始设定：stage1-diff / stage2-speed
        如果 auto_feature_select=True：
            - 在当前 r 下，对每个 state、每个 feature 仍然用 gamma 加权拟合
              但谁是 relevant 由 r 决定，其实不影响 Gaussian 本身，只影响 emission 里
              它和 irrelevant 的组合方式。
        """
        # 收集所有 demo 的 feature 和 responsibilities
        all_F = []
        all_gamma0 = []
        all_gamma1 = []
        for X, gamma in zip(self.demos, gammas):
            F = self._features_for_demo_matrix(X)  # (T,2)
            all_F.append(F)
            all_gamma0.append(gamma[:, 0])
            all_gamma1.append(gamma[:, 1])

        F_all = np.concatenate(all_F, axis=0)          # (N,2)
        w0_all = np.concatenate(all_gamma0, axis=0)    # (N,)
        w1_all = np.concatenate(all_gamma1, axis=0)    # (N,)

        # 对每个 state, feature 拟合 Gaussian
        for m in range(self.num_features):
            phi_m = F_all[:, m]

            # state 0
            self.feature_models[0][m].m_update(phi_m, w0_all)

            # state 1
            self.feature_models[1][m].m_update(phi_m, w1_all)

        # 更新可视化区间（仍然用 model_stage1 / model_stage2）
        self.L1, self.U1 = self._interval_from_model(self.model_stage1)
        self.L2, self.U2 = self._interval_from_model(self.model_stage2)

    # ------------------------------------------------------------------
    # M-step: hard-EM 更新 feature mask r
    # ------------------------------------------------------------------
    def _mstep_update_feature_mask(self, gammas):
        """
        对 r 做 hard-EM 更新：
          对每个 state k, feature m 独立比较:
            Q_rel  = E[ log N_relevant ] + log p(r=1)
            Q_irrel= E[ log N_irrel   ] + log p(r=0)

          这里我们改成按“平均 log-likelihood”来比较：
            Q_rel_avg   = (E[ log N_relevant ]) / N_eff  + log p(r=1)
            Q_irrel_avg = (E[ log N_irrel   ]) / N_eff

          其中 log p(r=1) = -r_sparse_lambda, log p(r=0)=0，
          相当于对每个 r[k,m]=1 加一个 L0 型的惩罚。
        """
        if not self.auto_feature_select:
            return

        # 收集 feature 矩阵和 gamma
        all_F = []
        all_gamma0 = []
        all_gamma1 = []
        for X, gamma in zip(self.demos, gammas):
            F = self._features_for_demo_matrix(X)  # (T,2) 或 (T,num_features)
            all_F.append(F)
            all_gamma0.append(gamma[:, 0])
            all_gamma1.append(gamma[:, 1])

        F_all = np.concatenate(all_F, axis=0)  # (N,num_features)
        w0_all = np.concatenate(all_gamma0, axis=0)
        w1_all = np.concatenate(all_gamma1, axis=0)

        # 计算 irrelevant 的 logpdf
        ll_irrel_all = np.zeros_like(F_all)
        for m in range(self.num_features):
            ll_irrel_all[:, m] = self._log_irrelevant(F_all[:, m])

        # 对每个 state & feature 独立做判断
        for k in range(self.num_states):
            w_all = w0_all if k == 0 else w1_all

            # 有效样本数（这个 state 实际“看见”的权重总和）
            N_eff = float(np.sum(w_all))

            # 极端情况：这个 state 基本没人用，就保持原来的 r 不动
            if N_eff <= 1e-8:
                continue

            for m in range(self.num_features):
                phi_m = F_all[:, m]
                ll_rel = self.feature_models[k][m].logpdf(phi_m)
                ll_ir = ll_irrel_all[:, m]

                # 原始的期望 log-likelihood（总和）
                Q_rel_sum = np.sum(w_all * ll_rel)
                Q_irrel_sum = np.sum(w_all * ll_ir)

                # ★ 改成按“平均值”比较，并在 relevant 这边减去 lambda 惩罚
                Q_rel_avg = Q_rel_sum / (N_eff + 1e-8) - self.r_sparse_lambda
                Q_irrel_avg = Q_irrel_sum / (N_eff + 1e-8)

                if Q_rel_avg >= Q_irrel_avg:
                    self.r[k, m] = 1
                else:
                    self.r[k, m] = 0

    # ------------------------------------------------------------------
    # M-step: goals
    # ------------------------------------------------------------------
    def _mstep_update_goals(self, gammas, xis_list, aux_list):
        if self.prog_weight <= 0.0 and self.trans_weight <= 0.0:
            return

        for _ in range(self.vmf_steps):
            g1_grad_vmf = np.zeros_like(self.g1)
            g2_grad_vmf = np.zeros_like(self.g2)
            w1_sum = 0.0
            w2_sum = 0.0

            # vMF gradients
            if self.prog_weight > 0.0:
                for X, gamma in zip(self.demos, gammas):
                    T = len(X)
                    for t in range(T - 1):
                        w1 = gamma[t, 0]
                        w2 = gamma[t, 1]
                        if w1 > 0:
                            g1_grad_vmf += w1 * vmf_grad_wrt_g(
                                X[t : t + 2], self.g1, self.prog_kappa1
                            )
                            w1_sum += w1
                        if w2 > 0:
                            g2_grad_vmf += w2 * vmf_grad_wrt_g(
                                X[t : t + 2], self.g2, self.prog_kappa2
                            )
                            w2_sum += w2

            if w1_sum > 1e-12:
                g1_grad_vmf /= w1_sum
            if w2_sum > 1e-12:
                g2_grad_vmf /= w2_sum

            # Transition attraction-only gradient for g1
            g1_grad_trans = np.zeros_like(self.g1)
            w_trans_sum = 0.0

            if self.trans_weight > 0.0:
                eps = self.trans_eps
                delta = max(self.delta, 1e-6)

                for X, xi, aux in zip(self.demos, xis_list, aux_list):
                    if xi is None or aux is None or len(xi) == 0:
                        continue
                    p12 = aux["p12"]
                    Tm1 = xi.shape[0]

                    for t in range(Tm1):
                        n01 = xi[t, 0, 1]  # attraction ONLY
                        if n01 <= 0:
                            continue

                        p = p12[t]
                        coef = (p - eps) / (p * (delta**2) + 1e-12)
                        g1_grad_trans += n01 * coef * (X[t] - self.g1)
                        w_trans_sum += n01

                if w_trans_sum > 1e-12:
                    g1_grad_trans /= w_trans_sum

            g1_grad = (
                self.g1_vmf_weight * self.prog_weight * g1_grad_vmf
                + self.g1_trans_weight * self.trans_weight * g1_grad_trans
            )
            g2_grad = self.prog_weight * g2_grad_vmf

            if self.g_grad_clip is not None:
                n1 = np.linalg.norm(g1_grad)
                if n1 > self.g_grad_clip:
                    g1_grad *= self.g_grad_clip / (n1 + 1e-12)
                n2 = np.linalg.norm(g2_grad)
                if n2 > self.g_grad_clip:
                    g2_grad *= self.g_grad_clip / (n2 + 1e-12)

            g1_new = self.g1 + self.vmf_lr * g1_grad
            g2_new = self.g2 + self.vmf_lr * g2_grad
            self.g1 = (1 - self.g_step) * self.g1 + self.g_step * g1_new
            self.g2 = (1 - self.g_step) * self.g2 + self.g_step * g2_new

        self.g1_hist.append(self.g1.copy())
        self.g2_hist.append(self.g2.copy())

    # ------------------------------------------------------------------
    # M-step: delta
    # ------------------------------------------------------------------
    def _mstep_update_delta(self, xis_list, aux_list):
        if not self.learn_delta:
            return

        eps = self.trans_eps
        delta = max(self.delta, 1e-6)
        a = 1.0 - 2.0 * eps

        grad_delta = 0.0
        count = 0.0

        for xi, aux in zip(xis_list, aux_list):
            if xi is None or aux is None:
                continue
            p12 = aux["p12"]
            dists = aux["dists"]
            Tm1 = xi.shape[0]

            for t in range(Tm1):
                n00 = xi[t, 0, 0]
                n01 = xi[t, 0, 1]
                p = p12[t]
                d = dists[t]

                exp_term = np.exp(-0.5 * (d**2) / (delta**2))
                dp_ddelta = a * exp_term * (d**2) / (delta**3 + 1e-12)

                grad_delta += (n01 / (p + 1e-12) - n00 / (1 - p + 1e-12)) * dp_ddelta
                count += n00 + n01

        if count > 1e-12:
            grad_delta /= count

        self.delta = float(np.clip(delta + self.lr_delta * grad_delta, 1e-4, 2.0))

    # ------------------------------------------------------------------
    def fit(self, max_iter=30, verbose=True):
        posts = None
        for it in range(max_iter):
            gammas, xis_list, aux_list = [], [], []
            alphas, betas = [], []
            total_ll = 0.0
            total_feat_ll = 0.0
            total_prog_ll = 0.0
            total_trans_ll = 0.0

            # -------- E-step --------
            for X in self.demos:
                ll_emit = self._emission_loglik(X)
                logA, aux = self._transition_logprob(X, return_aux=True)
                gamma, xi, ll, alpha, beta = self._forward_backward(ll_emit, logA)

                gammas.append(gamma)
                xis_list.append(xi)
                aux_list.append(aux)
                alphas.append(alpha)
                betas.append(beta)
                total_ll += ll

                # feature expected ll（为了拆分 loss_feat）
                F = self._features_for_demo_matrix(X)
                phi1 = F[:, 0]
                phi2 = F[:, 1]

                # 按当前的 feature_models + r 重新算一遍 feature log-likelihood
                ll_feat_state = np.zeros_like(gamma)
                # 先准备 relevant/irrelevant logpdf
                ll_state_feat = np.zeros((self.num_states, self.num_features, len(X)))
                ll_irrel = np.zeros((self.num_features, len(X)))
                for k in range(self.num_states):
                    for m in range(self.num_features):
                        ll_state_feat[k, m] = self.feature_models[k][m].logpdf(F[:, m])
                for m in range(self.num_features):
                    ll_irrel[m] = self._log_irrelevant(F[:, m])

                for k in range(self.num_states):
                    ll_k = np.zeros(len(X))
                    for m in range(self.num_features):
                        if not self.auto_feature_select:
                            # 原始设定：stage0-dist, stage1-speed
                            if k == 0:
                                if m == 0:
                                    ll_k += ll_state_feat[0, 0]
                                else:
                                    ll_k += ll_irrel[1]
                            else:
                                if m == 1:
                                    ll_k += ll_state_feat[1, 1]
                                else:
                                    ll_k += ll_irrel[0]
                            break
                        else:
                            if self.r[k, m] == 1:
                                ll_k += ll_state_feat[k, m]
                            else:
                                ll_k += ll_irrel[m]
                    ll_feat_state[:, k] = ll_k

                feat_ll = np.sum(gamma * ll_feat_state)
                total_feat_ll += self.feat_weight * feat_ll

                # progress expected ll
                T = len(X)
                if self.prog_weight > 0 and T > 1:
                    D = X.shape[1]
                    logC1 = vmf_logC_d(self.prog_kappa1, D)
                    logC2 = vmf_logC_d(self.prog_kappa2, D)
                    Vs = _unit(X[1:] - X[:-1])
                    U1 = _unit(self.g1[None, :] - X[:-1])
                    U2 = _unit(self.g2[None, :] - X[:-1])
                    cos1 = np.sum(Vs * U1, axis=1)
                    cos2 = np.sum(Vs * U2, axis=1)
                    llp1 = self.prog_weight * (logC1 + self.prog_kappa1 * cos1)
                    llp2 = self.prog_weight * (logC2 + self.prog_kappa2 * cos2)
                    total_prog_ll += np.sum(
                        gamma[:-1, 0] * llp1 + gamma[:-1, 1] * llp2
                    )

                # transition expected ll
                p12 = aux["p12"]
                for t in range(T - 1):
                    n00 = xi[t, 0, 0]
                    n01 = xi[t, 0, 1]
                    total_trans_ll += n01 * np.log(p12[t] + 1e-12) + n00 * np.log(
                        1 - p12[t] + 1e-12
                    )

            self.loss_loglik.append(total_ll)
            self.loss_feat.append(total_feat_ll)
            self.loss_prog.append(total_prog_ll)
            self.loss_trans.append(total_trans_ll)

            # ====================== 计算 metrics ======================
            # segmentation metrics
            taus_map = []
            if gammas is not None:
                for gamma in gammas:
                    idx = np.where(gamma[:, 1] > 0.5)[0]
                    tau_hat = (
                        int(idx[0]) if len(idx) > 0 else int(np.argmax(gamma[:, 1]))
                    )
                    taus_map.append(tau_hat)

            mae_list = []
            nmae_list = []
            if self.true_taus is not None:
                for tau_hat, tau_true, X in zip(
                    taus_map, self.true_taus, self.demos
                ):
                    if tau_true is None:
                        continue
                    tau_true = int(tau_true)
                    T = len(X)
                    err = abs(tau_hat - tau_true)
                    mae_list.append(err)
                    nmae_list.append(err / max(T, 1))

            mae_tau = float(np.mean(mae_list)) if len(mae_list) > 0 else np.nan
            nmae_tau = float(np.mean(nmae_list)) if len(nmae_list) > 0 else np.nan
            self.metric_tau_mae.append(mae_tau)
            self.metric_tau_nmae.append(nmae_tau)

            # goal error
            if hasattr(self.env, "subgoal") and hasattr(self.env, "goal"):
                g1_true = np.asarray(self.env.subgoal, float)
                g2_true = np.asarray(self.env.goal, float)
                e_g1 = float(np.linalg.norm(self.g1 - g1_true))
                e_g2 = float(np.linalg.norm(self.g2 - g2_true))
            else:
                e_g1 = np.nan
                e_g2 = np.nan
            self.metric_g1_err.append(e_g1)
            self.metric_g2_err.append(e_g2)

            # constraint threshold relative error
            if (self.oracle_d_safe_raw is not None) and (
                self.oracle_v2_max_raw is not None
            ):
                mu1 = float(self.model_stage1.mu)
                sig1 = float(self.model_stage1.sigma)
                mu2 = float(self.model_stage2.mu)
                sig2 = float(self.model_stage2.sigma)

                k1 = 1.0
                k2 = 1.0
                z1_low = mu1 - k1 * sig1
                z2_high = mu2 + k2 * sig2

                d_safe_raw = float(self._inv1(z1_low))
                v2_max_raw = float(self._inv2(z2_high))

                e_d = abs(d_safe_raw - self.oracle_d_safe_raw)
                e_v = abs(v2_max_raw - self.oracle_v2_max_raw)

                rel_d = e_d / (abs(self.oracle_d_safe_raw) + 1e-8)
                rel_v = e_v / (abs(self.oracle_v2_max_raw) + 1e-8)
            else:
                rel_d = np.nan
                rel_v = np.nan

            self.metric_d_relerr.append(float(rel_d))
            self.metric_v_relerr.append(float(rel_v))

            # -------- M-step --------
            self._mstep_update_features(gammas)
            self._mstep_update_feature_mask(gammas)
            self._mstep_update_goals(gammas, xis_list, aux_list)
            self._mstep_update_delta(xis_list, aux_list)

            posts = gammas

            if verbose:
                print(
                    f"Iter {it}: MAP_cutpoints={taus_map}, "
                    f"loglik={total_ll:.2f}, feat={total_feat_ll:.2f}, prog={total_prog_ll:.2f}, trans={total_trans_ll:.2f} | "
                    f"delta={self.delta:.4f}, g1={np.round(self.g1, 3)}, g2={np.round(self.g2, 3)}, "
                    f"r={self.r.tolist()} | "
                    f"MAE_tau={mae_tau:.3f}, NMAE_tau={nmae_tau:.3f}, "
                    f"e_g1={e_g1:.3f}, e_g2={e_g2:.3f}, "
                    f"RelErr_d={rel_d:.3f}, RelErr_v={rel_v:.3f}"
                )

            if self.plot_every is not None:
                if (it + 1) % self.plot_every == 0 or it == max_iter - 1:
                    plot_results_4panel(
                        self, taus_map, it, gammas, alphas, betas, xis_list, aux_list
                    )

        return posts
