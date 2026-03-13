# methods/gmm_hmm_core.py
# ------------------------------------------------------------
# Baseline GMM-HMM core model.
# Works for 2D/3D and keeps the plotting/evaluation hooks expected by the
# surrounding benchmark code.
# ------------------------------------------------------------

import numpy as np
from utils.models import GaussianModel
from visualization.plot4panel import plot_results_4panel

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

class GMMHMM:
    def __init__(
        self,
        demos,
        env,
        true_taus=None,
        g1_init=None,
        g2_init=None,
        tau_init=None,          # ★ 外部给的初始 cutpoints (shared random / shared heuristic)

        gmm_K=3,
        gmm_reg=1e-6,
        fixed_sigma_irrelevant=5.0,
        feat_weight=1.0,
        x_weight=1.0,

        A_init=None,
        pi_init=None,
        use_xy_vel=False,
        standardize_feat=True,

        plot_every=200,
        plot_dir="outputs/plots",
        q_low=0.1,
        q_high=0.9,

        g1_vis_alpha=1.0,
    ):
        self.demos = list(demos)
        self.env = env
        self.true_taus = list(true_taus) if true_taus is not None else [None] * len(
            self.demos
        )

        self.K_state = 2
        self.use_xy_vel = bool(use_xy_vel)

        self.dim_x_raw = self.demos[0].shape[1]  # 2D/3D
        self.dim_x = self.dim_x_raw * 2 if self.use_xy_vel else self.dim_x_raw

        self.gmm_K = int(gmm_K)
        self.gmm_reg = float(gmm_reg)

        # tau_init：由外部实验（比如 exp_init_sensitivity）传入
        if tau_init is not None:
            tau_init = np.asarray(tau_init, dtype=int)
            assert (
                len(tau_init) == len(self.demos)
            ), "tau_init 长度必须等于 demos 数量"
            self.tau_init = tau_init
        else:
            self.tau_init = None

        self.sigma_irrel = float(fixed_sigma_irrelevant)
        self.feat_weight = float(feat_weight)
        self.prog_weight = 0
        self.x_weight = float(x_weight)

        self.standardize_feat = bool(standardize_feat)
        self.plot_every = plot_every
        self.plot_dir = plot_dir
        if self.plot_every is not None and plt is None:
            print("[GMMHMM] matplotlib is not installed; plots will not be generated.")
        self.q_low = float(q_low)
        self.q_high = float(q_high)

        # ---------------- g1 / g2 仅用于可视化 ----------------
        if g1_init is None:
            self.g1 = np.mean([X[len(X) // 2] for X in self.demos], axis=0)
        else:
            self.g1 = np.array(g1_init, float)

        if g2_init is None:
            self.g2 = np.mean([X[-1] for X in self.demos], axis=0)
        else:
            self.g2 = np.array(g2_init, float)

        self.g1_vis_alpha = float(g1_vis_alpha)
        self.g1_hist = [self.g1.copy()]
        self.g2_hist = [self.g2.copy()]

        # ---------------- feature stats (distance, speed) ----------------
        d_all, s_all = [], []
        for X in self.demos:
            dists, speeds = env.compute_features_all(X)
            d_all.append(dists)
            s_all.append(speeds)
        d_all = np.concatenate(d_all).astype(float)
        s_all = np.concatenate(s_all).astype(float)

        self.f1_mean = float(np.nanmean(d_all))
        self.f1_std = float(np.nanstd(d_all) + 1e-8)
        self.f2_mean = float(np.nanmean(s_all))
        self.f2_std = float(np.nanstd(s_all) + 1e-8)

        def z1(v):
            v = np.asarray(v, float)
            return (v - self.f1_mean) / self.f1_std if self.standardize_feat else v

        def z2(v):
            v = np.asarray(v, float)
            return (v - self.f2_mean) / self.f2_std if self.standardize_feat else v

        self._z1, self._z2 = z1, z2
        self._inv1 = lambda v: np.asarray(v, float) * self.f1_std + self.f1_mean
        self._inv2 = lambda v: np.asarray(v, float) * self.f2_std + self.f2_mean
        self.num_states = 2
        self.num_features = 2
        self.feature_ids = np.array([0, 1], dtype=int)
        self.feature_types = ["gauss", "gauss"]
        self.feat_mean = np.array([self.f1_mean, self.f2_mean], dtype=float)
        self.feat_std = np.array([self.f1_std, self.f2_std], dtype=float)

        self._init_oracle_constraints()

        # ---------------- per-state feature Gaussians ----------------
        # state0: distance relevant, speed irrelevant
        # state1: speed relevant, distance irrelevant
        self.model_feat = [
            [
                GaussianModel(mu=0.0, sigma=1.0),
                GaussianModel(
                    mu=0.0, sigma=self.sigma_irrel, fixed_sigma=self.sigma_irrel
                ),
            ],
            [
                GaussianModel(
                    mu=0.0, sigma=self.sigma_irrel, fixed_sigma=self.sigma_irrel
                ),
                GaussianModel(mu=0.0, sigma=1.0),
            ],
        ]

        mu1 = float(np.median(z1(d_all)))
        sig1 = float(np.median(np.abs(z1(d_all) - mu1)) + 1e-3)
        mu2 = float(np.median(z2(s_all)))
        sig2 = float(np.median(np.abs(z2(s_all) - mu2)) + 1e-3)

        # state0, feature0: distance
        self.model_feat[0][0].mu = mu1
        self.model_feat[0][0].sigma = sig1
        # state1, feature1: speed
        self.model_feat[1][1].mu = mu2
        self.model_feat[1][1].sigma = sig2

        # ==== 兼容 SegCons / plot4panel 的接口 ====
        # 主特征高斯（stage1: distance, stage2: speed）
        self.model_stage1 = self.model_feat[0][0]
        self.model_stage2 = self.model_feat[1][1]
        self.feature_models = self.model_feat
        self.r = np.array([[1, 0], [0, 1]], dtype=int)

        # loss 分量（GMM-HMM 没有拆分，只占位，让 plot4panel 不报错）
        self.loss_loglik = []

        # 为了和 SegCons 对齐，这里也准备占位
        self.loss_feat = []
        self.loss_prog = []
        self.loss_trans = []

        # ---- metrics (per-iteration) ----
        self.metric_tau_mae = []
        self.metric_tau_nmae = []
        self.metric_g1_err = []
        self.metric_g2_err = []
        self.metric_d_relerr = []  # baseline 不学约束，这两个我们填 nan
        self.metric_v_relerr = []

        # ---------------- GMM init ----------------
        self.gmm_weights = [
            np.ones(self.gmm_K) / self.gmm_K for _ in range(self.K_state)
        ]
        self.gmm_means = []
        self.gmm_covs = []

        demos_aug = [self._X_for_gmm(X) for X in self.demos]
        Xcat = np.concatenate(demos_aug, axis=0)

        if self.tau_init is None:
            # 原版：全局随机采样 init
            for k in range(self.K_state):
                idx = np.random.choice(len(Xcat), self.gmm_K, replace=False)
                means = Xcat[idx].copy()
                covs = np.stack(
                    [np.eye(self.dim_x) * (0.2**2) for _ in range(self.gmm_K)], axis=0
                )
                self.gmm_means.append(means)
                self.gmm_covs.append(covs)
        else:
            # 使用 tau_init 分段，给 state0/state1 各自用对应 segment 初始化
            seg1_list, seg2_list = [], []
            for X, tau in zip(self.demos, self.tau_init):
                X_aug = self._X_for_gmm(X)
                tau = int(tau)
                tau = np.clip(tau, 1, len(X_aug) - 2)
                seg1_list.append(X_aug[: tau + 1])
                seg2_list.append(X_aug[tau + 1 :])

            seg1 = np.concatenate(seg1_list, axis=0)
            seg2 = np.concatenate(seg2_list, axis=0)

            def _init_gmm_from_segment(seg):
                # seg: (N, dim_x)
                N = len(seg)
                if N < self.gmm_K:
                    idx = np.random.choice(N, self.gmm_K, replace=True)
                else:
                    idx = np.random.choice(N, self.gmm_K, replace=False)
                means = seg[idx].copy()
                cov_base = np.cov(seg.T) + 1e-4 * np.eye(self.dim_x)
                covs = np.stack(
                    [cov_base.copy() for _ in range(self.gmm_K)], axis=0
                )
                return means, covs

            m0, C0 = _init_gmm_from_segment(seg1)
            m1, C1 = _init_gmm_from_segment(seg2)
            self.gmm_means = [m0, m1]
            self.gmm_covs = [C0, C1]

            print("[GMMHMM init] Using tau_init to initialize GMM means/covs.")

        # ---------------- HMM transitions & initial state ----------------
        if A_init is None:
            A = np.array([[0.8, 0.2], [0.0, 1.0]], float)
        else:
            A = np.array(A_init, float)
            A[1, 0] = 0.0
            A[1, 1] = 1.0
            A[0] = A[0] / (A[0].sum() + 1e-12)
        self.logA = np.log(A + 1e-12)

        if pi_init is None:
            pi = np.array([0.5, 0.5], float)
        else:
            pi = np.array(pi_init, float)
        self.logpi = np.log(pi + 1e-12)

    # ------------------------------------------------------------
    def _augment_with_velocity(self, X):
        vel = np.zeros_like(X)
        vel[:-1] = X[1:] - X[:-1]
        vel[-1] = vel[-2]
        return np.concatenate([X, vel], axis=1)

    def _X_for_gmm(self, X):
        return self._augment_with_velocity(X) if self.use_xy_vel else X

    def _features_for_demo(self, X):
        dists, speeds = self.env.compute_features_all(X)
        T = len(X)
        if len(speeds) < T:
            speeds = np.concatenate([speeds, [speeds[-1]]])
        d_z = self._z1(dists)
        s_z = self._z2(speeds)
        return d_z, s_z

    def _features_for_demo_matrix(self, X):
        d_z, s_z = self._features_for_demo(X)
        return np.stack([d_z, s_z], axis=1)

    def _log_irrelevant(self, z):
        z = np.asarray(z, dtype=float)
        sigma = float(self.sigma_irrel)
        return -0.5 * (np.log(2 * np.pi * sigma * sigma) + (z * z) / (sigma * sigma + 1e-12))

    # ------------------------------------------------------------
    # 兼容 SegCons：给 plot4panel 用的可视化边界
    # ------------------------------------------------------------
    def get_bounds_for_plot(self, k_sigma=2):
        """
        返回：
          L1_raw, U1_raw: stage1 距离特征在原始单位下的 [L, U]
          L2_raw, U2_raw: stage2 速度特征在原始单位下的 [L, U]
        """
        L1, U1 = self.model_stage1.interval(self.q_low, self.q_high)
        L2, U2 = self.model_stage2.interval(self.q_low, self.q_high)

        L1_raw, U1_raw = self._inv1(L1), self._inv1(U1)
        L2_raw, U2_raw = self._inv2(L2), self._inv2(U2)
        return float(L1_raw), float(U1_raw), float(L2_raw), float(U2_raw)

    # ------------------------------------------------------------
    def _log_gauss(self, x, mu, cov):
        D = x.shape[0]
        cov = cov + self.gmm_reg * np.eye(D)
        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov) + 1e-12
        diff = x - mu
        q = diff.T @ inv @ diff
        return -0.5 * (D * np.log(2 * np.pi) + np.log(det) + q)

    def _gmm_logpdf(self, x, weights, means, covs):
        lps = []
        for w, mu, cov in zip(weights, means, covs):
            lps.append(np.log(w + 1e-12) + self._log_gauss(x, mu, cov))
        return np.logaddexp.reduce(lps)

    # ------------------------------------------------------------
    def _emission_loglik(self, X):
        T = len(X)
        d_z, s_z = self._features_for_demo(X)
        X_aug = self._X_for_gmm(X)

        ll_emit = np.zeros((T, 2))
        for k in range(2):
            ll_x = np.array(
                [
                    self._gmm_logpdf(
                        X_aug[t],
                        self.gmm_weights[k],
                        self.gmm_means[k],
                        self.gmm_covs[k],
                    )
                    for t in range(T)
                ]
            )
            ll_f1 = self.model_feat[k][0].logpdf(d_z)
            ll_f2 = self.model_feat[k][1].logpdf(s_z)
            ll_emit[:, k] = self.x_weight * ll_x + self.feat_weight * (ll_f1 + ll_f2)
        return ll_emit

    def estimate_constraint_thresholds(self, k_sigma=1.0):
        """
        返回:
          d_safe_min_est, v2_max_est  (都是 raw 单位下的数值)
        """
        # stage1: distance -> model_feat[0][0]
        mu1 = float(self.model_feat[0][0].mu)
        sig1 = float(self.model_feat[0][0].sigma)
        z1_low = mu1 - k_sigma * sig1  # z-space 下界
        d_safe_min_est = float(self._inv1(z1_low))  # 还原到原 distance 空间

        # stage2: speed -> model_feat[1][1]
        mu2 = float(self.model_feat[1][1].mu)
        sig2 = float(self.model_feat[1][1].sigma)
        z2_high = mu2 + k_sigma * sig2  # z-space 上界
        v2_max_est = float(self._inv2(z2_high))  # 还原到原 speed 空间

        return d_safe_min_est, v2_max_est

    # ------------------------------------------------------------
    def _transition_logprob(self, X, return_aux=False):
        """
        对齐 SegCons 的接口：
          - 默认：只返回 logA（原行为不变）
          - return_aux=True 时，额外返回 aux dict，供 plot4panel 画 panel4 用
        """
        T = len(X)
        if T <= 1:
            logA = np.zeros((0, 2, 2))
            if return_aux:
                aux = {
                    "p12": np.zeros(T),
                    "dists": np.zeros(T),
                }
                return logA, aux
            return logA

        logA = np.broadcast_to(self.logA, (T - 1, 2, 2)).copy()
        logA[:, 1, 0] = -np.inf
        logA[:, 1, 1] = 0.0

        if not return_aux:
            return logA

        # 构造一个“平的” p12 曲线，只是为了 panel4 能画一个东西
        p01 = float(np.exp(self.logA[0, 1]))
        p12 = np.full(T, p01, dtype=float)
        aux = {
            "p12": p12,
            "dists": np.zeros(T),
        }
        return logA, aux

    # ------------------------------------------------------------
    def _forward_backward(self, ll_emit, logA):
        T, K = ll_emit.shape
        alpha = np.full((T, K), -np.inf)
        alpha[0] = self.logpi + ll_emit[0]
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

        # ★ 现在把 alpha, beta 一起返回
        return alpha, beta, gamma, xi, loglik

    # ------------------------------------------------------------
    def _mstep_update_transitions(self, gammas, xis):
        g0 = np.sum([g[0] for g in gammas], axis=0)
        g0 = g0 / (g0.sum() + 1e-12)
        self.logpi = np.log(g0 + 1e-12)

        xi_sum = np.zeros((2, 2))
        g_sum = np.zeros((2,))
        for xi, g in zip(xis, gammas):
            xi_sum += xi.sum(axis=0)
            g_sum += g[:-1].sum(axis=0)

        A = np.zeros((2, 2))
        A[0, 0] = xi_sum[0, 0] / (g_sum[0] + 1e-12)
        A[0, 1] = xi_sum[0, 1] / (g_sum[0] + 1e-12)
        A[0] = A[0] / (A[0].sum() + 1e-12)
        A[1, 0] = 0.0
        A[1, 1] = 1.0
        self.logA = np.log(A + 1e-12)

    def _mstep_update_features(self, gammas):
        d_all, s_all, w_all = [], [], []
        for X, g in zip(self.demos, gammas):
            d_z, s_z = self._features_for_demo(X)
            d_all.append(d_z)
            s_all.append(s_z)
            w_all.append(g)
        d_all = np.concatenate(d_all)
        s_all = np.concatenate(s_all)
        w_all = np.concatenate(w_all)

        # state0 distance, state1 speed
        self.model_feat[0][0].m_update(d_all, w_all[:, 0])
        self.model_feat[1][1].m_update(s_all, w_all[:, 1])

    def _mstep_update_gmms(self, gammas):
        demos_aug = [self._X_for_gmm(X) for X in self.demos]
        Xcat = np.concatenate(demos_aug, axis=0)
        Wcat_states = np.concatenate(gammas, axis=0)

        for state in range(2):
            Wcat = Wcat_states[:, state]
            N = len(Xcat)
            R = np.zeros((N, self.gmm_K))
            for n in range(N):
                logp = []
                for k in range(self.gmm_K):
                    logp.append(
                        np.log(self.gmm_weights[state][k] + 1e-12)
                        + self._log_gauss(
                            Xcat[n],
                            self.gmm_means[state][k],
                            self.gmm_covs[state][k],
                        )
                    )
                m = np.max(logp)
                p = np.exp(logp - m)
                p /= p.sum()
                R[n] = p

            Nk = np.sum(Wcat[:, None] * R, axis=0) + 1e-12
            new_w = Nk / Nk.sum()

            new_mu = np.zeros((self.gmm_K, self.dim_x))
            new_cov = np.zeros((self.gmm_K, self.dim_x, self.dim_x))
            for k in range(self.gmm_K):
                wkr = Wcat * R[:, k]
                s = wkr.sum() + 1e-12
                new_mu[k] = (wkr[:, None] * Xcat).sum(axis=0) / s
                diff = Xcat - new_mu[k]
                new_cov[k] = (diff * (wkr[:, None])).T @ diff / s + self.gmm_reg * np.eye(
                    self.dim_x
                )

            self.gmm_weights[state] = new_w
            self.gmm_means[state] = new_mu
            self.gmm_covs[state] = new_cov

    # ------------------------------------------------------------
    def fit(self, max_iter=30, verbose=True):
        posts = None
        for it in range(max_iter):
            gammas, xis, aux_list = [], [], []
            alphas, betas = [], []

            total_ll = 0.0

            for X in self.demos:
                ll_emit = self._emission_loglik(X)
                logA = self._transition_logprob(X, return_aux=False)
                alpha, beta, gamma, xi, ll = self._forward_backward(ll_emit, logA)

                alphas.append(alpha)
                betas.append(beta)
                gammas.append(gamma)
                xis.append(xi)
                total_ll += ll

                q = xi[:, 0, 1]
                qn = q / (q.max() + 1e-12)
                aux_list.append({"p12": np.concatenate([qn, [qn[-1]]])})

            # --------- 记录 loss ----------
            self.loss_loglik.append(total_ll)
            self.loss_feat.append(0.0)
            self.loss_prog.append(0.0)
            self.loss_trans.append(0.0)

            # --------- 计算当前 iteration 的 MAP taus ----------
            taus_map = []
            for gamma in gammas:
                idx = np.where(gamma[:, 1] > 0.5)[0]
                tau_hat = int(idx[0]) if len(idx) > 0 else int(np.argmax(gamma[:, 1]))
                taus_map.append(tau_hat)
            taus_map = np.array(taus_map, dtype=int)

            # --------- 更新 “可视化用 subgoal”（均值） ----------
            # 不参与训练，只为了画图和 metric g1_err
            sub_pts = []
            goal_pts = []
            for X, tau_hat in zip(self.demos, taus_map):
                T = len(X)
                tau_hat = max(1, min(T - 1, int(tau_hat)))
                sub_pts.append(X[tau_hat])
                goal_pts.append(X[-1])  # 简单：每条 demo 的最后一个点

            sub_pts = np.stack(sub_pts, axis=0)
            goal_pts = np.stack(goal_pts, axis=0)
            self.g1_vis = sub_pts.mean(axis=0)
            self.g2_vis = goal_pts.mean(axis=0)
            self.g1_hist.append(self.g1_vis.copy())
            self.g2_hist.append(self.g2_vis.copy())

            # --------- 计算 metrics ----------
            # 1) tau MAE / NMAE
            if self.true_taus is not None and self.true_taus[0] is not None:
                true_taus_arr = np.array(self.true_taus, dtype=int)
                abs_err = np.abs(taus_map - true_taus_arr)
                tau_mae = float(abs_err.mean())

                # NMAE: 归一化到 [0,1]（用每条轨迹长度）
                lengths = np.array([len(X) for X in self.demos], dtype=float)
                tau_nmae = float((abs_err / (lengths - 1)).mean())
            else:
                tau_mae = np.nan
                tau_nmae = np.nan

            # 2) g1/g2 误差（和 env 里 true subgoal/goal 比）
            #    2D: env.subgoal, env.goal
            #    3D: env.subgoal_xy + subgoal_z 等
            try:
                if hasattr(self.env, "subgoal"):  # 2D env
                    g1_true = np.asarray(self.env.subgoal, dtype=float)
                    g2_true = np.asarray(self.env.goal, dtype=float)
                else:  # 3D env
                    g1_true = np.array(
                        [self.env.subgoal_xy[0], self.env.subgoal_xy[1], self.env.subgoal_z],
                        dtype=float,
                    )
                    g2_true = np.array(
                        [self.env.goal_xy[0], self.env.goal_xy[1], self.env.goal_z],
                        dtype=float,
                    )
                g1_err = float(np.linalg.norm(self.g1_vis - g1_true))
                g2_err = float(np.linalg.norm(self.g2_vis - g2_true))
            except Exception:
                g1_err = np.nan
                g2_err = np.nan

            # 3) baseline 不学 constraint，这里明确写 nan，避免误解
            try:
                d_est, v_est = self.estimate_constraint_thresholds(k_sigma=1.0)  # 或 2.0，看你 SegCons 那边用多少σ

                if hasattr(self, "oracle_d_safe_raw"):
                    d_true = float(self.oracle_d_safe_raw)
                    d_relerr = abs(d_est - d_true) / (abs(d_true) + 1e-8)
                else:
                    d_relerr = np.nan

                if hasattr(self, "oracle_v2_max_raw"):
                    v_true = float(self.oracle_v2_max_raw)
                    v_relerr = abs(v_est - v_true) / (abs(v_true) + 1e-8)
                else:
                    v_relerr = np.nan

            except Exception:
                d_relerr = np.nan
                v_relerr = np.nan

            self.metric_d_relerr.append(d_relerr)
            self.metric_v_relerr.append(v_relerr)

            self.metric_tau_mae.append(tau_mae)
            self.metric_tau_nmae.append(tau_nmae)
            self.metric_g1_err.append(g1_err)
            self.metric_g2_err.append(g2_err)

            # --------- M-step ----------
            self._mstep_update_transitions(gammas, xis)
            self._mstep_update_features(gammas)
            self._mstep_update_gmms(gammas)
            posts = gammas

            if verbose:
                print(
                    f"Iter {it}: MAP_cutpoints={taus_map.tolist()}, "
                    f"loglik={total_ll:.2f}, "
                    f"MAE_tau={tau_mae:.3f}, g1_err={g1_err:.3f}, g2_err={g2_err:.3f}"
                )

            # --------- 画 4-panel ----------
            if self.plot_every is not None:
                if (it + 1) % self.plot_every == 0 or it == max_iter - 1:
                    plot_results_4panel(
                        self,
                        taus_map,
                        it,
                        gammas,
                        alphas,
                        betas,
                        xis,
                        aux_list,
                    )

        return posts

    def _init_oracle_constraints(self):
        """
        Estimate raw-space oracle thresholds from true cutpoints for comparison only.
        """
        self.oracle_d_safe_raw = None
        self.oracle_v2_max_raw = None

        if self.true_taus is None or all(t is None for t in self.true_taus):
            return

        stage1_dists = []
        stage2_speeds = []

        for X, tau_true in zip(self.demos, self.true_taus):
            if tau_true is None:
                continue
            tau_true = int(tau_true)
            d_raw, s_raw = self.env.compute_features_all(X)
            T = len(X)
            tau_true = np.clip(tau_true, 0, T - 1)

            stage1_dists.append(d_raw[: tau_true + 1])
            s_idx_start = min(max(tau_true, 0), len(s_raw) - 1)
            stage2_speeds.append(s_raw[s_idx_start:])

        if len(stage1_dists) == 0 or len(stage2_speeds) == 0:
            return

        stage1_dists = np.concatenate(stage1_dists).astype(float)
        stage2_speeds = np.concatenate(stage2_speeds).astype(float)

        z_d = self._z1(stage1_dists)
        z_s = self._z2(stage2_speeds)

        mu1_or = float(np.mean(z_d))
        sig1_or = float(np.std(z_d) + 1e-8)
        mu2_or = float(np.mean(z_s))
        sig2_or = float(np.std(z_s) + 1e-8)

        z1_low_or = mu1_or - sig1_or
        z2_high_or = mu2_or + sig2_or

        self.oracle_d_safe_raw = float(self._inv1(z1_low_or))
        self.oracle_v2_max_raw = float(self._inv2(z2_high_or))
