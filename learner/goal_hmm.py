# learner/goal_hmm.py
# ------------------------------------------------------------
# GoalHMM3D (升级版)
#
# - 两状态 left-to-right HMM
# - 发射概率:
#     对每个 state k / feature m:
#         r[k,m]=1: 使用高斯 N_relevant(k,m)
#         r[k,m]=0: 使用宽高斯背景 N_irrelevant(m)
#   特征都在 z 空间（逐特征 zero-mean / unit-ish variance）
#
# - 进度项: vMF 对齐 g1/g2
# - 转移: 围绕 g1 的高斯 bump, learnable delta
# - g1/g2 用 vMF + transition gradient 更新
# - r: (K=2, M=num_features) 的 0/1 mask, 初始为全 0，由 EM 中的 hard-EM 更新
#
# - 可选:
#     feature_ids: 使用 env 的哪些 feature 维度（索引）
#     main_feat_stage1_raw: 在 env feature 里的“主约束维度”(比如 distance)
#     main_feat_stage2_raw: 在 env feature 里的“主约束维度”(比如 speed)
#   只在 metric + 可视化时用；EM 内部对所有 feature 对称处理。
# ------------------------------------------------------------

import numpy as np
from plots.plot4panel import plot_results_4panel, plot_feature_model_debug

from utils.vmf import _unit, vmf_logC_d, vmf_grad_wrt_g
from utils.models import GaussianModel, MarginExpLowerEmission

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

        # feature 相关
        feature_ids=None,           # 在 env 的所有 feature 里选哪些维度 (None=全部)
        main_feat_stage1_raw=0,     # env feature 中表示 "distance-like" 的原始索引
        main_feat_stage2_raw=1,     # env feature 中表示 "speed-like" 的原始索引
        feature_types=None,

        # weights
        feat_weight=1.0,
        prog_weight=1.0,
        trans_weight=1.0,

        # vMF kappas
        prog_kappa1=8.0,
        prog_kappa2=6.0,

        # irrelevant feature width in z-space
        fixed_sigma_irrelevant=1.0,

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

        # feature mask (auto selection)
        auto_feature_select=True,
        r_sparse_lambda=1.7,       # 对每个 r[k,m]=1 的惩罚（L0 风格）

        # plotting bands（只在 main_feat_stage1/2 上用）
        q_low=0.1,
        q_high=0.9,
        width_reg=0.0,

        plot_every=200,
    ):
        self.demos = list(demos)
        self.env = env
        self.true_taus = (
            list(true_taus) if true_taus is not None else [None] * len(self.demos)
        )

        # tau_init：外部指定 cutpoints 的选项（可不管）
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

        # temperature on progress（保留接口）
        self.tau = 1.0

        # ---------------- Feature 结构初始化 ----------------
        self.feature_ids = feature_ids           # env 的原始 feature 维度索引
        self.main_feat_stage1_raw = int(main_feat_stage1_raw)
        self.main_feat_stage2_raw = int(main_feat_stage2_raw)
        self.feature_types_raw = feature_types

        # 先做全局 feature 预处理（确定 num_features / z 变换）
        self._init_feature_preprocessing()

        # HMM 状态数量（目前写死 2 个 stage）
        self.num_states = 2
        self.num_features = self.F_dim  # 选中后的 feature 维度数

        # 根据 feature_types 为每个 state×feature 构造 emission model
        self.feature_models = []
        for k in range(self.num_states):
            row = []
            for m in range(self.num_features):
                kind = self.feature_types[m]
                if kind == "gauss":
                    row.append(GaussianModel(mu=None, sigma=None, fixed_sigma=None))
                elif kind == "margin_exp_lower":
                    # 下界约束，比如距离要大（d >= b）
                    row.append(MarginExpLowerEmission(b_init=0.0, lam_init=1.0))
                # 如果你实现了上界，也可以扩展：
                # elif kind == "margin_exp_upper":
                #     row.append(MarginExpUpperEmission(b_init=0.0, lam_init=1.0))
                else:
                    raise ValueError(f"Unknown emission type '{kind}' for feature {m}")
            self.feature_models.append(row)

        # r[k,m]∈{0,1}，初始为全 0（你要求的设定）
        self.auto_feature_select = bool(auto_feature_select)
        self.r_sparse_lambda = float(r_sparse_lambda)
        self.r = np.zeros((self.num_states, self.num_features), dtype=int)

        # “主约束维度”在选中 feature 里的索引（用于 plot & metric）
        self.main_feat_stage1 = self.feature_ids.index(self.main_feat_stage1_raw)
        self.main_feat_stage2 = self.feature_ids.index(self.main_feat_stage2_raw)

        # 方便可视化的别名：stage1 = state0/main_feat_stage1，stage2 = state1/main_feat_stage2
        self.model_stage1 = self.feature_models[0][self.main_feat_stage1]
        self.model_stage2 = self.feature_models[1][self.main_feat_stage2]

        # init_taus 记录初始化分段，供 debug
        self.init_taus = None

        # 初始化 g1/g2 + feature_models
        self._init_goals_and_features(g1_init, g2_init)

        # 以 main_feat_stage1/2 为主，初始化绘图区间
        self.L1, self.U1 = self._interval_from_model(self.model_stage1)
        self.L2, self.U2 = self._interval_from_model(self.model_stage2)

        # 日志
        self.g1_hist = [self.g1.copy()]
        self.g2_hist = [self.g2.copy()]
        self.loss_loglik = []
        self.loss_feat = []
        self.loss_prog = []
        self.loss_trans = []

        # metrics
        self.metric_tau_mae = []
        self.metric_tau_nmae = []

        self.metric_g1_err = []
        self.metric_g2_err = []

        self.metric_d_relerr = []   # distance 阈值误差（relative）
        self.metric_v_relerr = []   # speed 阈值误差（relative）

        # 基于 true_taus 的 oracle 约束（只在 main_feat_stage1/2 上）
        self._init_oracle_constraints()

    # ==========================================================
    # Feature 预处理
    # ==========================================================
    def _compute_all_features_raw(self, X):
        """
        从 env 取所有 raw features:
          - 若 env 有 compute_all_features_matrix(traj)：
                返回 shape=(T, M_raw)
          - 否则回退到旧接口 compute_features_all(traj)，得到 (dists, speeds)
        """
        if hasattr(self.env, "compute_all_features_matrix"):
            F = self.env.compute_all_features_matrix(X)
            return np.asarray(F, float)  # (T, M_raw)
        else:
            # 兼容旧版本：只用 (distance, speed)
            d, s = self.env.compute_features_all(X)  # d len=T, s len=T-1
            T = len(X)
            d = np.asarray(d, float)
            s = np.asarray(s, float)
            if len(s) < T:
                s = np.concatenate([s, [s[-1]]])
            F = np.stack([d, s], axis=1)  # (T,2)
            return F

    def _init_feature_preprocessing(self):
        """
        聚合所有 demo 的 raw features，
        决定：
          - feature_ids （如果 None 就用全部维度）
          - 每个 feature 的全局 mean/std（用于 z 变换）
        """
        all_raw = []
        for X in self.demos:
            F_raw = self._compute_all_features_raw(X)   # (T, M_raw)
            all_raw.append(F_raw)
        all_raw = np.concatenate(all_raw, axis=0)       # (N_total, M_raw)
        N_all, M_raw = all_raw.shape

        # 若未指定 feature_ids，则用所有维度
        if self.feature_ids is None:
            self.feature_ids = list(range(M_raw))
        self.feature_ids = list(self.feature_ids)

        # 选中的 feature 原始数据
        F_sel = all_raw[:, self.feature_ids]  # (N_total, M)
        self.F_dim = F_sel.shape[1]

        # 逐 feature 计算 mean/std，用于 z 变换
        self.feat_mean = np.mean(F_sel, axis=0).astype(float)
        self.feat_std = (np.std(F_sel, axis=0) + 1e-8).astype(float)

        # ---- 根据 raw feature_types 映射到选中 feature 的 type 列表 ----
        # 最终 self.feature_types: 长度 = self.F_dim，每个是 "gauss" / "margin_exp_lower" 等
        if self.feature_types_raw is None:
            # 默认全部 Gaussian
            self.feature_types = ["gauss"] * self.F_dim
        else:
            types = []
            # 允许 feature_types_raw 是 list/tuple（按 raw index）、也可以是 dict
            if isinstance(self.feature_types_raw, dict):
                for fid in self.feature_ids:
                    types.append(self.feature_types_raw.get(fid, "gauss"))
            else:
                # 假设可以按 raw 索引 index
                for fid in self.feature_ids:
                    types.append(self.feature_types_raw[fid])
            self.feature_types = types

        # main feature 在选中的 feature 中必须存在
        assert self.main_feat_stage1_raw in self.feature_ids, \
            f"main_feat_stage1_raw={self.main_feat_stage1_raw} 不在 feature_ids={self.feature_ids} 中"
        assert self.main_feat_stage2_raw in self.feature_ids, \
            f"main_feat_stage2_raw={self.main_feat_stage2_raw} 不在 feature_ids={self.feature_ids} 中"

    def _features_for_demo_matrix(self, X):
        """
        返回当前模型真正使用的 feature 矩阵（已 z 变换）:
            shape = (T, M)，M = len(feature_ids)
        """
        F_raw = self._compute_all_features_raw(X)        # (T, M_raw)
        F_sel = F_raw[:, self.feature_ids]               # (T, M)
        Z = (F_sel - self.feat_mean[None, :]) / self.feat_std[None, :]
        return Z

    # ==========================================================
    # Oracle constraints (仅用于 metric，不参与训练)
    # ==========================================================
    def _init_oracle_constraints(self):
        """
        用 true_taus 把 stage1 / stage2 的“主 feature”数据分开，
        在 z 空间拟合高斯，再用 mean±1σ 映射回 raw，得到 oracle d_safe / v2_max。
        注意：这里只看 main_feat_stage1_raw / main_feat_stage2_raw 两个维度。
        """
        self.oracle_d_safe_raw = None
        self.oracle_v2_max_raw = None

        if self.true_taus is None or all(t is None for t in self.true_taus):
            return

        stage1_vals = []
        stage2_vals = []

        for X, tau_true in zip(self.demos, self.true_taus):
            if tau_true is None:
                continue
            tau_true = int(tau_true)

            # 用旧接口拿 distance/speed（你之前代码里就是这么做的）
            d_raw, s_raw = self.env.compute_features_all(X)  # 一般是 distance, speed
            d_raw = np.asarray(d_raw, float)
            s_raw = np.asarray(s_raw, float)

            T = len(X)
            tau_true = np.clip(tau_true, 0, T - 1)

            # stage1: 0..tau_true 的 distance
            stage1_vals.append(d_raw[: tau_true + 1])

            # stage2: tau_true.. 的 speed
            s_idx_start = min(max(tau_true, 0), len(s_raw) - 1)
            stage2_vals.append(s_raw[s_idx_start:])

        if len(stage1_vals) == 0 or len(stage2_vals) == 0:
            return

        stage1_vals = np.concatenate(stage1_vals).astype(float)
        stage2_vals = np.concatenate(stage2_vals).astype(float)

        # 找到这两个 "物理 feature" 在当前选中 feature 里的索引
        idx1 = self.feature_ids.index(self.main_feat_stage1_raw)
        idx2 = self.feature_ids.index(self.main_feat_stage2_raw)

        # 用同一套 z 变换
        z1 = (stage1_vals - self.feat_mean[idx1]) / self.feat_std[idx1]
        z2 = (stage2_vals - self.feat_mean[idx2]) / self.feat_std[idx2]

        mu1_or = float(np.mean(z1))
        sig1_or = float(np.std(z1) + 1e-8)
        mu2_or = float(np.mean(z2))
        sig2_or = float(np.std(z2) + 1e-8)

        k1 = 1.0
        k2 = 1.0
        z1_low_or = mu1_or - k1 * sig1_or
        z2_high_or = mu2_or + k2 * sig2_or

        # 反变换回 raw
        self.oracle_d_safe_raw = float(z1_low_or * self.feat_std[idx1] + self.feat_mean[idx1])
        self.oracle_v2_max_raw = float(z2_high_or * self.feat_std[idx2] + self.feat_mean[idx2])

    # ==========================================================
    # 初始化 g1/g2 + feature_models
    # ==========================================================
    def _init_goals_and_features(self, g1_init, g2_init):
        """
        逻辑基本沿用你原来的版本：
          1) 先初始化 g1 和每条轨迹的 init_taus（切点）
          2) 用 init_taus 把每条轨迹分成 stage0/1，在 z 空间上
             对每个 state×feature 初始化 emission model：
               - 若有 init_from_data(xs, ws) 就调用它
               - 否则退回到旧的 median+MAD 方式给 mu/sigma
          3) 初始化 g2（终点的平均）
        """
        demos = self.demos
        true_taus = self.true_taus

        # ---------------- g1 & init_taus ----------------
        taus_init = None
        if g1_init is None:
            g1_init = "heuristic"

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
            # 显式坐标
            if isinstance(g1_init, (list, tuple, np.ndarray)) and not isinstance(
                    g1_init, str
            ):
                self.g1 = np.array(g1_init, dtype=float)

            elif isinstance(g1_init, str):
                mode = g1_init.lower()

                # true_tau
                if (
                        mode == "true_tau"
                        and true_taus is not None
                        and true_taus[0] is not None
                ):
                    taus_init = np.array(true_taus, dtype=int)
                    pts = [X[t] for X, t in zip(demos, taus_init)]
                    self.g1 = np.mean(np.stack(pts, axis=0), axis=0)

                # random (shared lambda)
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

                # heuristic：找一个让各轨迹在该点附近最聚集的 g1
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
                            taus_for_c.append(t_i)
                            d_list.append(float(dists_i[t_i]))

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
                        temp = 0.02
                        mean_ds = np.array([c[0] for c in candidates], dtype=float)
                        scores = -mean_ds / max(temp, 1e-6)
                        scores -= scores.max()
                        probs = np.exp(scores)
                        probs /= probs.sum() + 1e-12

                        idx = np.random.choice(len(candidates), p=probs)
                        _, best_center, best_taus = candidates[idx]
                        self.g1 = best_center
                        taus_init = best_taus

        if taus_init is None:
            taus = []
            for X in demos:
                dists = np.linalg.norm(X - self.g1[None, :], axis=1)
                t = int(np.argmin(dists))
                t = np.clip(t, 1, len(X) - 2)
                taus.append(t)
            taus_init = np.array(taus, dtype=int)

        self.init_taus = taus_init

        # 记录初始化时 g1 的 workspace variance
        sub_pts = np.stack([X[t] for X, t in zip(demos, taus_init)], axis=0)
        self.g1_init_var = float(
            np.mean(np.sum((sub_pts - self.g1[None, :]) ** 2, axis=1))
        )

        # ---------------- 使用 init_taus 初始化 feature_models ----------------
        # 在 z 空间中统计：每个 state / feature 的值
        feat_state0 = [[] for _ in range(self.num_features)]
        feat_state1 = [[] for _ in range(self.num_features)]

        for X, tau in zip(demos, self.init_taus):
            Fz = self._features_for_demo_matrix(X)  # (T, M)
            tau = int(np.clip(tau, 1, len(Fz) - 2))

            F0 = Fz[: tau + 1]
            F1 = Fz[tau + 1:]

            if F0.shape[0] > 0:
                for m in range(self.num_features):
                    feat_state0[m].append(F0[:, m])
            if F1.shape[0] > 0:
                for m in range(self.num_features):
                    feat_state1[m].append(F1[:, m])

        # 按 state×feature 初始化 emission：
        #   - 若有 init_from_data(xs, ws) 则调用
        #   - 否则退回到 median + MAD 初始化 mu/sigma（兼容旧 Gaussian）
        for m in range(self.num_features):
            # --- state 0 ---
            xs0 = feat_state0[m] if len(feat_state0[m]) > 0 else [np.zeros(1, dtype=float)]
            model0 = self.feature_models[0][m]
            if hasattr(model0, "init_from_data"):
                model0.init_from_data(xs0, ws=None)
            else:
                vals0 = np.concatenate(xs0).astype(float)
                med0 = np.median(vals0)
                mad0 = np.median(np.abs(vals0 - med0)) + 1e-3
                model0.mu = float(med0)
                model0.sigma = float(mad0)

            # --- state 1 ---
            xs1 = feat_state1[m] if len(feat_state1[m]) > 0 else [np.zeros(1, dtype=float)]
            model1 = self.feature_models[1][m]
            if hasattr(model1, "init_from_data"):
                model1.init_from_data(xs1, ws=None)
            else:
                vals1 = np.concatenate(xs1).astype(float)
                med1 = np.median(vals1)
                mad1 = np.median(np.abs(vals1 - med1)) + 1e-3
                model1.mu = float(med1)
                model1.sigma = float(mad1)

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

    # ==========================================================
    # Irrelevant feature log-pdf（宽高斯背景）
    # ==========================================================
    def _log_irrelevant(self, phi):
        """
        phi: 1D ndarray (z-space)
        N(0, sigma_irrel^2) 的 logpdf
        """
        sig = self.sigma_irrel
        c = -0.5 * np.log(2 * np.pi * sig ** 2)
        phi = np.asarray(phi, float)
        return c - 0.5 * (phi ** 2) / (sig ** 2 + 1e-12)

    # ==========================================================
    # Emission log-likelihood （所有 feature 对称）
    # ==========================================================
    def _emission_loglik(self, X, return_parts=False):
        """
        return:
          - ll_emit: shape (T,2)
          - 若 return_parts=True, 同时返回
              ll_feat: shape (T,2)
              ll_prog: shape (T,2)
        """
        T = len(X)
        Fz = self._features_for_demo_matrix(X)  # (T,M)
        M = self.num_features
        K = self.num_states

        # ---------- feature 部分 ----------
        ll_feat_k = np.zeros((T, K))

        # 先算所有 feature 的背景 logpdf（不依赖 state）
        ll_irrel = np.zeros((T, M))
        for m in range(M):
            ll_irrel[:, m] = self._log_irrelevant(Fz[:, m])

        # 再算各 state 的 relevant logpdf 并按 r 组合
        for k in range(K):
            tmp = np.zeros(T)
            for m in range(M):
                # relevant 高斯
                ll_rel = self.feature_models[k][m].logpdf(Fz[:, m])
                # 根据 r[k,m] 选择是用 relevant 还是 irrelevant
                tmp += self.r[k, m] * ll_rel + (1 - self.r[k, m]) * ll_irrel[:, m]
            ll_feat_k[:, k] = tmp

        # ---------- progress vMF 部分 ----------
        ll_prog = np.zeros((T, K))
        if self.prog_weight > 0.0 and T > 1:
            D = X.shape[1]
            logC1 = vmf_logC_d(self.prog_kappa1, D)
            logC2 = vmf_logC_d(self.prog_kappa2, D)

            Vs = _unit(X[1:] - X[:-1])
            U1 = _unit(self.g1[None, :] - X[:-1])
            U2 = _unit(self.g2[None, :] - X[:-1])

            cos1 = np.sum(Vs * U1, axis=1)
            cos2 = np.sum(Vs * U2, axis=1)
            ll_prog[:-1, 0] = self.prog_weight * (logC1 + self.prog_kappa1 * cos1)
            ll_prog[:-1, 1] = self.prog_weight * (logC2 + self.prog_kappa2 * cos2)

        # ---------- 总 emission ----------
        ll_emit = np.zeros((T, 2))
        ll_emit[:, 0] = self.feat_weight * ll_feat_k[:, 0] + self.tau * ll_prog[:, 0]
        ll_emit[:, 1] = self.feat_weight * ll_feat_k[:, 1] + self.tau * ll_prog[:, 1]

        if return_parts:
            return ll_emit, ll_feat_k, ll_prog
        else:
            return ll_emit

    # ==========================================================
    # Transition logprob（围绕 g1 的高斯 bump）
    # ==========================================================
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

    # ==========================================================
    # Forward-Backward
    # ==========================================================
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

    # ==========================================================
    # M-step: 更新 feature Gaussians（所有 feature 对称）
    # ==========================================================
    def _mstep_update_features(self, gammas):
        """
        对所有 state×feature 更新 emission 模型参数。

        约定：
          - 每个 emission model 都 *尽量* 实现 m_step_update(xs, ws) 接口；
          - 若没有 m_step_update，则退回到旧版 GaussianModel.m_update(y, w)。
        """
        K, M = self.num_states, self.num_features

        # 先把每个 demo 的 Fz & gamma 缓存下来，后面重复使用
        F_list = []
        gamma_list = []
        for X, gamma in zip(self.demos, gammas):
            Fz = self._features_for_demo_matrix(X)  # (T, M)
            F_list.append(Fz)
            gamma_list.append(gamma)  # (T, K)

        # ------- 更新所有 state × feature 的 emission 模型 -------
        for k in range(K):
            for m in range(M):
                xs = []
                ws = []
                for Fz, gamma in zip(F_list, gamma_list):
                    xs.append(Fz[:, m])  # 这一条 demo 在第 m 维的 feature
                    ws.append(gamma[:, k])  # 这一条 demo 对第 k 个 state 的 posterior

                model = self.feature_models[k][m]

                if hasattr(model, "m_step_update"):
                    # 统一接口：MarginExp / 新的 emission / 你改过的 Gaussian 都走这里
                    model.m_step_update(xs, ws)
                else:
                    # 向后兼容：老版 GaussianModel 只支持 m_update(y, w)
                    y_all = np.concatenate(xs, axis=0)
                    w_all = np.concatenate(ws, axis=0)
                    model.m_update(y_all, w_all)

        # ------- 同步主约束维度到 model_stage1/2（暂时仍假设这两个是高斯）-------
        fm_10 = self.feature_models[0][self.main_feat_stage1]  # stage1 的主 feature 模型
        fm_21 = self.feature_models[1][self.main_feat_stage2]  # stage2 的主 feature 模型

        # 这里先保留 Gaussian 特化逻辑：只有在主维度真的是 GaussianModel 时才更新
        from utils.models import GaussianModel  # 确保类型判断可用（也可以在文件头 import）

        if isinstance(fm_10, GaussianModel):
            self.model_stage1.mu = float(fm_10.mu)
            self.model_stage1.sigma = float(fm_10.sigma)
        if isinstance(fm_21, GaussianModel):
            self.model_stage2.mu = float(fm_21.mu)
            self.model_stage2.sigma = float(fm_21.sigma)

        # 如果主 feature 不是高斯（比如是 MarginExp），
        # 后面就应该改 _interval_from_model / 画图逻辑，用 b / lambda 来算 L/U。
        self.L1, self.U1 = self._interval_from_model(self.model_stage1)
        self.L2, self.U2 = self._interval_from_model(self.model_stage2)

    # ==========================================================
    # M-step: hard-EM 更新 feature mask r
    # ==========================================================
    def _mstep_update_feature_mask(self, gammas):
        """
        对每个 state k、feature m 独立做 hard-EM：
            Q_rel_avg   = (E[ log N_rel(k,m) ])   / N_eff - lambda
            Q_irrel_avg = (E[ log N_irrel(m) ])   / N_eff

        选更大的那个作为 r[k,m]。
        """
        if not self.auto_feature_select:
            return

        all_F = []
        all_gamma0 = []
        all_gamma1 = []
        for X, gamma in zip(self.demos, gammas):
            Fz = self._features_for_demo_matrix(X)
            all_F.append(Fz)
            all_gamma0.append(gamma[:, 0])
            all_gamma1.append(gamma[:, 1])

        F_all = np.concatenate(all_F, axis=0)      # (N,M)
        w0_all = np.concatenate(all_gamma0, axis=0)
        w1_all = np.concatenate(all_gamma1, axis=0)

        # 预计算 irrelevant logpdf（不依赖 state）
        ll_irrel_all = np.zeros_like(F_all)
        for m in range(self.num_features):
            ll_irrel_all[:, m] = self._log_irrelevant(F_all[:, m])

        for k in range(self.num_states):
            w_all = w0_all if k == 0 else w1_all
            N_eff = float(np.sum(w_all))
            if N_eff <= 1e-8:
                continue

            for m in range(self.num_features):
                phi_m = F_all[:, m]
                ll_rel = self.feature_models[k][m].logpdf(phi_m)
                ll_ir = ll_irrel_all[:, m]

                Q_rel_sum = np.sum(w_all * ll_rel)
                Q_irrel_sum = np.sum(w_all * ll_ir)

                Q_rel_avg = Q_rel_sum / (N_eff + 1e-8) - self.r_sparse_lambda
                Q_irrel_avg = Q_irrel_sum / (N_eff + 1e-8)

                if Q_rel_avg >= Q_irrel_avg:
                    self.r[k, m] = 1
                else:
                    self.r[k, m] = 0


            # if k == 0:
            #     self.r[0, :] = 0
            #     self.r[0, 0] = 1

    # ==========================================================
    # M-step: goals & delta（保持原有逻辑）
    # ==========================================================
    def _mstep_update_goals(self, gammas, xis_list, aux_list):
        if self.prog_weight <= 0.0 and self.trans_weight <= 0.0:
            return

        for _ in range(self.vmf_steps):
            g1_grad_vmf = np.zeros_like(self.g1)
            g2_grad_vmf = np.zeros_like(self.g2)
            w1_sum = 0.0
            w2_sum = 0.0

            # vMF gradient
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

            # transition gradient（attraction only）
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
                        n01 = xi[t, 0, 1]
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

    # ==========================================================
    # interval & bounds（用于绘图/metric）
    # ==========================================================
    def _interval_from_model(self, model):
        """
        根据 emission model 的 summary 计算一个 [L, U] 区间：
          - Gaussian: 调用 model.interval(q_low, q_high) 或退回到 μ±kσ
          - margin_exp_lower: 用分位数公式 b + m_q (m_q = -λ log(1-q))
          - 其他类型：先尝试 interval，否则退回 0,0

        注意：这是“可视化/metric”用的区间，不是 planner 用的硬约束。
        """
        info = {}
        if hasattr(model, "get_summary"):
            info = model.get_summary() or {}
        t = info.get("type", "unknown")

        L = None
        U = None

        # 1) 先尝试已有的 interval 接口（主要是 GaussianModel 兼容）
        if hasattr(model, "interval"):
            try:
                L0, U0 = model.interval(self.q_low, self.q_high)
                L, U = float(L0), float(U0)
            except Exception:
                L, U = None, None

        # 2) margin_exp_lower：用解析分位数覆盖掉上面的 L/U
        if t == "margin_exp_lower":
            b = float(info["b"])
            lam = float(info.get("lam", 1.0))

            q_low = np.clip(self.q_low, 0.0, 0.999999)
            q_high = np.clip(self.q_high, 0.0, 0.999999)

            # m_q = -λ log(1-q)，z_q = b + m_q
            m_low = -lam * np.log(max(1.0 - q_low, 1e-8))
            m_high = -lam * np.log(max(1.0 - q_high, 1e-8))
            L = b + m_low
            U = b + m_high

        # 3) Gaussian：如果上面没拿到 interval，就退回 μ±kσ
        if t == "gauss" and (L is None or U is None):
            mu = float(info.get("mu", 0.0))
            sigma = float(info.get("sigma", 1.0))
            # 这里用一个简单的系数（对应 q_low/q_high 的 roughly 1~2σ）
            k = 2.0
            L = mu - k * sigma
            U = mu + k * sigma

        # 4) 其他类型且没成功：给个默认值，避免报错
        if L is None or U is None:
            L, U = 0.0, 0.0

        # width_reg 收缩区间
        if self.width_reg > 0 and np.isfinite(L) and np.isfinite(U):
            c = 0.5 * (L + U)
            h = 0.5 * (U - L) / (1.0 + self.width_reg)
            L, U = c - h, c + h

        return float(L), float(U)

    def get_bounds_for_plot(self, k_sigma=2):
        """
        返回 (L1_raw, U1_raw, L2_raw, U2_raw)：
          - stage1 主 feature（一般是 distance）
          - stage2 主 feature（一般是 speed）

        k_sigma:
          - >0 且主维度是 Gaussian 时：用 μ ± k_sigma * sigma
          - 其他情况：用 self.L1/U1, self.L2/U2（由 _interval_from_model 给出）
        """
        idx1 = self.main_feat_stage1
        idx2 = self.main_feat_stage2

        use_gauss_sigma = (
                k_sigma is not None
                and k_sigma > 0
                and hasattr(self.model_stage1, "get_summary")
                and hasattr(self.model_stage2, "get_summary")
        )

        z1_low = self.L1
        z1_up = self.U1
        z2_low = self.L2
        z2_up = self.U2

        if use_gauss_sigma:
            info1 = self.model_stage1.get_summary()
            info2 = self.model_stage2.get_summary()
            if info1.get("type", "") == "gauss" and info2.get("type", "") == "gauss":
                mu1 = float(info1["mu"])
                sig1 = float(info1["sigma"])
                mu2 = float(info2["mu"])
                sig2 = float(info2["sigma"])

                z1_low = mu1 - k_sigma * sig1
                z1_up = mu1 + k_sigma * sig1
                z2_low = mu2 - k_sigma * sig2
                z2_up = mu2 + k_sigma * sig2

        # 从 z 空间映射回 raw 空间
        L1_raw = z1_low * self.feat_std[idx1] + self.feat_mean[idx1]
        U1_raw = z1_up * self.feat_std[idx1] + self.feat_mean[idx1]
        L2_raw = z2_low * self.feat_std[idx2] + self.feat_mean[idx2]
        U2_raw = z2_up * self.feat_std[idx2] + self.feat_mean[idx2]

        return float(L1_raw), float(U1_raw), float(L2_raw), float(U2_raw)

    # ==========================================================
    # 主 training loop (EM)
    # ==========================================================
    def fit(self, max_iter=30, verbose=True):
        posts = None
        for it in range(max_iter):
            gammas, xis_list, aux_list = [], [], []
            alphas, betas = [], []
            total_ll = 0.0
            total_feat_ll = 0.0
            total_prog_ll = 0.0
            total_trans_ll = 0.0

            # ---------------- E-step ----------------
            for X in self.demos:
                ll_emit, ll_feat_k, ll_prog = self._emission_loglik(X, return_parts=True)
                logA, aux = self._transition_logprob(X, return_aux=True)

                gamma, xi, ll, alpha, beta = self._forward_backward(ll_emit, logA)

                gammas.append(gamma)
                xis_list.append(xi)
                aux_list.append(aux)
                alphas.append(alpha)
                betas.append(beta)
                total_ll += ll

                # feature 的 expected LL
                total_feat_ll += self.feat_weight * float(np.sum(gamma * ll_feat_k))
                # progress 的 expected LL
                total_prog_ll += self.tau * float(np.sum(gamma * ll_prog))

                # transition 的 expected LL
                p12 = aux["p12"]
                T = len(X)
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

            # ---------------- metrics ----------------
            # segmentation
            taus_map = []
            for gamma in gammas:
                idx = np.where(gamma[:, 1] > 0.5)[0]
                tau_hat = int(idx[0]) if len(idx) > 0 else int(np.argmax(gamma[:, 1]))
                taus_map.append(tau_hat)

            mae_list, nmae_list = [], []
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

            # constraint threshold relative error（仅 main feature）
            if (self.oracle_d_safe_raw is not None) and (
                    self.oracle_v2_max_raw is not None
            ):
                idx1 = self.main_feat_stage1
                idx2 = self.main_feat_stage2

                info1 = (
                    self.model_stage1.get_summary()
                    if hasattr(self.model_stage1, "get_summary")
                    else {}
                )
                info2 = (
                    self.model_stage2.get_summary()
                    if hasattr(self.model_stage2, "get_summary")
                    else {}
                )
                t1 = info1.get("type", "")
                t2 = info2.get("type", "")

                k1 = 1.0
                k2 = 1.0

                # ---- stage1: distance lower bound ----
                z1_low = None
                if t1 == "gauss":
                    mu1 = float(info1["mu"])
                    sig1 = float(info1["sigma"])
                    z1_low = mu1 - k1 * sig1
                elif t1 == "margin_exp_lower":
                    # inequality: boundary 本身就是 z-space 的下界
                    z1_low = float(info1["b"])

                # ---- stage2: speed upper bound ----
                z2_high = None
                if t2 == "gauss":
                    mu2 = float(info2["mu"])
                    sig2 = float(info2["sigma"])
                    z2_high = mu2 + k2 * sig2
                # 如果将来你有 margin_exp_upper，可以在这里加分支

                # 计算 relative error（如果能定义的话）
                if z1_low is not None:
                    d_safe_raw = float(
                        z1_low * self.feat_std[idx1] + self.feat_mean[idx1]
                    )
                    e_d = abs(d_safe_raw - self.oracle_d_safe_raw)
                    rel_d = e_d / (abs(self.oracle_d_safe_raw) + 1e-8)
                else:
                    rel_d = np.nan

                if z2_high is not None:
                    v2_max_raw = float(
                        z2_high * self.feat_std[idx2] + self.feat_mean[idx2]
                    )
                    e_v = abs(v2_max_raw - self.oracle_v2_max_raw)
                    rel_v = e_v / (abs(self.oracle_v2_max_raw) + 1e-8)
                else:
                    rel_v = np.nan
            else:
                rel_d = np.nan
                rel_v = np.nan
            self.metric_d_relerr.append(float(rel_d))
            self.metric_v_relerr.append(float(rel_v))

            # ---------------- M-step ----------------
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
                    plot_feature_model_debug(self, posts, stages=(0, 1))

        return posts

