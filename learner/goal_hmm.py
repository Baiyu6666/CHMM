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
# - 转移: 围绕 g1 的高斯 bump, learnable trans_delta
# - g1/g2 用 vMF + transition gradient 更新
# - r: (K=2, M=num_features) 的 0/1 mask, 初始为全 0，由 EM 中的 hard-EM 更新
#
# - 可选:
#     feature_ids: 使用 env 的哪些 feature 维度（索引）

#   只在 metric + 可视化时用；EM 内部对所有 feature 对称处理。
# ------------------------------------------------------------

import numpy as np
from plots.plot4panel import plot_results_4panel, plot_feature_model_debug
from eval.constraint_eval import eval_goalhmm_auto
import torch
from utils.vmf import _unit, vmf_logC_d, vmf_grad_wrt_g
from utils.models import GaussianModel, MarginExpLowerEmission, ZeroMeanGaussianModel
import matplotlib.pyplot as plt

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
        feature_types=None,

        learned_features=None,  # list[LearnedFeatureBase] 或 None
        f_lr=1e-2,  # learnable feature 的学习率
        f_mstep_steps=5,  # 每个 EM 迭代里对 g 进行多少个梯度步

        # feature mask (auto selection)
        auto_feature_select=True,
        fixed_feature_mask=None,  # 如果 auto_feature_select=False，则用这个 mask 指定哪些 feature 用于哪一阶段
        r_sparse_lambda=0.3,  # 对每个 r[k,m]=1 的惩罚（L0 风格）

        # weights
        feat_weight=1.0,
        prog_weight=1.0,
        trans_weight=1.0,

        posterior_temp = 1.0,

        # vMF kappas
        prog_kappa1=8.0,
        prog_kappa2=6.0,

        # irrelevant feature width in z-space
        fixed_sigma_irrelevant=1.0,

        # transition bump
        trans_eps=1e-6,
        delta_init=0.15,
        trans_b_init=-1,
        learn_transition=False,  # None -> follow learn_delta
        lr_delta=5e-4,
        lr_b=5e-4,

        # g update
        g_steps=5,
        g_lr=5e-4,
        g_grad_clip=None,
        g1_vmf_weight=1.0,
        g1_trans_weight=1.0,

        plot_every=200,

        eval_fn=eval_goalhmm_auto,

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

        self.eval_fn = eval_fn  # 回调
        self.metrics_hist = {}

        # HMM / emission 参数
        self.feat_weight = float(feat_weight)
        self.prog_weight = float(prog_weight)
        self.trans_weight = float(trans_weight)
        self.posterior_temp = float(posterior_temp)

        self.prog_kappa1 = float(prog_kappa1)
        self.prog_kappa2 = float(prog_kappa2)
        self.sigma_irrel = float(fixed_sigma_irrelevant)  #bg分布按理说来自whole demos的统计，但是因为还做了归一化，就变成固定1方差了

        self.trans_eps = float(trans_eps)
        self.trans_delta = float(delta_init)
        self.trans_b = float(trans_b_init)

        # learning switches (keep old API working)
        self.learn_transition = bool(learn_transition)
        self.lr_delta = float(lr_delta)
        self.lr_b = float(lr_b)

        self.g_steps = int(g_steps)
        self.g_lr = float(g_lr)
        self.g_grad_clip = g_grad_clip
        self.g1_vmf_weight = float(g1_vmf_weight)
        self.g1_trans_weight = float(g1_trans_weight)

        self.plot_every = plot_every

        # ---------------- Learnable features ----------------
        # learned_features: list of LearnedFeatureBase
        self.learned_features = list(learned_features) if learned_features is not None else []

        self.f_lr = float(f_lr)
        self.f_mstep_steps = int(f_mstep_steps)

        # 只有在存在 learnable feature 时才需要 torch 和 optimizer
        self._has_learned = len(self.learned_features) > 0

        if self._has_learned:
            import torch
            params = []
            for lf in self.learned_features:
                params += list(lf.parameters())
            # 一个总的 optimizer 就够了
            self.g_optimizer = torch.optim.Adam(params, lr=self.f_lr)
        else:
            self.g_optimizer = None

        # ---------------- Feature 结构初始化 ----------------
        self.feature_ids = feature_ids
        self.feature_types_raw = feature_types

        # 先做全局 feature 预处理（确定 num_features / z 变换）
        self._init_feature_preprocessing()
        # 后面代码（num_states, feature_models, etc.）完全沿用你现在的版本……

        # HMM 状态数量（目前写死 2 个 stage）
        self.num_states = 2

        # 根据 feature_types 为每个 state×feature 构造 emission model
        self.feature_models = []
        for k in range(self.num_states):
            row = []
            for m in range(self.num_features):
                kind = self.feature_types[m]
                if kind == "gauss":
                    row.append(GaussianModel(mu=None, sigma=None, fixed_sigma=None))
                elif kind == "zero_gauss":
                    row.append(GaussianModel(mu=None, sigma=None, fixed_sigma=None))
                    # row.append(ZeroMeanGaussianModel(sigma=None, fixed_sigma=0.1))
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
        if not auto_feature_select:
            if fixed_feature_mask is not None:
                self.r = np.asarray(fixed_feature_mask, dtype=int)
                assert self.r.shape == (self.num_states, self.num_features)
            else:
                self.r = np.ones((self.num_states, self.num_features), dtype=int)

        # init_taus 记录初始化分段，供 debug
        self.init_taus = None

        # 初始化 g1/g2 + feature_models
        self._init_goals_and_features(g1_init, g2_init)

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

        if self._has_learned:
            all_X = [np.asarray(X, float) for X in self.demos]
            for lf in self.learned_features:
                lf.update_stats(all_X)

    # ==========================================================
    # Feature 预处理
    # ==========================================================
    def _compute_all_features_raw(self, X):
        """
        从 env 取所有 **物理 feature**，再附加所有 **learnable feature**：

          F_env : shape = (T, M_env)
          F_learn: 把每个 learnable feature 的输出视为一个额外维度，拼在后面
                   shape = (T, M_learn)

        返回:
          F_raw : shape = (T, M_env + M_learn)
        """
        X = np.asarray(X, float)

        # --- 1) env 物理 feature ---
        F_env = self.env.compute_all_features_matrix(X)
        F_env = np.asarray(F_env, float)  # (T, M_env)

        T, M_env = F_env.shape

        # --- 2) learnable features（由 GoalHMM 持有） ---
        if not self._has_learned:
            return F_env

        F_list = [F_env]
        for lf in self.learned_features:
            # lf.eval_numpy: (T,) -> reshape 成 (T,1)
            g = lf.eval_numpy(X)  # (T,)
            g = np.asarray(g, float).reshape(T, -1)
            F_list.append(g)

        F_raw = np.concatenate(F_list, axis=1)  # (T, M_env + n_learn)
        return F_raw

    def _init_feature_preprocessing(self):
        """
        只用 env feature 计算全局 mean/std；
        对 learnable feature，统一设 mean=0, std=1。
        """
        all_env = []
        M_env = None

        for X in self.demos:
            X = np.asarray(X, float)

            # --- env feature ---
            F_env = np.asarray(self.env.compute_all_features_matrix(X), float)

            if M_env is None:
                M_env = F_env.shape[1]
            all_env.append(F_env)

        all_env = np.concatenate(all_env, axis=0)  # (N_total, M_env)
        self.M_env = M_env
        M_learn = len(self.learned_features)
        M_raw = M_env + M_learn

        # 若未指定 feature_ids，则默认用所有维度（env + learned）
        if self.feature_ids is None:
            self.feature_ids = list(range(M_raw))
        self.feature_ids = list(self.feature_ids)

        self.num_features = len(self.feature_ids)

        # ----- 构造 feat_mean / feat_std -----
        feat_mean = np.zeros(M_raw, dtype=float)
        feat_std = np.ones(M_raw, dtype=float)

        # 前 M_env 维：用 env 数据统计；后 M_learn 维：mean=0, std=1（因为 lf 内部已经标准化）
        for j in range(M_env):
            vals = all_env[:, j]
            feat_mean[j] = float(np.mean(vals))
            feat_std[j] = float(np.std(vals) + 1e-8)

        self.feat_mean = feat_mean  # 注意这里的feat mean包含了所有feature，而不仅是feature ids对应的哪些
        self.feat_std = feat_std

        # ----- feature_types 映射 -----
        if self.feature_types_raw is None:
            # 默认：fixed(env) -> gauss；learned -> zero_gauss（仅对出现在 feature_ids 里的维度生效）
            self.feature_types = []
            for fid in self.feature_ids:
                if fid < M_env:
                    self.feature_types.append("gauss")
                else:
                    self.feature_types.append("zero_gauss")
        else:
            types = []
            if isinstance(self.feature_types_raw, dict):
                for fid in self.feature_ids:
                    types.append(self.feature_types_raw.get(fid, "gauss"))
            else:
                for fid in self.feature_ids:
                    types.append(self.feature_types_raw[fid])
            self.feature_types = types

    def _features_for_demo_matrix(self, X):
        """
        返回当前模型真正使用的 feature 矩阵（已 z 变换）:
            shape = (T, M)，M = len(feature_ids)
        """
        F_raw = self._compute_all_features_raw(X)        # (T, M_raw)
        F_sel = F_raw[:, self.feature_ids]               # (T, M)
        Z = (F_sel - self.feat_mean[None,  self.feature_ids]) / self.feat_std[None,  self.feature_ids]
        return Z

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
                    lam = np.clip(np.random.rand(), 0.1, .9)
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

        if self._has_learned and self.g_optimizer is not None:
            gammas_init = []
            for X, tau in zip(self.demos, self.init_taus):
                T = len(X)
                tau = int(np.clip(tau, 1, T - 2))
                gamma = np.zeros((T, 2), dtype=float)
                gamma[:tau + 1, 0] = 1.0
                gamma[tau + 1:, 1] = 1.0
                gammas_init.append(gamma)

            # do a few gradient steps to make learned residual meaningful
            self._mstep_update_learned_features(gammas_init)

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
            ll_prog[:-1, 0] = (logC1 + self.prog_kappa1 * cos1)
            ll_prog[:-1, 1] = (logC2 + self.prog_kappa2 * cos2)

        # ---------- 总 emission ----------
        ll_emit = np.zeros((T, 2))
        ll_emit[:, 0] = self.feat_weight * ll_feat_k[:, 0] + self.prog_weight * ll_prog[:, 0]
        ll_emit[:, 1] = self.feat_weight * ll_feat_k[:, 1] + self.prog_weight *  ll_prog[:, 1]

        if return_parts:
            return ll_emit, self.feat_weight * ll_feat_k, self.prog_weight * ll_prog
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
                return logA, {"dists": np.zeros(T), "p12": np.zeros(T), "logit": np.zeros(T)}
            return logA

        # log-linear transition (no progress term here):
        #   logit p(0->1) = b - ||x_t - g1||^2 / (2*trans_delta^2)
        dists = np.linalg.norm(X - self.g1[None, :], axis=1)
        eps = self.trans_eps
        delta = max(self.trans_delta, 1e-6)
        b = float(getattr(self, "trans_b", 0.0))

        logit = b - 0.5 * (dists ** 2) / (delta ** 2)

        # sigmoid (stable)
        p12 = 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))
        p12 = np.clip(p12, eps, 1.0 - eps)

        logA = np.full((T - 1, 2, 2), -np.inf)
        logA[:, 0, 1] = np.log(p12[:-1])
        logA[:, 0, 0] = np.log(1.0 - p12[:-1])
        logA[:, 1, 1] = 0.0
        logA[:, 1, 0] = -np.inf

        if return_aux:
            return logA, {"p12": p12, "dists": dists, "logit": logit}
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
            Fz = self._features_for_demo_matrix(X)  # (T, M) # 这里的feature都是z空间的，包括learned feature
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
                model.m_step_update(xs, ws)

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

    # ==========================================================
    # M-step: goals & trans_delta（保持原有逻辑）
    # ==========================================================
    def _mstep_update_goals(self, gammas, xis_list, aux_list):
        if self.prog_weight <= 0.0 and self.trans_weight <= 0.0:
            return

        for _ in range(self.g_steps):
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

            # transition gradient
            g1_grad_trans = np.zeros_like(self.g1)
            w_trans_sum = 0.0

            if self.trans_weight > 0.0:
                for X, gamma, xi, aux in zip(self.demos, gammas, xis_list, aux_list):
                    if xi is None or aux is None or len(xi) == 0:
                        continue
                    p12 = aux["p12"]
                    Tm1 = xi.shape[0]
                    delta = max(self.trans_delta, 1e-6)
                    rts = []
                    for t in range(Tm1):
                        # r_t = dQ/deta_t
                        r_t = xi[t, 0, 1] - gamma[t, 0] * p12[t]
                        rts.append(r_t)
                        g1_grad_trans += r_t * (X[t] - self.g1) / (delta ** 2)
                        w_trans_sum += gamma[t, 0]


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


            self.g1 = self.g1 + self.g_lr * g1_grad
            self.g2 = self.g2 + self.g_lr * g2_grad

        self.g1_hist.append(self.g1.copy())
        self.g2_hist.append(self.g2.copy())

        # plt.figure(figsize=(6, 4))
        # plt.plot(np.arange(0, Tm1), rts, label=f"r_t ")
        # plt.plot(np.arange(0, Tm1), xi[:, 0, 1], "--", label=f"xi01 ")
        # plt.plot(np.arange(0, Tm1 + 1), p12, ":", label=f"p12 ")
        # plt.plot(np.arange(0, Tm1 + 1), p12 * gamma[:, 0], ":", label=f"p12*gamma ")
        # plt.legend()
        # plt.show()

    def _mstep_update_transition(self, gammas, xis_list, aux_list):
        """Update log-linear transition parameters (trans_delta, b).

        Transition model (left-to-right):
            p01(t) = sigmoid( b - ||x_t - g1||^2 / (2*trans_delta^2) )
            p00(t) = 1 - p01(t)
            p11 = 1, p10 = 0
        """
        if not getattr(self, "learn_transition", False):
            return

        eps = float(self.trans_eps)
        delta = max(float(self.trans_delta), 1e-6)
        b = float(getattr(self, "trans_b", 0.0))

        grad_b = 0.0
        grad_delta = 0.0
        w_sum = 0.0

        # Q_trans = sum_t [ xi01 log p + xi00 log(1-p) ]
        # dQ/deta = xi01 - gamma0 * p  (since xi00+xi01 = gamma0)
        # eta = b - 0.5*d^2/trans_delta^2
        # d eta / d b = 1
        # d eta / d trans_delta = d^2 / trans_delta^3
        for X, gamma, xi, aux in zip(self.demos, gammas, xis_list, aux_list):
            if xi is None or aux is None or len(xi) == 0:
                continue
            p12 = np.asarray(aux["p12"], float)
            dists = np.asarray(aux["dists"], float)
            Tm1 = xi.shape[0]
            for t in range(Tm1):
                p = float(np.clip(p12[t], eps, 1.0 - eps))
                d = float(dists[t])
                r_t = float(xi[t, 0, 1] - gamma[t, 0] * p)

                grad_b += r_t
                grad_delta += r_t * (d ** 2) / (delta ** 3 + 1e-12)
                w_sum += gamma[t, 0]

        if w_sum > 1e-12:
            grad_b /= w_sum
            grad_delta /= w_sum

        # gradient ascent
        self.trans_b = float(b + self.lr_b * grad_b)
        self.trans_delta = float(np.clip(delta + self.lr_delta * grad_delta, 1e-4, 2.0))

    def _mstep_update_learned_features(self, gammas):
        if not self._has_learned or self.g_optimizer is None:
            return

        for _ in range(self.f_mstep_steps):
            self.g_optimizer.zero_grad()
            total_loss = torch.tensor(0.0, dtype=torch.float32)

            for lf in self.learned_features:
                k = int(getattr(lf, "state_index", 0))
                for X, gamma in zip(self.demos, gammas):
                    X_np = np.asarray(X, float)
                    Xt = torch.from_numpy(X_np.astype(np.float32))  # (T,D)

                    g_raw = lf(Xt).squeeze(-1)  # 原始 residual
                    gamma_k = torch.from_numpy(
                        np.asarray(gamma[:, k], dtype=np.float32)
                    )

                    total_loss = total_loss + torch.sum(gamma_k * (g_raw ** 2))

            if total_loss.requires_grad:
                total_loss.backward()
                self.g_optimizer.step()

        # 参数 θ 更新完之后，用新的 g_raw 重新估计 mean/std
        all_X = [np.asarray(X, float) for X in self.demos]
        for lf in self.learned_features:
            lf.update_stats(all_X)

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
                temp = self.posterior_temp
                ll_emit /= temp
                ll_feat_k /= temp
                logA /= temp
                ll_prog /= temp

                gamma, xi, ll, alpha, beta = self._forward_backward(ll_emit , logA)

                gammas.append(gamma)
                xis_list.append(xi)
                aux_list.append(aux)
                alphas.append(alpha)
                betas.append(beta)
                total_ll += ll

                # feature 的 expected LL
                total_feat_ll +=  float(np.sum(gamma * ll_feat_k))
                # progress 的 expected LL
                total_prog_ll += float(np.sum(gamma * ll_prog))

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

            # 先统一算一份当前迭代的 MAP cutpoints，方便 eval & plot 用
            taus_hat = []
            for xi in xis_list:
                xi01 = xi[:, 0, 1]
                tau_hat = int(np.argmax(xi01))
                taus_hat.append(tau_hat)

            if self.eval_fn is not None:
                metrics = self.eval_fn(self, gammas, xis_list)  # dict
                if not hasattr(self, "metrics_hist") or self.metrics_hist is None:
                    self.metrics_hist = {}
                for name, value in metrics.items():
                    self.metrics_hist.setdefault(name, []).append(value)

                # ---------------- plotting ----------------
            if self.plot_every is not None:
                if (it) % self.plot_every == 0 or it == max_iter - 1:
                    # panel2 现在会从 learner.metrics_hist 里自动读所有曲线
                    plot_results_4panel(
                        self, taus_hat, it, gammas, alphas, betas, xis_list, aux_list
                    )
                    # debug 图这里直接传 gammas，不再依赖老的 metric 列表
                    # try:
                    #     plot_feature_model_debug(self, gammas, stages=(0, 1))
                    # except NameError:
                    #     # 如果你有时候不 import 这个函数，就静默跳过
                    #     pass
                    # try:
                    #     self._plot_learned_sine_xy(gammas)  # xs: list of trajectories, each shape (T, D)
                    # except Exception as e:
                    #     print("[GoalHMM] _plot_learned_sine_xy skipped:", repr(e))


            # ---------------- M-step ----------------
            self._mstep_update_learned_features(gammas)
            self._mstep_update_features(gammas)
            self._mstep_update_feature_mask(gammas)
            self._mstep_update_goals(gammas, xis_list, aux_list)
            self._mstep_update_transition(gammas, xis_list, aux_list)
            print(self.r)

            posts = gammas

            if verbose:
                # 打印时可以顺手从 metrics 里拿几项
                msg = (
                    f"Iter {it}: MAP_cutpoints={taus_hat}, "
                    f"loglik={total_ll:.2f}, feat={total_feat_ll:.2f}, "
                    f"prog={total_prog_ll:.2f}, trans={total_trans_ll:.2f}, "
                    f"trans_delta={self.trans_delta:.4f}, b={getattr(self, 'trans_b', 0.0):.3f}，"
                    f"g1={np.round(self.g1, 3)}, g2={np.round(self.g2, 3)}, r={self.r.tolist()}"
                )
                if hasattr(self, "metrics_hist") and self.metrics_hist:
                    last_metrics = {k: v[-1] for k, v in self.metrics_hist.items() if len(v) > 0}
                    msg += " | " + ", ".join(
                        f"{k}={last_metrics[k]:.3f}"
                        for k in sorted(last_metrics.keys())
                    )
                print(msg)
        return posts

    def _plot_learned_sine_xy(self, gammas):
        """Plot learned sine curve in *original xy space* with demos.

        Triggered only when:
          - there exists a learned feature whose name contains 'sine'
          - that learned feature is included in self.feature_ids

        Important: the learned feature is a residual g_raw = y - y_hat.
        In z-space we additionally fit a Gaussian with mean mu_z.
        To draw the sine curve in original space, we must shift the curve by
        mu_raw = mu_z * std_ + mean_ (otherwise it will be offset).
        """
        state0_xy = _collect_xy_for_state(self.demos, gammas, state_id=0, thresh=0.6)
        x_min, x_max = float(np.min(state0_xy[:, 0])), float(np.max(state0_xy[:, 0]))
        xs = np.linspace(x_min, x_max, 400)

        # plot one figure per sine learned feature (usually just one)
        for i, lf in enumerate(self.learned_features):
            name = str(getattr(lf, "name", "")).lower()
            if "sine" not in name:
                continue

            fid = int(getattr(self, "M_env", 0)) + int(i)  # raw feature id in concatenated space
            if fid not in self.feature_ids:
                # if you didn't include it in feature_ids, no need to plot
                continue
            m_local = int(self.feature_ids.index(fid))
            k = int(getattr(lf, "state_index", 0))

            # --- read learned sine params ---
            # SineCorridorResidual: y_hat = A sin(w x + phi) + bias
            A = float(lf.A.detach().cpu().item()) if hasattr(lf, "A") else 1.0
            w = float(lf.w.detach().cpu().item()) if hasattr(lf, "w") else 1.0
            phi = float(lf.phi.detach().cpu().item()) if hasattr(lf, "phi") else 0.0
            bias = float(lf.bias.detach().cpu().item()) if hasattr(lf, "bias") else 0.0

            # --- add Gaussian mean of residual (convert from z -> raw) ---
            mu_z = 0.0
            try:
                gm = self.feature_models[k][m_local]
                if hasattr(gm, "mu") and gm.mu is not None:
                    mu_z = float(gm.mu)
            except Exception:
                mu_z = 0.0

            # lf stores internal stats for raw residual: z = (g_raw - mean_) / std_
            mu_raw = mu_z * float(lf.std_.detach().cpu().item()) + float(lf.mean_.detach().cpu().item())

            ys = A * np.sin(w * xs + phi) + bias + mu_raw

            plt.figure(figsize=(8, 6))
            for X in self.demos:
                X = np.asarray(X, float)
                if X.ndim != 2 or X.shape[1] < 2:
                    continue
                plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.25, c='green')
            plt.scatter(state0_xy[:, 0], state0_xy[:, 1], s=10, label="demos (state0)", c='blue')

            plt.plot(xs, ys, linewidth=2.0)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(
                f"Learned sine in xy (feature='{getattr(lf, 'name', 'sine')}', "
                f"state={k})\n"
                f"y = A sin(w x + phi) + bias + mu_raw,  mu_raw={mu_raw:.4f} (from mu_z={mu_z:.4f})"
            )
            plt.grid(True, alpha=0.25)

            plt.show()


# ---- helper: collect points assigned to a specific state ----
def _collect_xy_for_state(demos, gammas, state_id=0, thresh=0.6):
    """
    demos: list of (T, D)
    gammas: list of (T, K) responsibilities
    return: (N, 2) xy points with gamma[:, state_id] > thresh
    """
    if gammas is None:
        raise RuntimeError("No state responsibilities found (self.gammas / self.gamma_list missing).")

    xy_list = []
    for X, G in zip(demos, gammas):
        X = np.asarray(X, float)
        G = np.asarray(G, float)

        if X.ndim != 2 or X.shape[1] < 2:
            continue
        if G.ndim != 2 or G.shape[0] != X.shape[0]:
            continue
        if state_id >= G.shape[1]:
            continue

        mask = G[:, state_id] > thresh
        if np.any(mask):
            xy_list.append(X[mask, :2])

    if len(xy_list) == 0:
        return None
    return np.concatenate(xy_list, axis=0)



