# methods/cores/cghmm_core.py
# ------------------------------------------------------------
# Baseline GMM-HMM core model.
# Works for 2D/3D and keeps the plotting/evaluation hooks expected by the
# surrounding benchmark code.
# ------------------------------------------------------------

import numpy as np
from evaluation import eval_goalhmm_auto
from utils.models import GaussianModel, MarginExpLowerEmission, ZeroMeanGaussianModel
from visualization.plot4panel import plot_results_4panel
from ..common.tau_init import extract_taus_hat, resolve_tau_init_for_demos
from ..base import format_training_log

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

class CGHMM:
    def __init__(
        self,
        demos,
        env,
        true_taus=None,
        g2_init=None,
        tau_init=None,
        tau_init_mode="uniform_taus",
        seed=0,

        gmm_K=3,
        gmm_reg=1e-6,
        fixed_sigma_irrelevant=1.0,
        feat_weight=1.0,
        x_weight=1.0,
        selected_raw_feature_ids=None,
        feature_model_types=None,
        fixed_feature_mask=None,

        A_init=None,
        pi_init=None,
        use_xy_vel=False,
        standardize_x=True,
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
        self.standardize_x = bool(standardize_x)

        self.dim_x_raw = self.demos[0].shape[1]  # 2D/3D
        self.dim_x = self.dim_x_raw * 2 if self.use_xy_vel else self.dim_x_raw

        self.gmm_K = int(gmm_K)
        self.gmm_reg = float(gmm_reg)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        self.tau_init_mode = str(tau_init_mode)
        self.tau_init = resolve_tau_init_for_demos(
            self.demos,
            tau_init=tau_init,
            tau_init_mode=self.tau_init_mode,
            env=self.env,
            seed=self.seed,
            use_velocity=self.use_xy_vel,
            vel_weight=1.0,
            standardize=self.standardize_x,
            use_env_features=True,
            selected_raw_feature_ids=selected_raw_feature_ids,
        )

        self.sigma_irrel = float(fixed_sigma_irrelevant)
        self.feat_weight = float(feat_weight)
        self.prog_weight = 0
        self.x_weight = float(x_weight)

        self.standardize_feat = bool(standardize_feat)
        self.plot_every = plot_every
        self.plot_dir = plot_dir
        self.plot_context = "cghmm"
        if self.plot_every is not None and plt is None:
            print("[CGHMM] matplotlib is not installed; plots will not be generated.")
        self.q_low = float(q_low)
        self.q_high = float(q_high)

        self.g1 = np.mean([X[t] for X, t in zip(self.demos, self.tau_init)], axis=0)

        if g2_init is None:
            self.g2 = np.mean([X[-1] for X in self.demos], axis=0)
        else:
            self.g2 = np.array(g2_init, float)

        self.g1_vis_alpha = float(g1_vis_alpha)
        self.g1_hist = [self.g1.copy()]
        self.g2_hist = [self.g2.copy()]
        self.selected_raw_feature_ids = selected_raw_feature_ids
        self.feature_model_types_raw = feature_model_types
        self.fixed_feature_mask = fixed_feature_mask

        self.num_states = 2
        self._init_feature_preprocessing()
        self._init_gmm_preprocessing()
        self.feature_models = []
        for k in range(self.num_states):
            row = []
            for m in range(self.num_features):
                kind = self.feature_model_types[m]
                if kind == "gauss":
                    row.append(GaussianModel(mu=None, sigma=None, fixed_sigma=None))
                elif kind == "zero_gauss":
                    row.append(ZeroMeanGaussianModel(sigma=None, fixed_sigma=None))
                elif kind == "margin_exp_lower":
                    row.append(MarginExpLowerEmission(b_init=0.0, lam_init=1.0))
                else:
                    raise ValueError(f"Unknown emission type '{kind}' for feature {m}")
            self.feature_models.append(row)
        if self.fixed_feature_mask is None:
            self.r = np.ones((self.num_states, self.num_features), dtype=int)
        else:
            self.r = np.asarray(self.fixed_feature_mask, dtype=int)
            if self.r.shape != (self.num_states, self.num_features):
                raise ValueError("fixed_feature_mask must have shape (2, num_selected_features).")
        self.model_feat = self.feature_models
        self._initialize_feature_models()

        # English comment omitted during cleanup.
        self.loss_loglik = []
        self.metrics_hist = {}
        self.loss_label = "Log-likelihood"

        # English comment omitted during cleanup.
        self.loss_feat = []
        self.loss_prog = []
        self.loss_trans = []

        # ---------------- GMM init ----------------
        self.gmm_weights = [
            np.ones(self.gmm_K) / self.gmm_K for _ in range(self.K_state)
        ]
        self.gmm_means = []
        self.gmm_covs = []

        seg1_list, seg2_list = [], []
        for X, tau in zip(self.demos, self.tau_init):
            X_aug = self._X_for_gmm(X)
            tau = int(np.clip(int(tau), 1, len(X_aug) - 2))
            seg1_list.append(X_aug[: tau + 1])
            seg2_list.append(X_aug[tau + 1 :])

        seg1 = np.concatenate(seg1_list, axis=0)
        seg2 = np.concatenate(seg2_list, axis=0)

        def _init_gmm_from_segment(seg):
            N = len(seg)
            if N < self.gmm_K:
                idx = self.rng.choice(N, self.gmm_K, replace=True)
            else:
                idx = self.rng.choice(N, self.gmm_K, replace=False)
            means = seg[idx].copy()
            cov_base = np.cov(seg.T) + 1e-4 * np.eye(self.dim_x)
            covs = np.stack([cov_base.copy() for _ in range(self.gmm_K)], axis=0)
            return means, covs

        m0, C0 = _init_gmm_from_segment(seg1)
        m1, C1 = _init_gmm_from_segment(seg2)
        self.gmm_means = [m0, m1]
        self.gmm_covs = [C0, C1]

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

    def _get_env_feature_schema(self, m_env):
        schema = None
        if hasattr(self.env, "get_feature_schema"):
            schema = self.env.get_feature_schema()
        elif hasattr(self.env, "feature_schema"):
            schema = getattr(self.env, "feature_schema")

        normalized = []
        if schema is not None:
            for i, item in enumerate(schema):
                spec = dict(item)
                fid = int(spec.get("id", i))
                spec["id"] = fid
                spec["column_idx"] = int(spec.get("column_idx", i))
                spec.setdefault("name", f"f{fid}")
                spec.setdefault("description", "")
                normalized.append(spec)

        if len(normalized) != int(m_env):
            normalized = [
                {"id": j, "column_idx": j, "name": f"f{j}", "description": ""}
                for j in range(int(m_env))
            ]
        return normalized

    def _resolve_feature_id(self, value):
        if isinstance(value, str):
            for spec in self.raw_feature_specs:
                if spec["name"] == value:
                    return int(spec["raw_id"])
            raise KeyError(f"Unknown feature name '{value}'.")
        return int(value)

    def _resolve_feature_model_types(self):
        if self.feature_model_types_raw is None:
            return [spec["default_type"] for spec in self.feature_specs]

        if isinstance(self.feature_model_types_raw, dict):
            out = []
            for spec in self.feature_specs:
                raw_id = spec["raw_id"]
                name = spec["name"]
                out.append(
                    self.feature_model_types_raw.get(
                        raw_id,
                        self.feature_model_types_raw.get(name, spec["default_type"]),
                    )
                )
            return out

        values = list(self.feature_model_types_raw)
        if len(values) == len(self.feature_specs):
            return values
        if self.feature_specs and max(spec["raw_id"] for spec in self.feature_specs) < len(values):
            return [values[spec["raw_id"]] for spec in self.feature_specs]
        raise ValueError(
            "feature_model_types list must either match len(selected_raw_feature_ids) or be indexable by raw feature id."
        )

    def _compute_all_features_raw(self, X):
        return np.asarray(self.env.compute_all_features_matrix(np.asarray(X, float)), float)

    def _init_feature_preprocessing(self):
        all_env = []
        m_env = None
        for X in self.demos:
            F_env = self._compute_all_features_raw(X)
            if m_env is None:
                m_env = F_env.shape[1]
            all_env.append(F_env)
        all_env = np.concatenate(all_env, axis=0)
        self.M_env = m_env

        self.env_feature_schema = self._get_env_feature_schema(m_env)
        self.raw_feature_specs = []
        for spec in self.env_feature_schema:
            self.raw_feature_specs.append(
                {
                    "raw_id": int(spec["id"]),
                    "column_idx": int(spec.get("column_idx", spec["id"])),
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "source": "env",
                    "default_type": "gauss",
                }
            )
        self.raw_feature_spec_by_id = {spec["raw_id"]: spec for spec in self.raw_feature_specs}

        if self.selected_raw_feature_ids is None:
            self.selected_raw_feature_ids = [spec["raw_id"] for spec in self.raw_feature_specs]
        self.selected_raw_feature_ids = [
            self._resolve_feature_id(fid) for fid in list(self.selected_raw_feature_ids)
        ]
        self.num_features = len(self.selected_raw_feature_ids)
        self.feature_specs = []
        for local_idx, raw_id in enumerate(self.selected_raw_feature_ids):
            if raw_id not in self.raw_feature_spec_by_id:
                raise KeyError(f"Unknown raw feature id {raw_id}.")
            spec = dict(self.raw_feature_spec_by_id[raw_id])
            spec["local_idx"] = local_idx
            self.feature_specs.append(spec)
        self.raw_id_to_local_idx = {spec["raw_id"]: spec["local_idx"] for spec in self.feature_specs}
        self.raw_id_to_column_idx = {spec["raw_id"]: spec["column_idx"] for spec in self.raw_feature_specs}
        self.feature_name_to_local_idx = {spec["name"]: spec["local_idx"] for spec in self.feature_specs}

        self.feat_mean = np.zeros(m_env, dtype=float)
        self.feat_std = np.ones(m_env, dtype=float)
        for j in range(m_env):
            vals = all_env[:, j]
            self.feat_mean[j] = float(np.mean(vals))
            self.feat_std[j] = float(np.std(vals) + 1e-8)

        self.feature_model_types = self._resolve_feature_model_types()
        for spec, kind in zip(self.feature_specs, self.feature_model_types):
            spec["type"] = kind

    def _initialize_feature_models(self):
        if self.tau_init is None:
            taus = np.asarray([max(1, len(X) // 2) for X in self.demos], dtype=int)
        else:
            taus = np.asarray(self.tau_init, dtype=int)
        per_state_xs = [[[] for _ in range(self.num_features)] for _ in range(self.num_states)]
        per_state_ws = [[[] for _ in range(self.num_features)] for _ in range(self.num_states)]
        for X, tau in zip(self.demos, taus):
            Fz = self._features_for_demo_matrix(X)
            T = len(X)
            tau = int(np.clip(tau, 1, T - 2))
            gamma0 = np.zeros(T, dtype=float)
            gamma1 = np.zeros(T, dtype=float)
            gamma0[: tau + 1] = 1.0
            gamma1[tau + 1 :] = 1.0
            for m in range(self.num_features):
                phi = Fz[:, m]
                per_state_xs[0][m].append(phi)
                per_state_xs[1][m].append(phi)
                per_state_ws[0][m].append(gamma0)
                per_state_ws[1][m].append(gamma1)
        for k in range(self.num_states):
            for m in range(self.num_features):
                if self.r[k, m] == 1:
                    self.feature_models[k][m].init_from_data(per_state_xs[k][m], per_state_ws[k][m])

    def _feature_index(self, raw_id):
        return self.raw_id_to_local_idx.get(int(raw_id))

    # ------------------------------------------------------------
    def _init_gmm_preprocessing(self):
        all_x = []
        for X in self.demos:
            all_x.append(self._X_for_gmm_raw(X))
        X_all = np.concatenate(all_x, axis=0)
        if self.standardize_x:
            self.x_mean = X_all.mean(axis=0, keepdims=True)
            self.x_std = X_all.std(axis=0, keepdims=True) + 1e-8
        else:
            self.x_mean = np.zeros((1, self.dim_x), dtype=float)
            self.x_std = np.ones((1, self.dim_x), dtype=float)

    def _augment_with_velocity(self, X):
        vel = np.zeros_like(X)
        vel[:-1] = X[1:] - X[:-1]
        vel[-1] = vel[-2]
        return np.concatenate([X, vel], axis=1)

    def _X_for_gmm_raw(self, X):
        X = np.asarray(X, float)
        return self._augment_with_velocity(X) if self.use_xy_vel else X

    def _X_for_gmm(self, X):
        X_raw = self._X_for_gmm_raw(X)
        return (X_raw - self.x_mean) / self.x_std

    def _features_for_demo_matrix(self, X):
        F_raw = self._compute_all_features_raw(X)
        selected_column_indices = [self.raw_id_to_column_idx[rid] for rid in self.selected_raw_feature_ids]
        F_sel = F_raw[:, selected_column_indices]
        Z = (F_sel - self.feat_mean[None, selected_column_indices]) / self.feat_std[
            None, selected_column_indices
        ]
        return Z

    def _log_irrelevant(self, phi):
        phi = np.asarray(phi, dtype=float)
        sig = float(self.sigma_irrel)
        sig2 = sig * sig + 1e-12
        c = -0.5 * np.log(2.0 * np.pi * sig2)
        return c - 0.5 * (phi * phi) / sig2

    # ------------------------------------------------------------
    # English comment omitted during cleanup.
    # ------------------------------------------------------------
    def get_bounds_for_plot(self, k_sigma=2):
        """
        English documentation omitted during cleanup.
          English documentation omitted during cleanup.
          English documentation omitted during cleanup.
        """
        idx_stage1 = self._feature_index(0)
        idx_stage2 = self._feature_index(1)
        if idx_stage1 is None or idx_stage2 is None:
            return np.nan, np.nan, np.nan, np.nan
        L1, U1 = self.feature_models[0][idx_stage1].interval(self.q_low, self.q_high)
        L2, U2 = self.feature_models[1][idx_stage2].interval(self.q_low, self.q_high)
        raw1 = self.selected_raw_feature_ids[idx_stage1]
        raw2 = self.selected_raw_feature_ids[idx_stage2]
        col1 = self.raw_id_to_column_idx[raw1]
        col2 = self.raw_id_to_column_idx[raw2]
        L1_raw = L1 * self.feat_std[col1] + self.feat_mean[col1]
        U1_raw = U1 * self.feat_std[col1] + self.feat_mean[col1]
        L2_raw = L2 * self.feat_std[col2] + self.feat_mean[col2]
        U2_raw = U2 * self.feat_std[col2] + self.feat_mean[col2]
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
        Fz = self._features_for_demo_matrix(X)
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
            ll_f = np.zeros(T, dtype=float)
            for m in range(self.num_features):
                if self.r[k, m] == 1:
                    ll_f += self.feature_models[k][m].logpdf(Fz[:, m])
            ll_emit[:, k] = self.x_weight * ll_x + self.feat_weight * ll_f
        return ll_emit

    def estimate_constraint_thresholds(self, k_sigma=1.0):
        """
        English documentation omitted during cleanup.
          English documentation omitted during cleanup.
        """
        idx1 = self._feature_index(0)
        idx2 = self._feature_index(1)
        if idx1 is None or idx2 is None:
            return np.nan, np.nan
        model1 = self.feature_models[0][idx1]
        model2 = self.feature_models[1][idx2]
        mu1 = float(getattr(model1, "mu", 0.0))
        sig1 = float(getattr(model1, "sigma", 1.0))
        z1_low = mu1 - k_sigma * sig1  # z-space   
        raw1 = self.selected_raw_feature_ids[idx1]
        col1 = self.raw_id_to_column_idx[raw1]
        d_safe_min_est = float(z1_low * self.feat_std[col1] + self.feat_mean[col1])

        mu2 = float(getattr(model2, "mu", 0.0))
        sig2 = float(getattr(model2, "sigma", 1.0))
        z2_high = mu2 + k_sigma * sig2  # z-space   
        raw2 = self.selected_raw_feature_ids[idx2]
        col2 = self.raw_id_to_column_idx[raw2]
        v2_max_est = float(z2_high * self.feat_std[col2] + self.feat_mean[col2])

        return d_safe_min_est, v2_max_est

    # ------------------------------------------------------------
    def _transition_logprob(self, X, return_aux=False):
        """
        English documentation omitted during cleanup.
          English documentation omitted during cleanup.
          English documentation omitted during cleanup.
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

        # English comment omitted during cleanup.
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

        # English comment omitted during cleanup.
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
        per_state_xs = [[[] for _ in range(self.num_features)] for _ in range(self.num_states)]
        per_state_ws = [[[] for _ in range(self.num_features)] for _ in range(self.num_states)]
        for X, g in zip(self.demos, gammas):
            Fz = self._features_for_demo_matrix(X)
            for m in range(self.num_features):
                phi = Fz[:, m]
                for k in range(self.num_states):
                    if self.r[k, m] == 1:
                        per_state_xs[k][m].append(phi)
                        per_state_ws[k][m].append(g[:, k])
        for k in range(self.num_states):
            for m in range(self.num_features):
                if self.r[k, m] == 1 and per_state_xs[k][m]:
                    self.feature_models[k][m].m_step_update(
                        per_state_xs[k][m],
                        per_state_ws[k][m],
                    )

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

            # --------- M-step ----------
            self._mstep_update_transitions(gammas, xis)
            self._mstep_update_features(gammas)
            self._mstep_update_gmms(gammas)
            # Re-run E-step so recorded losses/metrics reflect updated parameters.
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

            self.loss_loglik.append(total_ll)
            self.loss_feat.append(0.0)
            self.loss_prog.append(0.0)
            self.loss_trans.append(0.0)

            taus_map = np.asarray(extract_taus_hat(gammas, xis), dtype=int)
            sub_pts = []
            goal_pts = []
            for X, tau_hat in zip(self.demos, taus_map):
                T = len(X)
                tau_hat = max(1, min(T - 2, int(tau_hat)))
                sub_pts.append(X[tau_hat])
                goal_pts.append(X[-1])

            self.g1 = np.mean(np.stack(sub_pts, axis=0), axis=0)
            self.g2 = np.mean(np.stack(goal_pts, axis=0), axis=0)
            self.g1_hist.append(self.g1.copy())
            self.g2_hist.append(self.g2.copy())
            posts = gammas

            metrics = eval_goalhmm_auto(self, gammas, xis)
            for name, value in metrics.items():
                self.metrics_hist.setdefault(name, []).append(value)

            should_log = ((it + 1) % 10 == 0) or (it == max_iter - 1)
            if verbose and should_log:
                print(
                    format_training_log(
                        "CGHMM",
                        it,
                        losses={"loss": total_ll},
                        metrics=metrics,
                        extras={"taus": taus_map.tolist()},
                    )
                )

            # English comment omitted during cleanup.
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
