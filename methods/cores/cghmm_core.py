# methods/cores/cghmm_core.py
# ------------------------------------------------------------
# Baseline GMM-HMM core model.
# Works for 2D/3D and keeps the plotting/evaluation hooks expected by the
# surrounding benchmark code.
# ------------------------------------------------------------

import numpy as np
from evaluation import eval_goalhmm_auto
from utils.models import (
    GaussianModel,
    MarginExpLowerEmission,
    MarginExpLowerLeftHNEmission,
    MarginExpUpperEmission,
    MarginExpUpperRightHNEmission,
    StudentTModel,
    ZeroMeanGaussianModel,
)
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
        true_cutpoints=None,
        n_states=2,
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
        self.num_states = int(n_states)
        if self.num_states < 2:
            raise ValueError("CGHMM requires at least 2 states.")
        self.true_cutpoints = self._normalize_true_cutpoints(true_taus=true_taus, true_cutpoints=true_cutpoints)
        self.true_taus = [
            None if cuts is None or len(cuts) != 1 else int(cuts[0])
            for cuts in self.true_cutpoints
        ]

        self.K_state = self.num_states
        self.use_xy_vel = bool(use_xy_vel)
        self.standardize_x = bool(standardize_x)

        self.dim_x_raw = self.demos[0].shape[1]  # 2D/3D
        self.dim_x = self.dim_x_raw * 2 if self.use_xy_vel else self.dim_x_raw

        self.gmm_K = int(gmm_K)
        self.gmm_reg = float(gmm_reg)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        self.tau_init_mode = str(tau_init_mode)
        self.tau_init = None
        if self.num_states == 2:
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
        self.stage_ends_ = self._initial_stage_ends(tau_init=tau_init)
        self.initial_stage_ends_ = [list(map(int, ends)) for ends in self.stage_ends_]
        self.segmentation_history_ = [[list(map(int, ends)) for ends in self.stage_ends_]]

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

        self.g1_vis_alpha = float(g1_vis_alpha)
        self.selected_raw_feature_ids = selected_raw_feature_ids
        self.feature_model_types_raw = feature_model_types
        self.fixed_feature_mask = fixed_feature_mask

        self._init_feature_preprocessing()
        self._init_gmm_preprocessing()
        self.feature_models = []
        for k in range(self.num_states):
            row = []
            for m in range(self.num_features):
                kind = str(self.feature_model_types[m]).lower()
                if kind in {"gauss", "gaussian"}:
                    row.append(GaussianModel(mu=None, sigma=None, fixed_sigma=None))
                elif kind in {"student_t", "studentt", "t"}:
                    row.append(StudentTModel(mu=None, sigma=None))
                elif kind in {"zero_gauss", "zero_gaussian"}:
                    row.append(ZeroMeanGaussianModel(sigma=None, fixed_sigma=None))
                elif kind in {"margin_exp_lower", "marginexp", "margin_exp"}:
                    row.append(MarginExpLowerEmission(b_init=0.0, lam_init=1.0))
                elif kind in {"margin_exp_lower_left_hn", "marginexp_left_hn", "margin_exp_left_hn"}:
                    row.append(MarginExpLowerLeftHNEmission(b_init=0.0, lam_init=1.0))
                elif kind in {"margin_exp_upper", "marginexp_upper", "margin_exp_upper"}:
                    row.append(MarginExpUpperEmission(b_init=0.0, lam_init=1.0))
                elif kind in {"margin_exp_upper_right_hn", "marginexp_upper_right_hn", "margin_exp_upper_right_hn"}:
                    row.append(MarginExpUpperRightHNEmission(b_init=0.0, lam_init=1.0))
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
        self._initialize_stage_subgoals(g2_init=g2_init)

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

        per_state_segments = [[] for _ in range(self.num_states)]
        for X, stage_ends in zip(self.demos, self.stage_ends_):
            X_aug = self._X_for_gmm(X)
            start = 0
            for k, end in enumerate(stage_ends):
                per_state_segments[k].append(X_aug[start : int(end) + 1])
                start = int(end) + 1
        for k in range(self.num_states):
            seg = np.concatenate(per_state_segments[k], axis=0)
            means, covs = _init_gmm_from_segment(seg)
            self.gmm_means.append(means)
            self.gmm_covs.append(covs)

        # ---------------- HMM transitions & initial state ----------------
        if A_init is None:
            A = np.zeros((self.num_states, self.num_states), float)
            for i in range(self.num_states):
                A[i, i] = 0.8
                if i + 1 < self.num_states:
                    A[i, i + 1] = 0.2
                else:
                    A[i, i] = 1.0
            A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        else:
            A = np.array(A_init, float)
            if A.shape != (self.num_states, self.num_states):
                raise ValueError(f"A_init must have shape {(self.num_states, self.num_states)}.")
            for i in range(self.num_states):
                A[i, :i] = 0.0
                if i == self.num_states - 1:
                    A[i, :] = 0.0
                    A[i, i] = 1.0
                else:
                    A[i] = A[i] / (A[i].sum() + 1e-12)
        self.logA = np.log(A + 1e-12)

        if pi_init is None:
            pi = np.zeros(self.num_states, float)
            pi[0] = 1.0
        else:
            pi = np.array(pi_init, float)
        self.logpi = np.log(pi + 1e-12)

    def _normalize_true_cutpoints(self, true_taus=None, true_cutpoints=None):
        if true_cutpoints is not None:
            out = []
            for X, cuts in zip(self.demos, true_cutpoints):
                if cuts is None:
                    out.append(None)
                    continue
                arr = np.asarray(cuts, dtype=int).reshape(-1)
                arr = np.sort(arr[(arr >= 0) & (arr < len(X) - 1)])
                out.append(arr.astype(int))
            return out
        if true_taus is not None:
            return [None if tau is None else np.asarray([int(tau)], dtype=int) for tau in true_taus]
        return [None for _ in self.demos]

    def _initial_stage_ends(self, tau_init=None):
        out = []
        min_seg_len = 3
        mode = str(self.tau_init_mode).lower()
        shared_stage_proportions = None
        if self.num_states > 2 and mode in {"random_taus", "random_stage_ends", "random"}:
            shared_stage_proportions = self.rng.dirichlet(np.full(self.num_states, 0.5, dtype=float))
        for demo_idx, X in enumerate(self.demos):
            T = len(X)
            if self.num_states == 2:
                if tau_init is not None:
                    tau = int(np.asarray(tau_init, dtype=int)[demo_idx])
                elif self.tau_init is not None:
                    tau = int(self.tau_init[demo_idx])
                else:
                    tau = max(1, T // 2)
                tau = int(np.clip(tau, 1, T - 2))
                out.append([tau, T - 1])
                continue
            if T < self.num_states * min_seg_len:
                raise ValueError(
                    f"Sequence length {T} is too short for {self.num_states} states "
                    f"with minimum segment length {min_seg_len}."
                )
            if mode in {"random_taus", "random_stage_ends", "random"}:
                extra = T - self.num_states * min_seg_len
                if self.num_states > 1:
                    if extra > 0:
                        desired = extra * shared_stage_proportions
                        extra_parts = np.floor(desired).astype(int)
                        remainder = int(extra - np.sum(extra_parts))
                        if remainder > 0:
                            frac_order = np.argsort(desired - extra_parts)[::-1]
                            extra_parts[frac_order[:remainder]] += 1
                    else:
                        extra_parts = np.zeros(self.num_states, dtype=int)
                    lengths = extra_parts + min_seg_len
                    ends = np.cumsum(lengths) - 1
                else:
                    ends = np.asarray([T - 1], dtype=int)
                ends[-1] = T - 1
            else:
                ends = np.linspace(0, T, self.num_states + 1, dtype=int)[1:] - 1
                ends[-1] = T - 1
                for k in range(self.num_states - 1):
                    min_end = (k + 1) * min_seg_len - 1
                    max_end = ends[k + 1] - min_seg_len
                    ends[k] = int(np.clip(ends[k], min_end, max_end))
            out.append([int(x) for x in np.asarray(ends, dtype=int).tolist()])
        return out

    def _initialize_stage_subgoals(self, g2_init=None):
        self.stage_subgoals = []
        for stage_idx in range(self.num_states):
            pts = [np.asarray(X[int(stage_ends[stage_idx])], dtype=float) for X, stage_ends in zip(self.demos, self.stage_ends_)]
            if stage_idx == self.num_states - 1 and g2_init is not None:
                sg = np.array(g2_init, float)
            else:
                sg = np.mean(np.stack(pts, axis=0), axis=0)
            self.stage_subgoals.append(sg)
        self.stage_subgoals_hist = [[sg.copy() for sg in self.stage_subgoals]]
        self.g1 = self.stage_subgoals[0].copy()
        self.g2 = self.stage_subgoals[1].copy() if self.num_states >= 2 else self.stage_subgoals[0].copy()
        self.g1_hist = [self.g1.copy()]
        self.g2_hist = [self.g2.copy()]

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
        per_state_xs = [[[] for _ in range(self.num_features)] for _ in range(self.num_states)]
        per_state_ws = [[[] for _ in range(self.num_features)] for _ in range(self.num_states)]
        for X, stage_ends in zip(self.demos, self.stage_ends_):
            Fz = self._features_for_demo_matrix(X)
            T = len(X)
            gammas = np.zeros((self.num_states, T), dtype=float)
            start = 0
            for stage_idx, end in enumerate(stage_ends):
                gammas[stage_idx, start : int(end) + 1] = 1.0
                start = int(end) + 1
            for m in range(self.num_features):
                phi = Fz[:, m]
                for stage_idx in range(self.num_states):
                    per_state_xs[stage_idx][m].append(phi)
                    per_state_ws[stage_idx][m].append(gammas[stage_idx])
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

        ll_emit = np.zeros((T, self.num_states))
        for k in range(self.num_states):
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
            logA = np.zeros((0, self.num_states, self.num_states))
            if return_aux:
                aux = {}
                if self.num_states == 2:
                    aux["p12"] = np.zeros(T)
                    aux["dists"] = np.zeros(T)
                return logA, aux
            return logA

        logA = np.broadcast_to(self.logA, (T - 1, self.num_states, self.num_states)).copy()
        for i in range(self.num_states):
            logA[:, i, :i] = -np.inf
        logA[:, self.num_states - 1, :] = -np.inf
        logA[:, self.num_states - 1, self.num_states - 1] = 0.0

        if not return_aux:
            return logA

        aux = {}
        if self.num_states == 2:
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

        xi_sum = np.zeros((self.num_states, self.num_states))
        g_sum = np.zeros((self.num_states,))
        for xi, g in zip(xis, gammas):
            xi_sum += xi.sum(axis=0)
            g_sum += g[:-1].sum(axis=0)

        A = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_states):
            if i == self.num_states - 1:
                A[i, i] = 1.0
                continue
            for j in range(i, min(i + 2, self.num_states)):
                A[i, j] = xi_sum[i, j] / (g_sum[i] + 1e-12)
            A[i] = A[i] / (A[i].sum() + 1e-12)
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

        for state in range(self.num_states):
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

                if self.num_states == 2 and len(xi) > 0:
                    q = xi[:, 0, 1]
                    qn = q / (q.max() + 1e-12)
                    aux_list.append({"p12": np.concatenate([qn, [qn[-1]]])})
                else:
                    aux_list.append({})

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

                if self.num_states == 2 and len(xi) > 0:
                    q = xi[:, 0, 1]
                    qn = q / (q.max() + 1e-12)
                    aux_list.append({"p12": np.concatenate([qn, [qn[-1]]])})
                else:
                    aux_list.append({})

            self.loss_loglik.append(total_ll)
            self.loss_feat.append(0.0)
            self.loss_prog.append(0.0)
            self.loss_trans.append(0.0)

            labels = [np.argmax(gamma, axis=1).astype(int) for gamma in gammas]
            prev_stage_ends = [list(map(int, ends)) for ends in self.stage_ends_]
            new_stage_ends = []
            for prev_ends, z in zip(prev_stage_ends, labels):
                ends = np.where(np.diff(z) != 0)[0].astype(int).tolist() + [len(z) - 1]
                if len(ends) != self.num_states:
                    ends = prev_ends
                new_stage_ends.append([int(x) for x in ends])
            self.stage_ends_ = new_stage_ends
            self.segmentation_history_.append([list(map(int, ends)) for ends in self.stage_ends_])
            points_per_stage = [[] for _ in range(self.num_states)]
            for X, stage_ends in zip(self.demos, self.stage_ends_):
                for stage_idx, end in enumerate(stage_ends):
                    points_per_stage[stage_idx].append(np.asarray(X[int(end)], dtype=float))
            self.stage_subgoals = [np.mean(np.stack(pts, axis=0), axis=0) for pts in points_per_stage]
            self.stage_subgoals_hist.append([sg.copy() for sg in self.stage_subgoals])
            self.g1 = self.stage_subgoals[0].copy()
            self.g2 = self.stage_subgoals[1].copy() if self.num_states >= 2 else self.stage_subgoals[0].copy()
            self.g1_hist.append(self.g1.copy())
            self.g2_hist.append(self.g2.copy())
            posts = gammas

            metrics = eval_goalhmm_auto(self, gammas, xis)
            for name, value in metrics.items():
                if np.isscalar(value):
                    value_f = float(value)
                    if np.isfinite(value_f):
                        self.metrics_hist.setdefault(name, []).append(value_f)

            should_log = ((it + 1) % 10 == 0) or (it == max_iter - 1)
            if verbose and should_log:
                print(
                    format_training_log(
                        "CGHMM",
                        it,
                        losses={"loss": total_ll},
                        metrics=metrics,
                        extras={"stage_ends": self.stage_ends_},
                    )
                )

            # English comment omitted during cleanup.
            if self.plot_every is not None:
                if self.stage_ends_ is not None:
                    boundary_like = [
                        [int(x) for x in np.asarray(ends, dtype=int)[:-1]]
                        if self.num_states > 2
                        else int(np.asarray(ends, dtype=int)[0])
                        for ends in self.stage_ends_
                    ]
                else:
                    taus_map = np.asarray(extract_taus_hat(gammas, xis), dtype=int)
                    boundary_like = taus_map
                if (it + 1) % self.plot_every == 0 or it == max_iter - 1:
                    plot_results_4panel(
                        self,
                        boundary_like,
                        it,
                        gammas,
                        alphas,
                        betas,
                        xis,
                        aux_list,
                    )

        return posts
