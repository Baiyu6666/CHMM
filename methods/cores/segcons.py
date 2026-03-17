# methods/segcons.py
# ------------------------------------------------------------
# English comment omitted during cleanup.
#
# English comment omitted during cleanup.
# English comment omitted during cleanup.
# English comment omitted during cleanup.
# English comment omitted during cleanup.
# English comment omitted during cleanup.
# English comment omitted during cleanup.
#
# English comment omitted during cleanup.
# English comment omitted during cleanup.
# English comment omitted during cleanup.
# English comment omitted during cleanup.
#
# English comment omitted during cleanup.
# English comment omitted during cleanup.

# English comment omitted during cleanup.
# ------------------------------------------------------------

import numpy as np
from visualization.plot4panel import plot_results_4panel, plot_feature_model_debug
from evaluation import eval_goalhmm_auto
from ..base import format_training_log
from ..common.tau_init import extract_taus_hat, resolve_tau_init_for_demos
from utils.vmf import _unit, vmf_logC_d, vmf_grad_wrt_g
from utils.models import GaussianModel, MarginExpLowerEmission, ZeroMeanGaussianModel

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

class SegmentConstraintModel:
    def __init__(
        self,
        demos,
        env,
        true_taus=None,

        # init
        g2_init=None,
        tau_init=None,
        tau_init_mode="uniform_taus",
        seed=0,

        # English comment omitted during cleanup.
        selected_raw_feature_ids=None,  #   env     raw feature        (None=  )
        feature_model_types=None,

        # feature mask (auto selection)
        auto_feature_select=True,
        fixed_feature_mask=None,  #    auto_feature_select=False,     mask      feature       
        r_sparse_lambda=0.3,  #     r[k,m]=1    (L0   )

        # weights
        feat_weight=1.0,
        prog_weight=1.0,
        trans_weight=1.0,
        use_subgoal_consistency=True,
        use_transition_term=True,
        progress_term_type="vmf",

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
        plot_dir="outputs/plots",

        eval_fn=eval_goalhmm_auto,

    ):
        self.demos = list(demos)
        self.env = env
        self.true_taus = (
            list(true_taus) if true_taus is not None else [None] * len(self.demos)
        )

        # English comment omitted during cleanup.
        self.tau_init = None
        if tau_init is not None:
            tau_init = np.asarray(tau_init, dtype=int)
            assert len(tau_init) == len(self.demos), "tau_init        demos   "
            self.tau_init = tau_init
        self.tau_init_mode = str(tau_init_mode)
        self.seed = int(seed)

        self.eval_fn = eval_fn  #   
        self.metrics_hist = {}
        self.plot_dir = plot_dir
        self.plot_context = "joint"

        # English comment omitted during cleanup.
        self.feat_weight = float(feat_weight)
        self.use_subgoal_consistency = bool(use_subgoal_consistency)
        self.use_transition_term = bool(use_transition_term)
        self.progress_term_type = str(progress_term_type)
        if self.progress_term_type not in {"vmf", "distance_delta"}:
            raise ValueError(
                f"Unknown progress_term_type '{self.progress_term_type}'. "
                "Expected one of {'vmf', 'distance_delta'}."
            )
        self.prog_weight = float(prog_weight) if self.use_subgoal_consistency else 0.0
        self.trans_weight = float(trans_weight) if self.use_transition_term else 0.0
        self.posterior_temp = float(posterior_temp)

        self.prog_kappa1 = float(prog_kappa1)
        self.prog_kappa2 = float(prog_kappa2)
        self.sigma_irrel = float(fixed_sigma_irrelevant)  #bg       whole demos   ,          ,     1   

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
        if self.plot_every is not None and plt is None:
            print("[SegmentConstraintModel] matplotlib is not installed; plots will not be generated.")

        # English comment omitted during cleanup.
        self.selected_raw_feature_ids = selected_raw_feature_ids
        self.feature_model_types_raw = feature_model_types

        # English comment omitted during cleanup.
        self._init_feature_preprocessing()
        # English comment omitted during cleanup.

        # English comment omitted during cleanup.
        self.num_states = 2

        # English comment omitted during cleanup.
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
                    # English comment omitted during cleanup.
                    row.append(MarginExpLowerEmission(b_init=0.0, lam_init=1.0))
                # English comment omitted during cleanup.
                # elif kind == "margin_exp_upper":
                #     row.append(MarginExpUpperEmission(b_init=0.0, lam_init=1.0))
                else:
                    raise ValueError(f"Unknown emission type '{kind}' for feature {m}")
            self.feature_models.append(row)
        self.progress_models = None
        if self.progress_term_type == "distance_delta":
            self.progress_models = [
                GaussianModel(mu=0.05, sigma=0.05, fixed_sigma=None),
                GaussianModel(mu=0.05, sigma=0.05, fixed_sigma=None),
            ]

        # English comment omitted during cleanup.
        self.auto_feature_select = bool(auto_feature_select)
        self.r_sparse_lambda = float(r_sparse_lambda)
        self.r = np.zeros((self.num_states, self.num_features), dtype=int)
        if self.auto_feature_select and fixed_feature_mask is not None:
            raise ValueError("fixed_feature_mask requires auto_feature_select=False.")
        if not auto_feature_select:
            if fixed_feature_mask is not None:
                self.r = np.asarray(fixed_feature_mask, dtype=int)
                assert self.r.shape == (self.num_states, self.num_features)
            else:
                self.r = np.ones((self.num_states, self.num_features), dtype=int)

        # English comment omitted during cleanup.
        self.init_taus = None

        # English comment omitted during cleanup.
        self._init_goals_and_features(g2_init)

        # English comment omitted during cleanup.
        self.g1_hist = [self.g1.copy()]
        self.g2_hist = [self.g2.copy()]
        self.loss_loglik = []
        self.loss_feat = []
        self.loss_prog = []
        self.loss_trans = []

    # ==========================================================
    # English comment omitted during cleanup.
    # ==========================================================
    def _compute_all_features_raw(self, X):
        """
        English documentation omitted during cleanup.

        English documentation omitted during cleanup.
          F_raw : shape = (T, M_env)
        """
        X = np.asarray(X, float)
        F_env = self.env.compute_all_features_matrix(X)
        return np.asarray(F_env, float)

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

    def _init_feature_preprocessing(self):
        """
        English documentation omitted during cleanup.
        English documentation omitted during cleanup.
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

        self.env_feature_schema = self._get_env_feature_schema(M_env)
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

        # English comment omitted during cleanup.
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

        # English comment omitted during cleanup.
        feat_mean = np.zeros(M_env, dtype=float)
        feat_std = np.ones(M_env, dtype=float)

        # English comment omitted during cleanup.
        for j in range(M_env):
            vals = all_env[:, j]
            feat_mean[j] = float(np.mean(vals))
            feat_std[j] = float(np.std(vals) + 1e-8)

        self.feat_mean = feat_mean  #      feat mean     feature,    feature ids     
        self.feat_std = feat_std

        # English comment omitted during cleanup.
        self.feature_model_types = self._resolve_feature_model_types()
        for spec, kind in zip(self.feature_specs, self.feature_model_types):
            spec["type"] = kind

    def _features_for_demo_matrix(self, X):
        """
        English documentation omitted during cleanup.
            English documentation omitted during cleanup.
        """
        F_raw = self._compute_all_features_raw(X)        # (T, M_raw)
        selected_column_indices = [self.raw_id_to_column_idx[rid] for rid in self.selected_raw_feature_ids]
        F_sel = F_raw[:, selected_column_indices]               # (T, M)
        Z = (
            F_sel - self.feat_mean[None, selected_column_indices]
        ) / self.feat_std[None, selected_column_indices]
        return Z

    # ==========================================================
    # English comment omitted during cleanup.
    # ==========================================================
    def _clip_tau(self, X, tau):
        return int(np.clip(int(tau), 1, len(X) - 2))

    def _resolve_init_taus(self):
        return resolve_tau_init_for_demos(
            self.demos,
            tau_init=self.tau_init,
            tau_init_mode=self.tau_init_mode,
            env=self.env,
            seed=self.seed,
            use_velocity=False,
            vel_weight=1.0,
            standardize=False,
            use_env_features=True,
            selected_raw_feature_ids=self.selected_raw_feature_ids,
        )

    def _run_estep_once(self):
        gammas, xis_list, aux_list = [], [], []
        alphas, betas = [], []
        total_ll = 0.0
        total_feat_ll = 0.0
        total_prog_ll = 0.0
        total_trans_ll = 0.0
        taus_hat = []

        for X in self.demos:
            ll_emit, ll_feat_k, ll_prog = self._emission_loglik(X, return_parts=True)
            logA, aux = self._transition_logprob(X, return_aux=True)
            temp = self.posterior_temp
            ll_emit /= temp
            ll_feat_k /= temp
            logA /= temp
            ll_prog /= temp

            gamma, xi, ll, alpha, beta = self._forward_backward(ll_emit, logA)
            gammas.append(gamma)
            xis_list.append(xi)
            aux_list.append(aux)
            alphas.append(alpha)
            betas.append(beta)
            total_ll += ll
            total_feat_ll += float(np.sum(gamma * ll_feat_k))
            total_prog_ll += float(np.sum(gamma * ll_prog))

            p12 = aux["p12"]
            T = len(X)
            for t in range(T - 1):
                n00 = xi[t, 0, 0]
                n01 = xi[t, 0, 1]
                total_trans_ll += n01 * np.log(p12[t] + 1e-12) + n00 * np.log(1 - p12[t] + 1e-12)

            taus_hat.append(extract_taus_hat([gamma], [xi])[0])

        return {
            "gammas": gammas,
            "xis_list": xis_list,
            "aux_list": aux_list,
            "alphas": alphas,
            "betas": betas,
            "total_ll": total_ll,
            "total_feat_ll": total_feat_ll,
            "total_prog_ll": total_prog_ll,
            "total_trans_ll": total_trans_ll,
            "taus_hat": taus_hat,
        }

    def _init_g1_from_taus(self, taus_init):
        sub_pts = np.stack([X[t] for X, t in zip(self.demos, taus_init)], axis=0)
        self.g1 = np.mean(sub_pts, axis=0)
        self.g1_init_var = float(np.mean(np.sum((sub_pts - self.g1[None, :]) ** 2, axis=1)))

    def _init_feature_models_from_taus(self, taus_init):
        feat_state0 = [[] for _ in range(self.num_features)]
        feat_state1 = [[] for _ in range(self.num_features)]

        for X, tau in zip(self.demos, taus_init):
            Fz = self._features_for_demo_matrix(X)
            tau = self._clip_tau(Fz, tau)
            F0 = Fz[: tau + 1]
            F1 = Fz[tau + 1 :]

            if F0.shape[0] > 0:
                for m in range(self.num_features):
                    feat_state0[m].append(F0[:, m])
            if F1.shape[0] > 0:
                for m in range(self.num_features):
                    feat_state1[m].append(F1[:, m])

        for m in range(self.num_features):
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

    def _init_g2_from_taus(self, taus_init, g2_init):
        if isinstance(g2_init, (list, tuple, np.ndarray)) and not isinstance(g2_init, str):
            self.g2 = np.array(g2_init, dtype=float)
            return
        terminal_points = []
        for X, tau in zip(self.demos, taus_init):
            tau = self._clip_tau(X, tau)
            terminal_points.append(np.asarray(X[tau + 1 :], float)[-1])
        self.g2 = np.mean(terminal_points, axis=0)

    def _init_goals_and_features(self, g2_init):
        taus_init = self._resolve_init_taus()
        self.init_taus = taus_init
        self._init_g1_from_taus(taus_init)
        self._init_feature_models_from_taus(taus_init)
        self._init_g2_from_taus(taus_init, g2_init)
        self._init_progress_models_from_taus(taus_init)

    def _progress_delta(self, X, goal):
        X = np.asarray(X, float)
        if len(X) <= 1:
            return np.zeros(0, dtype=float)
        d0 = np.linalg.norm(X[:-1] - goal[None, :], axis=1)
        d1 = np.linalg.norm(X[1:] - goal[None, :], axis=1)
        return d0 - d1

    def _progress_grad_distance_delta(self, x0, x1, goal, model):
        x0 = np.asarray(x0, float)
        x1 = np.asarray(x1, float)
        goal = np.asarray(goal, float)
        r0 = goal - x0
        r1 = goal - x1
        n0 = max(float(np.linalg.norm(r0)), 1e-12)
        n1 = max(float(np.linalg.norm(r1)), 1e-12)
        y = n0 - n1
        mu = float(model.mu)
        sigma = max(float(model.sigma), 1e-6)
        dy_dg = (r0 / n0) - (r1 / n1)
        coeff = -(y - mu) / (sigma ** 2)
        return coeff * dy_dg

    def _init_progress_models_from_taus(self, taus_init):
        if self.progress_models is None:
            return
        ys_state = [[], []]
        for X, tau in zip(self.demos, taus_init):
            tau = self._clip_tau(X, tau)
            y0 = self._progress_delta(X[: tau + 1], self.g1)
            y1 = self._progress_delta(X[tau:], self.g2)
            if len(y0) > 0:
                ys_state[0].append(y0)
            if len(y1) > 0:
                ys_state[1].append(y1)
        for k in range(self.num_states):
            xs = ys_state[k] if ys_state[k] else [np.zeros(1, dtype=float)]
            self.progress_models[k].init_from_data(xs, ws=None)

    # ==========================================================
    # English comment omitted during cleanup.
    # ==========================================================
    def _log_irrelevant(self, phi):
        """
        phi: 1D ndarray (z-space)
        English documentation omitted during cleanup.
        """
        sig = self.sigma_irrel
        c = -0.5 * np.log(2 * np.pi * sig ** 2)
        phi = np.asarray(phi, float)
        return c - 0.5 * (phi ** 2) / (sig ** 2 + 1e-12)

    # ==========================================================
    # English comment omitted during cleanup.
    # ==========================================================
    def _emission_loglik(self, X, return_parts=False):
        """
        return:
          - ll_emit: shape (T,2)
          English documentation omitted during cleanup.
              ll_feat: shape (T,2)
              ll_prog: shape (T,2)
        """
        T = len(X)
        Fz = self._features_for_demo_matrix(X)  # (T,M)
        M = self.num_features
        K = self.num_states

        # English comment omitted during cleanup.
        ll_feat_k = np.zeros((T, K))

        # English comment omitted during cleanup.
        ll_irrel = np.zeros((T, M))
        for m in range(M):
            ll_irrel[:, m] = self._log_irrelevant(Fz[:, m])

        # English comment omitted during cleanup.
        for k in range(K):
            tmp = np.zeros(T)
            for m in range(M):
                # English comment omitted during cleanup.
                ll_rel = self.feature_models[k][m].logpdf(Fz[:, m])
                # English comment omitted during cleanup.
                tmp += self.r[k, m] * ll_rel + (1 - self.r[k, m]) * ll_irrel[:, m]
            ll_feat_k[:, k] = tmp

        # English comment omitted during cleanup.
        ll_prog = np.zeros((T, K))
        if self.prog_weight > 0.0 and T > 1:
            if self.progress_term_type == "vmf":
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
            else:
                y0 = self._progress_delta(X, self.g1)
                y1 = self._progress_delta(X, self.g2)
                ll_prog[:-1, 0] = self.progress_models[0].logpdf(y0)
                ll_prog[:-1, 1] = self.progress_models[1].logpdf(y1)

        # English comment omitted during cleanup.
        ll_emit = np.zeros((T, 2))
        ll_emit[:, 0] = self.feat_weight * ll_feat_k[:, 0] + self.prog_weight * ll_prog[:, 0]
        ll_emit[:, 1] = self.feat_weight * ll_feat_k[:, 1] + self.prog_weight *  ll_prog[:, 1]

        if return_parts:
            return ll_emit, self.feat_weight * ll_feat_k, self.prog_weight * ll_prog
        else:
            return ll_emit

    # ==========================================================
    # English comment omitted during cleanup.
    # ==========================================================
    def _transition_logprob(self, X, return_aux=False):
        T = len(X)
        if T <= 1:
            logA = np.zeros((0, 2, 2))
            if return_aux:
                return logA, {"dists": np.zeros(T), "p12": np.zeros(T), "logit": np.zeros(T)}
            return logA

        if self.trans_weight <= 0.0:
            p12 = np.full(T, 0.5, dtype=float)
            logA = np.full((T - 1, 2, 2), -np.inf)
            logA[:, 0, 1] = np.log(0.5)
            logA[:, 0, 0] = np.log(0.5)
            logA[:, 1, 1] = 0.0
            logA[:, 1, 0] = -np.inf
            if return_aux:
                return logA, {"dists": np.zeros(T), "p12": p12, "logit": np.zeros(T)}
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
    # English comment omitted during cleanup.
    # ==========================================================
    def _mstep_update_features(self, gammas):
        """
        English documentation omitted during cleanup.

        English documentation omitted during cleanup.
          English documentation omitted during cleanup.
          English documentation omitted during cleanup.
        """
        K, M = self.num_states, self.num_features

        # English comment omitted during cleanup.
        F_list = []
        gamma_list = []
        for X, gamma in zip(self.demos, gammas):
            Fz = self._features_for_demo_matrix(X)  # (T, M)
            F_list.append(Fz)
            gamma_list.append(gamma)  # (T, K)

        # English comment omitted during cleanup.
        for k in range(K):
            for m in range(M):
                xs = []
                ws = []
                for Fz, gamma in zip(F_list, gamma_list):
                    xs.append(Fz[:, m])  #     demo    m    feature
                    ws.append(gamma[:, k])  #     demo    k   state   posterior

                model = self.feature_models[k][m]
                model.m_step_update(xs, ws)

    def _mstep_update_progress_models(self, gammas):
        if self.progress_models is None or self.prog_weight <= 0.0:
            return
        ys_state = [[], []]
        ws_state = [[], []]
        for X, gamma in zip(self.demos, gammas):
            y0 = self._progress_delta(X, self.g1)
            y1 = self._progress_delta(X, self.g2)
            if len(y0) > 0:
                ys_state[0].append(y0)
                ws_state[0].append(gamma[:-1, 0])
            if len(y1) > 0:
                ys_state[1].append(y1)
                ws_state[1].append(gamma[:-1, 1])
        for k in range(self.num_states):
            if ys_state[k]:
                self.progress_models[k].m_step_update(ys_state[k], ws_state[k])

    # ==========================================================
    # English comment omitted during cleanup.
    # ==========================================================
    def _mstep_update_feature_mask(self, gammas):
        """
        English documentation omitted during cleanup.
            Q_rel_avg   = (E[ log N_rel(k,m) ])   / N_eff - lambda
            Q_irrel_avg = (E[ log N_irrel(m) ])   / N_eff

        English documentation omitted during cleanup.
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

        # English comment omitted during cleanup.
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
    # English comment omitted during cleanup.
    # ==========================================================
    def _mstep_update_goals(self, gammas, xis_list, aux_list):
        if self.prog_weight <= 0.0 and self.trans_weight <= 0.0:
            return

        for _ in range(self.g_steps):
            g1_grad_vmf = np.zeros_like(self.g1)
            g2_grad_vmf = np.zeros_like(self.g2)
            w1_sum = 0.0
            w2_sum = 0.0

            # progress gradient
            if self.prog_weight > 0.0:
                for X, gamma in zip(self.demos, gammas):
                    T = len(X)
                    for t in range(T - 1):
                        w1 = gamma[t, 0]
                        w2 = gamma[t, 1]
                        if w1 > 0:
                            if self.progress_term_type == "vmf":
                                g1_grad_vmf += w1 * vmf_grad_wrt_g(
                                    X[t : t + 2], self.g1, self.prog_kappa1
                                )
                            else:
                                g1_grad_vmf += w1 * self._progress_grad_distance_delta(
                                    X[t], X[t + 1], self.g1, self.progress_models[0]
                                )
                            w1_sum += w1
                        if w2 > 0:
                            if self.progress_term_type == "vmf":
                                g2_grad_vmf += w2 * vmf_grad_wrt_g(
                                    X[t : t + 2], self.g2, self.prog_kappa2
                                )
                            else:
                                g2_grad_vmf += w2 * self._progress_grad_distance_delta(
                                    X[t], X[t + 1], self.g2, self.progress_models[1]
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

    # ==========================================================
    # English comment omitted during cleanup.
    # ==========================================================
    def fit(self, max_iter=30, verbose=True):
        posts = None
        for it in range(max_iter):
            estep_pre = self._run_estep_once()
            gammas = estep_pre["gammas"]
            xis_list = estep_pre["xis_list"]
            aux_list = estep_pre["aux_list"]

            self._mstep_update_features(gammas)
            self._mstep_update_feature_mask(gammas)
            self._mstep_update_progress_models(gammas)
            self._mstep_update_goals(gammas, xis_list, aux_list)
            self._mstep_update_transition(gammas, xis_list, aux_list)

            estep_post = self._run_estep_once()
            posts = estep_post["gammas"]
            self.loss_loglik.append(estep_post["total_ll"])
            self.loss_feat.append(estep_post["total_feat_ll"])
            self.loss_prog.append(estep_post["total_prog_ll"])
            self.loss_trans.append(estep_post["total_trans_ll"])

            metrics = {}
            if self.eval_fn is not None:
                metrics = self.eval_fn(self, estep_post["gammas"], estep_post["xis_list"])
                if not hasattr(self, "metrics_hist") or self.metrics_hist is None:
                    self.metrics_hist = {}
                for name, value in metrics.items():
                    self.metrics_hist.setdefault(name, []).append(value)

            if self.plot_every is not None:
                if (it) % self.plot_every == 0 or it == max_iter - 1:
                    plot_results_4panel(
                        self,
                        estep_post["taus_hat"],
                        it,
                        estep_post["gammas"],
                        estep_post["alphas"],
                        estep_post["betas"],
                        estep_post["xis_list"],
                        estep_post["aux_list"],
                    )

            should_log = ((it + 1) % 10 == 0) or (it == max_iter - 1)
            if verbose and should_log:
                print(
                    format_training_log(
                        "SEGCONS",
                        it,
                        losses={
                            "loss": estep_post["total_ll"],
                            "feat": estep_post["total_feat_ll"],
                            "prog": estep_post["total_prog_ll"],
                            "trans": estep_post["total_trans_ll"],
                        },
                        metrics=metrics,
                        extras={
                            "prog_term": self.progress_term_type if self.use_subgoal_consistency else "off",
                            "trans_term": "on" if self.use_transition_term else "off",
                            "trans_delta": f"{self.trans_delta:.4f}",
                            "b": f"{getattr(self, 'trans_b', 0.0):.3f}",
                            "taus": estep_post["taus_hat"],
                            "r": self.r.tolist(),
                        },
                    )
                )
        return posts
