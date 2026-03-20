from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from evaluation import eval_goalhmm_auto
from methods.base import format_training_log
from methods.common.tau_init import clip_tau_for_sequence, resolve_tau_init_for_demos
from utils.models import GaussianModel, MarginExpLowerEmission, StudentTModel, ZeroMeanGaussianModel
from visualization import plot_ccp_progress_boundary_profile, plot_ccp_progress_heatmaps, plot_ccp_results_4panel
from visualization.ccp_4panel import plt as ccp_plot_plt


def _feature_schema(env):
    if hasattr(env, "get_feature_schema"):
        return list(env.get_feature_schema())
    schema = getattr(env, "feature_schema", None)
    return list(schema) if schema is not None else None


def _resolve_selected_feature_columns(env, selected_raw_feature_ids):
    schema = _feature_schema(env)
    if selected_raw_feature_ids is None:
        if schema is not None:
            return [int(spec.get("column_idx", i)) for i, spec in enumerate(schema)]
        raise ValueError("env feature schema is required when selected_raw_feature_ids is None.")

    if schema is None:
        raise ValueError("env feature schema is required when selected_raw_feature_ids is provided.")

    name_to_column = {}
    id_to_column = {}
    for i, spec in enumerate(schema):
        column_idx = int(spec.get("column_idx", i))
        name_to_column[str(spec.get("name", f"f{i}"))] = column_idx
        id_to_column[int(spec.get("id", i))] = column_idx

    out = []
    for value in selected_raw_feature_ids:
        if isinstance(value, str):
            if value not in name_to_column:
                raise KeyError(f"Unknown feature name '{value}'.")
            out.append(name_to_column[value])
        else:
            raw_id = int(value)
            if raw_id not in id_to_column:
                raise KeyError(f"Unknown raw feature id '{raw_id}'.")
            out.append(id_to_column[raw_id])
    return out


def _safe_log(x, eps=1e-8):
    return np.log(np.clip(x, eps, 1.0))


def _hard_gammas_from_stage_ends(lengths: Sequence[int], stage_ends_per_demo: Sequence[Sequence[int]], K: int):
    gammas: List[np.ndarray] = []
    for T, stage_ends in zip(lengths, stage_ends_per_demo):
        gamma = np.zeros((int(T), int(K)), dtype=float)
        start = 0
        for k, end in enumerate(stage_ends):
            gamma[start : end + 1, k] = 1.0
            start = int(end) + 1
        gammas.append(gamma)
    return gammas


def _segment_lengths(stage_ends_per_demo: Sequence[Sequence[int]]) -> List[List[int]]:
    out: List[List[int]] = []
    for stage_ends in stage_ends_per_demo:
        prev = -1
        cur: List[int] = []
        for end in stage_ends:
            cur.append(int(end) - prev)
            prev = int(end)
        out.append(cur)
    return out


def _state_summary_prefix(states_list: Sequence[np.ndarray]):
    prefix_sum = []
    prefix_sq_sum = []
    for X in states_list:
        X = np.asarray(X, dtype=float)
        prefix = np.zeros((len(X) + 1, X.shape[1]), dtype=float)
        prefix_sq = np.zeros((len(X) + 1, X.shape[1]), dtype=float)
        prefix[1:] = np.cumsum(X, axis=0)
        prefix_sq[1:] = np.cumsum(X * X, axis=0)
        prefix_sum.append(prefix)
        prefix_sq_sum.append(prefix_sq)
    return prefix_sum, prefix_sq_sum


@dataclass
class StageGainMLP:
    input_dim: int
    hidden_dim: int
    rng: np.random.RandomState

    def __post_init__(self):
        scale1 = 1.0 / math.sqrt(max(self.input_dim, 1))
        scale2 = 1.0 / math.sqrt(max(self.hidden_dim, 1))
        self.W1 = self.rng.randn(self.input_dim, self.hidden_dim) * scale1
        self.b1 = np.zeros(self.hidden_dim, dtype=float)
        self.W2 = self.rng.randn(self.hidden_dim, 1) * scale2
        self.b2 = np.zeros(1, dtype=float)

    def predict_score(self, X):
        X = np.asarray(X, dtype=float)
        H = np.tanh(X @ self.W1 + self.b1[None, :])
        logits = H @ self.W2 + self.b2[None, :]
        return logits[:, 0]

    def predict_prob(self, X):
        scores = self.predict_score(X)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))

    def fit_ranking(self, X_pos, X_neg, steps=100, lr=1e-2, weight_decay=1e-4):
        X_pos = np.asarray(X_pos, dtype=float)
        X_neg = np.asarray(X_neg, dtype=float)
        if len(X_pos) == 0 or len(X_neg) == 0:
            return
        n = float(len(X_pos))
        for _ in range(int(steps)):
            H_pos_pre = X_pos @ self.W1 + self.b1[None, :]
            H_pos = np.tanh(H_pos_pre)
            score_pos = (H_pos @ self.W2 + self.b2[None, :])[:, 0]

            H_neg_pre = X_neg @ self.W1 + self.b1[None, :]
            H_neg = np.tanh(H_neg_pre)
            score_neg = (H_neg @ self.W2 + self.b2[None, :])[:, 0]

            diff = score_pos - score_neg
            coeff = -1.0 / (1.0 + np.exp(np.clip(diff, -40.0, 40.0)))
            coeff = coeff[:, None] / n

            dscore_pos = coeff
            dscore_neg = -coeff

            dW2 = H_pos.T @ dscore_pos + H_neg.T @ dscore_neg + weight_decay * self.W2
            db2 = np.sum(dscore_pos + dscore_neg, axis=0)

            dH_pos = dscore_pos @ self.W2.T
            dH_neg = dscore_neg @ self.W2.T
            dH_pos_pre = dH_pos * (1.0 - H_pos * H_pos)
            dH_neg_pre = dH_neg * (1.0 - H_neg * H_neg)
            dW1 = X_pos.T @ dH_pos_pre + X_neg.T @ dH_neg_pre + weight_decay * self.W1
            db1 = np.sum(dH_pos_pre + dH_neg_pre, axis=0)

            self.W2 -= float(lr) * dW2
            self.b2 -= float(lr) * db2
            self.W1 -= float(lr) * dW1
            self.b1 -= float(lr) * db1


class ConstraintCompletionProgressModel:
    def __init__(
        self,
        demos,
        env,
        true_taus=None,
        n_states=2,
        tau_init=None,
        tau_init_mode="uniform_taus",
        seed=0,
        selected_raw_feature_ids=None,
        feature_model_types=None,
        auto_feature_select=False,
        fixed_feature_mask=None,
        r_sparse_lambda=0.0,
        lambda_constraint=1.0,
        lambda_end=0.5,
        lambda_progress=1.0,
        progress_term_type="ranking",
        progress_delta_scale=20.0,
        duration_min=None,
        duration_max=None,
        duration_slack=0,
        progress_rank_hidden_dim=32,
        progress_rank_steps=100,
        progress_rank_lr=1e-2,
        progress_rank_weight_decay=1e-4,
        hard_negative_radius=2,
        random_negative_per_demo=4,
        end_precision_floor=1e-3,
        plot_every=None,
        plot_dir="outputs/plots",
        eval_fn=eval_goalhmm_auto,
    ):
        self.demos = [np.asarray(X, dtype=float) for X in demos]
        self.env = env
        self.true_taus = list(true_taus) if true_taus is not None else [None] * len(self.demos)
        self.num_states = int(n_states)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.tau_init = None
        self.tau_init_mode = str(tau_init_mode)
        self.selected_raw_feature_ids = selected_raw_feature_ids
        self.feature_model_types_raw = feature_model_types
        self.auto_feature_select = bool(auto_feature_select)
        self.fixed_feature_mask = fixed_feature_mask
        self.r_sparse_lambda = float(r_sparse_lambda)
        self.lambda_constraint = float(lambda_constraint)
        self.lambda_end = float(lambda_end)
        self.lambda_progress = float(lambda_progress)
        self.progress_term_type = str(progress_term_type)
        if self.progress_term_type not in {"ranking", "delta"}:
            raise ValueError(
                f"Unknown progress_term_type '{self.progress_term_type}'. "
                "Expected one of {'ranking', 'delta'}."
            )
        self.progress_delta_scale = max(float(progress_delta_scale), 1e-6)
        self.duration_slack = int(duration_slack)
        self.progress_rank_hidden_dim = int(progress_rank_hidden_dim)
        self.progress_rank_steps = int(progress_rank_steps)
        self.progress_rank_lr = float(progress_rank_lr)
        self.progress_rank_weight_decay = float(progress_rank_weight_decay)
        self.hard_negative_radius = int(hard_negative_radius)
        self.random_negative_per_demo = int(random_negative_per_demo)
        self.end_precision_floor = float(end_precision_floor)
        self.plot_every = plot_every
        self.plot_dir = plot_dir
        self.eval_fn = eval_fn
        if self.plot_every is not None and ccp_plot_plt is None:
            print("[CCP] matplotlib is not installed; CCP 4-panel plots will not be generated.")

        if self.num_states == 2:
            self.tau_init = resolve_tau_init_for_demos(
                self.demos,
                tau_init=tau_init,
                tau_init_mode=self.tau_init_mode,
                env=self.env,
                seed=self.seed,
                use_velocity=False,
                vel_weight=1.0,
                standardize=False,
                use_env_features=True,
                selected_raw_feature_ids=selected_raw_feature_ids,
            )
        elif tau_init is not None:
            self.tau_init = np.asarray(tau_init, dtype=int)

        self._init_feature_preprocessing()
        self.feature_models = self._build_feature_models()
        self.r = np.ones((self.num_states, self.num_features), dtype=int)
        if fixed_feature_mask is not None:
            mask = np.asarray(fixed_feature_mask, dtype=int)
            if mask.shape != self.r.shape:
                raise ValueError(f"fixed_feature_mask must have shape {self.r.shape}.")
            self.r = mask.copy()
        self._resolve_duration_bounds(duration_min, duration_max)
        self.duration_means = np.ones(self.num_states, dtype=float)
        self.duration_stds = np.ones(self.num_states, dtype=float)
        self.end_mu = np.zeros((self.num_states, self.state_dim), dtype=float)
        self.end_precision = np.ones((self.num_states, self.state_dim), dtype=float)
        self.progress_rank_models: List[StageGainMLP | None] = [None] * self.num_states
        self.metrics_hist: Dict[str, List[float]] = {}
        self.loss_total: List[float] = []
        self.loss_constraint: List[float] = []
        self.loss_end: List[float] = []
        self.loss_progress: List[float] = []
        self.g1_hist: List[np.ndarray] = []
        self.g2_hist: List[np.ndarray] = []
        self.segmentation_history: List[List[List[int]]] = []
        self._init_goal_placeholders()

    def _init_goal_placeholders(self):
        if hasattr(self.env, "subgoal"):
            self.g1 = np.full_like(np.asarray(self.env.subgoal, dtype=float), np.nan, dtype=float)
        if hasattr(self.env, "goal"):
            self.g2 = np.full_like(np.asarray(self.env.goal, dtype=float), np.nan, dtype=float)

    def _normalize_feature_model_types(self, num_features):
        types = self.feature_model_types_raw
        if types is None:
            return ["gaussian"] * int(num_features)
        if isinstance(types, dict):
            schema = _feature_schema(self.env) or []
            column_to_name = {}
            column_to_id = {}
            for i, spec in enumerate(schema):
                column_idx = int(spec.get("column_idx", i))
                column_to_name[column_idx] = str(spec.get("name", f"f{i}"))
                column_to_id[column_idx] = int(spec.get("id", i))
            resolved = []
            for column_idx in self.selected_feature_columns:
                name = column_to_name.get(int(column_idx))
                raw_id = column_to_id.get(int(column_idx))
                if name in types:
                    resolved.append(types[name])
                elif raw_id in types:
                    resolved.append(types[raw_id])
                else:
                    resolved.append("gaussian")
            return resolved
        if len(types) != int(num_features):
            raise ValueError("feature_model_types must match the number of selected features.")
        return list(types)

    def _build_feature_models(self):
        feature_model_types = self._normalize_feature_model_types(self.num_features)
        self.feature_model_types = feature_model_types
        rows = []
        for _ in range(self.num_states):
            cur = []
            for kind in feature_model_types:
                kind = str(kind).lower()
                if kind in {"gauss", "gaussian"}:
                    cur.append(GaussianModel(mu=0.0, sigma=1.0))
                elif kind in {"student_t", "studentt", "t"}:
                    cur.append(StudentTModel(mu=0.0, sigma=1.0))
                elif kind in {"zero_gauss", "zero_gaussian"}:
                    cur.append(ZeroMeanGaussianModel(sigma=1.0))
                elif kind in {"margin_exp_lower", "marginexp", "margin_exp"}:
                    cur.append(MarginExpLowerEmission(b_init=0.0, lam_init=1.0))
                else:
                    raise ValueError(f"Unsupported feature model type '{kind}'.")
            rows.append(cur)
        return rows

    def _init_feature_preprocessing(self):
        if self.env is None:
            raise ValueError("CCP requires a dataset env with feature API.")
        raw_features = [np.asarray(self.env.compute_all_features_matrix(X), dtype=float) for X in self.demos]
        if not raw_features:
            raise ValueError("CCP requires at least one demo.")
        self.raw_feature_dim = int(raw_features[0].shape[1])
        schema = _feature_schema(self.env)
        if schema is None:
            schema = [{"id": i, "column_idx": i, "name": f"f{i}"} for i in range(self.raw_feature_dim)]
        self.raw_feature_specs = list(schema)
        self.selected_feature_columns = _resolve_selected_feature_columns(self.env, self.selected_raw_feature_ids)
        self.selected_feature_columns = [int(i) for i in self.selected_feature_columns]
        self.num_features = len(self.selected_feature_columns)

        full_stack = np.concatenate(raw_features, axis=0)
        self.feat_mean = np.mean(full_stack, axis=0)
        self.feat_std = np.std(full_stack, axis=0) + 1e-8
        self.standardized_features = []
        self.prefix_sum = []
        self.prefix_sq_sum = []
        for F_raw in raw_features:
            F_sel = F_raw[:, self.selected_feature_columns]
            mean_sel = self.feat_mean[self.selected_feature_columns]
            std_sel = self.feat_std[self.selected_feature_columns]
            F_std = (F_sel - mean_sel[None, :]) / std_sel[None, :]
            self.standardized_features.append(F_std)
            prefix = np.zeros((len(F_std) + 1, self.num_features), dtype=float)
            prefix_sq = np.zeros((len(F_std) + 1, self.num_features), dtype=float)
            prefix[1:] = np.cumsum(F_std, axis=0)
            prefix_sq[1:] = np.cumsum(F_std * F_std, axis=0)
            self.prefix_sum.append(prefix)
            self.prefix_sq_sum.append(prefix_sq)
        self.state_dim = int(self.demos[0].shape[1])
        self.state_prefix_sum, self.state_prefix_sq_sum = _state_summary_prefix(self.demos)

        self.raw_id_to_column_idx = {}
        self.raw_id_to_local_idx = {}
        for spec_idx, spec in enumerate(self.raw_feature_specs):
            raw_id = int(spec.get("id", spec_idx))
            col_idx = int(spec.get("column_idx", spec_idx))
            self.raw_id_to_column_idx[raw_id] = col_idx
        selected_column_to_local = {int(col): i for i, col in enumerate(self.selected_feature_columns)}
        for spec_idx, spec in enumerate(self.raw_feature_specs):
            raw_id = int(spec.get("id", spec_idx))
            col_idx = int(spec.get("column_idx", spec_idx))
            if col_idx in selected_column_to_local:
                self.raw_id_to_local_idx[raw_id] = int(selected_column_to_local[col_idx])

    def _resolve_duration_bounds(self, duration_min, duration_max):
        self.duration_min = self._broadcast_stage_value(duration_min, default=1, dtype=int)
        self.duration_max = self._broadcast_stage_value(duration_max, default=None, dtype=int)
        max_lengths = np.asarray([len(X) for X in self.demos], dtype=int)
        total_max = int(np.max(max_lengths))
        for k in range(self.num_states):
            if self.duration_max[k] is None:
                self.duration_max[k] = total_max
        self.duration_max = np.asarray(self.duration_max, dtype=int)

    def _broadcast_stage_value(self, value, default, dtype=float):
        if value is None:
            return [default for _ in range(self.num_states)]
        if np.isscalar(value):
            return [dtype(value) for _ in range(self.num_states)]
        value = list(value)
        if len(value) != self.num_states:
            raise ValueError(f"Expected {self.num_states} stage values, got {len(value)}.")
        return [default if v is None else dtype(v) for v in value]

    def _default_stage_ends(self, T):
        if self.num_states < 1:
            raise ValueError("n_states must be positive.")
        boundaries = []
        for k in range(1, self.num_states):
            frac = float(k) / float(self.num_states)
            cut = int(round(frac * (T - 1)))
            cut = int(np.clip(cut, k, T - (self.num_states - k) - 1))
            boundaries.append(cut)
        return self._stage_ends_from_cuts(boundaries, T)

    def _stage_ends_from_cuts(self, cuts, T):
        cuts = [int(c) for c in cuts]
        out = cuts + [int(T) - 1]
        for k in range(len(out) - 1):
            if out[k] >= out[k + 1]:
                out[k] = max(k, out[k + 1] - 1)
        return out

    def _initialize_stage_ends(self):
        stage_ends_per_demo = []
        if self.tau_init is not None and self.num_states == 2:
            if len(self.tau_init) != len(self.demos):
                raise ValueError("tau_init must match the number of demos.")
            for X, tau in zip(self.demos, self.tau_init):
                tau = clip_tau_for_sequence(X, int(tau))
                stage_ends_per_demo.append([tau, len(X) - 1])
            return stage_ends_per_demo
        for X in self.demos:
            stage_ends_per_demo.append(self._default_stage_ends(len(X)))
        return stage_ends_per_demo

    def _segment_bounds_from_stage_ends(self, stage_ends):
        starts = []
        ends = []
        prev = -1
        for end in stage_ends:
            starts.append(prev + 1)
            ends.append(int(end))
            prev = int(end)
        return starts, ends

    def _segment_mean_std(self, demo_idx, s, e):
        count = float(e - s + 1)
        total = self.prefix_sum[demo_idx][e + 1] - self.prefix_sum[demo_idx][s]
        total_sq = self.prefix_sq_sum[demo_idx][e + 1] - self.prefix_sq_sum[demo_idx][s]
        mean = total / max(count, 1.0)
        var = np.maximum(total_sq / max(count, 1.0) - mean * mean, 1e-8)
        return mean, np.sqrt(var)

    def _segment_state_mean_std(self, demo_idx, s, e):
        count = float(e - s + 1)
        total = self.state_prefix_sum[demo_idx][e + 1] - self.state_prefix_sum[demo_idx][s]
        total_sq = self.state_prefix_sq_sum[demo_idx][e + 1] - self.state_prefix_sq_sum[demo_idx][s]
        mean = total / max(count, 1.0)
        var = np.maximum(total_sq / max(count, 1.0) - mean * mean, 1e-8)
        return mean, np.sqrt(var)

    def _frame_constraint_nll(self, stage_idx, frame_features):
        frame_features = np.asarray(frame_features, dtype=float)
        total = np.zeros(frame_features.shape[0], dtype=float)
        count = 0.0
        for m, model in enumerate(self.feature_models[stage_idx]):
            if not self.r[stage_idx, m]:
                continue
            total -= np.asarray(model.logpdf(frame_features[:, m]), dtype=float)
            count += 1.0
        if count <= 0.0:
            return np.zeros(frame_features.shape[0], dtype=float)
        return total / count

    def _constraint_loss(self, demo_idx, stage_idx, s, e):
        return float(np.sum(self._frame_constraint_nll(stage_idx, self.standardized_features[demo_idx][s : e + 1])))

    def _end_loss(self, demo_idx, stage_idx, e):
        x_e = self.demos[demo_idx][e]
        diff = x_e - self.end_mu[stage_idx]
        return float(np.sum(self.end_precision[stage_idx] * diff * diff))

    def _duration_loss(self, stage_idx, duration):
        duration = float(duration)
        min_len = int(self.duration_min[stage_idx])
        max_len = int(self.duration_max[stage_idx])
        if duration < min_len or duration > max_len:
            return float("inf")
        return 0.0

    def _segment_feature_vector(self, demo_idx, stage_idx, s, e):
        x_s = self.demos[demo_idx][s]
        x_e = self.demos[demo_idx][e]
        duration = float(e - s + 1)
        duration_norm = duration / max(float(len(self.demos[demo_idx])), 1.0)
        return np.concatenate(
            [
                x_e - x_s,
                x_e - self.end_mu[stage_idx],
                np.array([duration_norm], dtype=float),
            ],
            axis=0,
        )

    def synthetic_segment_feature_vector(self, stage_idx, x_s, x_e, duration=None):
        if duration is None:
            duration = self.duration_means[stage_idx] if len(self.duration_means) > stage_idx else 1.0
        x_s = np.asarray(x_s, dtype=float).reshape(-1)
        x_e = np.asarray(x_e, dtype=float).reshape(-1)
        duration_norm = float(duration) / max(float(np.mean([len(X) for X in self.demos])), 1.0)
        return np.concatenate(
            [
                x_e - x_s,
                x_e - self.end_mu[stage_idx],
                np.array([duration_norm], dtype=float),
            ],
            axis=0,
        )

    def synthetic_progress_prob(self, stage_idx, x_s, x_e, duration=None):
        if self.progress_term_type == "delta":
            cost = self._softplus_progress_delta(self._delta_progress_change(stage_idx, x_s, x_e))
            return float(np.exp(-cost))
        model = self.progress_rank_models[stage_idx]
        if model is None:
            return 0.5
        z = self.synthetic_segment_feature_vector(stage_idx, x_s, x_e, duration=duration)[None, :]
        return float(model.predict_prob(z)[0])

    def synthetic_progress_score(self, stage_idx, x_s, x_e, duration=None):
        if self.progress_term_type == "delta":
            return float(-self._softplus_progress_delta(self._delta_progress_change(stage_idx, x_s, x_e)))
        model = self.progress_rank_models[stage_idx]
        if model is None:
            return 0.0
        z = self.synthetic_segment_feature_vector(stage_idx, x_s, x_e, duration=duration)[None, :]
        return float(model.predict_score(z)[0])

    def _progress_prob(self, demo_idx, stage_idx, s, e):
        if self.progress_term_type == "delta":
            cost = self._progress_loss(demo_idx, stage_idx, s, e)
            return float(np.exp(-cost))
        model = self.progress_rank_models[stage_idx]
        if model is None:
            return 0.5
        z = self._segment_feature_vector(demo_idx, stage_idx, s, e)[None, :]
        return float(model.predict_prob(z)[0])

    def _progress_score(self, demo_idx, stage_idx, s, e):
        if self.progress_term_type == "delta":
            return float(-self._progress_loss(demo_idx, stage_idx, s, e))
        model = self.progress_rank_models[stage_idx]
        if model is None:
            return 0.0
        z = self._segment_feature_vector(demo_idx, stage_idx, s, e)[None, :]
        return float(model.predict_score(z)[0])

    def _progress_loss(self, demo_idx, stage_idx, s, e):
        if self.progress_term_type == "delta":
            return float(np.sum(self._segment_step_progress_costs(demo_idx, stage_idx, s, e)))
        score = self._progress_score(demo_idx, stage_idx, s, e)
        return float(np.log1p(np.exp(-np.clip(score, -40.0, 40.0))))

    # Backward-compatible aliases while external naming moves to "progress".
    def synthetic_gain_prob(self, stage_idx, x_s, x_e, duration=None):
        return self.synthetic_progress_prob(stage_idx, x_s, x_e, duration=duration)

    def synthetic_gain_score(self, stage_idx, x_s, x_e, duration=None):
        return self.synthetic_progress_score(stage_idx, x_s, x_e, duration=duration)

    def _delta_progress_change(self, stage_idx, x_s, x_e):
        center = np.asarray(self.end_mu[stage_idx], dtype=float)
        x_s = np.asarray(x_s, dtype=float)
        x_e = np.asarray(x_e, dtype=float)
        d_start = float(np.linalg.norm(x_s - center))
        d_end = float(np.linalg.norm(x_e - center))
        return d_end - d_start

    def _softplus_progress_delta(self, delta):
        scaled = self.progress_delta_scale * float(delta)
        return float(np.log1p(np.exp(np.clip(scaled, -40.0, 40.0))) / self.progress_delta_scale)

    def _step_progress_cost(self, stage_idx, x_t, x_next):
        delta = self._delta_progress_change(stage_idx, x_t, x_next)
        return self._softplus_progress_delta(delta)

    def _segment_step_progress_costs(self, demo_idx, stage_idx, s, e):
        if e < s:
            return np.zeros(0, dtype=float)
        X = self.demos[demo_idx]
        start_t = max(int(s) - 1, 0)
        end_t = min(int(e) - 1, len(X) - 2)
        if end_t < start_t:
            return np.zeros(0, dtype=float)
        return np.asarray(
            [self._step_progress_cost(stage_idx, X[t], X[t + 1]) for t in range(start_t, end_t + 1)],
            dtype=float,
        )

    def _segment_progress_change(self, demo_idx, stage_idx, s, e):
        return self._delta_progress_change(stage_idx, self.demos[demo_idx][s], self.demos[demo_idx][e])

    def _segment_cost_parts(self, demo_idx, stage_idx, s, e):
        duration = e - s + 1
        parts = {
            "constraint": self._constraint_loss(demo_idx, stage_idx, s, e),
            "end": self._end_loss(demo_idx, stage_idx, e),
            "progress": self._progress_loss(demo_idx, stage_idx, s, e),
            "duration": self._duration_loss(stage_idx, duration),
        }
        total = (
            self.lambda_constraint * parts["constraint"]
            + self.lambda_end * parts["end"]
            + self.lambda_progress * parts["progress"]
        )
        parts["total"] = float(total)
        return parts

    def _update_feature_models(self, stage_ends_per_demo):
        segments = [[] for _ in range(self.num_states)]
        for demo_idx, stage_ends in enumerate(stage_ends_per_demo):
            starts, ends = self._segment_bounds_from_stage_ends(stage_ends)
            for k, (s, e) in enumerate(zip(starts, ends)):
                segments[k].append(self.standardized_features[demo_idx][s : e + 1])
        for k in range(self.num_states):
            for m in range(self.num_features):
                xs = [seg[:, m] for seg in segments[k] if len(seg) > 0]
                if xs:
                    self.feature_models[k][m].m_step_update(xs)

        if self.auto_feature_select and self.fixed_feature_mask is None:
            global_mean = np.mean(np.concatenate(self.standardized_features, axis=0), axis=0)
            scores = np.zeros((self.num_states, self.num_features), dtype=float)
            for k in range(self.num_states):
                for m in range(self.num_features):
                    mu = getattr(self.feature_models[k][m], "mu", 0.0)
                    scores[k, m] = abs(float(mu) - float(global_mean[m]))
            self.r = (scores >= np.median(scores, axis=1, keepdims=True)).astype(int)
            self.r[self.r.sum(axis=1) == 0, :] = 1

    def _update_completion_region(self, stage_ends_per_demo):
        for k in range(self.num_states):
            endpoints = []
            for demo_idx, stage_ends in enumerate(stage_ends_per_demo):
                endpoints.append(self.demos[demo_idx][stage_ends[k]])
            E = np.asarray(endpoints, dtype=float)
            self.end_mu[k] = np.mean(E, axis=0)
            var = np.var(E, axis=0) + self.end_precision_floor
            self.end_precision[k] = 1.0 / np.maximum(var, self.end_precision_floor)
        if self.num_states >= 1:
            self.g1 = np.asarray(self.end_mu[0], dtype=float).copy()
            self.g1_hist.append(self.g1.copy())
        if self.num_states >= 2:
            self.g2 = np.asarray(self.end_mu[1], dtype=float).copy()
            self.g2_hist.append(self.g2.copy())

    def _update_duration_prior(self, stage_ends_per_demo):
        lengths = np.asarray(_segment_lengths(stage_ends_per_demo), dtype=float)
        self.duration_means = np.mean(lengths, axis=0)
        self.duration_stds = np.std(lengths, axis=0) + 1.0

        if self.duration_slack > 0:
            lower = np.maximum(np.floor(self.duration_means - self.duration_slack).astype(int), 1)
            upper = np.ceil(self.duration_means + self.duration_slack).astype(int)
            self.duration_min = np.maximum(self.duration_min, lower)
            self.duration_max = np.maximum(self.duration_min, np.minimum(self.duration_max, upper))

    def _sample_negative_segments(self, demo_idx, pos_s, pos_e, stage_idx):
        T = len(self.demos[demo_idx])
        samples = []
        radius = max(self.hard_negative_radius, 0)
        for ds in range(-radius, radius + 1):
            for de in range(-radius, radius + 1):
                if ds == 0 and de == 0:
                    continue
                s = int(np.clip(pos_s + ds, 0, T - 1))
                e = int(np.clip(pos_e + de, s, T - 1))
                if s == pos_s and e == pos_e:
                    continue
                if e - s + 1 < self.duration_min[stage_idx] or e - s + 1 > self.duration_max[stage_idx]:
                    continue
                samples.append((s, e))
        pos_len = pos_e - pos_s + 1
        for _ in range(self.random_negative_per_demo):
            jitter = self.rng.randint(-radius - 1, radius + 2)
            length = int(np.clip(pos_len + jitter, self.duration_min[stage_idx], min(self.duration_max[stage_idx], T)))
            s = int(self.rng.randint(0, max(T - length + 1, 1)))
            e = s + length - 1
            if s == pos_s and e == pos_e:
                continue
            samples.append((s, e))
        return samples

    def _update_progress_models(self, stage_ends_per_demo):
        if self.progress_term_type != "ranking":
            return
        for k in range(self.num_states):
            X_pos = []
            X_neg = []
            for demo_idx, stage_ends in enumerate(stage_ends_per_demo):
                starts, ends = self._segment_bounds_from_stage_ends(stage_ends)
                pos_s = starts[k]
                pos_e = ends[k]
                z_pos = self._segment_feature_vector(demo_idx, k, pos_s, pos_e)
                for s, e in self._sample_negative_segments(demo_idx, pos_s, pos_e, k):
                    X_pos.append(z_pos)
                    X_neg.append(self._segment_feature_vector(demo_idx, k, s, e))
                for other_k, (s, e) in enumerate(zip(starts, ends)):
                    if other_k == k:
                        continue
                    X_pos.append(z_pos)
                    X_neg.append(self._segment_feature_vector(demo_idx, k, s, e))
            X_pos = np.asarray(X_pos, dtype=float)
            X_neg = np.asarray(X_neg, dtype=float)
            if len(X_pos) == 0 or len(X_neg) == 0:
                continue
            if self.progress_rank_models[k] is None or self.progress_rank_models[k].input_dim != X_pos.shape[1]:
                self.progress_rank_models[k] = StageGainMLP(X_pos.shape[1], self.progress_rank_hidden_dim, self.rng)
            self.progress_rank_models[k].fit_ranking(
                X_pos,
                X_neg,
                steps=self.progress_rank_steps,
                lr=self.progress_rank_lr,
                weight_decay=self.progress_rank_weight_decay,
            )

    def _run_dp_for_demo(self, demo_idx):
        T = len(self.demos[demo_idx])
        K = self.num_states
        dp = np.full((K + 1, T + 1), np.inf, dtype=float)
        prev = np.full((K + 1, T + 1), -1, dtype=int)
        dp[0, 0] = 0.0
        for k in range(1, K + 1):
            min_len = int(self.duration_min[k - 1])
            max_len = int(min(self.duration_max[k - 1], T))
            for t in range(1, T + 1):
                s_min = max((k - 1), t - max_len)
                s_max = t - min_len
                if s_max < s_min:
                    continue
                best = np.inf
                arg = -1
                for s in range(s_min, s_max + 1):
                    if not np.isfinite(dp[k - 1, s]):
                        continue
                    parts = self._segment_cost_parts(demo_idx, k - 1, s, t - 1)
                    val = dp[k - 1, s] + parts["total"]
                    if val < best:
                        best = val
                        arg = s
                dp[k, t] = best
                prev[k, t] = arg

        if not np.isfinite(dp[K, T]):
            raise RuntimeError(
                f"CCP DP failed on demo {demo_idx}. Check duration bounds for T={T}, K={K}."
            )
        ends = []
        t = T
        for k in range(K, 0, -1):
            s = int(prev[k, t])
            ends.append(t - 1)
            t = s
        ends.reverse()
        return ends

    def _compute_total_objective(self, stage_ends_per_demo):
        totals = {"constraint": 0.0, "end": 0.0, "progress": 0.0, "duration": 0.0, "total": 0.0}
        for demo_idx, stage_ends in enumerate(stage_ends_per_demo):
            starts, ends = self._segment_bounds_from_stage_ends(stage_ends)
            for k, (s, e) in enumerate(zip(starts, ends)):
                parts = self._segment_cost_parts(demo_idx, k, s, e)
                for key in totals:
                    totals[key] += float(parts[key])
        return totals

    def _taus_hat_from_stage_ends(self, stage_ends_per_demo):
        if self.num_states != 2:
            return None
        return [int(stage_ends[0]) for stage_ends in stage_ends_per_demo]

    def fit(self, max_iter=30, verbose=True):
        stage_ends_per_demo = self._initialize_stage_ends()
        self.segmentation_history = [[list(item) for item in stage_ends_per_demo]]
        for it in range(int(max_iter)):
            prev_stage_ends = [list(item) for item in stage_ends_per_demo]
            self._update_feature_models(stage_ends_per_demo)
            self._update_completion_region(stage_ends_per_demo)
            self._update_duration_prior(stage_ends_per_demo)
            self._update_progress_models(stage_ends_per_demo)
            stage_ends_per_demo = [self._run_dp_for_demo(i) for i in range(len(self.demos))]
            self.segmentation_history.append([list(item) for item in stage_ends_per_demo])
            self.stage_ends_ = [list(item) for item in stage_ends_per_demo]

            totals = self._compute_total_objective(stage_ends_per_demo)
            self.loss_total.append(totals["total"])
            self.loss_constraint.append(totals["constraint"])
            self.loss_end.append(totals["end"])
            self.loss_progress.append(totals["progress"])

            gammas = _hard_gammas_from_stage_ends([len(X) for X in self.demos], stage_ends_per_demo, self.num_states)
            metrics = self.eval_fn(self, gammas, None) if self.eval_fn is not None else {}
            for name, value in metrics.items():
                self.metrics_hist.setdefault(name, []).append(value)
            should_log = ((it + 1) % 10 == 0) or (it == int(max_iter) - 1)
            if verbose and should_log:
                taus_hat = self._taus_hat_from_stage_ends(stage_ends_per_demo)
                print(
                    format_training_log(
                        "CCP",
                        it,
                        losses={
                            "loss": totals["total"],
                            "constraint": totals["constraint"],
                            "end": totals["end"],
                            "progress": totals["progress"],
                        },
                        metrics=metrics,
                        extras={"taus": taus_hat if taus_hat is not None else "multi-stage"},
                    )
                )
            should_plot = self.plot_every is not None and (
                ((it + 1) % int(self.plot_every) == 0) or (it == int(max_iter) - 1)
            )
            if should_plot:
                plot_ccp_results_4panel(self, it + 1)
                plot_ccp_progress_heatmaps(self, it + 1)
                plot_ccp_progress_boundary_profile(self, it + 1, demo_idx=0)

        self.stage_ends_ = [list(item) for item in stage_ends_per_demo]
        return _hard_gammas_from_stage_ends([len(X) for X in self.demos], stage_ends_per_demo, self.num_states)
