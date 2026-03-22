from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from evaluation import eval_goalhmm_auto
from methods.base import format_training_log
from utils.models import GaussianModel, MarginExpLowerEmission, StudentTModel, ZeroMeanGaussianModel
from visualization.scdp_4panel import plt as scdp_plot_plt, plot_scdp_results_4panel, plot_scdp_results_4panel_overview
from visualization.scdp_activation import plot_scdp_activation_masks


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


def _hard_gammas_from_stage_ends(lengths: Sequence[int], stage_ends_per_demo: Sequence[Sequence[int]], num_states: int):
    gammas: List[np.ndarray] = []
    for T, stage_ends in zip(lengths, stage_ends_per_demo):
        gamma = np.zeros((int(T), int(num_states)), dtype=float)
        start = 0
        for k, end in enumerate(stage_ends):
            gamma[start : end + 1, k] = 1.0
            start = int(end) + 1
        gammas.append(gamma)
    return gammas


def _normalize_true_cutpoints(
    demos: Sequence[np.ndarray],
    true_taus: Sequence[int] | None = None,
    true_cutpoints: Sequence[Sequence[int]] | None = None,
):
    normalized = []
    if true_cutpoints is not None:
        for X, cuts in zip(demos, true_cutpoints):
            if cuts is None:
                normalized.append(None)
                continue
            arr = np.asarray(cuts, dtype=int).reshape(-1)
            if arr.size == 0:
                normalized.append(np.zeros(0, dtype=int))
                continue
            arr = np.sort(arr)
            arr = arr[(arr >= 0) & (arr < len(X) - 1)]
            normalized.append(arr.astype(int))
        return normalized
    if true_taus is not None:
        for tau in true_taus:
            if tau is None:
                normalized.append(None)
            else:
                normalized.append(np.asarray([int(tau)], dtype=int))
        return normalized
    return [None for _ in demos]


def _mean_abs_centered_dispersion(values) -> float:
    xs = np.asarray(values, dtype=float).reshape(-1)
    if xs.size == 0:
        return np.nan
    center = float(np.median(xs))
    return float(np.mean(np.abs(xs - center)))


@dataclass
class _StageParams:
    model_summaries: List[dict]
    subgoal: np.ndarray
    active_mask: np.ndarray | None = None
    feature_scores: np.ndarray | None = None
    feature_constraint_costs: np.ndarray | None = None


class SegmentConsensusDPModel:
    def __init__(
        self,
        demos,
        env,
        true_taus=None,
        true_cutpoints=None,
        n_states=2,
        seed=0,
        selected_raw_feature_ids=None,
        feature_model_types=None,
        fixed_feature_mask=None,
        lambda_eq_constraint=1.0,
        lambda_ineq_constraint=1.0,
        lambda_progress=1.0,
        lambda_subgoal_consensus=1.0,
        lambda_param_consensus=1.0,
        lambda_feature_score_consensus=1.0,
        consensus_schedule="linear",
        progress_delta_scale=20.0,
        duration_min=None,
        duration_max=None,
        sigma_floor=0.1,
        lam_floor=0.1,
        feature_activation_mode="fixed_mask",
        equality_dispersion_ratio_threshold=0.1,
        equality_dispersion_uncertainty_c=0.1,
        inequality_score_activation_threshold=-0.5,
        fixed_true_cutpoint_prefix=0,
        plot_every=None,
        plot_dir="outputs/plots",
        eval_fn=eval_goalhmm_auto,
    ):
        self.demos = [np.asarray(X, dtype=float) for X in demos]
        self.env = env
        self.true_cutpoints = _normalize_true_cutpoints(self.demos, true_taus=true_taus, true_cutpoints=true_cutpoints)
        self.true_taus = [
            None if cuts is None or len(cuts) != 1 else int(cuts[0])
            for cuts in self.true_cutpoints
        ]
        self.num_states = int(n_states)
        if self.num_states < 2:
            raise ValueError("scdp requires at least 2 stages.")
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.selected_raw_feature_ids = selected_raw_feature_ids
        self.feature_model_types_raw = feature_model_types
        self.fixed_feature_mask = fixed_feature_mask
        self.lambda_eq_constraint = float(lambda_eq_constraint)
        self.lambda_ineq_constraint = float(lambda_ineq_constraint)
        self.lambda_progress = float(lambda_progress)
        self.lambda_subgoal_consensus = float(lambda_subgoal_consensus)
        self.lambda_param_consensus = float(lambda_param_consensus)
        self.lambda_feature_score_consensus = float(lambda_feature_score_consensus)
        self.consensus_schedule = str(consensus_schedule)
        self.progress_delta_scale = max(float(progress_delta_scale), 1e-6)
        self.sigma_floor = max(float(sigma_floor), 1e-6)
        self.lam_floor = max(float(lam_floor), 1e-6)
        feature_activation_mode = str(feature_activation_mode).lower()
        if feature_activation_mode not in {"fixed_mask", "score"}:
            raise ValueError("feature_activation_mode must be one of {'fixed_mask', 'score'}.")
        self.feature_activation_mode = feature_activation_mode
        self.use_score_mode = self.feature_activation_mode == "score"
        self.equality_dispersion_ratio_threshold = float(equality_dispersion_ratio_threshold)
        self.equality_dispersion_uncertainty_c = float(equality_dispersion_uncertainty_c)
        self.inequality_score_activation_threshold = float(inequality_score_activation_threshold)
        self.fixed_true_cutpoint_prefix = max(int(fixed_true_cutpoint_prefix), 0)
        self.plot_every = plot_every
        self.plot_dir = plot_dir
        self.eval_fn = eval_fn
        if self.plot_every is not None and scdp_plot_plt is None:
            print("[SCDP] matplotlib is not installed; SCDP 4-panel plots will not be generated.")

        self._init_feature_preprocessing()
        self.feature_model_types = self._normalize_feature_model_types(self.num_features)
        self.r = np.ones((self.num_states, self.num_features), dtype=int)
        if fixed_feature_mask is not None:
            mask = np.asarray(fixed_feature_mask, dtype=int)
            if mask.shape != self.r.shape:
                raise ValueError(f"fixed_feature_mask must have shape {self.r.shape}.")
            self.r = mask.copy()

        self.duration_min = self._broadcast_stage_value(duration_min, default=1, dtype=int)
        self.duration_max = self._broadcast_stage_value(duration_max, default=None, dtype=int)
        total_max = int(max(len(X) for X in self.demos))
        for k in range(self.num_states):
            if self.duration_max[k] is None:
                self.duration_max[k] = total_max
        self.duration_max = np.asarray(self.duration_max, dtype=int)

        self.loss_total: List[float] = []
        self.loss_constraint: List[float] = []
        self.loss_progress: List[float] = []
        self.loss_subgoal_consensus: List[float] = []
        self.loss_param_consensus: List[float] = []
        self.loss_feature_score_consensus: List[float] = []
        self.metrics_hist: Dict[str, List[float]] = {}
        self.segmentation_history: List[List[List[int]]] = []
        self.subgoal_consensus_lambda_hist: List[float] = []
        self.param_consensus_lambda_hist: List[float] = []
        self.feature_score_consensus_lambda_hist: List[float] = []
        self.current_subgoal_consensus_lambda = 0.0
        self.current_param_consensus_lambda = 0.0
        self.current_feature_score_consensus_lambda = 0.0
        self.current_stage_params_per_demo: List[List[_StageParams]] = []
        self.demo_r_matrices_: List[np.ndarray] = []
        self.current_demo_cost_breakdown: List[Dict[str, float]] = []
        self.stage_subgoals: List[np.ndarray] = [np.full(self.state_dim, np.nan, dtype=float) for _ in range(self.num_states)]
        self.stage_subgoals_hist: List[List[np.ndarray]] = []
        self.g1 = np.full(self.state_dim, np.nan, dtype=float)
        self.g2 = np.full(self.state_dim, np.nan, dtype=float)
        self.g1_hist: List[np.ndarray] = []
        self.g2_hist: List[np.ndarray] = []
        self.stage_ends_: List[List[int]] = []
        self.feature_models = self._build_feature_model_grid()
        self.shared_param_vectors: List[List[np.ndarray | None]] = [[None for _ in range(self.num_features)] for _ in range(self.num_states)]
        self.shared_stage_subgoals: List[np.ndarray] = [np.zeros(self.state_dim, dtype=float) for _ in range(self.num_states)]
        self.shared_r_mean: np.ndarray | None = None
        self.shared_feature_score_mean: np.ndarray | None = None
        self.demo_feature_score_matrices_: List[np.ndarray] = []
        self.posthoc_activation_summary_: Dict[str, object] | None = None

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

    def _init_feature_preprocessing(self):
        if self.env is None:
            raise ValueError("scdp requires a dataset env with feature API.")
        raw_features = [np.asarray(self.env.compute_all_features_matrix(X), dtype=float) for X in self.demos]
        if not raw_features:
            raise ValueError("scdp requires at least one demo.")
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
        for F_raw in raw_features:
            F_sel = F_raw[:, self.selected_feature_columns]
            mean_sel = self.feat_mean[self.selected_feature_columns]
            std_sel = self.feat_std[self.selected_feature_columns]
            self.standardized_features.append((F_sel - mean_sel[None, :]) / std_sel[None, :])
        self.state_dim = int(self.demos[0].shape[1])

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

    def _broadcast_stage_value(self, value, default, dtype=float):
        if value is None:
            return [default for _ in range(self.num_states)]
        if np.isscalar(value):
            return [dtype(value) for _ in range(self.num_states)]
        value = list(value)
        if len(value) != self.num_states:
            raise ValueError(f"Expected {self.num_states} stage values, got {len(value)}.")
        return [default if v is None else dtype(v) for v in value]

    def _build_feature_model_grid(self):
        rows = []
        for _ in range(self.num_states):
            cur = []
            for kind in self.feature_model_types:
                cur.append(self._make_model_from_kind(kind))
            rows.append(cur)
        return rows

    def _make_model_from_kind(self, kind):
        kind = str(kind).lower()
        if kind in {"gauss", "gaussian"}:
            return GaussianModel(mu=0.0, sigma=1.0)
        if kind in {"student_t", "studentt", "t"}:
            return StudentTModel(mu=0.0, sigma=1.0)
        if kind in {"zero_gauss", "zero_gaussian"}:
            return ZeroMeanGaussianModel(sigma=1.0)
        if kind in {"margin_exp_lower", "marginexp", "margin_exp"}:
            return MarginExpLowerEmission(b_init=0.0, lam_init=max(self.lam_floor, 1.0))
        raise ValueError(f"Unsupported feature model type '{kind}'.")

    def _fit_local_model(self, kind, xs):
        xs = np.asarray(xs, dtype=float).reshape(-1)
        kind = str(kind).lower()
        if kind in {"gauss", "gaussian"}:
            mu = float(np.mean(xs))
            sigma = float(max(np.std(xs), self.sigma_floor))
            return GaussianModel(mu=mu, sigma=sigma)
        if kind in {"student_t", "studentt", "t"}:
            model = StudentTModel(mu=0.0, sigma=max(self.sigma_floor, 1.0))
            model.m_step_update([xs])
            model.sigma = max(float(model.sigma), self.sigma_floor)
            model._update_interval()
            return model
        if kind in {"zero_gauss", "zero_gaussian"}:
            sigma = float(max(np.sqrt(np.mean(xs * xs)), self.sigma_floor))
            return ZeroMeanGaussianModel(sigma=sigma)
        if kind in {"margin_exp_lower", "marginexp", "margin_exp"}:
            model = MarginExpLowerEmission(b_init=0.0, lam_init=max(self.lam_floor, 1.0))
            model.m_step_update([xs])
            model.lam = max(float(model.lam), self.lam_floor)
            model._update_interval()
            return model
        raise ValueError(f"Unsupported feature model type '{kind}'.")

    def _segment_bounds_from_stage_ends(self, stage_ends):
        bounds = []
        start = 0
        for end in stage_ends:
            end = int(end)
            bounds.append((start, end))
            start = end + 1
        return bounds

    def _stage_ends_from_cutpoints(self, cutpoints, T):
        cutpoints = np.asarray(cutpoints, dtype=int).reshape(-1)
        return [int(x) for x in cutpoints.tolist()] + [int(T - 1)]

    def _progress_cost(self, X, s, e, subgoal):
        if e <= s:
            return 0.0
        subgoal = np.asarray(subgoal, dtype=float)
        total = 0.0
        start_t = max(int(s), 0)
        end_t = min(int(e) - 1, len(X) - 2)
        for t in range(start_t, end_t + 1):
            d0 = float(np.linalg.norm(X[t] - subgoal))
            d1 = float(np.linalg.norm(X[t + 1] - subgoal))
            delta = d1 - d0
            total += float(np.log1p(np.exp(np.clip(self.progress_delta_scale * delta, -60.0, 60.0))) / self.progress_delta_scale)
        return total

    def _fit_segment_stage(self, demo_idx, stage_idx, s, e):
        F = self.standardized_features[demo_idx][s : e + 1]
        F_demo = self.standardized_features[demo_idx]
        summaries = []
        active_mask = np.zeros(self.num_features, dtype=int)
        feature_scores = np.zeros(self.num_features, dtype=float)
        feature_constraint_costs = np.zeros(self.num_features, dtype=float)
        active_fit_losses = [None for _ in range(self.num_features)]
        for feat_idx, kind in enumerate(self.feature_model_types):
            model = self._fit_local_model(kind, F[:, feat_idx])
            summary = model.get_summary()
            summaries.append(summary)
            fitted_loss = -np.asarray(model.logpdf(F[:, feat_idx]), dtype=float)
            fitted_step = float(np.mean(fitted_loss))
            baseline_model = GaussianModel(
                mu=float(np.mean(F[:, feat_idx])),
                sigma=float(max(np.std(F[:, feat_idx]), 1e-6)),
            )
            baseline_loss = -np.asarray(baseline_model.logpdf(F[:, feat_idx]), dtype=float)
            baseline_step = float(np.mean(baseline_loss))
            ll_gain = baseline_step - fitted_step
            if self.use_score_mode:
                kind_l = str(kind).lower()
                if kind_l in {"gauss", "gaussian", "student_t", "studentt", "t", "zero_gauss", "zero_gaussian"}:
                    stage_dispersion = float(_mean_abs_centered_dispersion(F[:, feat_idx]))
                    uncertainty_bonus = float(self.equality_dispersion_uncertainty_c / np.sqrt(max(len(F[:, feat_idx]), 1)))
                    score = stage_dispersion + uncertainty_bonus
                else:
                    score = -ll_gain
                feature_scores[feat_idx] = float(score)
                weight = self.lambda_eq_constraint if self._is_equality_feature(feat_idx) else self.lambda_ineq_constraint
                if self._is_equality_feature(feat_idx):
                    avg_step_cost = min(float(score) - float(self.equality_dispersion_ratio_threshold), 0.0)
                    feature_constraint_costs[feat_idx] = float(weight * len(F[:, feat_idx]) * avg_step_cost)
                else:
                    avg_step_cost = min(float(score) - float(self.inequality_score_activation_threshold), 0.0)
                    feature_constraint_costs[feat_idx] = float(weight * len(F[:, feat_idx]) * avg_step_cost)
            else:
                if self.r[stage_idx, feat_idx]:
                    feature_scores[feat_idx] = -ll_gain
                    active_mask[feat_idx] = 1
                    active_fit_losses[feat_idx] = fitted_loss
        if not self.use_score_mode:
            n_active = int(np.sum(active_mask))
            if n_active > 0:
                for feat_idx, kind in enumerate(self.feature_model_types):
                    if not active_mask[feat_idx]:
                        continue
                    weight = self.lambda_eq_constraint if self._is_equality_feature(feat_idx) else self.lambda_ineq_constraint
                    fitted_loss = np.asarray(active_fit_losses[feat_idx], dtype=float)
                    feature_constraint_costs[feat_idx] = float(weight * np.sum(fitted_loss) / n_active)
        constraint_cost = float(np.sum(feature_constraint_costs))
        subgoal = np.asarray(self.demos[demo_idx][e], dtype=float)
        progress_cost = self._progress_cost(self.demos[demo_idx], s, e, subgoal)
        return _StageParams(
            model_summaries=summaries,
            subgoal=subgoal,
            active_mask=active_mask,
            feature_scores=feature_scores,
            feature_constraint_costs=feature_constraint_costs,
        ), constraint_cost, progress_cost

    def _is_equality_feature(self, feat_idx: int) -> bool:
        kind_l = str(self.feature_model_types[feat_idx]).lower()
        return kind_l in {"gauss", "gaussian", "student_t", "studentt", "t", "zero_gauss", "zero_gaussian"}

    def _compute_posthoc_activation_summary(self):
        if not self.demo_feature_score_matrices_:
            return None

        scores = np.asarray(self.demo_feature_score_matrices_, dtype=float)
        thresholds = np.zeros((self.num_states, self.num_features), dtype=float)
        for feat_idx in range(self.num_features):
            thr = self.equality_dispersion_ratio_threshold if self._is_equality_feature(feat_idx) else self.inequality_score_activation_threshold
            thresholds[:, feat_idx] = float(thr)

        activated = scores < thresholds[None, :, :]
        activation_rate = np.mean(activated.astype(float), axis=0)
        feature_names = []
        for local_idx in range(self.num_features):
            selected_col = int(self.selected_feature_columns[local_idx])
            name = None
            for i, spec in enumerate(self.raw_feature_specs):
                if int(spec.get("column_idx", i)) == selected_col:
                    name = str(spec.get("name", f"f{local_idx}"))
                    break
            feature_names.append(name or f"f{local_idx}")

        by_stage = []
        for stage_idx in range(self.num_states):
            stage_items = []
            for feat_idx in range(self.num_features):
                stage_items.append(
                    {
                        "feature_idx": int(feat_idx),
                        "feature_name": feature_names[feat_idx],
                        "score_type": "local_dispersion" if self._is_equality_feature(feat_idx) else "-ll_gain",
                        "threshold": float(thresholds[stage_idx, feat_idx]),
                        "activation_rate": float(activation_rate[stage_idx, feat_idx]),
                        "mean_score": float(np.mean(scores[:, stage_idx, feat_idx])),
                        "median_score": float(np.median(scores[:, stage_idx, feat_idx])),
                        "activated_demo_indices": [
                            int(i) for i in np.where(activated[:, stage_idx, feat_idx])[0].tolist()
                        ],
                    }
                )
            by_stage.append(stage_items)

        return {
            "thresholds": thresholds.tolist(),
            "activation_rate_matrix": activation_rate.tolist(),
            "activated_mask_per_demo": activated.astype(int).tolist(),
            "by_stage": by_stage,
        }

    def _summary_to_vector(self, kind, summary):
        kind = str(kind).lower()
        if kind in {"gauss", "gaussian"}:
            return np.asarray([float(summary["mu"]), float(np.log(max(float(summary["sigma"]), self.sigma_floor)))], dtype=float)
        if kind in {"student_t", "studentt", "t"}:
            return np.asarray([float(summary["mu"]), float(np.log(max(float(summary["sigma"]), self.sigma_floor)))], dtype=float)
        if kind in {"zero_gauss", "zero_gaussian"}:
            return np.asarray([float(np.log(max(float(summary["sigma"]), self.sigma_floor)))], dtype=float)
        if kind in {"margin_exp_lower", "marginexp", "margin_exp"}:
            return np.asarray([float(summary["b"]), float(np.log(max(float(summary["lam"]), self.lam_floor)))], dtype=float)
        raise ValueError(f"Unsupported feature model type '{kind}'.")

    def _vector_to_model(self, kind, vec):
        kind = str(kind).lower()
        vec = np.asarray(vec, dtype=float).reshape(-1)
        if kind in {"gauss", "gaussian"}:
            return GaussianModel(mu=float(vec[0]), sigma=float(np.exp(vec[1])))
        if kind in {"student_t", "studentt", "t"}:
            return StudentTModel(mu=float(vec[0]), sigma=float(np.exp(vec[1])))
        if kind in {"zero_gauss", "zero_gaussian"}:
            return ZeroMeanGaussianModel(sigma=float(np.exp(vec[0])))
        if kind in {"margin_exp_lower", "marginexp", "margin_exp"}:
            return MarginExpLowerEmission(b_init=float(vec[0]), lam_init=float(np.exp(vec[1])))
        raise ValueError(f"Unsupported feature model type '{kind}'.")

    def _subgoal_consensus_cost(self, candidate_stage_params, shared_stage_subgoals):
        total = 0.0
        for stage_idx in range(self.num_states):
            diff = np.asarray(candidate_stage_params[stage_idx].subgoal, dtype=float) - np.asarray(shared_stage_subgoals[stage_idx], dtype=float)
            total += float(np.dot(diff, diff))
        return total

    def _param_consensus_cost(self, candidate_stage_params, shared_param_vectors):
        if self.use_score_mode:
            return 0.0
        total = 0.0
        for stage_idx in range(self.num_states):
            for feat_idx, kind in enumerate(self.feature_model_types):
                if not self.r[stage_idx, feat_idx]:
                    continue
                local_vec = self._summary_to_vector(kind, candidate_stage_params[stage_idx].model_summaries[feat_idx])
                shared_vec = shared_param_vectors[stage_idx][feat_idx]
                if shared_vec is None:
                    continue
                delta = local_vec - shared_vec
                total += float(np.dot(delta, delta))
        return total

    def _r_consensus_cost(self, candidate_stage_params, shared_r_mean):
        if shared_r_mean is None:
            return 0.0
        total = 0.0
        for stage_idx in range(self.num_states):
            active_mask = candidate_stage_params[stage_idx].active_mask
            if active_mask is None:
                continue
            delta = np.asarray(active_mask, dtype=float) - np.asarray(shared_r_mean[stage_idx], dtype=float)
            total += float(np.dot(delta, delta))
        return total

    def _feature_score_consensus_cost(self, candidate_stage_params, shared_feature_score_mean):
        if not self.use_score_mode or shared_feature_score_mean is None:
            return 0.0
        total = 0.0
        for stage_idx in range(self.num_states):
            scores = candidate_stage_params[stage_idx].feature_scores
            if scores is None:
                continue
            delta = np.asarray(scores, dtype=float) - np.asarray(shared_feature_score_mean[stage_idx], dtype=float)
            total += float(np.dot(delta, delta))
        return total

    def _segment_stage_cost_info(
        self,
        demo_idx,
        stage_idx,
        s,
        e,
        lam_subgoal_consensus,
        lam_param_consensus,
        lam_feature_score_consensus,
        shared_stage_subgoals,
        shared_param_vectors,
        shared_r_mean,
        shared_feature_score_mean=None,
    ):
        stage_len = int(e - s + 1)
        if stage_len < int(self.duration_min[stage_idx]) or stage_len > int(self.duration_max[stage_idx]):
            return None
        stage_params, constraint_cost, progress_cost = self._fit_segment_stage(demo_idx, stage_idx, s, e)
        subgoal_consensus_cost = 0.0
        if lam_subgoal_consensus > 0.0:
            diff = np.asarray(stage_params.subgoal, dtype=float) - np.asarray(shared_stage_subgoals[stage_idx], dtype=float)
            subgoal_consensus_cost = float(np.dot(diff, diff))
        param_consensus_cost = 0.0
        if lam_param_consensus > 0.0 and not self.use_score_mode:
            for feat_idx, kind in enumerate(self.feature_model_types):
                if not self.r[stage_idx, feat_idx]:
                    continue
                shared_vec = shared_param_vectors[stage_idx][feat_idx]
                if shared_vec is None:
                    continue
                local_vec = self._summary_to_vector(kind, stage_params.model_summaries[feat_idx])
                delta = local_vec - shared_vec
                param_consensus_cost += float(np.dot(delta, delta))
        feature_score_consensus_cost = 0.0
        if lam_feature_score_consensus > 0.0:
            if self.use_score_mode:
                scores = stage_params.feature_scores
                if scores is not None and shared_feature_score_mean is not None:
                    delta = np.asarray(scores, dtype=float) - np.asarray(shared_feature_score_mean[stage_idx], dtype=float)
                    feature_score_consensus_cost = float(np.dot(delta, delta))
            else:
                active_mask = stage_params.active_mask
                if active_mask is not None and shared_r_mean is not None:
                    delta = np.asarray(active_mask, dtype=float) - np.asarray(shared_r_mean[stage_idx], dtype=float)
                    feature_score_consensus_cost = float(np.dot(delta, delta))
        weighted_total = (
            float(constraint_cost)
            + self.lambda_progress * float(progress_cost)
            + lam_subgoal_consensus * float(subgoal_consensus_cost)
            + lam_param_consensus * float(param_consensus_cost)
            + lam_feature_score_consensus * float(feature_score_consensus_cost)
        )
        return {
            "stage_idx": int(stage_idx),
            "s": int(s),
            "e": int(e),
            "stage_params": stage_params,
            "constraint": float(constraint_cost),
            "progress": float(progress_cost),
            "subgoal_consensus": float(subgoal_consensus_cost),
            "param_consensus": float(param_consensus_cost),
            "feature_score_consensus": float(feature_score_consensus_cost),
            "weighted_total": float(weighted_total),
        }

    def _best_segmentation_info(
        self,
        demo_idx,
        lam_subgoal_consensus,
        lam_param_consensus,
        lam_feature_score_consensus,
        shared_stage_subgoals,
        shared_param_vectors,
        shared_r_mean,
        shared_feature_score_mean=None,
        fixed_cutpoints=None,
    ):
        X = self.demos[demo_idx]
        T = len(X)
        fixed_cutpoints_arr = np.asarray([], dtype=int) if fixed_cutpoints is None else np.asarray(fixed_cutpoints, dtype=int).reshape(-1)
        if fixed_cutpoints_arr.size > 0:
            fixed_cutpoints_arr = np.sort(fixed_cutpoints_arr)
            fixed_cutpoints_arr = fixed_cutpoints_arr[(fixed_cutpoints_arr >= 0) & (fixed_cutpoints_arr < T - 1)]
        n_fixed = min(int(fixed_cutpoints_arr.size), self.num_states - 1)
        fixed_cutpoints_arr = fixed_cutpoints_arr[:n_fixed]
        suffix_min = np.zeros(self.num_states + 1, dtype=int)
        suffix_max = np.zeros(self.num_states + 1, dtype=int)
        for k in range(self.num_states - 1, -1, -1):
            suffix_min[k] = suffix_min[k + 1] + int(self.duration_min[k])
            suffix_max[k] = suffix_max[k + 1] + int(self.duration_max[k])

        cache = {}

        def seg_info(stage_idx, s, e):
            key = (int(stage_idx), int(s), int(e))
            if key not in cache:
                cache[key] = self._segment_stage_cost_info(
                    demo_idx=demo_idx,
                    stage_idx=stage_idx,
                    s=s,
                    e=e,
                    lam_subgoal_consensus=lam_subgoal_consensus,
                    lam_param_consensus=lam_param_consensus,
                    lam_feature_score_consensus=lam_feature_score_consensus,
                    shared_stage_subgoals=shared_stage_subgoals,
                    shared_param_vectors=shared_param_vectors,
                    shared_r_mean=shared_r_mean,
                    shared_feature_score_mean=shared_feature_score_mean,
                )
            return cache[key]

        inf = float("inf")
        best = np.full((self.num_states, T), inf, dtype=float)
        back = np.full((self.num_states, T), -1, dtype=int)

        for stage_idx in range(self.num_states):
            for e in range(T):
                if stage_idx < n_fixed and int(e) != int(fixed_cutpoints_arr[stage_idx]):
                    continue
                remaining_after = int(T - e - 1)
                if remaining_after < int(suffix_min[stage_idx + 1]) or remaining_after > int(suffix_max[stage_idx + 1]):
                    continue
                if stage_idx == 0:
                    s = 0
                    info = seg_info(stage_idx, s, e)
                    if info is not None:
                        best[stage_idx, e] = float(info["weighted_total"])
                    continue

                for prev_end in range(stage_idx - 1, e):
                    prev_total = float(best[stage_idx - 1, prev_end])
                    if not np.isfinite(prev_total):
                        continue
                    s = int(prev_end + 1)
                    info = seg_info(stage_idx, s, e)
                    if info is None:
                        continue
                    total = prev_total + float(info["weighted_total"])
                    if total < float(best[stage_idx, e]):
                        best[stage_idx, e] = total
                        back[stage_idx, e] = int(prev_end)

        final_end = T - 1
        if not np.isfinite(best[self.num_states - 1, final_end]):
            raise RuntimeError(f"No feasible segmentation found for demo {demo_idx}.")

        stage_ends = [final_end]
        cur_end = final_end
        for stage_idx in range(self.num_states - 1, 0, -1):
            prev_end = int(back[stage_idx, cur_end])
            if prev_end < 0:
                raise RuntimeError(f"Broken DP backpointer for demo {demo_idx}, stage {stage_idx}.")
            stage_ends.append(prev_end)
            cur_end = prev_end
        stage_ends = sorted(int(x) for x in stage_ends)
        bounds = self._segment_bounds_from_stage_ends(stage_ends)
        stage_infos = [seg_info(stage_idx, s, e) for stage_idx, (s, e) in enumerate(bounds)]
        return {
            "cutpoints": [int(x) for x in stage_ends[:-1]],
            "stage_ends": [int(x) for x in stage_ends],
            "stage_params": [info["stage_params"] for info in stage_infos],
            "constraint": float(sum(info["constraint"] for info in stage_infos)),
            "progress": float(sum(info["progress"] for info in stage_infos)),
            "subgoal_consensus": float(sum(info["subgoal_consensus"] for info in stage_infos)),
            "param_consensus": float(sum(info["param_consensus"] for info in stage_infos)),
            "feature_score_consensus": float(sum(info["feature_score_consensus"] for info in stage_infos)),
            "total": float(best[self.num_states - 1, final_end]),
        }

    def _candidate_cost(
        self,
        demo_idx,
        stage_ends,
        lam_subgoal_consensus,
        lam_param_consensus,
        lam_feature_score_consensus,
        shared_stage_subgoals,
        shared_param_vectors,
        shared_r_mean,
        shared_feature_score_mean=None,
    ):
        bounds = self._segment_bounds_from_stage_ends(stage_ends)
        candidate_stage_params = []
        constraint_cost = 0.0
        progress_cost = 0.0
        for stage_idx, (s, e) in enumerate(bounds):
            stage_len = int(e - s + 1)
            if stage_len < int(self.duration_min[stage_idx]) or stage_len > int(self.duration_max[stage_idx]):
                return None
            stage_params, c_cost, p_cost = self._fit_segment_stage(demo_idx, stage_idx, s, e)
            candidate_stage_params.append(stage_params)
            constraint_cost += c_cost
            progress_cost += p_cost
        subgoal_consensus_cost = 0.0
        if lam_subgoal_consensus > 0.0:
            subgoal_consensus_cost = self._subgoal_consensus_cost(candidate_stage_params, shared_stage_subgoals)
        param_consensus_cost = 0.0
        if lam_param_consensus > 0.0:
            param_consensus_cost = self._param_consensus_cost(candidate_stage_params, shared_param_vectors)
        feature_score_consensus_cost = 0.0
        if lam_feature_score_consensus > 0.0:
            if self.use_score_mode:
                feature_score_consensus_cost = self._feature_score_consensus_cost(candidate_stage_params, shared_feature_score_mean)
            else:
                feature_score_consensus_cost = self._r_consensus_cost(candidate_stage_params, shared_r_mean)
        total = (
            constraint_cost
            + self.lambda_progress * progress_cost
            + lam_subgoal_consensus * subgoal_consensus_cost
            + lam_param_consensus * param_consensus_cost
            + lam_feature_score_consensus * feature_score_consensus_cost
        )
        return {
            "cutpoints": [int(x) for x in stage_ends[:-1]],
            "stage_ends": [int(x) for x in stage_ends],
            "stage_params": candidate_stage_params,
            "constraint": float(constraint_cost),
            "progress": float(progress_cost),
            "subgoal_consensus": float(subgoal_consensus_cost),
            "param_consensus": float(param_consensus_cost),
            "feature_score_consensus": float(feature_score_consensus_cost),
            "total": float(total),
        }

    def _shared_from_selected(self, selected_infos):
        shared_stage_subgoals = []
        for stage_idx in range(self.num_states):
            shared_stage_subgoals.append(
                np.mean([info["stage_params"][stage_idx].subgoal for info in selected_infos], axis=0)
            )
        shared_param_vectors = [[None for _ in range(self.num_features)] for _ in range(self.num_states)]
        if self.use_score_mode:
            return shared_stage_subgoals, shared_param_vectors
        for stage_idx in range(self.num_states):
            for feat_idx, kind in enumerate(self.feature_model_types):
                if not self.r[stage_idx, feat_idx]:
                    continue
                vectors = [
                    self._summary_to_vector(kind, info["stage_params"][stage_idx].model_summaries[feat_idx])
                    for info in selected_infos
                ]
                shared_param_vectors[stage_idx][feat_idx] = np.mean(np.stack(vectors, axis=0), axis=0)
        return shared_stage_subgoals, shared_param_vectors

    def _apply_shared_state(self, shared_stage_subgoals, shared_param_vectors):
        self.shared_stage_subgoals = [np.asarray(x, dtype=float).copy() for x in shared_stage_subgoals]
        self.shared_param_vectors = [[None if v is None else np.asarray(v, dtype=float).copy() for v in row] for row in shared_param_vectors]
        self.stage_subgoals = [np.asarray(x, dtype=float).copy() for x in shared_stage_subgoals]
        self.stage_subgoals_hist.append([x.copy() for x in self.stage_subgoals])
        if self.num_states >= 1:
            self.g1 = self.stage_subgoals[0].copy()
            self.g1_hist.append(self.g1.copy())
        if self.num_states >= 2:
            self.g2 = self.stage_subgoals[1].copy()
            self.g2_hist.append(self.g2.copy())
        for stage_idx in range(self.num_states):
            for feat_idx, kind in enumerate(self.feature_model_types):
                vec = shared_param_vectors[stage_idx][feat_idx]
                if vec is None:
                    continue
                self.feature_models[stage_idx][feat_idx] = self._vector_to_model(kind, vec)

    def _current_scheduled_lambda(self, base_lambda, iteration, max_iter):
        frac = 1.0 if max_iter <= 1 else float(iteration) / float(max_iter - 1)
        if self.consensus_schedule == "linear":
            return float(base_lambda) * frac
        if self.consensus_schedule == "quadratic":
            return float(base_lambda) * frac * frac
        raise ValueError(f"Unsupported consensus_schedule '{self.consensus_schedule}'.")

    def fit(self, max_iter=30, verbose=True):
        self.stage_ends_ = []
        self.shared_r_mean = None
        self.shared_feature_score_mean = None
        self.stage_subgoals_hist = []
        self.g1_hist = []
        self.g2_hist = []
        shared_stage_subgoals = [np.zeros(self.state_dim, dtype=float) for _ in range(self.num_states)]
        shared_param_vectors = [[None for _ in range(self.num_features)] for _ in range(self.num_states)]

        for iteration in range(int(max_iter)):
            lam_subgoal_consensus = self._current_scheduled_lambda(self.lambda_subgoal_consensus, iteration, int(max_iter))
            lam_param_consensus = self._current_scheduled_lambda(self.lambda_param_consensus, iteration, int(max_iter))
            lam_feature_score_consensus = self._current_scheduled_lambda(
                self.lambda_feature_score_consensus,
                iteration,
                int(max_iter),
            )
            self.subgoal_consensus_lambda_hist.append(float(lam_subgoal_consensus))
            self.param_consensus_lambda_hist.append(float(lam_param_consensus))
            self.feature_score_consensus_lambda_hist.append(float(lam_feature_score_consensus))
            self.current_subgoal_consensus_lambda = float(lam_subgoal_consensus)
            self.current_param_consensus_lambda = float(lam_param_consensus)
            self.current_feature_score_consensus_lambda = float(lam_feature_score_consensus)
            selected_infos = []
            for demo_idx in range(len(self.demos)):
                fixed_cutpoints = None
                if self.fixed_true_cutpoint_prefix > 0 and self.true_cutpoints[demo_idx] is not None:
                    fixed_cutpoints = self.true_cutpoints[demo_idx][: self.fixed_true_cutpoint_prefix]
                selected_infos.append(
                    self._best_segmentation_info(
                        demo_idx=demo_idx,
                        lam_subgoal_consensus=lam_subgoal_consensus,
                        lam_param_consensus=lam_param_consensus,
                        lam_feature_score_consensus=lam_feature_score_consensus,
                        shared_stage_subgoals=shared_stage_subgoals,
                        shared_param_vectors=shared_param_vectors,
                        shared_r_mean=self.shared_r_mean,
                        shared_feature_score_mean=self.shared_feature_score_mean,
                        fixed_cutpoints=fixed_cutpoints,
                    )
                )

            self.stage_ends_ = [list(info["stage_ends"]) for info in selected_infos]
            self.current_stage_params_per_demo = [list(info["stage_params"]) for info in selected_infos]
            self.current_demo_cost_breakdown = [
                {
                    "constraint": float(info["constraint"]),
                    "progress": self.lambda_progress * float(info["progress"]),
                    "subgoal_consensus": lam_subgoal_consensus * float(info["subgoal_consensus"]),
                    "param_consensus": lam_param_consensus * float(info["param_consensus"]),
                    "feature_score_consensus": lam_feature_score_consensus * float(info.get("feature_score_consensus", 0.0)),
                    "total": float(info["total"]),
                }
                for info in selected_infos
            ]
            self.demo_r_matrices_ = [
                np.stack([stage_params.active_mask for stage_params in info["stage_params"]], axis=0)
                for info in selected_infos
            ]
            self.demo_feature_score_matrices_ = [
                np.stack([stage_params.feature_scores for stage_params in info["stage_params"]], axis=0)
                for info in selected_infos
            ]
            if self.use_score_mode and self.demo_feature_score_matrices_:
                self.shared_feature_score_mean = np.mean(np.stack(self.demo_feature_score_matrices_, axis=0), axis=0)
                self.posthoc_activation_summary_ = self._compute_posthoc_activation_summary()
            self.segmentation_history.append([list(item) for item in self.stage_ends_])
            shared_stage_subgoals, shared_param_vectors = self._shared_from_selected(selected_infos)
            self._apply_shared_state(shared_stage_subgoals, shared_param_vectors)

            total_constraint = float(np.sum([info["constraint"] for info in selected_infos]))
            total_progress = float(np.sum([info["progress"] for info in selected_infos]))
            total_subgoal_consensus = float(np.sum([info["subgoal_consensus"] for info in selected_infos]))
            total_param_consensus = float(np.sum([info["param_consensus"] for info in selected_infos]))
            total_feature_score_consensus = float(np.sum([info.get("feature_score_consensus", 0.0) for info in selected_infos]))
            total_loss = float(
                total_constraint
                + self.lambda_progress * total_progress
                + lam_subgoal_consensus * total_subgoal_consensus
                + lam_param_consensus * total_param_consensus
                + lam_feature_score_consensus * total_feature_score_consensus
            )
            self.loss_constraint.append(total_constraint)
            self.loss_progress.append(total_progress)
            self.loss_subgoal_consensus.append(total_subgoal_consensus)
            self.loss_param_consensus.append(total_param_consensus)
            self.loss_feature_score_consensus.append(total_feature_score_consensus)
            self.loss_total.append(total_loss)

            gammas = _hard_gammas_from_stage_ends([len(X) for X in self.demos], self.stage_ends_, self.num_states)
            metrics = self.eval_fn(self, gammas, None) if self.eval_fn is not None and self.num_states == 2 else {}
            for name, value in metrics.items():
                if np.isscalar(value):
                    value_f = float(value)
                    if np.isfinite(value_f):
                        self.metrics_hist.setdefault(name, []).append(value_f)

            if verbose:
                print(
                    format_training_log(
                        "scdp",
                        iteration,
                        losses={
                            "total": total_loss,
                            "constraint": total_constraint,
                            "progress": total_progress,
                            "subgoal_consensus": total_subgoal_consensus,
                            "param_consensus": total_param_consensus,
                            "feature_score_consensus": total_feature_score_consensus,
                        },
                        metrics=metrics,
                        extras={
                            "lam_subgoal_consensus": f"{lam_subgoal_consensus:.3f}",
                            "lam_param_consensus": f"{lam_param_consensus:.3f}",
                            "lam_feature_score_consensus": f"{lam_feature_score_consensus:.3f}",
                        },
                    )
                )
            if self.plot_every is not None and (iteration + 1) % int(self.plot_every) == 0:
                if self.use_score_mode:
                    plot_scdp_activation_masks(self, iteration + 1)

        if self.plot_every is not None:
            final_it = int(max_iter)
            plot_scdp_results_4panel_overview(self, final_it)
            for demo_idx in range(len(self.demos)):
                plot_scdp_results_4panel(self, final_it, demo_idx=demo_idx)

        return _hard_gammas_from_stage_ends([len(X) for X in self.demos], self.stage_ends_, self.num_states)
