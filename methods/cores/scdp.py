from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from evaluation import eval_goalhmm_auto
from methods.base import format_training_log
from methods.common.tau_init import clip_tau_for_sequence, resolve_tau_init_for_demos
from utils.models import GaussianModel, MarginExpLowerEmission, StudentTModel, ZeroMeanGaussianModel
from visualization.scdp_4panel import plt as scdp_plot_plt, plot_scdp_results_4panel
from visualization.scdp_activation import plot_scdp_activation_masks, plot_scdp_demo_avg_costs


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


def _shortest_coverage_width(values, coverage: float = 0.7) -> float:
    xs = np.sort(np.asarray(values, dtype=float).reshape(-1))
    n = xs.size
    if n == 0:
        return np.nan
    if n == 1:
        return 0.0
    coverage = float(np.clip(coverage, 1e-6, 1.0))
    window = max(int(np.ceil(coverage * n)), 1)
    if window >= n:
        return float(xs[-1] - xs[0])
    widths = xs[window - 1 :] - xs[: n - window + 1]
    return float(np.min(widths))


@dataclass
class _StageParams:
    model_summaries: List[dict]
    subgoal: np.ndarray
    active_mask: np.ndarray | None = None


class SegmentConsensusDPModel:
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
        fixed_feature_mask=None,
        lambda_constraint=1.0,
        lambda_progress=1.0,
        lambda_consensus=1.0,
        lambda_r_consensus=1.0,
        consensus_schedule="linear",
        progress_delta_scale=20.0,
        duration_min=None,
        duration_max=None,
        sigma_floor=0.1,
        lam_floor=0.1,
        auto_feature_activation=False,
        demo_local_baseline_gate=False,
        equality_w70_ratio_threshold=0.2,
        plot_every=None,
        plot_dir="outputs/plots",
        eval_fn=eval_goalhmm_auto,
    ):
        if int(n_states) != 2:
            raise ValueError("scdp currently supports exactly 2 stages.")
        self.demos = [np.asarray(X, dtype=float) for X in demos]
        self.env = env
        self.true_taus = list(true_taus) if true_taus is not None else [None] * len(self.demos)
        self.num_states = 2
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.tau_init_mode = str(tau_init_mode)
        self.selected_raw_feature_ids = selected_raw_feature_ids
        self.feature_model_types_raw = feature_model_types
        self.fixed_feature_mask = fixed_feature_mask
        self.lambda_constraint = float(lambda_constraint)
        self.lambda_progress = float(lambda_progress)
        self.lambda_consensus = float(lambda_consensus)
        self.lambda_r_consensus = float(lambda_r_consensus)
        self.consensus_schedule = str(consensus_schedule)
        self.progress_delta_scale = max(float(progress_delta_scale), 1e-6)
        self.sigma_floor = max(float(sigma_floor), 1e-6)
        self.lam_floor = max(float(lam_floor), 1e-6)
        self.auto_feature_activation = bool(auto_feature_activation or demo_local_baseline_gate)
        self.equality_w70_ratio_threshold = float(equality_w70_ratio_threshold)
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
        self.loss_consensus: List[float] = []
        self.loss_r_consensus: List[float] = []
        self.metrics_hist: Dict[str, List[float]] = {}
        self.segmentation_history: List[List[List[int]]] = []
        self.consensus_lambda_hist: List[float] = []
        self.current_consensus_lambda = 0.0
        self.current_stage_params_per_demo: List[List[_StageParams]] = []
        self.demo_r_matrices_: List[np.ndarray] = []
        self.current_demo_cost_breakdown: List[Dict[str, float]] = []

        self.g1 = np.full(self.state_dim, np.nan, dtype=float)
        self.g2 = np.full(self.state_dim, np.nan, dtype=float)
        self.g1_hist: List[np.ndarray] = []
        self.g2_hist: List[np.ndarray] = []
        self.stage_ends_: List[List[int]] = []
        self.feature_models = self._build_feature_model_grid()
        self.shared_param_vectors: List[List[np.ndarray | None]] = [[None for _ in range(self.num_features)] for _ in range(self.num_states)]
        self.shared_stage_subgoals: List[np.ndarray] = [np.zeros(self.state_dim, dtype=float) for _ in range(self.num_states)]
        self.shared_r_mean: np.ndarray | None = None

        if tau_init is not None:
            self.tau_init = np.asarray(tau_init, dtype=int)
        else:
            self.tau_init = resolve_tau_init_for_demos(
                self.demos,
                tau_init=None,
                tau_init_mode=self.tau_init_mode,
                env=self.env,
                seed=self.seed,
                use_velocity=False,
                vel_weight=1.0,
                standardize=False,
                use_env_features=True,
                selected_raw_feature_ids=selected_raw_feature_ids,
            )

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

    def _segment_bounds_from_tau(self, tau, T):
        tau = int(tau)
        return [(0, tau), (tau + 1, T - 1)]

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
        chosen_losses = []
        active_mask = np.zeros(self.num_features, dtype=int)
        for feat_idx, kind in enumerate(self.feature_model_types):
            model = self._fit_local_model(kind, F[:, feat_idx])
            summary = model.get_summary()
            summaries.append(summary)
            fitted_loss = -np.asarray(model.logpdf(F[:, feat_idx]), dtype=float)
            if self.auto_feature_activation:
                kind_l = str(kind).lower()
                if kind_l in {"gauss", "gaussian", "student_t", "studentt", "t"}:
                    stage_w70 = float(_shortest_coverage_width(F[:, feat_idx], coverage=0.7))
                    demo_global_w70 = float(max(_shortest_coverage_width(F_demo[:, feat_idx], coverage=0.7), 1e-6))
                    w70_ratio = stage_w70 / demo_global_w70
                    if w70_ratio < self.equality_w70_ratio_threshold:
                        chosen_losses.append(fitted_loss)
                        active_mask[feat_idx] = 1
                else:
                    baseline_model = GaussianModel(
                        mu=float(np.mean(F[:, feat_idx])),
                        sigma=float(max(np.std(F[:, feat_idx]), 1e-6)),
                    )
                    baseline_loss = -np.asarray(baseline_model.logpdf(F[:, feat_idx]), dtype=float)
                    if float(np.mean(fitted_loss)) <= float(np.mean(baseline_loss)):
                        chosen_losses.append(fitted_loss)
                        active_mask[feat_idx] = 1
            else:
                if self.r[stage_idx, feat_idx]:
                    chosen_losses.append(fitted_loss)
                    active_mask[feat_idx] = 1
        if chosen_losses:
            constraint_cost = float(np.sum(np.mean(np.stack(chosen_losses, axis=1), axis=1)))
        else:
            constraint_cost = 0.0
        subgoal = np.asarray(self.demos[demo_idx][e if stage_idx == 0 else -1], dtype=float)
        progress_cost = self._progress_cost(self.demos[demo_idx], s, e, subgoal)
        return _StageParams(model_summaries=summaries, subgoal=subgoal, active_mask=active_mask), constraint_cost, progress_cost

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

    def _consensus_cost(self, candidate_stage_params, shared_stage_subgoals, shared_param_vectors):
        total = 0.0
        for stage_idx in range(self.num_states):
            diff = np.asarray(candidate_stage_params[stage_idx].subgoal, dtype=float) - np.asarray(shared_stage_subgoals[stage_idx], dtype=float)
            total += float(np.dot(diff, diff))
            for feat_idx, kind in enumerate(self.feature_model_types):
                if self.auto_feature_activation:
                    if candidate_stage_params[stage_idx].active_mask is None or not candidate_stage_params[stage_idx].active_mask[feat_idx]:
                        continue
                elif not self.r[stage_idx, feat_idx]:
                    continue
                local_vec = self._summary_to_vector(kind, candidate_stage_params[stage_idx].model_summaries[feat_idx])
                shared_vec = shared_param_vectors[stage_idx][feat_idx]
                if shared_vec is None:
                    continue
                delta = local_vec - shared_vec
                total += float(np.dot(delta, delta))
        return total

    def _r_consensus_cost(self, candidate_stage_params, shared_r_mean):
        if not self.auto_feature_activation or shared_r_mean is None:
            return 0.0
        total = 0.0
        for stage_idx in range(self.num_states):
            active_mask = candidate_stage_params[stage_idx].active_mask
            if active_mask is None:
                continue
            delta = np.asarray(active_mask, dtype=float) - np.asarray(shared_r_mean[stage_idx], dtype=float)
            total += float(np.dot(delta, delta))
        return total

    def _candidate_cost(self, demo_idx, tau, lam_consensus, lam_r_consensus, shared_stage_subgoals, shared_param_vectors, shared_r_mean):
        bounds = self._segment_bounds_from_tau(tau, len(self.demos[demo_idx]))
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
        consensus_cost = 0.0
        if lam_consensus > 0.0:
            consensus_cost = self._consensus_cost(candidate_stage_params, shared_stage_subgoals, shared_param_vectors)
        r_consensus_cost = 0.0
        if lam_r_consensus > 0.0:
            r_consensus_cost = self._r_consensus_cost(candidate_stage_params, shared_r_mean)
        total = (
            self.lambda_constraint * constraint_cost
            + self.lambda_progress * progress_cost
            + lam_consensus * consensus_cost
            + lam_r_consensus * r_consensus_cost
        )
        return {
            "tau": int(tau),
            "stage_ends": [int(tau), len(self.demos[demo_idx]) - 1],
            "stage_params": candidate_stage_params,
            "constraint": float(constraint_cost),
            "progress": float(progress_cost),
            "consensus": float(consensus_cost),
            "r_consensus": float(r_consensus_cost),
            "total": float(total),
        }

    def _initial_stage_ends(self):
        stage_ends = []
        if self.tau_init is not None:
            if len(self.tau_init) != len(self.demos):
                raise ValueError("tau_init must match the number of demos.")
            for X, tau in zip(self.demos, self.tau_init):
                cut = clip_tau_for_sequence(X, int(tau))
                stage_ends.append([int(cut), len(X) - 1])
            return stage_ends
        for X in self.demos:
            tau = clip_tau_for_sequence(X, int(round(0.5 * (len(X) - 1))))
            stage_ends.append([int(tau), len(X) - 1])
        return stage_ends

    def _shared_from_selected(self, selected_infos):
        shared_stage_subgoals = []
        for stage_idx in range(self.num_states):
            shared_stage_subgoals.append(
                np.mean([info["stage_params"][stage_idx].subgoal for info in selected_infos], axis=0)
            )
        shared_param_vectors = [[None for _ in range(self.num_features)] for _ in range(self.num_states)]
        for stage_idx in range(self.num_states):
            for feat_idx, kind in enumerate(self.feature_model_types):
                if self.auto_feature_activation:
                    vectors = [
                        self._summary_to_vector(kind, info["stage_params"][stage_idx].model_summaries[feat_idx])
                        for info in selected_infos
                        if info["stage_params"][stage_idx].active_mask is not None and info["stage_params"][stage_idx].active_mask[feat_idx]
                    ]
                    if not vectors:
                        continue
                else:
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
        self.g1 = np.asarray(shared_stage_subgoals[0], dtype=float).copy()
        self.g2 = np.asarray(shared_stage_subgoals[1], dtype=float).copy()
        self.g1_hist.append(self.g1.copy())
        self.g2_hist.append(self.g2.copy())
        for stage_idx in range(self.num_states):
            for feat_idx, kind in enumerate(self.feature_model_types):
                vec = shared_param_vectors[stage_idx][feat_idx]
                if vec is None:
                    continue
                self.feature_models[stage_idx][feat_idx] = self._vector_to_model(kind, vec)

    def _current_consensus_lambda(self, iteration, max_iter):
        frac = 1.0 if max_iter <= 1 else float(iteration) / float(max_iter - 1)
        if self.consensus_schedule == "linear":
            return self.lambda_consensus * frac
        if self.consensus_schedule == "quadratic":
            return self.lambda_consensus * frac * frac
        raise ValueError(f"Unsupported consensus_schedule '{self.consensus_schedule}'.")

    def fit(self, max_iter=30, verbose=True):
        self.stage_ends_ = self._initial_stage_ends()
        init_infos = []
        zero_goals = [np.zeros(self.state_dim, dtype=float), np.zeros(self.state_dim, dtype=float)]
        zero_params = [[None for _ in range(self.num_features)] for _ in range(self.num_states)]
        for demo_idx, stage_ends in enumerate(self.stage_ends_):
            init_infos.append(self._candidate_cost(demo_idx, int(stage_ends[0]), 0.0, 0.0, zero_goals, zero_params, None))
        shared_stage_subgoals, shared_param_vectors = self._shared_from_selected(init_infos)
        self._apply_shared_state(shared_stage_subgoals, shared_param_vectors)
        if self.auto_feature_activation and init_infos:
            self.shared_r_mean = np.mean(
                np.stack(
                    [
                        np.stack([stage_params.active_mask for stage_params in info["stage_params"]], axis=0)
                        for info in init_infos
                    ],
                    axis=0,
                ),
                axis=0,
            )
        else:
            self.shared_r_mean = None

        for iteration in range(int(max_iter)):
            lam_consensus = self._current_consensus_lambda(iteration, int(max_iter))
            lam_r_consensus = self._current_consensus_lambda(iteration, int(max_iter)) * self.lambda_r_consensus
            self.consensus_lambda_hist.append(float(lam_consensus))
            self.current_consensus_lambda = float(lam_consensus)
            selected_infos = []
            for demo_idx, X in enumerate(self.demos):
                best_info = None
                for tau in range(1, len(X) - 1):
                    info = self._candidate_cost(
                        demo_idx,
                        tau,
                        lam_consensus,
                        lam_r_consensus,
                        shared_stage_subgoals,
                        shared_param_vectors,
                        self.shared_r_mean,
                    )
                    if info is None:
                        continue
                    if best_info is None or info["total"] < best_info["total"]:
                        best_info = info
                if best_info is None:
                    raise RuntimeError(f"No feasible segmentation found for demo {demo_idx}.")
                selected_infos.append(best_info)

            self.stage_ends_ = [list(info["stage_ends"]) for info in selected_infos]
            self.current_stage_params_per_demo = [list(info["stage_params"]) for info in selected_infos]
            self.current_demo_cost_breakdown = [
                {
                    "constraint": self.lambda_constraint * float(info["constraint"]),
                    "progress": self.lambda_progress * float(info["progress"]),
                    "consensus": lam_consensus * float(info["consensus"]),
                    "r_consensus": lam_r_consensus * float(info.get("r_consensus", 0.0)),
                    "total": float(info["total"]),
                }
                for info in selected_infos
            ]
            self.demo_r_matrices_ = [
                np.stack([stage_params.active_mask for stage_params in info["stage_params"]], axis=0)
                for info in selected_infos
            ]
            if self.auto_feature_activation and self.demo_r_matrices_:
                self.shared_r_mean = np.mean(np.stack(self.demo_r_matrices_, axis=0), axis=0)
            self.segmentation_history.append([list(item) for item in self.stage_ends_])
            shared_stage_subgoals, shared_param_vectors = self._shared_from_selected(selected_infos)
            self._apply_shared_state(shared_stage_subgoals, shared_param_vectors)

            total_constraint = float(np.sum([info["constraint"] for info in selected_infos]))
            total_progress = float(np.sum([info["progress"] for info in selected_infos]))
            total_consensus = float(np.sum([info["consensus"] for info in selected_infos]))
            total_r_consensus = float(np.sum([info.get("r_consensus", 0.0) for info in selected_infos]))
            total_loss = float(
                self.lambda_constraint * total_constraint
                + self.lambda_progress * total_progress
                + lam_consensus * total_consensus
                + lam_r_consensus * total_r_consensus
            )
            self.loss_constraint.append(total_constraint)
            self.loss_progress.append(total_progress)
            self.loss_consensus.append(total_consensus)
            self.loss_r_consensus.append(total_r_consensus)
            self.loss_total.append(total_loss)

            gammas = _hard_gammas_from_stage_ends([len(X) for X in self.demos], self.stage_ends_, self.num_states)
            metrics = self.eval_fn(self, gammas, None)
            for name, value in metrics.items():
                self.metrics_hist.setdefault(name, []).append(float(value))

            if verbose:
                print(
                    format_training_log(
                        "scdp",
                        iteration,
                        losses={
                            "total": total_loss,
                            "constraint": total_constraint,
                            "progress": total_progress,
                            "consensus": total_consensus,
                            "r_consensus": total_r_consensus,
                        },
                        metrics=metrics,
                        extras={
                            "lam_consensus": f"{lam_consensus:.3f}",
                            "lam_r_consensus": f"{lam_r_consensus:.3f}",
                        },
                    )
                )
            if self.plot_every is not None and (iteration + 1) % int(self.plot_every) == 0:
                plot_scdp_results_4panel(self, iteration + 1)
                if self.auto_feature_activation:
                    plot_scdp_activation_masks(self, iteration + 1)
                    plot_scdp_demo_avg_costs(self, iteration + 1)

        return _hard_gammas_from_stage_ends([len(X) for X in self.demos], self.stage_ends_, self.num_states)
