from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
import os
import time
from typing import Dict, List, Sequence

import numpy as np

from evaluation import evaluate_model_metrics
from methods.base import format_training_log
from utils.models import (
    GaussianModel,
    MarginExpLowerEmission,
    MarginExpLowerLeftHNEmission,
    MarginExpUpperEmission,
    MarginExpUpperRightHNEmission,
    StudentTModel,
    ZeroMeanGaussianModel,
)
from visualization.swcl_4panel import plt as swcl_plot_plt, plot_swcl_results_4panel
from visualization.swcl_activation import plot_swcl_activation_dynamics


_SWCL_PRECOMPUTE_MODEL = None


def _swcl_precompute_worker_init(model):
    global _SWCL_PRECOMPUTE_MODEL
    _SWCL_PRECOMPUTE_MODEL = model


def _swcl_precompute_worker_run(demo_idx, items):
    if _SWCL_PRECOMPUTE_MODEL is None:
        raise RuntimeError("SWCL precompute worker is not initialized.")
    return _SWCL_PRECOMPUTE_MODEL._compute_demo_segment_cache_batch(demo_idx, items)


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


def _hard_gammas_from_stage_ends(lengths: Sequence[int], stage_ends_per_demo: Sequence[Sequence[int]], num_stages: int):
    gammas: List[np.ndarray] = []
    for T, stage_ends in zip(lengths, stage_ends_per_demo):
        gamma = np.zeros((int(T), int(num_stages)), dtype=float)
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
    abs_dev = np.abs(xs - center)
    keep = max(int(np.ceil(0.8 * abs_dev.size)), 1)
    trimmed_abs_dev = np.partition(abs_dev, keep - 1)[:keep]
    return float(np.mean(trimmed_abs_dev))


def _geometric_median(points, *, max_iter: int = 100, tol: float = 1e-6):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array.")
    if len(pts) == 0:
        raise ValueError("points must contain at least one point.")
    if len(pts) == 1:
        return pts[0].copy()

    guess = np.mean(pts, axis=0)
    for _ in range(max(int(max_iter), 1)):
        diff = pts - guess[None, :]
        dist = np.linalg.norm(diff, axis=1)
        close_mask = dist < tol
        if np.any(close_mask):
            return pts[np.argmax(close_mask)].copy()
        inv_dist = 1.0 / np.maximum(dist, tol)
        next_guess = np.sum(pts * inv_dist[:, None], axis=0) / np.sum(inv_dist)
        if np.linalg.norm(next_guess - guess) < tol:
            return next_guess
        guess = next_guess
    return guess


@dataclass
class _StageParams:
    model_summaries: List[dict]
    subgoal: np.ndarray
    active_mask: np.ndarray | None = None
    feature_scores: np.ndarray | None = None
    feature_constraint_costs: np.ndarray | None = None
    selected_feature_kinds: List[str] | None = None
    param_vectors: List[np.ndarray | None] | None = None


class StageWiseConstraintLearningModel:
    def __init__(
        self,
        demos,
        env,
        true_taus=None,
        true_cutpoints=None,
        n_stages=2,
        seed=0,
        selected_raw_feature_ids=None,
        feature_model_types=None,
        fixed_feature_mask=None,
        lambda_eq_constraint=1.0,
        lambda_ineq_constraint=1.0,
        lambda_progress=1.0,
        lambda_subgoal_consensus=1.0,
        lambda_param_consensus=1.0,
        lambda_activation_consensus=1.0,
        lambda_feature_score_consensus=None,
        consensus_schedule="linear",
        progress_delta_scale=20.0,
        duration_min=None,
        duration_max=None,
        feature_activation_mode="fixed_mask",
        equality_score_mode="dispersion",
        equality_dispersion_ratio_threshold=0.1,
        constraint_core_trim=0,
        short_segment_penalty_c=0.1,
        inequality_score_activation_threshold=-0.5,
        activation_proto_temperature=0.1,
        joint_mask_search_max_masks=4096,
        fixed_true_cutpoint_prefix=0,
        fixed_true_cutpoint_indices=None,
        precompute_num_workers=None,
        plot_every=None,
        plot_dir="outputs/plots",
        eval_fn=evaluate_model_metrics,
        verbose=True,
    ):
        self.demos = [np.asarray(X, dtype=float) for X in demos]
        self.env = env
        self.true_cutpoints = _normalize_true_cutpoints(self.demos, true_taus=true_taus, true_cutpoints=true_cutpoints)
        self.true_taus = [
            None if cuts is None or len(cuts) != 1 else int(cuts[0])
            for cuts in self.true_cutpoints
        ]
        self.num_stages = int(n_stages)
        if self.num_stages < 2:
            raise ValueError("swcl requires at least 2 stages.")
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
        if lambda_feature_score_consensus is None:
            lambda_feature_score_consensus = lambda_activation_consensus
        self.lambda_activation_consensus = float(lambda_feature_score_consensus)
        self.consensus_schedule = str(consensus_schedule)
        self.progress_delta_scale = max(float(progress_delta_scale), 1e-6)
        feature_activation_mode = str(feature_activation_mode).lower()
        if feature_activation_mode not in {"fixed_mask", "score", "joint_mask_search"}:
            raise ValueError("feature_activation_mode must be one of {'fixed_mask', 'score', 'joint_mask_search'}.")
        self.feature_activation_mode = feature_activation_mode
        self.use_joint_mask_search = self.feature_activation_mode == "joint_mask_search"
        self.use_score_mode = self.feature_activation_mode in {"score", "joint_mask_search"}
        self.equality_score_mode = str(equality_score_mode).lower()
        if self.equality_score_mode not in {"dispersion", "gaussian_ll_gain"}:
            raise ValueError("equality_score_mode must be one of {'dispersion', 'gaussian_ll_gain'}.")
        self.equality_dispersion_ratio_threshold = float(equality_dispersion_ratio_threshold)
        self.constraint_core_trim = max(int(constraint_core_trim), 0)
        self.short_segment_penalty_c = float(short_segment_penalty_c)
        self.inequality_score_activation_threshold = float(inequality_score_activation_threshold)
        self.activation_proto_temperature = max(float(activation_proto_temperature), 1e-6)
        self.joint_mask_search_max_masks = max(int(joint_mask_search_max_masks), 1)
        self.fixed_true_cutpoint_prefix = max(int(fixed_true_cutpoint_prefix), 0)
        if fixed_true_cutpoint_indices is None:
            self.fixed_true_cutpoint_indices = []
        else:
            self.fixed_true_cutpoint_indices = sorted({int(idx) for idx in fixed_true_cutpoint_indices})
        if precompute_num_workers is None:
            cpu_count = os.cpu_count() or 1
            precompute_num_workers = min(max(cpu_count, 1), max(len(self.demos), 1))
        self.precompute_num_workers = max(int(precompute_num_workers), 1)
        max_boundary_idx = self.num_stages - 2
        for idx in self.fixed_true_cutpoint_indices:
            if idx < 0 or idx > max_boundary_idx:
                raise ValueError(
                    f"fixed_true_cutpoint_indices entries must lie in [0, {max_boundary_idx}], got {idx}."
                )
        self.plot_every = plot_every
        self.plot_dir = plot_dir
        self.eval_fn = eval_fn
        self.verbose = bool(verbose)
        if self.plot_every is not None and swcl_plot_plt is None:
            print("[SWCL] matplotlib is not installed; SWCL 4-panel plots will not be generated.")

        self._init_feature_preprocessing()
        self.feature_model_types = self._normalize_feature_model_types(self.num_features)
        if any(self._is_auto_constraint_feature(feat_idx) for feat_idx in range(self.num_features)) and not self.use_score_mode:
            raise ValueError("auto constraint type selection currently requires feature_activation_mode in {'score', 'joint_mask_search'}.")
        self.r = np.ones((self.num_stages, self.num_features), dtype=int)
        self.has_equality_feature = any(self._feature_supports_equality(feat_idx) for feat_idx in range(self.num_features))
        self.score_threshold_matrix = self._build_score_threshold_matrix()
        if fixed_feature_mask is not None:
            mask = np.asarray(fixed_feature_mask, dtype=int)
            if mask.shape != self.r.shape:
                raise ValueError(f"fixed_feature_mask must have shape {self.r.shape}.")
            self.r = mask.copy()

        self.duration_min = self._broadcast_stage_value(duration_min, default=1, dtype=int)
        self.duration_max = self._broadcast_stage_value(duration_max, default=None, dtype=int)
        total_max = int(max(len(X) for X in self.demos))
        for k in range(self.num_stages):
            if self.duration_max[k] is None:
                self.duration_max[k] = total_max
        self.duration_max = np.asarray(self.duration_max, dtype=int)

        self.loss_total: List[float] = []
        self.loss_constraint: List[float] = []
        self.loss_short_segment_penalty: List[float] = []
        self.loss_progress: List[float] = []
        self.loss_subgoal_consensus: List[float] = []
        self.loss_param_consensus: List[float] = []
        self.loss_activation_consensus: List[float] = []
        self.metrics_hist: Dict[str, List[float]] = {}
        self.segmentation_history: List[List[List[int]]] = []
        self.activation_rate_history: List[np.ndarray] = []
        self.subgoal_consensus_lambda_hist: List[float] = []
        self.param_consensus_lambda_hist: List[float] = []
        self.activation_consensus_lambda_hist: List[float] = []
        self.current_subgoal_consensus_lambda = 0.0
        self.current_param_consensus_lambda = 0.0
        self.current_activation_consensus_lambda = 0.0
        self.current_stage_params_per_demo: List[List[_StageParams]] = []
        self.demo_r_matrices_: List[np.ndarray] = []
        self.current_demo_cost_breakdown: List[Dict[str, float]] = []
        self.stage_subgoals: List[np.ndarray] = [np.full(self.state_dim, np.nan, dtype=float) for _ in range(self.num_stages)]
        self.stage_subgoals_hist: List[List[np.ndarray]] = []
        self.g1 = np.full(self.state_dim, np.nan, dtype=float)
        self.g2 = np.full(self.state_dim, np.nan, dtype=float)
        self.g1_hist: List[np.ndarray] = []
        self.g2_hist: List[np.ndarray] = []
        self.stage_ends_: List[List[int]] = []
        self.feature_models = self._build_feature_model_grid()
        self.shared_param_vectors: List[List[np.ndarray | None]] = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        self.shared_stage_subgoals: List[np.ndarray] = [np.zeros(self.state_dim, dtype=float) for _ in range(self.num_stages)]
        self.shared_r_mean: np.ndarray | None = None
        self.shared_feature_score_mean: np.ndarray | None = None
        self.shared_activation_proto: np.ndarray | None = None
        self.demo_feature_score_matrices_: List[np.ndarray] = []
        self.demo_activation_matrices_: List[np.ndarray] = []
        self.posthoc_activation_summary_: Dict[str, object] | None = None
        self._joint_mask_fallback_warned = False
        self.demo_activation_history: List[np.ndarray] = []
        self.activation_proto_history: List[np.ndarray] = []
        self._segment_stage_cache: Dict[tuple[int, int, int, int], tuple[_StageParams, float, float]] = {}
        self._segment_base_cache: Dict[tuple[int, int, int], tuple[_StageParams, float, float]] = {}

    @staticmethod
    def _kind_is_auto(kind) -> bool:
        return str(kind).lower() in {"auto", "auto_constraint", "auto_eq_ineq", "auto_constraint_type"}

    @staticmethod
    def _kind_is_equality(kind) -> bool:
        kind_l = str(kind).lower()
        return kind_l in {"gauss", "gaussian", "student_t", "studentt", "t", "zero_gauss", "zero_gaussian"}

    @staticmethod
    def _kind_is_lower(kind) -> bool:
        kind_l = str(kind).lower()
        return kind_l in {
            "margin_exp_lower", "marginexp", "margin_exp",
            "margin_exp_lower_left_hn", "marginexp_left_hn", "margin_exp_left_hn",
        }

    @staticmethod
    def _kind_is_upper(kind) -> bool:
        kind_l = str(kind).lower()
        return kind_l in {
            "margin_exp_upper", "marginexp_upper", "margin_exp_upper",
            "margin_exp_upper_right_hn", "marginexp_upper_right_hn", "margin_exp_upper_right_hn",
        }

    def _feature_supports_equality(self, feat_idx: int) -> bool:
        return self._kind_is_auto(self.feature_model_types[feat_idx]) or self._is_equality_feature(feat_idx)

    def _is_auto_constraint_feature(self, feat_idx: int) -> bool:
        return self._kind_is_auto(self.feature_model_types[feat_idx])

    def _auto_constraint_threshold(self) -> float:
        return 0.0

    def _score_threshold_for_kind(self, kind) -> float:
        if self._kind_is_auto(kind):
            return self._auto_constraint_threshold()
        if self._kind_is_equality(kind):
            return float(self._equality_score_threshold())
        return float(self.inequality_score_activation_threshold)

    def _stage_feature_kind(self, stage_params, feat_idx: int) -> str:
        kinds = getattr(stage_params, "selected_feature_kinds", None)
        if kinds is not None and feat_idx < len(kinds):
            kind = str(kinds[feat_idx]).lower()
            if kind and kind != "unconstrained":
                return kind
        base_kind = str(self.feature_model_types[feat_idx]).lower()
        if self._kind_is_auto(base_kind):
            return "student_t"
        return base_kind

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
            raise ValueError("swcl requires a dataset env with feature API.")
        raw_features = [np.asarray(self.env.compute_all_features_matrix(X), dtype=float) for X in self.demos]
        if not raw_features:
            raise ValueError("swcl requires at least one demo.")
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

        self.raw_feature_spec_by_column_idx = {}
        self.raw_id_to_column_idx = {}
        self.feature_specs = []
        self.raw_id_to_local_idx = {}
        self.feature_name_to_local_idx = {}
        for spec_idx, spec in enumerate(self.raw_feature_specs):
            raw_id = int(spec.get("id", spec_idx))
            col_idx = int(spec.get("column_idx", spec_idx))
            self.raw_feature_spec_by_column_idx[col_idx] = dict(spec)
            self.raw_id_to_column_idx[raw_id] = col_idx
        for local_idx, col_idx in enumerate(self.selected_feature_columns):
            raw_spec = dict(self.raw_feature_spec_by_column_idx.get(int(col_idx), {}))
            raw_id = int(raw_spec.get("id", col_idx))
            name = str(raw_spec.get("name", f"f{local_idx}"))
            spec = {
                "raw_id": raw_id,
                "column_idx": int(col_idx),
                "name": name,
                "description": str(raw_spec.get("description", "")),
                "local_idx": int(local_idx),
            }
            self.feature_specs.append(spec)
            self.raw_id_to_local_idx[raw_id] = int(local_idx)
            self.feature_name_to_local_idx[name] = int(local_idx)

    def _segment_base_cache_work_items_for_demo(self, demo_idx: int):
        X = self.demos[int(demo_idx)]
        T = len(X)
        fixed_cutpoints_by_stage = self._fixed_cutpoint_map_for_demo(int(demo_idx)) or {}
        items = []
        seen_base = set()
        for stage_idx in range(self.num_stages):
            min_len = int(self.duration_min[stage_idx])
            max_len = int(self.duration_max[stage_idx])
            fixed_e = fixed_cutpoints_by_stage.get(int(stage_idx))
            e_values = [int(fixed_e)] if fixed_e is not None else range(T)
            for e in e_values:
                if e < 0 or e >= T:
                    continue
                s_min = max(0, int(e) - max_len + 1)
                s_max = int(e) - min_len + 1
                if s_max < s_min:
                    continue
                for s in range(s_min, s_max + 1):
                    base_key = (int(demo_idx), int(s), int(e))
                    if base_key in self._segment_base_cache or base_key in seen_base:
                        continue
                    seen_base.add(base_key)
                    items.append(base_key)
        return items

    def _segment_stage_cache_work_items_for_demo(self, demo_idx: int):
        if self.use_score_mode:
            return self._segment_base_cache_work_items_for_demo(demo_idx)
        X = self.demos[int(demo_idx)]
        T = len(X)
        fixed_cutpoints_by_stage = self._fixed_cutpoint_map_for_demo(int(demo_idx)) or {}
        items = []
        for stage_idx in range(self.num_stages):
            min_len = int(self.duration_min[stage_idx])
            max_len = int(self.duration_max[stage_idx])
            fixed_e = fixed_cutpoints_by_stage.get(int(stage_idx))
            e_values = [int(fixed_e)] if fixed_e is not None else range(T)
            for e in e_values:
                if e < 0 or e >= T:
                    continue
                s_min = max(0, int(e) - max_len + 1)
                s_max = int(e) - min_len + 1
                if s_max < s_min:
                    continue
                for s in range(s_min, s_max + 1):
                    key = (int(demo_idx), int(stage_idx), int(s), int(e))
                    if key not in self._segment_stage_cache:
                        items.append(key)
        return items

    def _prepare_segment_stage_cache(self):
        demo_items = [self._segment_stage_cache_work_items_for_demo(demo_idx) for demo_idx in range(len(self.demos))]
        total_items = int(sum(len(items) for items in demo_items))
        if total_items <= 0:
            return
        if self.verbose:
            noun = "local segment summaries" if self.use_score_mode else "local segment costs"
            workers = min(self.precompute_num_workers, len(self.demos))
            print(f"[SWCL] preparing DP {noun}: {total_items} segments with {workers} workers...")
        start_time = time.time()
        overall_done = 0
        worker_count = min(self.precompute_num_workers, len(self.demos))
        if worker_count <= 1:
            for demo_idx, items in enumerate(demo_items):
                demo_total = len(items)
                if demo_total <= 0:
                    continue
                report_every = max(1, demo_total // 10)
                for local_idx, item in enumerate(items):
                    if self.use_score_mode:
                        _, s, e = item
                        self._fit_segment_base(demo_idx, s, e)
                    else:
                        _, stage_idx, s, e = item
                        self._fit_segment_stage(demo_idx, stage_idx, s, e)
                    overall_done += 1
                    if self.verbose and (
                        local_idx == 0
                        or local_idx + 1 == demo_total
                        or (local_idx + 1) % report_every == 0
                    ):
                        elapsed = time.time() - start_time
                        print(
                            f"[SWCL] DP prep demo {demo_idx + 1}/{len(self.demos)}: "
                            f"{local_idx + 1}/{demo_total} local segments | "
                            f"overall {overall_done}/{total_items} ({elapsed:.1f}s elapsed)"
                        )
        else:
            try:
                mp_context = mp.get_context("fork")
            except ValueError:
                mp_context = mp.get_context()
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp_context,
                initializer=_swcl_precompute_worker_init,
                initargs=(self,),
            ) as executor:
                future_to_demo = {
                    executor.submit(_swcl_precompute_worker_run, demo_idx, items): (demo_idx, len(items))
                    for demo_idx, items in enumerate(demo_items)
                    if len(items) > 0
                }
                for future in as_completed(future_to_demo):
                    demo_idx, demo_total = future_to_demo[future]
                    cache_entries = future.result()
                    if self.use_score_mode:
                        self._segment_base_cache.update(cache_entries)
                    else:
                        self._segment_stage_cache.update(cache_entries)
                    overall_done += int(demo_total)
                    if self.verbose:
                        elapsed = time.time() - start_time
                        print(
                            f"[SWCL] DP prep demo {demo_idx + 1}/{len(self.demos)}: "
                            f"{demo_total}/{demo_total} local segments | "
                            f"overall {overall_done}/{total_items} ({elapsed:.1f}s elapsed)"
                        )
        if self.verbose:
            elapsed = time.time() - start_time
            noun = "local segment summaries" if self.use_score_mode else "local segment costs"
            print(f"[SWCL] DP {noun} ready ({elapsed:.1f}s)")

    def _broadcast_stage_value(self, value, default, dtype=float):
        if value is None:
            return [default for _ in range(self.num_stages)]
        if np.isscalar(value):
            return [dtype(value) for _ in range(self.num_stages)]
        value = list(value)
        if len(value) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} stage values, got {len(value)}.")
        return [default if v is None else dtype(v) for v in value]

    def _build_feature_model_grid(self):
        rows = []
        for _ in range(self.num_stages):
            cur = []
            for kind in self.feature_model_types:
                cur.append(self._make_model_from_kind(kind))
            rows.append(cur)
        return rows

    def _make_model_from_kind(self, kind):
        kind = str(kind).lower()
        if self._kind_is_auto(kind) or kind == "unconstrained":
            return StudentTModel(mu=0.0, sigma=1.0)
        if kind in {"gauss", "gaussian"}:
            return GaussianModel(mu=0.0, sigma=1.0)
        if kind in {"student_t", "studentt", "t"}:
            return StudentTModel(mu=0.0, sigma=1.0)
        if kind in {"zero_gauss", "zero_gaussian"}:
            return ZeroMeanGaussianModel(sigma=1.0)
        if kind in {"margin_exp_lower", "marginexp", "margin_exp"}:
            return MarginExpLowerEmission(b_init=0.0, lam_init=1.0)
        if kind in {"margin_exp_lower_left_hn", "marginexp_left_hn", "margin_exp_left_hn"}:
            return MarginExpLowerLeftHNEmission(b_init=0.0, lam_init=1.0)
        if kind in {"margin_exp_upper", "marginexp_upper", "margin_exp_upper"}:
            return MarginExpUpperEmission(b_init=0.0, lam_init=1.0)
        if kind in {"margin_exp_upper_right_hn", "marginexp_upper_right_hn", "margin_exp_upper_right_hn"}:
            return MarginExpUpperRightHNEmission(b_init=0.0, lam_init=1.0)
        raise ValueError(f"Unsupported feature model type '{kind}'.")

    def _fit_local_model(self, kind, xs):
        xs = np.asarray(xs, dtype=float).reshape(-1)
        kind = str(kind).lower()
        if self._kind_is_auto(kind) or kind == "unconstrained":
            kind = "student_t"
        if kind in {"gauss", "gaussian"}:
            mu = float(np.mean(xs))
            sigma = float(max(np.std(xs), 1e-6))
            return GaussianModel(mu=mu, sigma=sigma)
        if kind in {"student_t", "studentt", "t"}:
            model = StudentTModel(mu=0.0, sigma=1.0)
            model.m_step_update([xs])
            model._update_interval()
            return model
        if kind in {"zero_gauss", "zero_gaussian"}:
            sigma = float(max(np.sqrt(np.mean(xs * xs)), 1e-6))
            return ZeroMeanGaussianModel(sigma=sigma)
        if kind in {"margin_exp_lower", "marginexp", "margin_exp"}:
            model = MarginExpLowerEmission(b_init=0.0, lam_init=1.0)
            model.m_step_update([xs])
            model._update_interval()
            return model
        if kind in {"margin_exp_lower_left_hn", "marginexp_left_hn", "margin_exp_left_hn"}:
            model = MarginExpLowerLeftHNEmission(b_init=0.0, lam_init=1.0)
            model.m_step_update([xs])
            model._update_interval()
            return model
        if kind in {"margin_exp_upper", "marginexp_upper", "margin_exp_upper"}:
            model = MarginExpUpperEmission(b_init=0.0, lam_init=1.0)
            model.m_step_update([xs])
            model._update_interval()
            return model
        if kind in {"margin_exp_upper_right_hn", "marginexp_upper_right_hn", "margin_exp_upper_right_hn"}:
            model = MarginExpUpperRightHNEmission(b_init=0.0, lam_init=1.0)
            model.m_step_update([xs])
            model._update_interval()
            return model
        raise ValueError(f"Unsupported feature model type '{kind}'.")

    def _fit_student_t_baseline(self, xs):
        xs = np.asarray(xs, dtype=float).reshape(-1)
        model = StudentTModel(mu=0.0, sigma=1.0)
        model.m_step_update([xs])
        model._update_interval()
        return model

    def _fit_auto_constraint_feature(self, feat_idx: int, values, F_demo, segment_median: float):
        values = np.asarray(values, dtype=float).reshape(-1)
        F_demo = np.asarray(F_demo, dtype=float).reshape(-1)
        candidate_kinds = ["student_t", "margin_exp_lower", "margin_exp_upper"]
        candidate_records = []
        for kind in candidate_kinds:
            model = self._fit_local_model(kind, values)
            summary = dict(model.get_summary())
            summary["segment_median"] = float(segment_median)
            kind_l = str(kind).lower()
            if self._kind_is_equality(kind_l) and self.equality_score_mode == "dispersion":
                score = float(_mean_abs_centered_dispersion(values))
                threshold = float(self._equality_score_threshold())
                weight = float(self.lambda_eq_constraint)
            elif self._kind_is_equality(kind_l) and self.equality_score_mode == "gaussian_ll_gain":
                local_gaussian = GaussianModel(
                    mu=float(np.mean(values)),
                    sigma=float(max(np.std(values), 1e-6)),
                )
                global_gaussian = GaussianModel(
                    mu=float(np.mean(F_demo)),
                    sigma=float(max(np.std(F_demo), 1e-6)),
                )
                local_loss = -np.asarray(local_gaussian.logpdf(values), dtype=float)
                global_loss = -np.asarray(global_gaussian.logpdf(values), dtype=float)
                score = float(np.mean(local_loss) - np.mean(global_loss))
                threshold = float(self._equality_score_threshold())
                weight = float(self.lambda_eq_constraint)
            else:
                baseline_model = self._fit_student_t_baseline(values)
                fitted_loss = -np.asarray(model.logpdf(values), dtype=float)
                baseline_loss = -np.asarray(baseline_model.logpdf(values), dtype=float)
                score = float(np.mean(fitted_loss) - np.mean(baseline_loss))
                threshold = float(self.inequality_score_activation_threshold)
                weight = float(self.lambda_ineq_constraint)
            margin = float(threshold - score)
            candidate_records.append(
                {
                    "kind": kind_l,
                    "summary": summary,
                    "score": float(score),
                    "threshold": float(threshold),
                    "margin": float(margin),
                    "weighted_margin": float(weight * margin),
                }
            )

        best_record = max(candidate_records, key=lambda item: float(item["weighted_margin"]))
        positive_records = [item for item in candidate_records if float(item["margin"]) > 0.0]
        chosen_record = max(positive_records, key=lambda item: float(item["weighted_margin"])) if positive_records else best_record
        active = int(bool(positive_records) and float(chosen_record["weighted_margin"]) > 0.0)
        chosen_kind = str(chosen_record["kind"]) if active else "unconstrained"
        best_weighted_margin = float(chosen_record["weighted_margin"])
        score = -best_weighted_margin
        constraint_cost = float(len(values) * min(-best_weighted_margin, 0.0))
        return {
            "summary": dict(chosen_record["summary"]),
            "selected_kind": chosen_kind,
            "score": float(score),
            "active": int(active),
            "constraint_cost": float(constraint_cost),
        }

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

    def _segment_core_bounds(self, s, e):
        s_i = int(s)
        e_i = int(e)
        trim = int(self.constraint_core_trim)
        if trim <= 0:
            return s_i, e_i
        core_s = min(s_i + trim, e_i)
        core_e = max(core_s, e_i - trim)
        return int(core_s), int(core_e)

    def _fixed_cutpoint_map_for_demo(self, demo_idx):
        if self.true_cutpoints[demo_idx] is None:
            return None

        true_cuts = np.asarray(self.true_cutpoints[demo_idx], dtype=int).reshape(-1)
        if true_cuts.size == 0:
            return None

        indices = set(range(min(self.fixed_true_cutpoint_prefix, true_cuts.size)))
        indices.update(idx for idx in self.fixed_true_cutpoint_indices if idx < true_cuts.size)
        if not indices:
            return None

        fixed_map = {int(idx): int(true_cuts[int(idx)]) for idx in sorted(indices)}
        fixed_values = [fixed_map[idx] for idx in sorted(fixed_map)]
        if fixed_values != sorted(fixed_values):
            raise ValueError(f"True cutpoints must be strictly increasing for demo {demo_idx}.")
        return fixed_map

    def _progress_cost(self, X, s, e, subgoal):
        if e <= s:
            return 0.0
        subgoal = np.asarray(subgoal, dtype=float)
        start_t = max(int(s), 0)
        end_t = min(int(e) - 1, len(X) - 2)
        if end_t < start_t:
            return 0.0
        segment = np.asarray(X[start_t : end_t + 2], dtype=float)
        d0 = np.linalg.norm(segment[:-1] - subgoal[None, :], axis=1)
        d1 = np.linalg.norm(segment[1:] - subgoal[None, :], axis=1)
        delta = np.clip(self.progress_delta_scale * (d1 - d0), -60.0, 60.0)
        return float(np.sum(np.log1p(np.exp(delta)) / self.progress_delta_scale))

    def _fit_segment_stage(self, demo_idx, stage_idx, s, e):
        if self.use_score_mode:
            return self._fit_segment_base(demo_idx, s, e)
        cache_key = (int(demo_idx), int(stage_idx), int(s), int(e))
        cached = self._segment_stage_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._fit_segment_base(demo_idx, s, e, stage_idx=stage_idx)
        self._segment_stage_cache[cache_key] = result
        return result

    def _compute_demo_segment_cache_batch(self, demo_idx, items):
        out = {}
        if self.use_score_mode:
            for _, s, e in items:
                out[(int(demo_idx), int(s), int(e))] = self._compute_segment_base_uncached(demo_idx, s, e)
            return out
        for _, stage_idx, s, e in items:
            out[(int(demo_idx), int(stage_idx), int(s), int(e))] = self._compute_segment_base_uncached(
                demo_idx,
                s,
                e,
                stage_idx=stage_idx,
            )
        return out

    def _fit_segment_base(self, demo_idx, s, e, stage_idx=None):
        base_cache_key = (int(demo_idx), int(s), int(e))
        if stage_idx is None:
            cached_base = self._segment_base_cache.get(base_cache_key)
            if cached_base is not None:
                return cached_base
        result = self._compute_segment_base_uncached(demo_idx, s, e, stage_idx=stage_idx)
        if stage_idx is None:
            self._segment_base_cache[base_cache_key] = result
        return result

    def _compute_segment_base_uncached(self, demo_idx, s, e, stage_idx=None):
        core_s, core_e = self._segment_core_bounds(s, e)
        F = self.standardized_features[demo_idx][core_s : core_e + 1]
        F_demo = self.standardized_features[demo_idx]
        summaries = []
        selected_feature_kinds = []
        param_vectors = []
        active_mask = np.zeros(self.num_features, dtype=int)
        feature_scores = np.zeros(self.num_features, dtype=float)
        feature_constraint_costs = np.zeros(self.num_features, dtype=float)
        active_fit_losses = [None for _ in range(self.num_features)]
        for feat_idx, kind in enumerate(self.feature_model_types):
            values = F[:, feat_idx]
            segment_median = float(np.median(values))
            if self._is_auto_constraint_feature(feat_idx):
                auto_info = self._fit_auto_constraint_feature(feat_idx, values, F_demo[:, feat_idx], segment_median)
                summaries.append(dict(auto_info["summary"]))
                selected_feature_kinds.append(str(auto_info["selected_kind"]))
                param_vectors.append(
                    self._summary_to_vector_or_none(str(auto_info["selected_kind"]), auto_info["summary"])
                )
                feature_scores[feat_idx] = float(auto_info["score"])
                active_mask[feat_idx] = int(auto_info["active"])
                feature_constraint_costs[feat_idx] = float(auto_info["constraint_cost"])
                continue
            is_equality_feature = self._is_equality_feature(feat_idx)
            if self.use_score_mode and is_equality_feature and self.equality_score_mode == "dispersion":
                kind_l = str(kind).lower()
                if kind_l in {"gauss", "gaussian"}:
                    model = GaussianModel(
                        mu=float(np.mean(values)),
                        sigma=float(max(np.std(values), 1e-6)),
                    )
                elif kind_l in {"student_t", "studentt", "t"}:
                    model = StudentTModel(
                        mu=float(np.median(values)),
                        sigma=float(max(np.std(values), 1e-6)),
                    )
                elif kind_l in {"zero_gauss", "zero_gaussian"}:
                    model = ZeroMeanGaussianModel(
                        sigma=float(max(np.sqrt(np.mean(values * values)), 1e-6)),
                    )
                else:
                    model = self._fit_local_model(kind, values)
                summary = dict(model.get_summary())
                summary["segment_median"] = segment_median
                summaries.append(summary)
                selected_feature_kinds.append(str(kind_l))
                param_vectors.append(self._summary_to_vector_or_none(kind_l, summary))
                score = float(_mean_abs_centered_dispersion(values))
                feature_scores[feat_idx] = score
                active_mask[feat_idx] = self._feature_active_from_score(feat_idx, score)
                avg_step_cost = min(score - float(self._equality_score_threshold()), 0.0)
                feature_constraint_costs[feat_idx] = float(self.lambda_eq_constraint * len(values) * avg_step_cost)
                continue

            model = self._fit_local_model(kind, values)
            kind_l = str(kind).lower()
            summary = dict(model.get_summary())
            summary["segment_median"] = segment_median
            summaries.append(summary)
            selected_feature_kinds.append(str(kind_l))
            param_vectors.append(self._summary_to_vector_or_none(kind_l, summary))
            fitted_loss = -np.asarray(model.logpdf(values), dtype=float)
            fitted_step = float(np.mean(fitted_loss))
            if self.use_score_mode:
                if is_equality_feature and self.equality_score_mode == "gaussian_ll_gain":
                    local_gaussian = GaussianModel(
                        mu=float(np.mean(values)),
                        sigma=float(max(np.std(values), 1e-6)),
                    )
                    global_gaussian = GaussianModel(
                        mu=float(np.mean(F_demo[:, feat_idx])),
                        sigma=float(max(np.std(F_demo[:, feat_idx]), 1e-6)),
                    )
                    local_loss = -np.asarray(local_gaussian.logpdf(values), dtype=float)
                    global_loss = -np.asarray(global_gaussian.logpdf(values), dtype=float)
                    score = float(np.mean(local_loss) - np.mean(global_loss))
                else:
                    if kind_l in {
                        "margin_exp_lower", "marginexp", "margin_exp",
                        "margin_exp_lower_left_hn", "marginexp_left_hn", "margin_exp_left_hn",
                        "margin_exp_upper", "marginexp_upper", "margin_exp_upper",
                        "margin_exp_upper_right_hn", "marginexp_upper_right_hn", "margin_exp_upper_right_hn",
                    }:
                        baseline_model = self._fit_student_t_baseline(values)
                    else:
                        baseline_model = GaussianModel(
                            mu=float(np.mean(values)),
                            sigma=float(max(np.std(values), 1e-6)),
                        )
                    baseline_loss = -np.asarray(baseline_model.logpdf(values), dtype=float)
                    baseline_step = float(np.mean(baseline_loss))
                    ll_gain = baseline_step - fitted_step
                    score = -ll_gain
                feature_scores[feat_idx] = float(score)
                active_mask[feat_idx] = self._feature_active_from_score(feat_idx, score)
                weight = self.lambda_eq_constraint if is_equality_feature else self.lambda_ineq_constraint
                if is_equality_feature:
                    avg_step_cost = min(float(score) - float(self._equality_score_threshold()), 0.0)
                    feature_constraint_costs[feat_idx] = float(weight * len(values) * avg_step_cost)
                else:
                    avg_step_cost = min(float(score) - float(self.inequality_score_activation_threshold), 0.0)
                    feature_constraint_costs[feat_idx] = float(weight * len(values) * avg_step_cost)
            else:
                if kind_l in {
                    "margin_exp_lower", "marginexp", "margin_exp",
                    "margin_exp_lower_left_hn", "marginexp_left_hn", "margin_exp_left_hn",
                    "margin_exp_upper", "marginexp_upper", "margin_exp_upper",
                    "margin_exp_upper_right_hn", "marginexp_upper_right_hn", "margin_exp_upper_right_hn",
                }:
                    baseline_model = self._fit_student_t_baseline(values)
                else:
                    baseline_model = GaussianModel(
                        mu=float(np.mean(values)),
                        sigma=float(max(np.std(values), 1e-6)),
                    )
                baseline_loss = -np.asarray(baseline_model.logpdf(values), dtype=float)
                baseline_step = float(np.mean(baseline_loss))
                ll_gain = baseline_step - fitted_step
                if self.r[int(stage_idx), feat_idx]:
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
        result = (
            _StageParams(
                model_summaries=summaries,
                subgoal=subgoal,
                active_mask=active_mask,
                feature_scores=feature_scores,
                feature_constraint_costs=feature_constraint_costs,
                selected_feature_kinds=selected_feature_kinds,
                param_vectors=param_vectors,
            ),
            constraint_cost,
            progress_cost,
        )
        return result

    def _is_equality_feature(self, feat_idx: int) -> bool:
        return self._kind_is_equality(self.feature_model_types[feat_idx])

    def _equality_score_threshold(self) -> float:
        return float(self.equality_dispersion_ratio_threshold)

    def _equality_score_type(self) -> str:
        if self.equality_score_mode == "gaussian_ll_gain":
            return "gaussian_ll_gain"
        return "local_dispersion"

    def _build_score_threshold_matrix(self) -> np.ndarray:
        thresholds = np.zeros((self.num_stages, self.num_features), dtype=float)
        for feat_idx in range(self.num_features):
            thr = self._score_threshold_for_kind(self.feature_model_types[feat_idx])
            thresholds[:, feat_idx] = float(thr)
        return thresholds

    def _soft_activation_from_scores(self, scores) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=float)
        logits = (self.score_threshold_matrix - scores_arr) / float(self.activation_proto_temperature)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))

    def _hard_activation_from_scores(self, scores) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=float)
        return (scores_arr < self.score_threshold_matrix).astype(float)

    def _feature_active_from_score(self, feat_idx: int, score: float) -> int:
        threshold = self._score_threshold_for_kind(self.feature_model_types[feat_idx])
        return int(float(score) < float(threshold))

    def _compute_posthoc_activation_summary(self):
        if not self.demo_feature_score_matrices_:
            return None

        scores = np.asarray(self.demo_feature_score_matrices_, dtype=float)
        thresholds = np.asarray(self.score_threshold_matrix, dtype=float)
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
        for stage_idx in range(self.num_stages):
            stage_items = []
            for feat_idx in range(self.num_features):
                stage_items.append(
                    {
                        "feature_idx": int(feat_idx),
                        "feature_name": feature_names[feat_idx],
                        "score_type": "auto_margin" if self._is_auto_constraint_feature(feat_idx) else (self._equality_score_type() if self._is_equality_feature(feat_idx) else "-ll_gain"),
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

    def _compute_current_activation_rate_matrix(self):
        if self.use_score_mode and self.demo_feature_score_matrices_:
            scores = np.asarray(self.demo_feature_score_matrices_, dtype=float)
            activated = scores < self.score_threshold_matrix[None, :, :]
            return np.mean(activated.astype(float), axis=0)
        if self.demo_r_matrices_:
            masks = np.asarray(self.demo_r_matrices_, dtype=float)
            return np.mean(masks, axis=0)
        return np.zeros((self.num_stages, self.num_features), dtype=float)

    def _summary_to_vector(self, kind, summary):
        kind = str(kind).lower()
        if kind in {"gauss", "gaussian"}:
            center = float(summary.get("segment_median", summary["mu"]))
            return np.asarray([center, float(np.log(max(float(summary["sigma"]), 1e-12)))], dtype=float)
        if kind in {"student_t", "studentt", "t"}:
            center = float(summary.get("segment_median", summary["mu"]))
            return np.asarray([center, float(np.log(max(float(summary["sigma"]), 1e-12)))], dtype=float)
        if kind in {"zero_gauss", "zero_gaussian"}:
            center = float(summary.get("segment_median", 0.0))
            return np.asarray([center, float(np.log(max(float(summary["sigma"]), 1e-12)))], dtype=float)
        if kind in {"margin_exp_lower", "marginexp", "margin_exp"}:
            return np.asarray([float(summary["b"]), float(np.log(max(float(summary["lam"]), 1e-12)))], dtype=float)
        if kind in {"margin_exp_lower_left_hn", "marginexp_left_hn", "margin_exp_left_hn"}:
            pi_left = float(np.clip(float(summary["pi_left"]), 1e-6, 1.0 - 1e-6))
            return np.asarray(
                [
                    float(summary["b"]),
                    float(np.log(max(float(summary["lam"]), 1e-12))),
                    float(np.log(max(float(summary["sigma_left"]), 1e-12))),
                    float(np.log(pi_left / max(1.0 - pi_left, 1e-12))),
                ],
                dtype=float,
            )
        if kind in {"margin_exp_upper", "marginexp_upper", "margin_exp_upper"}:
            return np.asarray([float(summary["b"]), float(np.log(max(float(summary["lam"]), 1e-12)))], dtype=float)
        if kind in {"margin_exp_upper_right_hn", "marginexp_upper_right_hn", "margin_exp_upper_right_hn"}:
            pi_right = float(np.clip(float(summary["pi_right"]), 1e-6, 1.0 - 1e-6))
            return np.asarray(
                [
                    float(summary["b"]),
                    float(np.log(max(float(summary["lam"]), 1e-12))),
                    float(np.log(max(float(summary["sigma_right"]), 1e-12))),
                    float(np.log(pi_right / max(1.0 - pi_right, 1e-12))),
                ],
                dtype=float,
            )
        raise ValueError(f"Unsupported feature model type '{kind}'.")

    def _vector_to_model(self, kind, vec):
        kind = str(kind).lower()
        vec = np.asarray(vec, dtype=float).reshape(-1)
        if kind in {"gauss", "gaussian"}:
            return GaussianModel(mu=float(vec[0]), sigma=float(np.exp(vec[1])))
        if kind in {"student_t", "studentt", "t"}:
            return StudentTModel(mu=float(vec[0]), sigma=float(np.exp(vec[1])))
        if kind in {"zero_gauss", "zero_gaussian"}:
            sigma_idx = 1 if vec.shape[0] >= 2 else 0
            return ZeroMeanGaussianModel(sigma=float(np.exp(vec[sigma_idx])))
        if kind in {"margin_exp_lower", "marginexp", "margin_exp"}:
            return MarginExpLowerEmission(b_init=float(vec[0]), lam_init=float(np.exp(vec[1])))
        if kind in {"margin_exp_lower_left_hn", "marginexp_left_hn", "margin_exp_left_hn"}:
            pi_left = 1.0 / (1.0 + np.exp(-float(vec[3])))
            return MarginExpLowerLeftHNEmission(
                b_init=float(vec[0]),
                lam_init=float(np.exp(vec[1])),
                sigma_left_init=float(np.exp(vec[2])),
                pi_left_init=float(pi_left),
            )
        if kind in {"margin_exp_upper", "marginexp_upper", "margin_exp_upper"}:
            return MarginExpUpperEmission(b_init=float(vec[0]), lam_init=float(np.exp(vec[1])))
        if kind in {"margin_exp_upper_right_hn", "marginexp_upper_right_hn", "margin_exp_upper_right_hn"}:
            pi_right = 1.0 / (1.0 + np.exp(-float(vec[3])))
            return MarginExpUpperRightHNEmission(
                b_init=float(vec[0]),
                lam_init=float(np.exp(vec[1])),
                sigma_right_init=float(np.exp(vec[2])),
                pi_right_init=float(pi_right),
            )
        raise ValueError(f"Unsupported feature model type '{kind}'.")

    def _param_consensus_dims(self, kind):
        kind = str(kind).lower()
        if kind in {
            "gauss",
            "gaussian",
            "student_t",
            "studentt",
            "t",
            "zero_gauss",
            "zero_gaussian",
            "margin_exp_lower",
            "marginexp",
            "margin_exp",
            "margin_exp_lower_left_hn",
            "marginexp_left_hn",
            "margin_exp_left_hn",
            "margin_exp_upper",
            "marginexp_upper",
            "margin_exp_upper",
            "margin_exp_upper_right_hn",
            "marginexp_upper_right_hn",
            "margin_exp_upper_right_hn",
        }:
            return (0,)
        return None

    def _summary_to_vector_or_none(self, kind, summary):
        try:
            return self._summary_to_vector(kind, summary)
        except ValueError:
            return None

    def _stage_feature_vector(self, stage_params, feat_idx: int):
        vectors = getattr(stage_params, "param_vectors", None)
        if vectors is not None and int(feat_idx) < len(vectors):
            vec = vectors[int(feat_idx)]
            if vec is not None:
                return np.asarray(vec, dtype=float)
        kind = self._stage_feature_kind(stage_params, feat_idx)
        return self._summary_to_vector_or_none(kind, stage_params.model_summaries[feat_idx])

    def _subgoal_consensus_cost(self, candidate_stage_params, shared_stage_subgoals):
        total = 0.0
        for stage_idx in range(self.num_stages):
            diff = np.asarray(candidate_stage_params[stage_idx].subgoal, dtype=float) - np.asarray(shared_stage_subgoals[stage_idx], dtype=float)
            total += float(np.dot(diff, diff))
        return total

    def _param_consensus_cost(
        self,
        candidate_stage_params,
        shared_param_vectors,
        shared_feature_score_mean=None,
        shared_r_mean=None,
    ):
        total = 0.0
        for stage_idx in range(self.num_stages):
            for feat_idx, _ in enumerate(self.feature_model_types):
                if self._is_auto_constraint_feature(feat_idx):
                    continue
                shared_active = None
                if shared_feature_score_mean is not None:
                    shared_active = int(np.rint(float(shared_feature_score_mean[stage_idx, feat_idx])))
                elif shared_r_mean is not None:
                    shared_active = int(np.rint(float(shared_r_mean[stage_idx, feat_idx])))
                if shared_active is not None and shared_active != 1:
                    continue
                active_mask = candidate_stage_params[stage_idx].active_mask
                if active_mask is None or not int(active_mask[feat_idx]):
                    continue
                kind = self._stage_feature_kind(candidate_stage_params[stage_idx], feat_idx)
                local_vec = self._stage_feature_vector(candidate_stage_params[stage_idx], feat_idx)
                if local_vec is None:
                    continue
                shared_vec = shared_param_vectors[stage_idx][feat_idx]
                if shared_vec is None:
                    continue
                dims = self._param_consensus_dims(kind)
                if dims is not None:
                    local_vec = local_vec[list(dims)]
                    shared_vec = np.asarray(shared_vec, dtype=float)[list(dims)]
                delta = local_vec - shared_vec
                total += float(np.dot(delta, delta))
        return total

    def _r_consensus_cost(self, candidate_stage_params, shared_r_mean):
        if shared_r_mean is None:
            return 0.0
        total = 0.0
        for stage_idx in range(self.num_stages):
            active_mask = candidate_stage_params[stage_idx].active_mask
            if active_mask is None:
                continue
            delta = np.asarray(active_mask, dtype=float) - np.asarray(shared_r_mean[stage_idx], dtype=float)
            total += float(np.dot(delta, delta))
        return total

    def _activation_consensus_cost(self, candidate_stage_params, shared_feature_score_mean):
        if not self.use_score_mode or shared_feature_score_mean is None:
            return 0.0
        local_scores = np.stack(
            [np.asarray(stage_params.feature_scores, dtype=float) for stage_params in candidate_stage_params],
            axis=0,
        )
        local_activation = self._hard_activation_from_scores(local_scores)
        total = 0.0
        for stage_idx in range(self.num_stages):
            delta = np.asarray(local_activation[stage_idx], dtype=float) - np.asarray(shared_feature_score_mean[stage_idx], dtype=float)
            total += float(np.dot(delta, delta))
        return total

    def _hard_activation_match(self, local_activation, shared_feature_score_mean, stage_idx=None):
        if shared_feature_score_mean is None:
            return True
        local_arr = np.asarray(local_activation, dtype=float)
        shared_arr = np.asarray(shared_feature_score_mean, dtype=float)
        if stage_idx is not None:
            return bool(np.array_equal(np.rint(local_arr).astype(int), np.rint(shared_arr[int(stage_idx)]).astype(int)))
        return bool(np.array_equal(np.rint(local_arr).astype(int), np.rint(shared_arr).astype(int)))

    def _segment_stage_cost_info(
        self,
        demo_idx,
        stage_idx,
        s,
        e,
        lam_subgoal_consensus,
        lam_param_consensus,
        lam_activation_consensus,
        shared_stage_subgoals,
        shared_param_vectors,
        shared_r_mean,
        shared_feature_score_mean=None,
    ):
        stage_len = int(e - s + 1)
        if stage_len < int(self.duration_min[stage_idx]) or stage_len > int(self.duration_max[stage_idx]):
            return None
        stage_params, constraint_cost, progress_cost = self._fit_segment_stage(demo_idx, stage_idx, s, e)
        short_segment_penalty = 0.0
        if self.use_score_mode and self.has_equality_feature:
            short_segment_penalty = float(self.short_segment_penalty_c / np.sqrt(max(stage_len, 1)))
        subgoal_consensus_cost = 0.0
        if lam_subgoal_consensus > 0.0:
            diff = np.asarray(stage_params.subgoal, dtype=float) - np.asarray(shared_stage_subgoals[stage_idx], dtype=float)
            subgoal_consensus_cost = float(np.dot(diff, diff))
        param_consensus_cost = 0.0
        if lam_param_consensus > 0.0:
            for feat_idx, _ in enumerate(self.feature_model_types):
                if self._is_auto_constraint_feature(feat_idx):
                    continue
                active_mask = stage_params.active_mask
                if active_mask is None or not int(active_mask[feat_idx]):
                    continue
                shared_active = None
                if shared_feature_score_mean is not None:
                    shared_active = int(np.rint(float(shared_feature_score_mean[stage_idx, feat_idx])))
                elif shared_r_mean is not None:
                    shared_active = int(np.rint(float(shared_r_mean[stage_idx, feat_idx])))
                if shared_active is not None and shared_active != 1:
                    continue
                shared_vec = shared_param_vectors[stage_idx][feat_idx]
                if shared_vec is None:
                    continue
                kind = self._stage_feature_kind(stage_params, feat_idx)
                local_vec = self._stage_feature_vector(stage_params, feat_idx)
                if local_vec is None:
                    continue
                dims = self._param_consensus_dims(kind)
                if dims is not None:
                    local_vec = local_vec[list(dims)]
                    shared_vec = np.asarray(shared_vec, dtype=float)[list(dims)]
                delta = local_vec - shared_vec
                param_consensus_cost += float(np.dot(delta, delta))
        activation_consensus_cost = 0.0
        if self.use_score_mode:
            scores = stage_params.feature_scores
            if scores is not None and shared_feature_score_mean is not None:
                local_activation = self._hard_activation_from_scores(
                    np.asarray(scores, dtype=float)[None, :]
                )[0]
                if self.use_joint_mask_search and not self._hard_activation_match(
                    local_activation,
                    shared_feature_score_mean,
                    stage_idx=stage_idx,
                ):
                    return None
                if lam_activation_consensus > 0.0:
                    delta = np.asarray(local_activation, dtype=float) - np.asarray(shared_feature_score_mean[stage_idx], dtype=float)
                    activation_consensus_cost = float(np.dot(delta, delta))
        elif lam_activation_consensus > 0.0:
            active_mask = stage_params.active_mask
            if active_mask is not None and shared_r_mean is not None:
                    delta = np.asarray(active_mask, dtype=float) - np.asarray(shared_r_mean[stage_idx], dtype=float)
                    activation_consensus_cost = float(np.dot(delta, delta))
        weighted_total = (
            float(constraint_cost)
            + float(short_segment_penalty)
            + self.lambda_progress * float(progress_cost)
            + lam_subgoal_consensus * float(subgoal_consensus_cost)
            + lam_param_consensus * float(param_consensus_cost)
            + lam_activation_consensus * float(activation_consensus_cost)
        )
        return {
            "stage_idx": int(stage_idx),
            "s": int(s),
            "e": int(e),
            "stage_params": stage_params,
            "constraint": float(constraint_cost),
            "short_segment_penalty": float(short_segment_penalty),
            "progress": float(progress_cost),
            "subgoal_consensus": float(subgoal_consensus_cost),
            "param_consensus": float(param_consensus_cost),
            "activation_consensus": float(activation_consensus_cost),
            "weighted_total": float(weighted_total),
        }

    def _best_segmentation_info(
        self,
        demo_idx,
        lam_subgoal_consensus,
        lam_param_consensus,
        lam_activation_consensus,
        shared_stage_subgoals,
        shared_param_vectors,
        shared_r_mean,
        shared_feature_score_mean=None,
        fixed_cutpoints_by_stage=None,
    ):
        X = self.demos[demo_idx]
        T = len(X)
        normalized_fixed_cutpoints = {}
        if fixed_cutpoints_by_stage:
            for stage_idx, cutpoint in dict(fixed_cutpoints_by_stage).items():
                stage_idx_i = int(stage_idx)
                cutpoint_i = int(cutpoint)
                if stage_idx_i < 0 or stage_idx_i >= self.num_stages - 1:
                    continue
                if cutpoint_i < 0 or cutpoint_i >= T - 1:
                    continue
                normalized_fixed_cutpoints[stage_idx_i] = cutpoint_i
        suffix_min = np.zeros(self.num_stages + 1, dtype=int)
        suffix_max = np.zeros(self.num_stages + 1, dtype=int)
        for k in range(self.num_stages - 1, -1, -1):
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
                    lam_activation_consensus=lam_activation_consensus,
                    shared_stage_subgoals=shared_stage_subgoals,
                    shared_param_vectors=shared_param_vectors,
                    shared_r_mean=shared_r_mean,
                    shared_feature_score_mean=shared_feature_score_mean,
                )
            return cache[key]

        inf = float("inf")
        best = np.full((self.num_stages, T), inf, dtype=float)
        back = np.full((self.num_stages, T), -1, dtype=int)

        for stage_idx in range(self.num_stages):
            for e in range(T):
                fixed_e = normalized_fixed_cutpoints.get(int(stage_idx))
                if fixed_e is not None and int(e) != int(fixed_e):
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
        if not np.isfinite(best[self.num_stages - 1, final_end]):
            raise RuntimeError(f"No feasible segmentation found for demo {demo_idx}.")

        stage_ends = [final_end]
        cur_end = final_end
        for stage_idx in range(self.num_stages - 1, 0, -1):
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
            "short_segment_penalty": float(sum(info.get("short_segment_penalty", 0.0) for info in stage_infos)),
            "progress": float(sum(info["progress"] for info in stage_infos)),
            "subgoal_consensus": float(sum(info["subgoal_consensus"] for info in stage_infos)),
            "param_consensus": float(sum(info["param_consensus"] for info in stage_infos)),
            "activation_consensus": float(sum(info["activation_consensus"] for info in stage_infos)),
            "total": float(best[self.num_stages - 1, final_end]),
        }

    def _candidate_cost(
        self,
        demo_idx,
        stage_ends,
        lam_subgoal_consensus,
        lam_param_consensus,
        lam_activation_consensus,
        shared_stage_subgoals,
        shared_param_vectors,
        shared_r_mean,
        shared_feature_score_mean=None,
    ):
        bounds = self._segment_bounds_from_stage_ends(stage_ends)
        candidate_stage_params = []
        constraint_cost = 0.0
        short_segment_penalty = 0.0
        progress_cost = 0.0
        for stage_idx, (s, e) in enumerate(bounds):
            stage_len = int(e - s + 1)
            if stage_len < int(self.duration_min[stage_idx]) or stage_len > int(self.duration_max[stage_idx]):
                return None
            stage_params, c_cost, p_cost = self._fit_segment_stage(demo_idx, stage_idx, s, e)
            candidate_stage_params.append(stage_params)
            constraint_cost += c_cost
            if self.use_score_mode and self.has_equality_feature:
                short_segment_penalty += float(self.short_segment_penalty_c / np.sqrt(max(stage_len, 1)))
            progress_cost += p_cost
        subgoal_consensus_cost = 0.0
        if lam_subgoal_consensus > 0.0:
            subgoal_consensus_cost = self._subgoal_consensus_cost(candidate_stage_params, shared_stage_subgoals)
        param_consensus_cost = 0.0
        if lam_param_consensus > 0.0:
            param_consensus_cost = self._param_consensus_cost(
                candidate_stage_params,
                shared_param_vectors,
                shared_feature_score_mean=shared_feature_score_mean,
                shared_r_mean=shared_r_mean,
            )
        activation_consensus_cost = 0.0
        if self.use_score_mode and self.use_joint_mask_search and shared_feature_score_mean is not None:
            local_scores = np.stack(
                [np.asarray(stage_params.feature_scores, dtype=float) for stage_params in candidate_stage_params],
                axis=0,
            )
            local_activation = self._hard_activation_from_scores(local_scores)
            if not self._hard_activation_match(local_activation, shared_feature_score_mean):
                return None
        if lam_activation_consensus > 0.0:
            if self.use_score_mode:
                activation_consensus_cost = self._activation_consensus_cost(candidate_stage_params, shared_feature_score_mean)
            else:
                activation_consensus_cost = self._r_consensus_cost(candidate_stage_params, shared_r_mean)
        total = (
            constraint_cost
            + short_segment_penalty
            + self.lambda_progress * progress_cost
            + lam_subgoal_consensus * subgoal_consensus_cost
            + lam_param_consensus * param_consensus_cost
            + lam_activation_consensus * activation_consensus_cost
        )
        return {
            "cutpoints": [int(x) for x in stage_ends[:-1]],
            "stage_ends": [int(x) for x in stage_ends],
            "stage_params": candidate_stage_params,
            "constraint": float(constraint_cost),
            "short_segment_penalty": float(short_segment_penalty),
            "progress": float(progress_cost),
            "subgoal_consensus": float(subgoal_consensus_cost),
            "param_consensus": float(param_consensus_cost),
            "activation_consensus": float(activation_consensus_cost),
            "total": float(total),
        }

    def _enumerate_shared_activation_masks(self):
        if self.fixed_feature_mask is not None:
            yield np.asarray(self.r, dtype=float).copy()
            return
        n_bits = int(self.num_stages * self.num_features)
        total_masks = 1 << n_bits
        if total_masks > int(self.joint_mask_search_max_masks):
            raise ValueError(
                "joint_mask_search would enumerate "
                f"{total_masks} shared masks for shape {(self.num_stages, self.num_features)}, "
                f"which exceeds joint_mask_search_max_masks={self.joint_mask_search_max_masks}."
            )
        bit_ids = np.arange(n_bits, dtype=np.uint64)
        for mask_id in range(total_masks):
            bits = ((np.uint64(mask_id) >> bit_ids) & np.uint64(1)).astype(float)
            yield bits.reshape(self.num_stages, self.num_features)

    def _majority_activation_mask_from_infos(self, selected_infos):
        if not selected_infos:
            return np.zeros((self.num_stages, self.num_features), dtype=float)
        activation_mats = [
            np.stack([np.asarray(stage_params.active_mask, dtype=float) for stage_params in info["stage_params"]], axis=0)
            for info in selected_infos
        ]
        return (np.mean(np.stack(activation_mats, axis=0), axis=0) > 0.5).astype(float)

    def _select_infos_with_shared_activation_mask(
        self,
        lam_subgoal_consensus,
        lam_param_consensus,
        lam_activation_consensus,
        shared_stage_subgoals,
        shared_param_vectors,
        shared_mask,
    ):
        selected_infos = []
        total = 0.0
        for demo_idx in range(len(self.demos)):
            fixed_cutpoints_by_stage = self._fixed_cutpoint_map_for_demo(demo_idx)
            try:
                info = self._best_segmentation_info(
                    demo_idx=demo_idx,
                    lam_subgoal_consensus=lam_subgoal_consensus,
                    lam_param_consensus=lam_param_consensus,
                    lam_activation_consensus=lam_activation_consensus,
                    shared_stage_subgoals=shared_stage_subgoals,
                    shared_param_vectors=shared_param_vectors,
                    shared_r_mean=None,
                    shared_feature_score_mean=shared_mask,
                    fixed_cutpoints_by_stage=fixed_cutpoints_by_stage,
                )
            except RuntimeError:
                return None, np.inf
            selected_infos.append(info)
            total += float(info["total"])
        return selected_infos, float(total)

    def _best_joint_activation_mask_selection(
        self,
        lam_subgoal_consensus,
        lam_param_consensus,
        lam_activation_consensus,
        shared_stage_subgoals,
        shared_param_vectors,
    ):
        if lam_activation_consensus <= 0.0 and self.fixed_feature_mask is None:
            selected_infos, _ = self._select_infos_with_shared_activation_mask(
                lam_subgoal_consensus=lam_subgoal_consensus,
                lam_param_consensus=lam_param_consensus,
                lam_activation_consensus=lam_activation_consensus,
                shared_stage_subgoals=shared_stage_subgoals,
                shared_param_vectors=shared_param_vectors,
                shared_mask=np.zeros((self.num_stages, self.num_features), dtype=float),
            )
            return selected_infos, self._majority_activation_mask_from_infos(selected_infos)

        best_total = np.inf
        best_infos = None
        best_mask = None
        for shared_mask in self._enumerate_shared_activation_masks():
            selected_infos, total = self._select_infos_with_shared_activation_mask(
                lam_subgoal_consensus=lam_subgoal_consensus,
                lam_param_consensus=lam_param_consensus,
                lam_activation_consensus=lam_activation_consensus,
                shared_stage_subgoals=shared_stage_subgoals,
                shared_param_vectors=shared_param_vectors,
                shared_mask=shared_mask,
            )
            if selected_infos is None:
                continue
            if float(total) < float(best_total):
                best_total = float(total)
                best_infos = selected_infos
                best_mask = np.asarray(shared_mask, dtype=float).copy()
        if best_infos is None or best_mask is None:
            if self.fixed_feature_mask is not None:
                if not self._joint_mask_fallback_warned:
                    print(
                        "[SWCL] warning: fixed_feature_mask is infeasible under hard joint activation matching; "
                        "falling back to the minimum-cost segmentation without activation matching."
                    )
                    self._joint_mask_fallback_warned = True
                fallback_infos = []
                for demo_idx in range(len(self.demos)):
                    fixed_cutpoints_by_stage = self._fixed_cutpoint_map_for_demo(demo_idx)
                    fallback_infos.append(
                        self._best_segmentation_info(
                            demo_idx=demo_idx,
                            lam_subgoal_consensus=lam_subgoal_consensus,
                            lam_param_consensus=lam_param_consensus,
                            lam_activation_consensus=0.0,
                            shared_stage_subgoals=shared_stage_subgoals,
                            shared_param_vectors=shared_param_vectors,
                            shared_r_mean=None,
                            shared_feature_score_mean=None,
                            fixed_cutpoints_by_stage=fixed_cutpoints_by_stage,
                        )
                    )
                return fallback_infos, np.asarray(self.r, dtype=float).copy()
            raise RuntimeError("joint_mask_search failed to find a feasible shared activation mask.")
        return best_infos, best_mask

    def _shared_from_selected(self, selected_infos, shared_activation_mask=None):
        shared_stage_subgoals = []
        for stage_idx in range(self.num_stages):
            stage_subgoals = np.asarray(
                [info["stage_params"][stage_idx].subgoal for info in selected_infos],
                dtype=float,
            )
            shared_stage_subgoals.append(_geometric_median(stage_subgoals))
        if shared_activation_mask is None:
            shared_activation_mask = self._majority_activation_mask_from_infos(selected_infos)
        shared_activation_mask = np.asarray(shared_activation_mask, dtype=float)
        shared_param_vectors = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        shared_param_kinds = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        for stage_idx in range(self.num_stages):
            for feat_idx, _ in enumerate(self.feature_model_types):
                if int(np.rint(float(shared_activation_mask[stage_idx, feat_idx]))) != 1:
                    continue
                active_stage_params = [
                    info["stage_params"][stage_idx]
                    for info in selected_infos
                    if info["stage_params"][stage_idx].active_mask is not None
                    and int(info["stage_params"][stage_idx].active_mask[feat_idx]) == 1
                ]
                if not active_stage_params:
                    continue
                if self._is_auto_constraint_feature(feat_idx):
                    kind_counts = {}
                    for stage_params in active_stage_params:
                        kind = self._stage_feature_kind(stage_params, feat_idx)
                        kind_counts[kind] = kind_counts.get(kind, 0) + 1
                    chosen_kind = sorted(kind_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                else:
                    chosen_kind = str(self.feature_model_types[feat_idx]).lower()
                vectors = [
                    self._stage_feature_vector(stage_params, feat_idx)
                    for stage_params in active_stage_params
                    if self._stage_feature_kind(stage_params, feat_idx) == chosen_kind
                ]
                vectors = [np.asarray(vec, dtype=float) for vec in vectors if vec is not None]
                if not vectors:
                    continue
                shared_param_kinds[stage_idx][feat_idx] = str(chosen_kind)
                shared_param_vectors[stage_idx][feat_idx] = np.median(np.stack(vectors, axis=0), axis=0)
        return shared_stage_subgoals, shared_param_vectors, shared_param_kinds

    def _apply_shared_state(self, shared_stage_subgoals, shared_param_vectors, shared_param_kinds=None):
        self.shared_stage_subgoals = [np.asarray(x, dtype=float).copy() for x in shared_stage_subgoals]
        self.shared_param_vectors = [[None if v is None else np.asarray(v, dtype=float).copy() for v in row] for row in shared_param_vectors]
        if shared_param_kinds is None:
            shared_param_kinds = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        self.shared_param_kinds = [[None if k is None else str(k) for k in row] for row in shared_param_kinds]
        self.stage_subgoals = [np.asarray(x, dtype=float).copy() for x in shared_stage_subgoals]
        self.stage_subgoals_hist.append([x.copy() for x in self.stage_subgoals])
        if self.num_stages >= 1:
            self.g1 = self.stage_subgoals[0].copy()
            self.g1_hist.append(self.g1.copy())
        if self.num_stages >= 2:
            self.g2 = self.stage_subgoals[1].copy()
            self.g2_hist.append(self.g2.copy())
        for stage_idx in range(self.num_stages):
            for feat_idx, base_kind in enumerate(self.feature_model_types):
                vec = shared_param_vectors[stage_idx][feat_idx]
                if vec is None:
                    continue
                kind = shared_param_kinds[stage_idx][feat_idx] or str(base_kind).lower()
                if self._kind_is_auto(kind) or kind == "unconstrained":
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
        self.shared_activation_proto = None
        self.demo_activation_matrices_ = []
        self.demo_activation_history = []
        self.activation_proto_history = []
        self.stage_subgoals_hist = []
        self.g1_hist = []
        self.g2_hist = []
        shared_stage_subgoals = [np.zeros(self.state_dim, dtype=float) for _ in range(self.num_stages)]
        shared_param_vectors = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        shared_param_kinds = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        shared_activation_mask = None
        self._prepare_segment_stage_cache()

        for iteration in range(int(max_iter)):
            lam_subgoal_consensus = self._current_scheduled_lambda(self.lambda_subgoal_consensus, iteration, int(max_iter))
            lam_param_consensus = self._current_scheduled_lambda(self.lambda_param_consensus, iteration, int(max_iter))
            lam_activation_consensus = self._current_scheduled_lambda(
                self.lambda_activation_consensus,
                iteration,
                int(max_iter),
            )
            self.subgoal_consensus_lambda_hist.append(float(lam_subgoal_consensus))
            self.param_consensus_lambda_hist.append(float(lam_param_consensus))
            self.activation_consensus_lambda_hist.append(float(lam_activation_consensus))
            self.current_subgoal_consensus_lambda = float(lam_subgoal_consensus)
            self.current_param_consensus_lambda = float(lam_param_consensus)
            self.current_activation_consensus_lambda = float(lam_activation_consensus)
            if self.use_joint_mask_search:
                selected_infos, shared_activation_mask = self._best_joint_activation_mask_selection(
                    lam_subgoal_consensus=lam_subgoal_consensus,
                    lam_param_consensus=lam_param_consensus,
                    lam_activation_consensus=lam_activation_consensus,
                    shared_stage_subgoals=shared_stage_subgoals,
                    shared_param_vectors=shared_param_vectors,
                )
            else:
                selected_infos = []
                for demo_idx in range(len(self.demos)):
                    fixed_cutpoints_by_stage = self._fixed_cutpoint_map_for_demo(demo_idx)
                    selected_infos.append(
                        self._best_segmentation_info(
                            demo_idx=demo_idx,
                            lam_subgoal_consensus=lam_subgoal_consensus,
                            lam_param_consensus=lam_param_consensus,
                            lam_activation_consensus=lam_activation_consensus,
                            shared_stage_subgoals=shared_stage_subgoals,
                            shared_param_vectors=shared_param_vectors,
                            shared_r_mean=self.shared_r_mean,
                            shared_feature_score_mean=self.shared_feature_score_mean,
                            fixed_cutpoints_by_stage=fixed_cutpoints_by_stage,
                        )
                    )

            self.stage_ends_ = [list(info["stage_ends"]) for info in selected_infos]
            self.current_stage_params_per_demo = [list(info["stage_params"]) for info in selected_infos]
            self.current_demo_cost_breakdown = [
                {
                    "constraint": float(info["constraint"]),
                    "short_segment_penalty": float(info.get("short_segment_penalty", 0.0)),
                    "progress": self.lambda_progress * float(info["progress"]),
                    "subgoal_consensus": lam_subgoal_consensus * float(info["subgoal_consensus"]),
                    "param_consensus": lam_param_consensus * float(info["param_consensus"]),
                    "activation_consensus": lam_activation_consensus * float(info.get("activation_consensus", 0.0)),
                    "total": float(info["total"]),
                }
                for info in selected_infos
            ]
            self.demo_r_matrices_ = [
                np.stack([stage_params.active_mask for stage_params in info["stage_params"]], axis=0)
                for info in selected_infos
            ]
            if not self.use_score_mode and self.demo_r_matrices_:
                self.shared_r_mean = np.mean(np.stack(self.demo_r_matrices_, axis=0), axis=0)
            else:
                self.shared_r_mean = None
            self.demo_feature_score_matrices_ = [
                np.stack([stage_params.feature_scores for stage_params in info["stage_params"]], axis=0)
                for info in selected_infos
            ]
            if self.use_score_mode and self.demo_feature_score_matrices_:
                self.demo_activation_matrices_ = [
                    self._hard_activation_from_scores(score_mat) for score_mat in self.demo_feature_score_matrices_
                ]
                activation_mats = np.stack(self.demo_activation_matrices_, axis=0)
                if self.use_joint_mask_search:
                    if shared_activation_mask is None:
                        shared_activation_mask = self._majority_activation_mask_from_infos(selected_infos)
                    self.shared_feature_score_mean = np.asarray(shared_activation_mask, dtype=float).copy()
                else:
                    self.shared_feature_score_mean = self._majority_activation_mask_from_infos(selected_infos)
                self.r = np.asarray(np.rint(self.shared_feature_score_mean), dtype=int)
                self.shared_activation_proto = np.asarray(self.shared_feature_score_mean, dtype=float).copy()
                self.demo_activation_history.append(np.asarray(activation_mats, dtype=float).copy())
                self.activation_proto_history.append(np.asarray(self.shared_activation_proto, dtype=float).copy())
                self.posthoc_activation_summary_ = self._compute_posthoc_activation_summary()
            self.activation_rate_history.append(np.asarray(self._compute_current_activation_rate_matrix(), dtype=float))
            self.segmentation_history.append([list(item) for item in self.stage_ends_])
            shared_activation_mask_for_update = self.shared_feature_score_mean if self.use_score_mode else self.shared_r_mean
            shared_stage_subgoals, shared_param_vectors, shared_param_kinds = self._shared_from_selected(
                selected_infos,
                shared_activation_mask=shared_activation_mask_for_update,
            )
            self._apply_shared_state(shared_stage_subgoals, shared_param_vectors, shared_param_kinds)

            total_constraint = float(np.sum([info["constraint"] for info in selected_infos]))
            total_short_segment_penalty = float(np.sum([info.get("short_segment_penalty", 0.0) for info in selected_infos]))
            total_progress = float(np.sum([info["progress"] for info in selected_infos]))
            total_subgoal_consensus = float(np.sum([info["subgoal_consensus"] for info in selected_infos]))
            total_param_consensus = float(np.sum([info["param_consensus"] for info in selected_infos]))
            total_activation_consensus = float(np.sum([info.get("activation_consensus", 0.0) for info in selected_infos]))
            total_loss = float(
                total_constraint
                + total_short_segment_penalty
                + self.lambda_progress * total_progress
                + lam_subgoal_consensus * total_subgoal_consensus
                + lam_param_consensus * total_param_consensus
                + lam_activation_consensus * total_activation_consensus
            )
            self.loss_constraint.append(total_constraint)
            self.loss_short_segment_penalty.append(total_short_segment_penalty)
            self.loss_progress.append(total_progress)
            self.loss_subgoal_consensus.append(total_subgoal_consensus)
            self.loss_param_consensus.append(total_param_consensus)
            self.loss_activation_consensus.append(total_activation_consensus)
            self.loss_total.append(total_loss)

            gammas = _hard_gammas_from_stage_ends([len(X) for X in self.demos], self.stage_ends_, self.num_stages)
            metrics = self.eval_fn(self, gammas, None) if self.eval_fn is not None and self.num_stages == 2 else {}
            for name, value in metrics.items():
                if np.isscalar(value):
                    value_f = float(value)
                    if np.isfinite(value_f):
                        self.metrics_hist.setdefault(name, []).append(value_f)

            if verbose:
                print(
                    format_training_log(
                        "swcl",
                        iteration,
                        losses={
                            "total": total_loss,
                            "constraint": total_constraint,
                            "short_segment_penalty": total_short_segment_penalty,
                            "progress": total_progress,
                            "subgoal_consensus": total_subgoal_consensus,
                            "param_consensus": total_param_consensus,
                            "activation_consensus": total_activation_consensus,
                        },
                        metrics=metrics,
                        extras={
                            "lam_subgoal_consensus": f"{lam_subgoal_consensus:.3f}",
                            "lam_param_consensus": f"{lam_param_consensus:.3f}",
                            "lam_activation_consensus": f"{lam_activation_consensus:.3f}",
                        },
                    )
                )
            if self.plot_every is not None and (iteration + 1) % int(self.plot_every) == 0:
                pass

        should_final_resegment = int(max_iter) > 0 and (
            self.lambda_subgoal_consensus > 0.0
            or self.lambda_param_consensus > 0.0
            or (self.use_score_mode and self.lambda_activation_consensus > 0.0)
            or self.use_joint_mask_search
        )
        if should_final_resegment:
            if self.use_joint_mask_search:
                final_selected_infos, shared_activation_mask = self._best_joint_activation_mask_selection(
                    lam_subgoal_consensus=self.current_subgoal_consensus_lambda,
                    lam_param_consensus=self.current_param_consensus_lambda,
                    lam_activation_consensus=self.current_activation_consensus_lambda,
                    shared_stage_subgoals=self.shared_stage_subgoals,
                    shared_param_vectors=self.shared_param_vectors,
                )
            else:
                final_selected_infos = []
                for demo_idx in range(len(self.demos)):
                    fixed_cutpoints_by_stage = self._fixed_cutpoint_map_for_demo(demo_idx)
                    final_selected_infos.append(
                        self._best_segmentation_info(
                            demo_idx=demo_idx,
                            lam_subgoal_consensus=self.current_subgoal_consensus_lambda,
                            lam_param_consensus=self.current_param_consensus_lambda,
                            lam_activation_consensus=self.current_activation_consensus_lambda,
                            shared_stage_subgoals=self.shared_stage_subgoals,
                            shared_param_vectors=self.shared_param_vectors,
                            shared_r_mean=self.shared_r_mean,
                            shared_feature_score_mean=self.shared_feature_score_mean,
                            fixed_cutpoints_by_stage=fixed_cutpoints_by_stage,
                        )
                    )

            self.stage_ends_ = [list(info["stage_ends"]) for info in final_selected_infos]
            self.current_stage_params_per_demo = [list(info["stage_params"]) for info in final_selected_infos]
            self.current_demo_cost_breakdown = [
                {
                    "constraint": float(info["constraint"]),
                    "short_segment_penalty": float(info.get("short_segment_penalty", 0.0)),
                    "progress": self.lambda_progress * float(info["progress"]),
                    "subgoal_consensus": self.current_subgoal_consensus_lambda * float(info["subgoal_consensus"]),
                    "param_consensus": self.current_param_consensus_lambda * float(info["param_consensus"]),
                    "activation_consensus": self.current_activation_consensus_lambda * float(info.get("activation_consensus", 0.0)),
                    "total": float(info["total"]),
                }
                for info in final_selected_infos
            ]
            self.demo_r_matrices_ = [
                np.stack([stage_params.active_mask for stage_params in info["stage_params"]], axis=0)
                for info in final_selected_infos
            ]
            if not self.use_score_mode and self.demo_r_matrices_:
                self.shared_r_mean = np.mean(np.stack(self.demo_r_matrices_, axis=0), axis=0)
            else:
                self.shared_r_mean = None
            self.demo_feature_score_matrices_ = [
                np.stack([stage_params.feature_scores for stage_params in info["stage_params"]], axis=0)
                for info in final_selected_infos
            ]
            if self.use_score_mode and self.demo_feature_score_matrices_:
                self.demo_activation_matrices_ = [
                    self._hard_activation_from_scores(score_mat) for score_mat in self.demo_feature_score_matrices_
                ]
                activation_mats = np.stack(self.demo_activation_matrices_, axis=0)
                if self.use_joint_mask_search:
                    if shared_activation_mask is None:
                        shared_activation_mask = self._majority_activation_mask_from_infos(final_selected_infos)
                    self.shared_feature_score_mean = np.asarray(shared_activation_mask, dtype=float).copy()
                else:
                    self.shared_feature_score_mean = self._majority_activation_mask_from_infos(final_selected_infos)
                self.r = np.asarray(np.rint(self.shared_feature_score_mean), dtype=int)
                self.shared_activation_proto = np.asarray(self.shared_feature_score_mean, dtype=float).copy()
                if self.demo_activation_history:
                    self.demo_activation_history[-1] = np.asarray(activation_mats, dtype=float).copy()
                else:
                    self.demo_activation_history.append(np.asarray(activation_mats, dtype=float).copy())
                if self.activation_proto_history:
                    self.activation_proto_history[-1] = np.asarray(self.shared_activation_proto, dtype=float).copy()
                else:
                    self.activation_proto_history.append(np.asarray(self.shared_activation_proto, dtype=float).copy())
                self.posthoc_activation_summary_ = self._compute_posthoc_activation_summary()
            if self.activation_rate_history:
                self.activation_rate_history[-1] = np.asarray(self._compute_current_activation_rate_matrix(), dtype=float)
            else:
                self.activation_rate_history.append(np.asarray(self._compute_current_activation_rate_matrix(), dtype=float))
            shared_activation_mask_for_update = self.shared_feature_score_mean if self.use_score_mode else self.shared_r_mean
            final_shared_stage_subgoals, final_shared_param_vectors, final_shared_param_kinds = self._shared_from_selected(
                final_selected_infos,
                shared_activation_mask=shared_activation_mask_for_update,
            )
            self._apply_shared_state(final_shared_stage_subgoals, final_shared_param_vectors, final_shared_param_kinds)

            final_total_constraint = float(np.sum([info["constraint"] for info in final_selected_infos]))
            final_total_short_segment_penalty = float(np.sum([info.get("short_segment_penalty", 0.0) for info in final_selected_infos]))
            final_total_progress = float(np.sum([info["progress"] for info in final_selected_infos]))
            final_total_subgoal_consensus = float(np.sum([info["subgoal_consensus"] for info in final_selected_infos]))
            final_total_param_consensus = float(np.sum([info["param_consensus"] for info in final_selected_infos]))
            final_total_activation_consensus = float(np.sum([info.get("activation_consensus", 0.0) for info in final_selected_infos]))
            final_total_loss = float(
                final_total_constraint
                + final_total_short_segment_penalty
                + self.lambda_progress * final_total_progress
                + self.current_subgoal_consensus_lambda * final_total_subgoal_consensus
                + self.current_param_consensus_lambda * final_total_param_consensus
                + self.current_activation_consensus_lambda * final_total_activation_consensus
            )
            self.loss_constraint[-1] = final_total_constraint
            self.loss_short_segment_penalty[-1] = final_total_short_segment_penalty
            self.loss_progress[-1] = final_total_progress
            self.loss_subgoal_consensus[-1] = final_total_subgoal_consensus
            self.loss_param_consensus[-1] = final_total_param_consensus
            self.loss_activation_consensus[-1] = final_total_activation_consensus
            self.loss_total[-1] = final_total_loss
            if self.num_stages == 2 and self.eval_fn is not None:
                final_gammas = _hard_gammas_from_stage_ends([len(X) for X in self.demos], self.stage_ends_, self.num_stages)
                final_metrics = self.eval_fn(self, final_gammas, None)
                for name, value in final_metrics.items():
                    if np.isscalar(value):
                        value_f = float(value)
                        if np.isfinite(value_f) and self.metrics_hist.get(name):
                            self.metrics_hist[name][-1] = value_f

        if self.plot_every is not None:
            final_it = int(max_iter)
            if self.use_score_mode:
                plot_swcl_activation_dynamics(self, final_it)
            for demo_idx in range(len(self.demos)):
                plot_swcl_results_4panel(self, final_it, demo_idx=demo_idx)

        return _hard_gammas_from_stage_ends([len(X) for X in self.demos], self.stage_ends_, self.num_stages)
