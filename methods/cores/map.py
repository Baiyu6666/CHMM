from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ModuleNotFoundError:
    plt = None
    Line2D = None
    Patch = None

from evaluation import evaluate_model_metrics
from methods.base import format_training_log
from methods.cores.swcl import (
    StageWiseConstraintLearningModel,
    _StageParams,
    _geometric_median,
    _hard_gammas_from_stage_ends,
)
from visualization.io import learner_plot_dir, save_figure
from visualization.map_plots import plot_map_final_outputs


_MAP_DEMO_MODEL = None


@dataclass
class _MAPModeFit:
    mode: str
    kind: str | None
    eta: float | None
    scale: float | None
    cost: float
    summary: dict
    vector: np.ndarray | None


def _map_demo_worker_init(model):
    global _MAP_DEMO_MODEL
    _MAP_DEMO_MODEL = model


def _map_demo_worker_run(demo_idx: int, phase: str, shared_state):
    if _MAP_DEMO_MODEL is None:
        raise RuntimeError("MAP demo worker is not initialized.")
    if phase == "free":
        cost_info_fn = _MAP_DEMO_MODEL._free_interval_cost_info
    elif phase == "shared":
        if shared_state is None:
            raise RuntimeError("MAP shared segmentation requires a shared model state.")
        _MAP_DEMO_MODEL._apply_map_shared_state(*shared_state)
        cost_info_fn = _MAP_DEMO_MODEL._shared_interval_cost_info
    else:
        raise ValueError(f"Unknown MAP demo segmentation phase: {phase!r}.")
    result = _MAP_DEMO_MODEL._best_segmentation_by_interval_cost(
        demo_idx=int(demo_idx),
        cost_info_fn=cost_info_fn,
        fixed_cutpoints_by_stage=_MAP_DEMO_MODEL._fixed_cutpoint_map_for_demo(int(demo_idx)),
    )
    return int(demo_idx), result


class StageWiseMAPConstraintLearningModel(StageWiseConstraintLearningModel):
    """Likelihood-based MAP stage-wise constraint learner.

    This class keeps the public learner shape used by the existing evaluation
    and plotting code, but replaces SWCL's local-fit plus consensus objective
    with the MAP.md alternating procedure:

    1. initialize each demo by DP with fully local/free feature modes;
    2. update shared modes using point-pooled or demo-balanced candidate
       parameters, with either direct score aggregation or majority voting;
    3. segment demos under the fixed shared prototype;
    4. update the shared prototype from the recovered intervals.
    """

    def __init__(
        self,
        *args,
        map_eq_sigma: float = 0.05,
        map_c_bg: float = 2.0,
        map_c_ineq: float = 0.0,
        map_eq_distribution: str = "gaussian",
        map_inactive_distribution: str = "gaussian",
        map_nu_eq: float = 3.0,
        map_nu_inactive: float = 3.0,
        map_nu_ineq: float = 3.0,
        map_boundary_quantile: float = 0.05,
        map_activation_prior=None,
        map_active_mode_prior=None,
        map_mode_aggregation: str = "shared_vote",
        map_vote_prior_scope: str = "shared",
        map_refit_winning_voters: bool = False,
        map_convergence_tol: float = 1e-6,
        map_demo_num_workers: int | None = None,
        map_mstep_boundary_trim: int = 0,
        map_progress_kappa: float | None = None,
        map_progress_kappa_max: float = 100.0,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        unsupported_weights = sorted(
            key
            for key in (
                "lambda_constraint",
                "lambda_eq_constraint",
                "lambda_ineq_constraint",
                "map_inactive_weight",
                "map_active_mode_penalty",
                "lambda_progress",
                "progress_delta_scale",
            )
            if key in kwargs
        )
        if unsupported_weights:
            raise ValueError(
                "MAP uses normalized likelihood costs; remove unsupported weighting parameters: "
                + ", ".join(unsupported_weights)
            )
        kwargs["lambda_progress"] = 1.0
        demos = kwargs.get("demos", args[0] if args else None)
        env = kwargs.get("env", args[1] if len(args) > 1 else None)
        precomputed_features = kwargs.get("precomputed_features")
        if demos is not None and env is not None and kwargs.get("feature_model_types") is None:
            selected = kwargs.get("selected_raw_feature_ids")
            if selected is None:
                if hasattr(env, "get_feature_schema"):
                    num_features = len(env.get_feature_schema())
                elif getattr(env, "feature_schema", None) is not None:
                    num_features = len(getattr(env, "feature_schema"))
                elif precomputed_features is not None:
                    num_features = int(np.asarray(precomputed_features[0], dtype=float).shape[1])
                else:
                    num_features = int(np.asarray(env.compute_all_features_matrix(np.asarray(demos[0]))).shape[1])
            else:
                num_features = len(selected)
            kwargs["feature_model_types"] = ["student_t"] * int(num_features)
        kwargs["feature_activation_mode"] = "score"
        kwargs["equality_score_mode"] = "dispersion"
        super().__init__(*args, **kwargs)

        self.method_name = "map"
        self.map_eq_sigma = max(float(map_eq_sigma), 1e-9)
        self.map_c_bg = max(float(map_c_bg), 1.0 + 1e-9)
        self.map_c_ineq = max(float(map_c_ineq), 0.0)
        self.map_eq_distribution = self._normalize_map_distribution(map_eq_distribution)
        self.map_inactive_distribution = self._normalize_map_distribution(map_inactive_distribution)
        self.map_nu_eq = max(float(map_nu_eq), 1e-6)
        self.map_nu_inactive = max(float(map_nu_inactive), 1e-6)
        self.map_nu_ineq = max(float(map_nu_ineq), 1e-6)
        self.map_boundary_quantile = float(np.clip(float(map_boundary_quantile), 1e-4, 0.49))
        self.map_activation_prior = self._normalize_feature_probability_vector(
            map_activation_prior,
            default=0.5,
            name="map_activation_prior",
        )
        self.map_active_mode_prior = self._normalize_active_mode_prior(map_active_mode_prior)
        self.map_mode_aggregation = self._normalize_map_mode_aggregation(map_mode_aggregation)
        self.map_vote_prior_scope = self._normalize_map_vote_prior_scope(map_vote_prior_scope)
        self.map_refit_winning_voters = bool(map_refit_winning_voters)
        self.map_convergence_tol = max(float(map_convergence_tol), 0.0)
        self.map_mstep_boundary_trim = max(int(map_mstep_boundary_trim), 0)
        if map_progress_kappa is not None:
            map_progress_kappa = float(map_progress_kappa)
            if not np.isfinite(map_progress_kappa) or map_progress_kappa < 0.0:
                raise ValueError("map_progress_kappa must be null or a finite nonnegative scalar.")
        self.map_progress_kappa = map_progress_kappa
        self.map_progress_kappa_max = float(map_progress_kappa_max)
        if not np.isfinite(self.map_progress_kappa_max) or self.map_progress_kappa_max <= 0.0:
            raise ValueError("map_progress_kappa_max must be a finite positive scalar.")
        initial_kappa = 0.0 if self.map_progress_kappa is None else float(self.map_progress_kappa)
        self.map_progress_kappas_ = np.full(self.num_stages, initial_kappa, dtype=float)
        self.map_progress_kappa_history_ = []
        if map_demo_num_workers is None:
            map_demo_num_workers = min(len(self.demos), os.cpu_count() or 1)
        if int(map_demo_num_workers) < 1:
            raise ValueError("map_demo_num_workers must be at least 1 or null for automatic selection.")
        self.map_demo_num_workers = min(int(map_demo_num_workers), max(len(self.demos), 1))
        self.score_threshold_matrix = np.zeros((self.num_stages, self.num_features), dtype=float)
        self.has_equality_feature = True
        self._map_free_segment_cache: dict[tuple[int, int, int], tuple[_StageParams, float]] = {}
        self._map_local_mode_cache: dict[tuple[int, int, int, int], Dict[str, _MAPModeFit]] = {}
        self._map_inactive_fit_cache: dict[tuple[int, int, int, int], _MAPModeFit] = {}
        self._map_interval_stats_cache: dict[tuple[int, int, int, int], dict] = {}
        self._map_shared_cost_cache: dict[tuple[int, int, int, int, int], float] = {}
        self._map_shared_eq_prefix_cache: dict[tuple[int, int, int, float], np.ndarray] = {}
        self._map_t_ppf_cache: dict[tuple[float, float], float] = {}
        self._map_shared_cache_version = 0
        self.map_shared_mode_costs_ = []
        self.map_shared_mode_votes_ = []
        self._map_last_mode_vote_diagnostic = {}

    def _student_t_ppf(self, q: float, nu: float) -> float:
        key = (round(float(q), 12), round(float(nu), 12))
        cached = self._map_t_ppf_cache.get(key)
        if cached is not None:
            return float(cached)
        value = StageWiseConstraintLearningModel._student_t_ppf(float(q), float(nu))
        self._map_t_ppf_cache[key] = float(value)
        return float(value)

    @staticmethod
    def _normalize_map_distribution(value: str) -> str:
        text = str(value).strip().lower().replace("-", "_")
        aliases = {
            "t": "student_t",
            "student": "student_t",
            "studentt": "student_t",
            "student_t": "student_t",
            "gauss": "gaussian",
            "normal": "gaussian",
            "gaussian": "gaussian",
        }
        if text not in aliases:
            raise ValueError("MAP distribution must be one of: student_t, gaussian.")
        return aliases[text]

    @staticmethod
    def _normalize_map_mode_aggregation(value: str) -> str:
        text = str(value).strip().lower().replace("-", "_")
        aliases = {
            "vote": "shared_vote",
            "demo_vote": "shared_vote",
            "shared_vote": "shared_vote",
            "pooled": "pooled",
            "pooled_nll": "pooled",
            "balanced": "demo_balanced_pooled",
            "balanced_pooled": "demo_balanced_pooled",
            "demo_balanced": "demo_balanced_pooled",
            "demo_balanced_pooled": "demo_balanced_pooled",
            "balanced_vote": "demo_balanced_vote",
            "demo_balanced_vote": "demo_balanced_vote",
        }
        if text not in aliases:
            raise ValueError(
                "map_mode_aggregation must be one of: shared_vote, pooled, "
                "demo_balanced_pooled, demo_balanced_vote."
            )
        return aliases[text]

    @staticmethod
    def _normalize_map_vote_prior_scope(value: str) -> str:
        text = str(value).strip().lower().replace("-", "_")
        aliases = {
            "shared": "shared",
            "global": "shared",
            "global_shared": "shared",
            "per_demo": "per_demo",
            "local": "per_demo",
        }
        if text not in aliases:
            raise ValueError("map_vote_prior_scope must be one of: shared, per_demo.")
        return aliases[text]

    def _normalize_feature_probability_vector(self, value, *, default: float, name: str) -> np.ndarray:
        if value is None:
            values = np.full(self.num_features, float(default), dtype=float)
        else:
            values = np.asarray(value, dtype=float)
            if values.ndim == 0:
                values = np.full(self.num_features, float(values), dtype=float)
            else:
                values = values.reshape(-1)
        if values.size != self.num_features:
            raise ValueError(f"{name} must contain {self.num_features} feature probabilities, got {values.size}.")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"{name} probabilities must be finite and lie in [0, 1].")
        return values.astype(float, copy=True)

    def _normalize_active_mode_prior(self, value) -> dict[str, np.ndarray]:
        if value is None:
            return {
                mode: np.full(self.num_features, 1.0 / 3.0, dtype=float)
                for mode in ("eq", "lb", "ub")
            }
        if not isinstance(value, dict):
            raise ValueError("map_active_mode_prior must be a dict with eq, lb, and ub feature vectors.")
        required = {"eq", "lb", "ub"}
        if set(value) != required:
            raise ValueError("map_active_mode_prior must contain exactly the keys: eq, lb, ub.")
        priors = {
            mode: self._normalize_feature_probability_vector(
                value[mode],
                default=1.0 / 3.0,
                name=f"map_active_mode_prior.{mode}",
            )
            for mode in ("eq", "lb", "ub")
        }
        totals = priors["eq"] + priors["lb"] + priors["ub"]
        if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("For every feature, map_active_mode_prior eq/lb/ub probabilities must sum to 1.")
        return priors

    def _mode_to_kind(self, mode: str) -> str | None:
        mode_l = str(mode).lower()
        if mode_l == "eq":
            return "gaussian" if self.map_eq_distribution == "gaussian" else "student_t"
        if mode_l == "lb":
            return "trunc_t_lower_z"
        if mode_l == "ub":
            return "trunc_t_upper_z"
        return None

    @staticmethod
    def _kind_to_mode(kind: str | None) -> str:
        kind_l = "" if kind is None else str(kind).lower()
        if kind_l in {"student_t", "studentt", "t", "gauss", "gaussian"}:
            return "eq"
        if "lower" in kind_l:
            return "lb"
        if "upper" in kind_l:
            return "ub"
        return "inactive"

    def _map_feature_name(self, feat_idx: int) -> str:
        selected_col = int(self.selected_feature_columns[int(feat_idx)])
        for i, spec in enumerate(getattr(self, "raw_feature_specs", [])):
            if int(spec.get("column_idx", i)) == selected_col:
                return str(spec.get("name", f"f{feat_idx}"))
        return f"f{feat_idx}"

    def _map_true_mode(self, stage_idx: int, feat_idx: int) -> str | None:
        env = getattr(self, "env", None)
        if env is None:
            return None
        if hasattr(env, "get_constraint_specs"):
            specs = list(env.get_constraint_specs())
        else:
            specs = list(getattr(env, "constraint_specs", []) or [])
        if not specs:
            return None
        feature_name = self._map_feature_name(int(feat_idx))
        semantic_to_mode = {
            "target_value": "eq",
            "target": "eq",
            "equality": "eq",
            "eq": "eq",
            "lower_bound": "lb",
            "lower": "lb",
            "ge": "lb",
            ">=": "lb",
            "upper_bound": "ub",
            "upper": "ub",
            "le": "ub",
            "<=": "ub",
        }
        for spec in specs:
            if int(spec.get("stage", -1)) != int(stage_idx):
                continue
            if str(spec.get("feature_name", "")) != feature_name:
                continue
            return semantic_to_mode.get(str(spec.get("semantics", "")).strip().lower())
        return "inactive"

    @staticmethod
    def _map_mode_markers(mode: str, *, best_mode: str, true_mode: str | None) -> str:
        best_marker = "*" if str(mode) == str(best_mode) else " "
        true_marker = "#" if true_mode is not None and str(mode) == str(true_mode) else " "
        return best_marker + true_marker

    @staticmethod
    def _map_student_t_pdf(xs, *, mu: float, sigma: float, nu: float) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        sigma = max(float(sigma), 1e-12)
        nu = max(float(nu), 1e-12)
        z = (xs - float(mu)) / sigma
        log_norm = (
            math.lgamma(0.5 * (nu + 1.0))
            - math.lgamma(0.5 * nu)
            - 0.5 * math.log(nu * math.pi)
            - math.log(sigma)
        )
        return np.exp(log_norm - 0.5 * (nu + 1.0) * np.log1p((z * z) / nu))

    @staticmethod
    def _map_gaussian_pdf(xs, *, mu: float, sigma: float) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        sigma = max(float(sigma), 1e-12)
        z = (xs - float(mu)) / sigma
        return np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sigma)

    @staticmethod
    def _map_soft_half_t_pdf(xs, *, b: float, scale: float, nu: float, softness: float, mode: str) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        scale = max(float(scale), 1e-12)
        nu = max(float(nu), 1e-12)
        softness = max(float(softness), 1e-12)
        sign = 1.0 if str(mode).lower() == "lb" else -1.0
        signed_slack = sign * (xs - float(b))
        log_t_norm = (
            math.log(2.0)
            + math.lgamma(0.5 * (nu + 1.0))
            - math.lgamma(0.5 * nu)
            - 0.5 * math.log(nu * math.pi)
            - math.log(scale)
        )
        half_t_at_zero = float(math.exp(min(log_t_norm, 700.0)))
        log_partition = math.log1p(max(half_t_at_zero * softness, 0.0))
        log_pdf = np.empty_like(xs, dtype=float)
        ok = signed_slack >= 0.0
        if np.any(ok):
            z = signed_slack[ok] / scale
            log_pdf[ok] = log_t_norm - 0.5 * (nu + 1.0) * np.log1p((z * z) / nu) - log_partition
        if np.any(~ok):
            log_pdf[~ok] = log_t_norm + signed_slack[~ok] / softness - log_partition
        return np.exp(np.maximum(log_pdf, -690.0))

    def _map_mode_fit_pdf(self, xs, fit: _MAPModeFit) -> np.ndarray:
        summary = dict(fit.summary)
        xs = np.asarray(xs, dtype=float)
        if fit.mode in {"inactive", "eq"}:
            if str(summary.get("distribution", "")).lower() == "gaussian":
                return self._map_gaussian_pdf(
                    xs,
                    mu=float(summary.get("mu", 0.0)),
                    sigma=float(summary.get("sigma", 1.0)),
                )
            return self._map_student_t_pdf(
                xs,
                mu=float(summary.get("mu", 0.0)),
                sigma=float(summary.get("sigma", 1.0)),
                nu=float(summary.get("nu", self.map_nu_eq if fit.mode == "eq" else self.map_nu_inactive)),
            )
        if fit.mode in {"lb", "ub"}:
            scale = float(summary.get("sigma", 1.0))
            return self._map_soft_half_t_pdf(
                xs,
                b=float(summary.get("b", 0.0)),
                scale=scale,
                nu=float(summary.get("nu", self.map_nu_ineq)),
                softness=float(self.truncated_z_soft_boundary_scale) * max(scale, 1e-12),
                mode=fit.mode,
            )
        return np.full_like(xs, np.nan, dtype=float)

    @staticmethod
    def _map_param_text(mode: str, vec) -> str:
        mode = str(mode)
        if mode == "inactive" or vec is None:
            return "-"
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if arr.size == 0:
            return "-"
        return f"{arr[0]:.3f}"

    def _map_should_plot_iteration(self, iteration: int) -> tuple[bool, int]:
        plot_it = max(int(iteration) + 1, 0)
        if self.disable_plots or self.plot_every is None or plt is None:
            return False, plot_it
        return (plot_it % int(self.plot_every) == 0), plot_it

    def _map_stage_interval_refs(self, selected_infos: Sequence[dict], stage_idx: int, feat_idx: int):
        refs = []
        for demo_idx, info in enumerate(selected_infos):
            bounds = self._segment_bounds_from_stage_ends([int(x) for x in info["stage_ends"]])
            s, e = bounds[int(stage_idx)]
            core_s, core_e = self._map_mstep_interval_bounds(int(s), int(e), stage_idx=int(stage_idx))
            xs = np.asarray(
                self.standardized_features[int(demo_idx)][int(core_s) : int(core_e) + 1, int(feat_idx)],
                dtype=float,
            ).reshape(-1)
            refs.append(
                {
                    "demo_idx": int(demo_idx),
                    "stage_idx": int(stage_idx),
                    "feat_idx": int(feat_idx),
                    "s": int(s),
                    "e": int(e),
                    "core_s": int(core_s),
                    "core_e": int(core_e),
                    "xs": xs,
                }
            )
        return refs

    def _map_mstep_interval_bounds(self, s: int, e: int, *, stage_idx: int) -> tuple[int, int]:
        core_s, core_e = self._segment_core_bounds(int(s), int(e))
        trim = int(self.map_mstep_boundary_trim)
        if trim <= 0:
            return int(core_s), int(core_e)
        left_trim = trim if int(stage_idx) > 0 else 0
        right_trim = trim if int(stage_idx) < int(self.num_stages) - 1 else 0
        fit_s = min(int(core_s) + left_trim, int(core_e))
        fit_e = max(int(fit_s), int(core_e) - right_trim)
        return int(fit_s), int(fit_e)

    @staticmethod
    def _map_optimization_bounds(values: np.ndarray) -> tuple[float, float]:
        values = np.asarray(values, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return -1.0, 1.0
        lo = float(np.min(values))
        hi = float(np.max(values))
        span = max(float(hi - lo), 1e-3)
        return lo - 0.5 * span, hi + 0.5 * span

    def _map_minimize_scalar(self, objective, bounds: tuple[float, float]) -> tuple[float, float]:
        lo, hi = float(bounds[0]), float(bounds[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -1.0, 1.0
        candidates = [lo, 0.5 * (lo + hi), hi]
        try:
            result = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
            if result.success and np.isfinite(result.fun):
                candidates.append(float(result.x))
        except Exception:
            pass
        scored = []
        for eta in candidates:
            try:
                value = float(objective(float(eta)))
            except Exception:
                value = float("inf")
            if np.isfinite(value):
                scored.append((value, float(eta)))
        if not scored:
            return float("inf"), 0.5 * (lo + hi)
        value, eta = min(scored, key=lambda item: item[0])
        return float(value), float(eta)

    @staticmethod
    def _demo_cost_weight(ref: dict, *, demo_balanced: bool) -> float:
        if not demo_balanced:
            return 1.0
        sample_count = int(np.asarray(ref["xs"], dtype=float).size)
        return 1.0 / float(max(sample_count, 1))

    def _pooled_inactive_mstep(
        self,
        refs: Sequence[dict],
        *,
        demo_balanced: bool = False,
    ) -> tuple[float, np.ndarray | None, str | None]:
        total = 0.0
        for ref in refs:
            fit = self._inactive_fit_cached(
                int(ref["demo_idx"]),
                int(ref["core_s"]),
                int(ref["core_e"]),
                int(ref["feat_idx"]),
            )
            total += self._demo_cost_weight(
                ref,
                demo_balanced=demo_balanced,
            ) * self._mode_fit_likelihood_cost(fit)
        return float(total), None, None

    def _pooled_eq_mstep(
        self,
        refs: Sequence[dict],
        *,
        demo_balanced: bool = False,
    ) -> tuple[float, np.ndarray | None, str | None]:
        pooled = np.concatenate([np.asarray(ref["xs"], dtype=float).reshape(-1) for ref in refs if ref["xs"].size])
        if pooled.size == 0:
            return float("inf"), None, None
        bounds = self._map_optimization_bounds(pooled)

        def objective(eta: float) -> float:
            total_nll = 0.0
            for ref in refs:
                xs = np.asarray(ref["xs"], dtype=float).reshape(-1)
                if xs.size == 0:
                    continue
                total_nll += self._demo_cost_weight(
                    ref,
                    demo_balanced=demo_balanced,
                ) * self._eq_sum_nll(xs, eta=float(eta))
            return float(total_nll)

        total, eta = self._map_minimize_scalar(objective, bounds)
        if not np.isfinite(total):
            return float("inf"), None, None
        vec = np.asarray([float(eta), math.log(float(self.map_eq_sigma))], dtype=float)
        return float(total), vec, self._mode_to_kind("eq")

    def _pooled_ineq_mstep(
        self,
        refs: Sequence[dict],
        mode: str,
        *,
        demo_balanced: bool = False,
    ) -> tuple[float, np.ndarray | None, str | None]:
        mode_l = str(mode).lower()
        if mode_l not in {"lb", "ub"}:
            raise ValueError(f"Unknown inequality mode '{mode}'.")
        pooled = np.concatenate([np.asarray(ref["xs"], dtype=float).reshape(-1) for ref in refs if ref["xs"].size])
        if pooled.size == 0:
            return float("inf"), None, None
        bounds = self._map_optimization_bounds(pooled)

        def objective(eta: float) -> float:
            total = 0.0
            for ref in refs:
                xs = np.asarray(ref["xs"], dtype=float).reshape(-1)
                if xs.size == 0:
                    continue
                stats = self._interval_feature_stats_cached(
                    int(ref["demo_idx"]),
                    int(ref["core_s"]),
                    int(ref["core_e"]),
                    int(ref["feat_idx"]),
                )
                fit = self._inequality_fit_from_stats(
                    xs,
                    mode=mode_l,
                    eta=float(eta),
                    stats=stats,
                )
                total += self._demo_cost_weight(
                    ref,
                    demo_balanced=demo_balanced,
                ) * self._mode_fit_likelihood_cost(fit)
            return float(total)

        total, eta = self._map_minimize_scalar(objective, bounds)
        if not np.isfinite(total):
            return float("inf"), None, None
        scales = []
        centers = []
        for ref in refs:
            xs = np.asarray(ref["xs"], dtype=float).reshape(-1)
            if xs.size == 0:
                continue
            stats = self._interval_feature_stats_cached(
                int(ref["demo_idx"]),
                int(ref["core_s"]),
                int(ref["core_e"]),
                int(ref["feat_idx"]),
            )
            fit = self._inequality_fit_from_stats(
                xs,
                mode=mode_l,
                eta=float(eta),
                stats=stats,
            )
            if fit.vector is not None:
                vec = np.asarray(fit.vector, dtype=float).reshape(-1)
                if vec.size >= 3:
                    centers.append(float(vec[1]))
                    scales.append(float(np.exp(vec[2])))
        center = float(np.median(centers)) if centers else float(np.median(pooled))
        scale_floor = max(float(self._truncated_z_scale_floor()), float(self.map_c_ineq * self.map_eq_sigma))
        scale = float(np.median(scales)) if scales else float(scale_floor)
        scale = max(scale, float(scale_floor), 1e-6)
        vec = np.asarray([float(eta), center, math.log(scale)], dtype=float)
        kind = "trunc_t_lower_z" if mode_l == "lb" else "trunc_t_upper_z"
        return float(total), vec, kind

    def _shared_candidate_demo_mean_nlls(
        self,
        ref: dict,
        vectors: dict[str, np.ndarray | None],
    ) -> dict[str, float]:
        xs = np.asarray(ref["xs"], dtype=float).reshape(-1)
        if xs.size == 0:
            return {mode: float("inf") for mode in ("inactive", "eq", "lb", "ub")}

        demo_idx = int(ref["demo_idx"])
        core_s = int(ref["core_s"])
        core_e = int(ref["core_e"])
        feat_idx = int(ref["feat_idx"])
        normalizer = float(xs.size)
        mean_nlls = {
            "inactive": self._mode_fit_likelihood_cost(
                self._inactive_fit_cached(demo_idx, core_s, core_e, feat_idx)
            )
            / normalizer,
        }

        eq_vector = vectors.get("eq")
        if eq_vector is None:
            mean_nlls["eq"] = float("inf")
        else:
            eq_eta = float(np.asarray(eq_vector, dtype=float).reshape(-1)[0])
            mean_nlls["eq"] = self._eq_sum_nll(xs, eta=eq_eta) / normalizer

        stats = self._interval_feature_stats_cached(demo_idx, core_s, core_e, feat_idx)
        for mode in ("lb", "ub"):
            vector = vectors.get(mode)
            if vector is None:
                mean_nlls[mode] = float("inf")
                continue
            eta = float(np.asarray(vector, dtype=float).reshape(-1)[0])
            fit = self._inequality_fit_from_stats(xs, mode=mode, eta=eta, stats=stats)
            mean_nlls[mode] = self._mode_fit_likelihood_cost(fit) / normalizer
        return {mode: float(value) for mode, value in mean_nlls.items()}

    def _shared_vote_mode(
        self,
        refs: Sequence[dict],
        vectors: dict[str, np.ndarray | None],
        prior_costs: dict[str, float],
        *,
        aggregation: str,
    ) -> tuple[str, dict]:
        modes = ("inactive", "eq", "lb", "ub")
        order = {mode: idx for idx, mode in enumerate(modes)}
        num_demos = len(refs)
        prior_divisor = float(max(num_demos, 1)) if self.map_vote_prior_scope == "shared" else 1.0
        prior_share = {
            mode: float(prior_costs[mode]) / prior_divisor
            for mode in modes
        }
        demo_votes = []
        demo_mean_nlls = []
        demo_vote_scores = []
        vote_counts = {mode: 0 for mode in modes}

        for ref in refs:
            mean_nlls = self._shared_candidate_demo_mean_nlls(ref, vectors)
            vote_scores = {
                mode: float(mean_nlls[mode] + prior_share[mode])
                for mode in modes
            }
            vote = min(modes, key=lambda mode: (float(vote_scores[mode]), order[mode]))
            vote_counts[vote] += 1
            demo_votes.append(str(vote))
            demo_mean_nlls.append(mean_nlls)
            demo_vote_scores.append(vote_scores)

        majority_required = num_demos // 2 + 1
        majority_modes = [mode for mode in modes if vote_counts[mode] >= majority_required]
        selected_mode = majority_modes[0] if majority_modes else "inactive"
        return selected_mode, {
            "aggregation": str(aggregation),
            "prior_scope": str(self.map_vote_prior_scope),
            "prior_divisor": float(prior_divisor),
            "selected_mode": str(selected_mode),
            "majority_required": int(majority_required),
            "vote_counts": {mode: int(vote_counts[mode]) for mode in modes},
            "demo_votes": demo_votes,
            "demo_mean_nlls": demo_mean_nlls,
            "demo_vote_scores": demo_vote_scores,
        }

    def _pooled_mode_mstep(self, selected_infos: Sequence[dict], stage_idx: int, feat_idx: int):
        refs = self._map_stage_interval_refs(selected_infos, int(stage_idx), int(feat_idx))
        aggregation = str(self.map_mode_aggregation)
        demo_balanced = aggregation in {"demo_balanced_pooled", "demo_balanced_vote"}
        costs: dict[str, float] = {}
        vectors: dict[str, np.ndarray | None] = {}
        kinds: dict[str, str | None] = {}

        costs["inactive"], vectors["inactive"], kinds["inactive"] = self._pooled_inactive_mstep(
            refs,
            demo_balanced=demo_balanced,
        )
        costs["eq"], vectors["eq"], kinds["eq"] = self._pooled_eq_mstep(
            refs,
            demo_balanced=demo_balanced,
        )
        costs["lb"], vectors["lb"], kinds["lb"] = self._pooled_ineq_mstep(
            refs,
            "lb",
            demo_balanced=demo_balanced,
        )
        costs["ub"], vectors["ub"], kinds["ub"] = self._pooled_ineq_mstep(
            refs,
            "ub",
            demo_balanced=demo_balanced,
        )
        likelihood_costs = dict(costs)
        prior_costs = {
            mode: self._mode_prior_cost(mode, int(feat_idx))
            for mode in ("inactive", "eq", "lb", "ub")
        }
        for mode in ("inactive", "eq", "lb", "ub"):
            costs[mode] = float(likelihood_costs[mode] + prior_costs[mode])

        order = {"inactive": 0, "eq": 1, "lb": 2, "ub": 3}
        aggregate_mode = min(costs, key=lambda item: (float(costs[item]), order[item]))
        if aggregation in {"shared_vote", "demo_balanced_vote"}:
            mode, diagnostic = self._shared_vote_mode(
                refs,
                vectors,
                prior_costs,
                aggregation=aggregation,
            )
            diagnostic["pre_refit_vector"] = (
                None if vectors[mode] is None else np.asarray(vectors[mode], dtype=float).tolist()
            )
            diagnostic["refit_enabled"] = bool(self.map_refit_winning_voters)
            diagnostic["refit_demo_indices"] = []
            if self.map_refit_winning_voters and mode != "inactive":
                voter_refs = [
                    ref
                    for ref, vote in zip(refs, diagnostic["demo_votes"])
                    if str(vote) == str(mode)
                ]
                diagnostic["refit_demo_indices"] = [int(ref["demo_idx"]) for ref in voter_refs]
                if mode == "eq":
                    _, vectors[mode], kinds[mode] = self._pooled_eq_mstep(
                        voter_refs,
                        demo_balanced=demo_balanced,
                    )
                elif mode in {"lb", "ub"}:
                    _, vectors[mode], kinds[mode] = self._pooled_ineq_mstep(
                        voter_refs,
                        mode,
                        demo_balanced=demo_balanced,
                    )
            diagnostic["post_refit_vector"] = (
                None if vectors[mode] is None else np.asarray(vectors[mode], dtype=float).tolist()
            )
        else:
            mode = aggregate_mode
            diagnostic = {
                "aggregation": aggregation,
                "selected_mode": str(mode),
                "majority_required": None,
                "vote_counts": {},
                "demo_votes": [],
                "demo_mean_nlls": [],
                "demo_vote_scores": [],
                "refit_enabled": False,
                "pre_refit_vector": None,
                "post_refit_vector": None,
                "refit_demo_indices": [],
            }
        diagnostic["aggregate_mode"] = str(aggregate_mode)
        diagnostic["pooled_mode"] = str(aggregate_mode)
        diagnostic["pooled_costs"] = {name: float(value) for name, value in costs.items()}
        self._map_last_mode_vote_diagnostic = diagnostic
        return mode, vectors[mode], kinds[mode], costs

    def _pooled_mode_fits_for_plot(self, selected_infos: Sequence[dict], stage_idx: int, feat_idx: int):
        refs = self._map_stage_interval_refs(selected_infos, int(stage_idx), int(feat_idx))
        demo_balanced = self.map_mode_aggregation in {"demo_balanced_pooled", "demo_balanced_vote"}
        pooled_parts = [np.asarray(ref["xs"], dtype=float).reshape(-1) for ref in refs if ref["xs"].size]
        if not pooled_parts:
            return None
        pooled = np.concatenate(pooled_parts, axis=0)
        costs: dict[str, float] = {}
        vectors: dict[str, np.ndarray | None] = {}
        kinds: dict[str, str | None] = {}
        costs["inactive"], vectors["inactive"], kinds["inactive"] = self._pooled_inactive_mstep(
            refs,
            demo_balanced=demo_balanced,
        )
        costs["eq"], vectors["eq"], kinds["eq"] = self._pooled_eq_mstep(
            refs,
            demo_balanced=demo_balanced,
        )
        costs["lb"], vectors["lb"], kinds["lb"] = self._pooled_ineq_mstep(
            refs,
            "lb",
            demo_balanced=demo_balanced,
        )
        costs["ub"], vectors["ub"], kinds["ub"] = self._pooled_ineq_mstep(
            refs,
            "ub",
            demo_balanced=demo_balanced,
        )
        likelihood_costs = dict(costs)
        prior_costs = {
            mode: self._mode_prior_cost(mode, int(feat_idx))
            for mode in ("inactive", "eq", "lb", "ub")
        }
        for mode in ("inactive", "eq", "lb", "ub"):
            costs[mode] = float(likelihood_costs[mode] + prior_costs[mode])

        fits: dict[str, _MAPModeFit] = {}
        inactive_display = self._inactive_fit(pooled, feat_idx=int(feat_idx))
        fits["inactive"] = _MAPModeFit(
            mode="inactive",
            kind=None,
            eta=None,
            scale=inactive_display.scale,
            cost=float(costs["inactive"]),
            summary=dict(inactive_display.summary),
            vector=None,
        )
        if vectors["eq"] is not None:
            eta = float(np.asarray(vectors["eq"], dtype=float).reshape(-1)[0])
            eq_display = self._equality_fit(pooled, eta=eta, feat_idx=int(feat_idx))
            fits["eq"] = _MAPModeFit(
                mode="eq",
                kind=kinds["eq"],
                eta=float(eta),
                scale=eq_display.scale,
                cost=float(costs["eq"]),
                summary=dict(eq_display.summary),
                vector=np.asarray(vectors["eq"], dtype=float),
            )
        for mode in ("lb", "ub"):
            vec = vectors[mode]
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=float).reshape(-1)
            if arr.size < 3:
                continue
            eta = float(arr[0])
            center = float(arr[1])
            scale = max(float(np.exp(arr[2])), 1e-12)
            kind = kinds[mode] or ("trunc_t_lower_z" if mode == "lb" else "trunc_t_upper_z")
            summary = {
                "type": str(kind),
                "mode": mode,
                "b": float(eta),
                "mu": float(center),
                "sigma": float(scale),
                "nu": float(self.map_nu_ineq),
                "soft_boundary_scale": float(self.truncated_z_soft_boundary_scale),
                "ineq_sigma_min": float(self.map_c_ineq * self.map_eq_sigma),
                "boundary_quantile": float(self.map_boundary_quantile if mode == "lb" else 1.0 - self.map_boundary_quantile),
            }
            fits[mode] = _MAPModeFit(
                mode=mode,
                kind=str(kind),
                eta=float(eta),
                scale=float(scale),
                cost=float(costs[mode]),
                summary=summary,
                vector=arr,
            )
        order = {"inactive": 0, "eq": 1, "lb": 2, "ub": 3}
        best_mode = min(costs, key=lambda item: (float(costs[item]), order[item]))
        shared_mode = self._kind_to_mode(self.shared_param_kinds[int(stage_idx)][int(feat_idx)])
        return {
            "refs": refs,
            "pooled": pooled,
            "fits": fits,
            "costs": costs,
            "likelihood_costs": likelihood_costs,
            "prior_costs": prior_costs,
            "best_mode": best_mode,
            "shared_mode": shared_mode,
            "cost_label": "demo-balanced NLL" if demo_balanced else "pooled NLL",
        }

    def _plot_map_pooled_mode_density_diagnostics(self, iteration: int, selected_infos: Sequence[dict], *, force: bool = False) -> None:
        should_plot, plot_it = self._map_should_plot_iteration(int(iteration))
        if not force and not should_plot:
            return
        if force:
            plot_it = max(int(iteration), 0)
        out_dir = learner_plot_dir(self)
        colors = {
            "inactive": "#6B7280",
            "eq": "#2563EB",
            "lb": "#059669",
            "ub": "#DC2626",
        }
        labels = {
            "inactive": "inactive/background",
            "eq": "equality",
            "lb": "lower-bound",
            "ub": "upper-bound",
        }
        legend_handles = []
        legend_labels = []
        if Line2D is not None and Patch is not None:
            legend_handles = [
                Patch(facecolor="#CBD5E1", edgecolor="#64748B", alpha=0.72),
                Line2D([0], [0], color=colors["inactive"], lw=1.35, linestyle="--", alpha=0.62),
                Line2D([0], [0], color=colors["eq"], lw=1.35, linestyle="--", alpha=0.62),
                Line2D([0], [0], color=colors["lb"], lw=1.35, linestyle="--", alpha=0.62),
                Line2D([0], [0], color=colors["ub"], lw=1.35, linestyle="--", alpha=0.62),
                Line2D([0], [0], color="#111827", lw=1.6, linestyle="-."),
            ]
            legend_labels = [
                "pooled stage-feature data",
                labels["inactive"],
                labels["eq"],
                labels["lb"],
                labels["ub"],
                "shared param",
            ]
        fig, axes = plt.subplots(
            self.num_stages,
            self.num_features,
            figsize=(4.2 * self.num_features, 2.95 * self.num_stages),
            squeeze=False,
        )
        for stage_idx in range(self.num_stages):
            for feat_idx in range(self.num_features):
                ax = axes[stage_idx][feat_idx]
                plot_info = self._pooled_mode_fits_for_plot(selected_infos, int(stage_idx), int(feat_idx))
                if plot_info is None:
                    ax.axis("off")
                    continue
                pooled = np.asarray(plot_info["pooled"], dtype=float).reshape(-1)
                if pooled.size == 0:
                    ax.axis("off")
                    continue
                q_lo, q_hi = np.quantile(pooled, [0.02, 0.98])
                span = max(float(q_hi - q_lo), 1e-6)
                lo = min(float(np.min(pooled)), float(q_lo) - 0.8 * span)
                hi = max(float(np.max(pooled)), float(q_hi) + 0.8 * span)
                for fit in plot_info["fits"].values():
                    vec = fit.vector
                    if vec is not None:
                        eta = float(np.asarray(vec, dtype=float).reshape(-1)[0])
                        lo = min(lo, eta)
                        hi = max(hi, eta)
                if hi <= lo:
                    lo -= 1e-3
                    hi += 1e-3
                grid = np.linspace(lo, hi, 450)
                ax_density = ax.twinx()
                ax_density.patch.set_alpha(0.0)
                hist_vals, _, _ = ax.hist(
                    pooled,
                    bins=min(max(int(np.sqrt(pooled.size)) + 2, 8), 28),
                    density=False,
                    color="#CBD5E1",
                    edgecolor="#64748B",
                    linewidth=0.5,
                    alpha=0.72,
                    label="pooled stage-feature data",
                )
                shared_mode = str(plot_info["shared_mode"])
                best_mode = str(plot_info["best_mode"])
                true_mode = self._map_true_mode(int(stage_idx), int(feat_idx))
                density_y_max = 0.0
                cost_lines = [
                    f"*=best  #=true  T=total  L={plot_info['cost_label']}  P=prior NLL"
                ]
                for mode in ("inactive", "eq", "lb", "ub"):
                    fit = plot_info["fits"].get(mode)
                    cost = float(plot_info["costs"].get(mode, np.nan))
                    if fit is not None:
                        pdf = self._map_mode_fit_pdf(grid, fit)
                        finite = np.isfinite(pdf)
                        if np.any(finite):
                            density_y_max = max(density_y_max, float(np.nanpercentile(pdf[finite], 99.0)))
                            ax_density.plot(
                                grid[finite],
                                pdf[finite],
                                color=colors[mode],
                                lw=2.7 if mode == shared_mode else 1.35,
                                alpha=0.95 if mode == shared_mode else 0.62,
                                linestyle="-" if mode == shared_mode else "--",
                                label=labels[mode],
                            )
                        if fit.vector is not None:
                            eta = float(np.asarray(fit.vector, dtype=float).reshape(-1)[0])
                            ax.axvline(eta, color=colors[mode], lw=1.0, alpha=0.35)
                    prefix = self._map_mode_markers(mode, best_mode=best_mode, true_mode=true_mode)
                    likelihood_cost = float(plot_info["likelihood_costs"].get(mode, np.nan))
                    prior_cost = float(plot_info["prior_costs"].get(mode, np.nan))
                    cost_lines.append(
                        f"{prefix}{mode}: T={cost:.1f}  L={likelihood_cost:.1f}  P={prior_cost:.2f}"
                    )
                shared_vec = self.shared_param_vectors[stage_idx][feat_idx]
                if shared_vec is not None:
                    shared_eta = float(np.asarray(shared_vec, dtype=float).reshape(-1)[0])
                    ax.axvline(shared_eta, color="#111827", lw=1.6, linestyle="-.", alpha=0.85, label="shared param")

                hist_y_max = float(np.nanmax(hist_vals)) if np.asarray(hist_vals).size else 0.0
                if hist_y_max > 0.0 and np.isfinite(hist_y_max):
                    ax.set_ylim(0.0, hist_y_max * 1.18)
                if density_y_max > 0.0 and np.isfinite(density_y_max):
                    ax_density.set_ylim(0.0, density_y_max * 1.18)
                title = (
                    f"s{stage_idx + 1} {self._map_feature_name(feat_idx)} "
                    f"pooled n={pooled.size} best={best_mode} shared={shared_mode} "
                    f"true={'?' if true_mode is None else '#' + true_mode}"
                )
                ax.set_title(title, fontsize=8)
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(cost_lines),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=6.5,
                    bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.84},
                )
                ax.tick_params(labelsize=6.5)
                ax_density.tick_params(labelsize=6.0, colors="#475569")
                if stage_idx == self.num_stages - 1:
                    ax.set_xlabel("standardized feature value", fontsize=7)
                if feat_idx == 0:
                    ax.set_ylabel("pooled count", fontsize=7)
                if feat_idx == self.num_features - 1:
                    ax_density.set_ylabel("density", fontsize=7, color="#475569")
                else:
                    ax_density.set_yticklabels([])
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                ncol=min(len(legend_handles), 6),
                fontsize=7,
                frameon=False,
            )
        fig.suptitle(f"MAP pooled shared mode fits | iter {plot_it:04d}", fontsize=12, y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), pad=0.7)
        save_figure(fig, out_dir / f"pooled_density_iter_{int(plot_it):04d}.png", dpi=180)

    def _plot_map_mode_density_diagnostics(
        self,
        iteration: int,
        selected_infos: Sequence[dict],
        *,
        force: bool = False,
    ) -> None:
        should_plot, plot_it = self._map_should_plot_iteration(int(iteration))
        if not force and not should_plot:
            return
        if force:
            plot_it = max(int(iteration), 0)
        out_dir = learner_plot_dir(self)
        colors = {
            "inactive": "#6B7280",
            "eq": "#2563EB",
            "lb": "#059669",
            "ub": "#DC2626",
        }
        labels = {
            "inactive": "inactive/background",
            "eq": "equality",
            "lb": "lower-bound",
            "ub": "upper-bound",
        }
        legend_handles = []
        legend_labels = []
        if Line2D is not None and Patch is not None:
            legend_handles = [
                Patch(facecolor="#CBD5E1", edgecolor="#64748B", alpha=0.7),
                Line2D([0], [0], color=colors["inactive"], lw=1.35, linestyle="--", alpha=0.62),
                Line2D([0], [0], color=colors["eq"], lw=1.35, linestyle="--", alpha=0.62),
                Line2D([0], [0], color=colors["lb"], lw=2.6, linestyle="-", alpha=0.95),
                Line2D([0], [0], color=colors["ub"], lw=1.35, linestyle="--", alpha=0.62),
                Line2D([0], [0], color="#111827", lw=1.6, linestyle="-."),
            ]
            legend_labels = [
                "segment data",
                labels["inactive"],
                labels["eq"],
                labels["lb"],
                labels["ub"],
                "shared param",
            ]
        shared_candidate_plot_infos = [
            [
                self._pooled_mode_fits_for_plot(selected_infos, int(stage_idx), int(feat_idx))
                for feat_idx in range(self.num_features)
            ]
            for stage_idx in range(self.num_stages)
        ]
        for demo_idx, info in enumerate(selected_infos):
            stage_ends = [int(x) for x in info["stage_ends"]]
            bounds = self._segment_bounds_from_stage_ends(stage_ends)
            legend_items = {}
            fig, axes = plt.subplots(
                self.num_stages,
                self.num_features,
                figsize=(4.0 * self.num_features, 2.85 * self.num_stages),
                squeeze=False,
            )
            for stage_idx, (s, e) in enumerate(bounds):
                core_s, core_e = self._map_mstep_interval_bounds(int(s), int(e), stage_idx=int(stage_idx))
                for feat_idx in range(self.num_features):
                    ax = axes[stage_idx][feat_idx]
                    xs = np.asarray(
                        self.standardized_features[int(demo_idx)][int(core_s) : int(core_e) + 1, int(feat_idx)],
                        dtype=float,
                    ).reshape(-1)
                    if xs.size == 0:
                        ax.axis("off")
                        continue
                    q_lo, q_hi = np.quantile(xs, [0.02, 0.98])
                    span = max(float(q_hi - q_lo), 1e-6)
                    lo = min(float(np.min(xs)), float(q_lo) - 0.8 * span)
                    hi = max(float(np.max(xs)), float(q_hi) + 0.8 * span)
                    pooled_plot_info = shared_candidate_plot_infos[int(stage_idx)][int(feat_idx)]
                    if pooled_plot_info is None:
                        ax.axis("off")
                        continue
                    candidates = self._shared_candidate_mode_fits_for_interval(
                        int(demo_idx),
                        int(core_s),
                        int(core_e),
                        int(feat_idx),
                        pooled_plot_info["fits"],
                    )
                    for fit in candidates.values():
                        vec = fit.vector
                        if vec is not None:
                            eta = float(np.asarray(vec, dtype=float).reshape(-1)[0])
                            lo = min(lo, eta)
                            hi = max(hi, eta)
                    if hi <= lo:
                        lo -= 1e-3
                        hi += 1e-3
                    grid = np.linspace(lo, hi, 400)
                    ax_density = ax.twinx()
                    ax_density.patch.set_alpha(0.0)
                    hist_vals, _, _ = ax.hist(
                        xs,
                        bins=min(max(int(np.sqrt(xs.size)) + 2, 6), 22),
                        density=False,
                        color="#CBD5E1",
                        edgecolor="#64748B",
                        linewidth=0.5,
                        alpha=0.7,
                        label="segment data",
                    )
                    selected_mode = self._kind_to_mode(self.shared_param_kinds[stage_idx][feat_idx])
                    order = {"inactive": 0, "eq": 1, "lb": 2, "ub": 3}
                    best_fit = min(candidates.values(), key=lambda item: (float(item.cost), order.get(item.mode, 99)))
                    best_mode = str(best_fit.mode)
                    true_mode = self._map_true_mode(int(stage_idx), int(feat_idx))
                    hist_y_max = float(np.nanmax(hist_vals)) if np.asarray(hist_vals).size else 0.0
                    density_y_max = 0.0
                    cost_lines = ["*=best shared-param fit  #=true"]
                    for mode in ("inactive", "eq", "lb", "ub"):
                        fit = candidates[mode]
                        pdf = self._map_mode_fit_pdf(grid, fit)
                        finite = np.isfinite(pdf)
                        if np.any(finite):
                            density_y_max = max(density_y_max, float(np.nanpercentile(pdf[finite], 99.0)))
                            ax_density.plot(
                                grid[finite],
                                pdf[finite],
                                color=colors[mode],
                                lw=2.6 if mode == selected_mode else 1.35,
                                alpha=0.95 if mode == selected_mode else 0.62,
                                linestyle="-" if mode == selected_mode else "--",
                                label=labels[mode],
                            )
                        if fit.vector is not None:
                            eta = float(np.asarray(fit.vector, dtype=float).reshape(-1)[0])
                            ax.axvline(
                                eta,
                                color=colors[mode],
                                lw=1.0,
                                alpha=0.35,
                            )
                        prefix = self._map_mode_markers(mode, best_mode=best_mode, true_mode=true_mode)
                        cost_lines.append(f"{prefix}{mode}: {float(fit.cost):.1f}")
                    shared_mode = self._kind_to_mode(self.shared_param_kinds[stage_idx][feat_idx])
                    shared_vec = self.shared_param_vectors[stage_idx][feat_idx]
                    if shared_vec is not None:
                        shared_eta = float(np.asarray(shared_vec, dtype=float).reshape(-1)[0])
                        ax.axvline(shared_eta, color="#111827", lw=1.6, linestyle="-.", alpha=0.85, label="shared param")
                    for axis in (ax, ax_density):
                        handles, handle_labels = axis.get_legend_handles_labels()
                        for handle, handle_label in zip(handles, handle_labels):
                            text = str(handle_label).strip() if handle_label is not None else ""
                            if text and not text.startswith("_") and text not in legend_items:
                                legend_items[text] = handle
                    title = (
                        f"s{stage_idx + 1} {self._map_feature_name(feat_idx)} "
                        f"[{int(s)},{int(e)}] demo-best={best_mode} shared={shared_mode} "
                        f"true={'?' if true_mode is None else '#' + true_mode}"
                    )
                    ax.set_title(title, fontsize=8)
                    ax.text(
                        0.02,
                        0.98,
                        "\n".join(cost_lines),
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=6.5,
                        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.82},
                    )
                    if hist_y_max > 0.0 and np.isfinite(hist_y_max):
                        ax.set_ylim(0.0, hist_y_max * 1.18)
                    if density_y_max > 0.0 and np.isfinite(density_y_max):
                        ax_density.set_ylim(0.0, density_y_max * 1.18)
                    ax.tick_params(labelsize=6.5)
                    ax_density.tick_params(labelsize=6.0, colors="#475569")
                    if stage_idx == self.num_stages - 1:
                        ax.set_xlabel("standardized feature value", fontsize=7)
                    if feat_idx == 0:
                        ax.set_ylabel("count", fontsize=7)
                    if feat_idx == self.num_features - 1:
                        ax_density.set_ylabel("density", fontsize=7, color="#475569")
                    else:
                        ax_density.set_yticklabels([])
            if legend_handles:
                fig.legend(
                    legend_handles,
                    legend_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.965),
                    ncol=min(len(legend_handles), 6),
                    fontsize=7,
                    frameon=False,
                )
            elif legend_items:
                fig.legend(
                    legend_items.values(),
                    legend_items.keys(),
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.965),
                    ncol=min(len(legend_items), 5),
                    fontsize=7,
                    frameon=False,
                )
            fig.suptitle(f"MAP shared-parameter mode costs | demo {demo_idx} | iter {plot_it:04d}", fontsize=11, y=0.995)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), pad=0.7)
            save_figure(fig, out_dir / f"density_demo_{int(demo_idx):02d}_iter_{int(plot_it):04d}.png", dpi=180)

    def _plot_map_vote_summary(
        self,
        iteration: int,
        selected_infos: Sequence[dict],
        total_loss: float,
        *,
        force: bool = False,
    ) -> None:
        should_plot, plot_it = self._map_should_plot_iteration(int(iteration))
        if not force and not should_plot:
            return
        if force:
            plot_it = max(int(iteration), 0)
        out_dir = learner_plot_dir(self)
        n_demos = len(selected_infos)
        col_labels = [f"d{idx}" for idx in range(n_demos)] + ["shared"]
        fig, axes = plt.subplots(
            self.num_stages,
            1,
            figsize=(max(7.5, 1.35 * (n_demos + 1)), max(2.2, 1.15 * self.num_stages * self.num_features)),
            squeeze=False,
        )
        for stage_idx in range(self.num_stages):
            ax = axes[stage_idx][0]
            ax.axis("off")
            rows = []
            row_labels = []
            for feat_idx in range(self.num_features):
                row_labels.append(self._map_feature_name(feat_idx))
                cells = []
                local_modes = []
                for info in selected_infos:
                    stage_params = info["stage_params"][stage_idx]
                    mode = self._kind_to_mode(self._stage_feature_kind(stage_params, feat_idx))
                    local_modes.append(mode)
                    vec = self._stage_feature_vector(stage_params, feat_idx)
                    cost = float(np.asarray(stage_params.feature_constraint_costs, dtype=float)[feat_idx])
                    cells.append(f"{mode}\np={self._map_param_text(mode, vec)}\nc={cost:.1f}")
                shared_mode = self._kind_to_mode(self.shared_param_kinds[stage_idx][feat_idx])
                shared_vec = self.shared_param_vectors[stage_idx][feat_idx]
                mstep_costs = {}
                try:
                    mstep_costs = dict(self.map_shared_mode_costs_[stage_idx][feat_idx])
                except Exception:
                    mstep_costs = {}
                costs_text = " ".join(
                    f"{mode}:{float(mstep_costs[mode]):.1f}"
                    for mode in ("inactive", "eq", "lb", "ub")
                    if mode in mstep_costs and np.isfinite(float(mstep_costs[mode]))
                )
                vote_info = {}
                try:
                    vote_info = dict(self.map_shared_mode_votes_[stage_idx][feat_idx])
                except Exception:
                    vote_info = {}
                vote_counts = dict(vote_info.get("vote_counts", {}))
                votes_text = " ".join(
                    f"{mode}:{int(vote_counts[mode])}"
                    for mode in ("inactive", "eq", "lb", "ub")
                    if int(vote_counts.get(mode, 0)) > 0
                )
                aggregation = str(vote_info.get("aggregation", self.map_mode_aggregation))
                shared_lines = [
                    f"{shared_mode} [{aggregation}]",
                    f"p={self._map_param_text(shared_mode, shared_vec)}",
                ]
                if votes_text:
                    shared_lines.append(f"votes {votes_text}")
                shared_lines.append(costs_text)
                cells.append("\n".join(shared_lines))
                rows.append(cells)
            table = ax.table(
                cellText=rows,
                rowLabels=row_labels,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
                rowLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.0, 1.55)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor("#CBD5E1")
                cell.set_linewidth(0.6)
                if row == 0 or col == -1:
                    cell.set_facecolor("#F1F5F9")
                elif col == n_demos:
                    cell.set_facecolor("#FEF3C7")
                else:
                    cell.set_facecolor("white")
            ax.set_title(f"stage {stage_idx + 1}", fontsize=10, pad=8)
        fig.suptitle(f"MAP M-step summary | iter {plot_it:04d} | total={float(total_loss):.3f}", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), pad=0.6)
        save_figure(fig, out_dir / f"map_mstep_summary_iter_{int(plot_it):04d}.png", dpi=190)

    def _plot_map_diagnostics(self, iteration: int, selected_infos: Sequence[dict], total_loss: float) -> None:
        try:
            self._plot_map_pooled_mode_density_diagnostics(int(iteration), selected_infos)
            self._plot_map_mode_density_diagnostics(int(iteration), selected_infos)
            self._plot_map_vote_summary(int(iteration), selected_infos, float(total_loss))
        except Exception as exc:
            if self.verbose:
                print(f"[MAP] diagnostic plots skipped at iter {int(iteration) + 1:04d}: {exc}")

    def _plot_map_final_pooled_diagnostics(self, iteration: int, selected_infos: Sequence[dict]) -> None:
        try:
            self._plot_map_pooled_mode_density_diagnostics(int(iteration), selected_infos, force=True)
        except Exception as exc:
            if self.verbose:
                print(f"[MAP] final pooled distribution plot skipped: {exc}")
        try:
            self._plot_map_mode_density_diagnostics(int(iteration), selected_infos, force=True)
        except Exception as exc:
            if self.verbose:
                print(f"[MAP] final per-demo distribution plots skipped: {exc}")
        try:
            total_loss = float(self.loss_total[-1]) if self.loss_total else float("nan")
            self._plot_map_vote_summary(int(iteration), selected_infos, total_loss, force=True)
        except Exception as exc:
            if self.verbose:
                print(f"[MAP] final M-step vote summary plot skipped: {exc}")

    def _mode_fit_likelihood_cost(self, fit: _MAPModeFit) -> float:
        return float(fit.summary.get("nll", 0.0))

    def _mode_prior_probability(self, mode: str, feat_idx: int) -> float:
        mode_l = str(mode).lower()
        feat_idx = int(feat_idx)
        activation_probability = float(self.map_activation_prior[feat_idx])
        if mode_l == "inactive":
            return float(1.0 - activation_probability)
        if mode_l not in self.map_active_mode_prior:
            raise ValueError(f"Unknown MAP mode '{mode}'.")
        return float(activation_probability * self.map_active_mode_prior[mode_l][feat_idx])

    def _mode_prior_cost(self, mode: str, feat_idx: int | None = None) -> float:
        mode_l = str(mode).lower()
        if feat_idx is None:
            return 0.0
        probability = self._mode_prior_probability(mode_l, int(feat_idx))
        if probability <= 0.0:
            return float("inf")
        return float(-math.log(probability))

    def _student_t_sum_nll(self, xs, *, mu: float, sigma: float, nu: float) -> float:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        if xs.size == 0:
            return 0.0
        return float(xs.size * self._student_t_profile_nll_from_params(xs, float(mu), float(sigma), float(nu)))

    def _student_t_nll_values(self, xs, *, mu: float, sigma: float, nu: float) -> np.ndarray:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        if xs.size == 0:
            return np.zeros(0, dtype=float)
        sigma = max(float(sigma), 1e-9)
        nu = max(float(nu), 1e-6)
        z = (xs - float(mu)) / sigma
        log_norm = (
            math.lgamma(0.5 * (nu + 1.0))
            - math.lgamma(0.5 * nu)
            - 0.5 * math.log(nu * math.pi)
            - math.log(sigma)
        )
        logpdf = log_norm - 0.5 * (nu + 1.0) * np.log1p((z * z) / nu)
        return -np.asarray(logpdf, dtype=float)

    def _gaussian_sum_nll(self, xs, *, mu: float, sigma: float) -> float:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        if xs.size == 0:
            return 0.0
        sigma = max(float(sigma), 1e-9)
        z = (xs - float(mu)) / sigma
        return float(np.sum(0.5 * z * z + math.log(sigma) + 0.5 * math.log(2.0 * math.pi)))

    def _gaussian_nll_values(self, xs, *, mu: float, sigma: float) -> np.ndarray:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        if xs.size == 0:
            return np.zeros(0, dtype=float)
        sigma = max(float(sigma), 1e-9)
        z = (xs - float(mu)) / sigma
        return np.asarray(0.5 * z * z + math.log(sigma) + 0.5 * math.log(2.0 * math.pi), dtype=float)

    def _eq_sum_nll(self, xs, *, eta: float) -> float:
        if self.map_eq_distribution == "gaussian":
            return self._gaussian_sum_nll(xs, mu=float(eta), sigma=float(self.map_eq_sigma))
        return self._student_t_sum_nll(xs, mu=float(eta), sigma=float(self.map_eq_sigma), nu=float(self.map_nu_eq))

    def _eq_nll_values(self, xs, *, eta: float) -> np.ndarray:
        if self.map_eq_distribution == "gaussian":
            return self._gaussian_nll_values(xs, mu=float(eta), sigma=float(self.map_eq_sigma))
        return self._student_t_nll_values(xs, mu=float(eta), sigma=float(self.map_eq_sigma), nu=float(self.map_nu_eq))

    def _inactive_fit(self, xs, *, feat_idx: int | None = None) -> _MAPModeFit:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        if xs.size == 0:
            mu = 0.0
            scale = float(self.map_c_bg * self.map_eq_sigma)
            nll = 0.0
        else:
            mu, scale, _ = self._robust_student_t_profile_params(xs, nu=self.map_nu_inactive)
            scale = max(float(scale), float(self.map_c_bg * self.map_eq_sigma))
            if self.map_inactive_distribution == "gaussian":
                nll = self._gaussian_sum_nll(xs, mu=mu, sigma=scale)
            else:
                nll = self._student_t_sum_nll(xs, mu=mu, sigma=scale, nu=self.map_nu_inactive)
        prior_cost = self._mode_prior_cost("inactive", feat_idx)
        summary = {
            "type": f"inactive_{self.map_inactive_distribution}",
            "distribution": self.map_inactive_distribution,
            "mode": "inactive",
            "mu": float(mu),
            "sigma": float(scale),
            "nu": float(self.map_nu_inactive),
            "nll": float(nll),
            "prior_nll": float(prior_cost),
            "bg_sigma_min": float(self.map_c_bg * self.map_eq_sigma),
        }
        return _MAPModeFit(
            mode="inactive",
            kind=None,
            eta=None,
            scale=float(scale),
            cost=float(nll + prior_cost),
            summary=summary,
            vector=None,
        )

    def _inactive_fit_from_stats(self, xs, stats: dict, *, feat_idx: int | None = None) -> _MAPModeFit:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        if xs.size == 0:
            return self._inactive_fit(xs, feat_idx=feat_idx)
        mu = float(stats["q50"])
        if self.map_inactive_distribution == "gaussian":
            unit_width = 2.5631031310892007
        else:
            unit_width = self._student_t_ppf(0.90, self.map_nu_inactive) - self._student_t_ppf(0.10, self.map_nu_inactive)
        scale = float(stats["q90"] - stats["q10"]) / max(float(unit_width), 1e-12)
        scale = max(float(scale), float(self.map_c_bg * self.map_eq_sigma), 1e-12)
        if self.map_inactive_distribution == "gaussian":
            nll = self._gaussian_sum_nll(xs, mu=mu, sigma=scale)
        else:
            nll = self._student_t_sum_nll(xs, mu=mu, sigma=scale, nu=self.map_nu_inactive)
        prior_cost = self._mode_prior_cost("inactive", feat_idx)
        summary = {
            "type": f"inactive_{self.map_inactive_distribution}",
            "distribution": self.map_inactive_distribution,
            "mode": "inactive",
            "mu": float(mu),
            "sigma": float(scale),
            "nu": float(self.map_nu_inactive),
            "nll": float(nll),
            "prior_nll": float(prior_cost),
            "bg_sigma_min": float(self.map_c_bg * self.map_eq_sigma),
        }
        return _MAPModeFit(
            mode="inactive",
            kind=None,
            eta=None,
            scale=float(scale),
            cost=float(nll + prior_cost),
            summary=summary,
            vector=None,
        )

    def _equality_fit(self, xs, *, eta: float | None = None, feat_idx: int | None = None) -> _MAPModeFit:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        eta_hat = float(np.median(xs)) if eta is None and xs.size else float(0.0 if eta is None else eta)
        nll = self._eq_sum_nll(xs, eta=eta_hat)
        prior_cost = self._mode_prior_cost("eq", feat_idx)
        cost = float(nll + prior_cost)
        kind = self._mode_to_kind("eq")
        summary = {
            "type": kind,
            "distribution": self.map_eq_distribution,
            "mode": "eq",
            "mu": float(eta_hat),
            "sigma": float(self.map_eq_sigma),
            "nu": float(self.map_nu_eq),
            "L": float(eta_hat - self.map_eq_sigma),
            "U": float(eta_hat + self.map_eq_sigma),
            "nll": float(nll),
            "prior_nll": float(prior_cost),
            "fixed_sigma": float(self.map_eq_sigma),
        }
        vector = np.asarray([float(eta_hat), math.log(float(self.map_eq_sigma))], dtype=float)
        return _MAPModeFit(
            mode="eq",
            kind=kind,
            eta=float(eta_hat),
            scale=float(self.map_eq_sigma),
            cost=cost,
            summary=summary,
            vector=vector,
        )

    def _equality_fit_from_stats(self, xs, stats: dict, *, feat_idx: int | None = None) -> _MAPModeFit:
        return self._equality_fit(xs, eta=float(stats["q50"]), feat_idx=feat_idx)

    def _signed_slack(self, xs, *, mode: str, eta: float):
        xs = np.asarray(xs, dtype=float).reshape(-1)
        if str(mode).lower() == "lb":
            return xs - float(eta)
        if str(mode).lower() == "ub":
            return float(eta) - xs
        raise ValueError(f"Unknown inequality mode '{mode}'.")

    def _inequality_profile_scale(self, xs, *, mode: str, eta: float) -> float:
        signed = self._signed_slack(xs, mode=mode, eta=float(eta))
        feasible_slack = np.maximum(np.asarray(signed, dtype=float), 0.0)
        scale, _ = self._half_t_quantile_profile_params(feasible_slack, nu=self.map_nu_ineq)
        scale_floor = max(float(self._truncated_z_scale_floor()), float(self.map_c_ineq * self.map_eq_sigma))
        return max(float(scale), float(scale_floor), 1e-6)

    def _inequality_sum_nll(self, xs, *, mode: str, eta: float, scale: float) -> float:
        direction = "lower" if str(mode).lower() == "lb" else "upper"
        softness = max(float(self.truncated_z_soft_boundary_scale) * float(scale), 1e-12)
        mean_nll = self._soft_half_t_profile_nll_on_x(
            xs,
            b=float(eta),
            scale=float(scale),
            nu=float(self.map_nu_ineq),
            softness=float(softness),
            direction=direction,
        )
        return float(len(np.asarray(xs).reshape(-1)) * mean_nll)

    def _inequality_fit(self, xs, *, mode: str, eta: float | None = None, feat_idx: int | None = None) -> _MAPModeFit:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        mode_l = str(mode).lower()
        if mode_l not in {"lb", "ub"}:
            raise ValueError(f"Unknown inequality mode '{mode}'.")
        if eta is None:
            q = self.map_boundary_quantile if mode_l == "lb" else 1.0 - self.map_boundary_quantile
            eta_hat = float(np.quantile(xs, q)) if xs.size else 0.0
        else:
            eta_hat = float(eta)
        scale = self._inequality_profile_scale(xs, mode=mode_l, eta=eta_hat)
        nll = self._inequality_sum_nll(xs, mode=mode_l, eta=eta_hat, scale=scale)
        prior_cost = self._mode_prior_cost(mode_l, feat_idx)
        cost = float(nll + prior_cost)
        q50 = float(np.median(xs)) if xs.size else float(eta_hat)
        kind = "trunc_t_lower_z" if mode_l == "lb" else "trunc_t_upper_z"
        summary = {
            "type": kind,
            "mode": mode_l,
            "b": float(eta_hat),
            "mu": float(q50),
            "sigma": float(scale),
            "nu": float(self.map_nu_ineq),
            "nll": float(nll),
            "prior_nll": float(prior_cost),
            "soft_boundary_scale": float(self.truncated_z_soft_boundary_scale),
            "ineq_sigma_min": float(self.map_c_ineq * self.map_eq_sigma),
            "boundary_quantile": float(self.map_boundary_quantile if mode_l == "lb" else 1.0 - self.map_boundary_quantile),
        }
        vector = np.asarray([float(eta_hat), float(q50), math.log(float(scale))], dtype=float)
        return _MAPModeFit(
            mode=mode_l,
            kind=kind,
            eta=float(eta_hat),
            scale=float(scale),
            cost=cost,
            summary=summary,
            vector=vector,
        )

    def _inequality_fit_from_stats(
        self,
        xs,
        *,
        mode: str,
        eta: float | None,
        stats: dict,
        feat_idx: int | None = None,
    ) -> _MAPModeFit:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        mode_l = str(mode).lower()
        if mode_l not in {"lb", "ub"}:
            raise ValueError(f"Unknown inequality mode '{mode}'.")
        if eta is None:
            eta_hat = float(stats["q_boundary_low"] if mode_l == "lb" else stats["q_boundary_high"])
        else:
            eta_hat = float(eta)
        q_scale = float(getattr(self, "truncated_z_half_t_scale_quantile", 0.9))
        if mode_l == "lb":
            slack_q = max(float(stats["q_scale_high"]) - eta_hat, 0.0)
        else:
            slack_q = max(eta_hat - float(stats["q_scale_low"]), 0.0)
        unit_q = self._student_t_ppf(0.5 + 0.5 * q_scale, self.map_nu_ineq)
        scale = slack_q / max(float(unit_q), 1e-12)
        scale_floor = max(float(self._truncated_z_scale_floor()), float(self.map_c_ineq * self.map_eq_sigma))
        scale = max(float(scale), float(scale_floor), 1e-6)
        nll = self._inequality_sum_nll(xs, mode=mode_l, eta=eta_hat, scale=scale)
        prior_cost = self._mode_prior_cost(mode_l, feat_idx)
        cost = float(nll + prior_cost)
        kind = "trunc_t_lower_z" if mode_l == "lb" else "trunc_t_upper_z"
        summary = {
            "type": kind,
            "mode": mode_l,
            "b": float(eta_hat),
            "mu": float(stats["q50"]),
            "sigma": float(scale),
            "nu": float(self.map_nu_ineq),
            "nll": float(nll),
            "prior_nll": float(prior_cost),
            "soft_boundary_scale": float(self.truncated_z_soft_boundary_scale),
            "ineq_sigma_min": float(self.map_c_ineq * self.map_eq_sigma),
            "boundary_quantile": float(self.map_boundary_quantile if mode_l == "lb" else 1.0 - self.map_boundary_quantile),
        }
        vector = np.asarray([float(eta_hat), float(stats["q50"]), math.log(float(scale))], dtype=float)
        return _MAPModeFit(
            mode=mode_l,
            kind=kind,
            eta=float(eta_hat),
            scale=float(scale),
            cost=cost,
            summary=summary,
            vector=vector,
        )

    def _local_mode_candidates(self, xs, *, feat_idx: int | None = None) -> Dict[str, _MAPModeFit]:
        return {
            "inactive": self._inactive_fit(xs, feat_idx=feat_idx),
            "eq": self._equality_fit(xs, feat_idx=feat_idx),
            "lb": self._inequality_fit(xs, mode="lb", feat_idx=feat_idx),
            "ub": self._inequality_fit(xs, mode="ub", feat_idx=feat_idx),
        }

    def _local_mode_candidates_cached(self, demo_idx: int, core_s: int, core_e: int, feat_idx: int) -> Dict[str, _MAPModeFit]:
        key = (int(demo_idx), int(core_s), int(core_e), int(feat_idx))
        cached = self._map_local_mode_cache.get(key)
        if cached is not None:
            return cached
        xs = self.standardized_features[int(demo_idx)][int(core_s) : int(core_e) + 1, int(feat_idx)]
        stats = self._interval_feature_stats_cached(int(demo_idx), int(core_s), int(core_e), int(feat_idx))
        out = {
            "inactive": self._inactive_fit_from_stats(xs, stats, feat_idx=int(feat_idx)),
            "eq": self._equality_fit_from_stats(xs, stats, feat_idx=int(feat_idx)),
            "lb": self._inequality_fit_from_stats(xs, mode="lb", eta=None, stats=stats, feat_idx=int(feat_idx)),
            "ub": self._inequality_fit_from_stats(xs, mode="ub", eta=None, stats=stats, feat_idx=int(feat_idx)),
        }
        self._map_local_mode_cache[key] = out
        if "inactive" in out:
            self._map_inactive_fit_cache[key] = out["inactive"]
        return out

    def _inactive_fit_cached(self, demo_idx: int, core_s: int, core_e: int, feat_idx: int) -> _MAPModeFit:
        key = (int(demo_idx), int(core_s), int(core_e), int(feat_idx))
        cached = self._map_inactive_fit_cache.get(key)
        if cached is not None:
            return cached
        xs = self.standardized_features[int(demo_idx)][int(core_s) : int(core_e) + 1, int(feat_idx)]
        stats = self._interval_feature_stats_cached(int(demo_idx), int(core_s), int(core_e), int(feat_idx))
        fit = self._inactive_fit_from_stats(xs, stats, feat_idx=int(feat_idx))
        self._map_inactive_fit_cache[key] = fit
        return fit

    def _interval_feature_stats_cached(self, demo_idx: int, core_s: int, core_e: int, feat_idx: int) -> dict:
        key = (int(demo_idx), int(core_s), int(core_e), int(feat_idx))
        cached = self._map_interval_stats_cache.get(key)
        if cached is not None:
            return cached
        xs = self.standardized_features[int(demo_idx)][int(core_s) : int(core_e) + 1, int(feat_idx)]
        xs = np.asarray(xs, dtype=float).reshape(-1)
        if xs.size == 0:
            stats = {
                "n": 0,
                "q10": 0.0,
                "q50": 0.0,
                "q90": 0.0,
                "q_boundary_low": 0.0,
                "q_boundary_high": 0.0,
                "q_scale_low": 0.0,
                "q_scale_high": 0.0,
            }
            self._map_interval_stats_cache[key] = stats
            return stats
        q_scale = float(np.clip(float(getattr(self, "truncated_z_half_t_scale_quantile", 0.9)), 0.5, 0.99))
        requests = {
            "q10": 0.10,
            "q50": 0.50,
            "q90": 0.90,
            "q_boundary_low": float(self.map_boundary_quantile),
            "q_boundary_high": 1.0 - float(self.map_boundary_quantile),
            "q_scale_low": 1.0 - q_scale,
            "q_scale_high": q_scale,
        }
        unique_q = sorted({float(np.clip(q, 0.0, 1.0)) for q in requests.values()})
        values = np.quantile(xs, unique_q)
        q_lookup = {q: float(v) for q, v in zip(unique_q, np.asarray(values, dtype=float).reshape(-1))}
        stats = {"n": int(xs.size)}
        for name, q in requests.items():
            stats[name] = float(q_lookup[float(np.clip(q, 0.0, 1.0))])
        self._map_interval_stats_cache[key] = stats
        return stats

    def _best_local_mode_fit(self, xs, *, feat_idx: int | None = None) -> _MAPModeFit:
        candidates = self._local_mode_candidates(xs, feat_idx=feat_idx)
        order = {"inactive": 0, "eq": 1, "lb": 2, "ub": 3}
        return min(candidates.values(), key=lambda item: (float(item.cost), order.get(item.mode, 99)))

    def _shared_candidate_mode_fits_for_interval(
        self,
        demo_idx: int,
        core_s: int,
        core_e: int,
        feat_idx: int,
        shared_candidate_fits: Dict[str, _MAPModeFit],
    ) -> Dict[str, _MAPModeFit]:
        xs = np.asarray(
            self.standardized_features[int(demo_idx)][int(core_s) : int(core_e) + 1, int(feat_idx)],
            dtype=float,
        ).reshape(-1)
        stats = self._interval_feature_stats_cached(int(demo_idx), int(core_s), int(core_e), int(feat_idx))
        eq_eta = float(np.asarray(shared_candidate_fits["eq"].vector, dtype=float).reshape(-1)[0])
        lb_eta = float(np.asarray(shared_candidate_fits["lb"].vector, dtype=float).reshape(-1)[0])
        ub_eta = float(np.asarray(shared_candidate_fits["ub"].vector, dtype=float).reshape(-1)[0])
        return {
            "inactive": self._inactive_fit_cached(int(demo_idx), int(core_s), int(core_e), int(feat_idx)),
            "eq": self._equality_fit(xs, eta=eq_eta, feat_idx=int(feat_idx)),
            "lb": self._inequality_fit_from_stats(
                xs,
                mode="lb",
                eta=lb_eta,
                stats=stats,
                feat_idx=int(feat_idx),
            ),
            "ub": self._inequality_fit_from_stats(
                xs,
                mode="ub",
                eta=ub_eta,
                stats=stats,
                feat_idx=int(feat_idx),
            ),
        }

    def _shared_mode_fit(self, xs, *, stage_idx: int, feat_idx: int) -> _MAPModeFit:
        kind = None
        eta = None
        try:
            kind = self.shared_param_kinds[int(stage_idx)][int(feat_idx)]
            vec = self.shared_param_vectors[int(stage_idx)][int(feat_idx)]
            if vec is not None:
                eta = float(np.asarray(vec, dtype=float).reshape(-1)[0])
        except Exception:
            kind = None
            eta = None
        mode = self._kind_to_mode(kind)
        if mode == "eq" and eta is not None:
            return self._equality_fit(xs, eta=eta, feat_idx=int(feat_idx))
        if mode in {"lb", "ub"} and eta is not None:
            return self._inequality_fit(xs, mode=mode, eta=eta, feat_idx=int(feat_idx))
        return self._inactive_fit(xs, feat_idx=int(feat_idx))

    def _shared_eq_prefix(self, demo_idx: int, feat_idx: int, eta: float) -> np.ndarray:
        key = (
            int(self._map_shared_cache_version),
            int(demo_idx),
            int(feat_idx),
            round(float(eta), 12),
        )
        cached = self._map_shared_eq_prefix_cache.get(key)
        if cached is not None:
            return cached
        xs = self.standardized_features[int(demo_idx)][:, int(feat_idx)]
        nll = self._eq_nll_values(xs, eta=float(eta))
        prefix = np.concatenate([[0.0], np.cumsum(np.asarray(nll, dtype=float), dtype=float)])
        self._map_shared_eq_prefix_cache[key] = prefix
        return prefix

    @staticmethod
    def _prefix_interval_sum(prefix: np.ndarray, s: int, e: int) -> float:
        return float(np.asarray(prefix, dtype=float)[int(e) + 1] - np.asarray(prefix, dtype=float)[int(s)])

    def _shared_feature_interval_cost(self, demo_idx: int, stage_idx: int, feat_idx: int, s: int, e: int) -> float:
        if int(feat_idx) in self.force_inactive_feature_indices:
            return 0.0
        core_s, core_e = self._segment_core_bounds(s, e)
        kind = None
        vec = None
        try:
            kind = self.shared_param_kinds[int(stage_idx)][int(feat_idx)]
            vec = self.shared_param_vectors[int(stage_idx)][int(feat_idx)]
        except Exception:
            kind = None
            vec = None
        mode = self._kind_to_mode(kind)
        if mode == "inactive" or vec is None:
            fit = self._inactive_fit_cached(int(demo_idx), int(core_s), int(core_e), int(feat_idx))
            return self._mode_fit_likelihood_cost(fit)

        eta = float(np.asarray(vec, dtype=float).reshape(-1)[0])
        if mode == "eq":
            prefix = self._shared_eq_prefix(int(demo_idx), int(feat_idx), float(eta))
            return self._prefix_interval_sum(prefix, int(core_s), int(core_e))

        if mode in {"lb", "ub"}:
            xs = self.standardized_features[int(demo_idx)][int(core_s) : int(core_e) + 1, int(feat_idx)]
            stats = self._interval_feature_stats_cached(int(demo_idx), int(core_s), int(core_e), int(feat_idx))
            fit = self._inequality_fit_from_stats(
                xs,
                mode=mode,
                eta=eta,
                stats=stats,
            )
            return self._mode_fit_likelihood_cost(fit)

        fit = self._inactive_fit_cached(int(demo_idx), int(core_s), int(core_e), int(feat_idx))
        return self._mode_fit_likelihood_cost(fit)

    def _shared_interval_constraint_cost(self, demo_idx: int, stage_idx: int, s: int, e: int) -> float:
        key = (
            int(self._map_shared_cache_version),
            int(demo_idx),
            int(stage_idx),
            int(s),
            int(e),
        )
        cached = self._map_shared_cost_cache.get(key)
        if cached is not None:
            return float(cached)
        total = 0.0
        for feat_idx in range(self.num_features):
            total += self._shared_feature_interval_cost(int(demo_idx), int(stage_idx), int(feat_idx), int(s), int(e))
        self._map_shared_cost_cache[key] = float(total)
        return float(total)

    def _compute_stage_feature_free_params_uncached(self, demo_idx: int, s: int, e: int) -> _StageParams:
        core_s, core_e = self._segment_core_bounds(s, e)
        F = self.standardized_features[int(demo_idx)][core_s : core_e + 1]
        summaries = []
        selected_kinds = []
        vectors = []
        active_mask = np.zeros(self.num_features, dtype=int)
        feature_scores = np.zeros(self.num_features, dtype=float)
        feature_constraint_costs = np.zeros(self.num_features, dtype=float)
        for feat_idx in range(self.num_features):
            xs = np.asarray(F[:, feat_idx], dtype=float).reshape(-1)
            candidates = self._local_mode_candidates_cached(int(demo_idx), int(core_s), int(core_e), int(feat_idx))
            inactive = candidates["inactive"]
            order = {"inactive": 0, "eq": 1, "lb": 2, "ub": 3}
            fit = min(candidates.values(), key=lambda item: (float(item.cost), order.get(item.mode, 99)))
            summaries.append(dict(fit.summary))
            selected_kinds.append(str(fit.kind) if fit.kind is not None else "inactive")
            vectors.append(None if fit.vector is None else np.asarray(fit.vector, dtype=float))
            active_mask[feat_idx] = int(fit.mode != "inactive")
            feature_scores[feat_idx] = float((fit.cost - inactive.cost) / max(xs.size, 1))
            feature_constraint_costs[feat_idx] = float(fit.cost)
        if self.force_inactive_feature_indices:
            for feat_idx in self.force_inactive_feature_indices:
                active_mask[feat_idx] = 0
                selected_kinds[feat_idx] = "inactive"
                vectors[feat_idx] = None
                feature_constraint_costs[feat_idx] = 0.0
        subgoal = np.asarray(self.demos[int(demo_idx)][int(e)], dtype=float)
        return _StageParams(
            model_summaries=summaries,
            subgoal=subgoal,
            active_mask=active_mask,
            feature_scores=feature_scores,
            feature_constraint_costs=feature_constraint_costs,
            selected_feature_kinds=selected_kinds,
            param_vectors=vectors,
        )

    @staticmethod
    def _map_progress_log_normalizer_ratio(kappa: float) -> float:
        kappa = float(kappa)
        if kappa <= 1e-6:
            kappa_sq = kappa * kappa
            return float(kappa_sq / 6.0 - (kappa_sq * kappa_sq) / 180.0)
        return float(kappa + math.log1p(-math.exp(-2.0 * kappa)) - math.log(2.0 * kappa))

    @staticmethod
    def _map_progress_values(X, s: int, e: int, subgoal) -> np.ndarray:
        if int(e) <= int(s):
            return np.empty(0, dtype=float)
        trajectory = np.asarray(X, dtype=float)
        start_t = max(int(s), 0)
        end_t = min(int(e) - 1, len(trajectory) - 2)
        if end_t < start_t:
            return np.empty(0, dtype=float)
        segment = trajectory[start_t : end_t + 2]
        steps = segment[1:] - segment[:-1]
        step_norms = np.linalg.norm(steps, axis=1)
        valid = step_norms > 1e-12
        if not np.any(valid):
            return np.empty(0, dtype=float)
        goal = np.asarray(subgoal, dtype=float).reshape(1, -1)
        distance_before = np.linalg.norm(segment[:-1] - goal, axis=1)
        distance_after = np.linalg.norm(segment[1:] - goal, axis=1)
        progress = (distance_before[valid] - distance_after[valid]) / step_norms[valid]
        return np.clip(np.asarray(progress, dtype=float), -1.0, 1.0)

    def _map_progress_cost(self, X, s: int, e: int, subgoal, stage_idx: int) -> float:
        progress = self._map_progress_values(X, int(s), int(e), subgoal)
        if progress.size == 0:
            return 0.0
        kappa = float(self.map_progress_kappas_[int(stage_idx)])
        log_normalizer_ratio = self._map_progress_log_normalizer_ratio(kappa)
        return float(progress.size * log_normalizer_ratio - kappa * float(np.sum(progress)))

    def _fit_map_progress_kappas(self, selected_infos: Sequence[dict]) -> np.ndarray:
        if self.map_progress_kappa is not None:
            return np.full(self.num_stages, float(self.map_progress_kappa), dtype=float)

        fitted = np.zeros(self.num_stages, dtype=float)
        for stage_idx in range(self.num_stages):
            stage_progress = []
            for demo_idx, info in enumerate(selected_infos):
                bounds = self._segment_bounds_from_stage_ends(info["stage_ends"])
                s, e = bounds[int(stage_idx)]
                values = self._map_progress_values(
                    self.demos[int(demo_idx)],
                    int(s),
                    int(e),
                    info["stage_params"][int(stage_idx)].subgoal,
                )
                if values.size:
                    stage_progress.append(values)
            if not stage_progress:
                continue
            mean_progress = float(np.mean(np.concatenate(stage_progress)))
            if mean_progress <= 0.0:
                continue

            def objective(kappa):
                return self._map_progress_log_normalizer_ratio(float(kappa)) - float(kappa) * mean_progress

            result = minimize_scalar(
                objective,
                bounds=(0.0, float(self.map_progress_kappa_max)),
                method="bounded",
                options={"xatol": 1e-8},
            )
            if result.success and np.isfinite(result.x):
                fitted[stage_idx] = float(np.clip(result.x, 0.0, self.map_progress_kappa_max))
        return fitted

    def _free_segment_fit(self, demo_idx: int, stage_idx: int, s: int, e: int) -> tuple[_StageParams, float, float]:
        key = (int(demo_idx), int(s), int(e))
        cached = self._map_free_segment_cache.get(key)
        if cached is None:
            stage_params = self._compute_stage_feature_free_params_uncached(int(demo_idx), int(s), int(e))
            constraint_cost = float(np.sum(stage_params.feature_constraint_costs))
            cached = (stage_params, constraint_cost)
            self._map_free_segment_cache[key] = cached
        stage_params, constraint_cost = cached
        progress_cost = self._map_progress_cost(
            self.demos[int(demo_idx)],
            int(s),
            int(e),
            stage_params.subgoal,
            int(stage_idx),
        )
        return stage_params, float(constraint_cost), float(progress_cost)

    def _stage_feature_free_params(self, demo_idx: int, s: int, e: int) -> _StageParams:
        key = (int(demo_idx), int(s), int(e))
        cached = self._map_free_segment_cache.get(key)
        if cached is None:
            stage_params = self._compute_stage_feature_free_params_uncached(int(demo_idx), int(s), int(e))
            constraint_cost = float(np.sum(stage_params.feature_constraint_costs))
            cached = (stage_params, constraint_cost)
            self._map_free_segment_cache[key] = cached
        return cached[0]

    def _fit_segment_stage(self, demo_idx, stage_idx, s, e):
        return self._free_segment_fit(int(demo_idx), int(stage_idx), int(s), int(e))

    def _free_interval_cost_info(self, demo_idx: int, stage_idx: int, s: int, e: int):
        stage_len = int(e - s + 1)
        if stage_len < int(self.duration_min[stage_idx]) or stage_len > int(self.duration_max[stage_idx]):
            return None
        stage_params, constraint_cost, progress_cost = self._free_segment_fit(
            int(demo_idx), int(stage_idx), int(s), int(e)
        )
        total = float(constraint_cost + progress_cost)
        return {
            "stage_idx": int(stage_idx),
            "s": int(s),
            "e": int(e),
            "stage_params": stage_params,
            "constraint": constraint_cost,
            "short_segment_penalty": 0.0,
            "progress": float(progress_cost),
            "param_consensus": 0.0,
            "activation_consensus": 0.0,
            "weighted_total": total,
        }

    def _shared_interval_cost_info(self, demo_idx: int, stage_idx: int, s: int, e: int):
        stage_len = int(e - s + 1)
        if stage_len < int(self.duration_min[stage_idx]) or stage_len > int(self.duration_max[stage_idx]):
            return None
        constraint_cost = self._shared_interval_constraint_cost(int(demo_idx), int(stage_idx), int(s), int(e))
        subgoal = np.asarray(self.demos[int(demo_idx)][int(e)], dtype=float)
        progress_cost = self._map_progress_cost(
            self.demos[int(demo_idx)], s, e, subgoal, int(stage_idx)
        )
        total = float(constraint_cost + progress_cost)
        return {
            "stage_idx": int(stage_idx),
            "s": int(s),
            "e": int(e),
            "constraint": constraint_cost,
            "short_segment_penalty": 0.0,
            "progress": float(progress_cost),
            "param_consensus": 0.0,
            "activation_consensus": 0.0,
            "weighted_total": total,
        }

    def _best_segmentation_by_interval_cost(self, demo_idx: int, cost_info_fn, fixed_cutpoints_by_stage=None):
        X = self.demos[int(demo_idx)]
        T = len(X)
        normalized_fixed_cutpoints = {}
        if fixed_cutpoints_by_stage:
            for stage_idx, cutpoint in dict(fixed_cutpoints_by_stage).items():
                stage_idx_i = int(stage_idx)
                cutpoint_i = int(cutpoint)
                if 0 <= stage_idx_i < self.num_stages - 1 and 0 <= cutpoint_i < T - 1:
                    normalized_fixed_cutpoints[stage_idx_i] = cutpoint_i

        if all(int(stage_idx) in normalized_fixed_cutpoints for stage_idx in range(self.num_stages - 1)):
            stage_ends = [int(normalized_fixed_cutpoints[stage_idx]) for stage_idx in range(self.num_stages - 1)] + [int(T - 1)]
            if any(stage_ends[idx] >= stage_ends[idx + 1] for idx in range(len(stage_ends) - 1)):
                raise RuntimeError(f"Fixed MAP cutpoints are not strictly increasing for demo {demo_idx}.")
            bounds = self._segment_bounds_from_stage_ends(stage_ends)
            cost_infos = []
            for stage_idx, (s, e) in enumerate(bounds):
                info = cost_info_fn(int(demo_idx), int(stage_idx), int(s), int(e))
                if info is None:
                    raise RuntimeError(f"Fixed MAP segmentation violates duration constraints for demo {demo_idx}.")
                cost_infos.append(info)
            local_stage_params = [
                self._stage_feature_free_params(int(demo_idx), int(s), int(e))
                for s, e in bounds
            ]
            return {
                "cutpoints": [int(x) for x in stage_ends[:-1]],
                "stage_ends": [int(x) for x in stage_ends],
                "stage_params": local_stage_params,
                "constraint": float(sum(info["constraint"] for info in cost_infos)),
                "short_segment_penalty": 0.0,
                "progress": float(sum(info["progress"] for info in cost_infos)),
                "param_consensus": 0.0,
                "activation_consensus": 0.0,
                "total": float(sum(info["weighted_total"] for info in cost_infos)),
            }

        suffix_min = np.zeros(self.num_stages + 1, dtype=int)
        suffix_max = np.zeros(self.num_stages + 1, dtype=int)
        for k in range(self.num_stages - 1, -1, -1):
            suffix_min[k] = suffix_min[k + 1] + int(self.duration_min[k])
            suffix_max[k] = suffix_max[k + 1] + int(self.duration_max[k])

        cache = {}

        def seg_info(stage_idx, s, e):
            key = (int(stage_idx), int(s), int(e))
            if key not in cache:
                cache[key] = cost_info_fn(int(demo_idx), int(stage_idx), int(s), int(e))
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
                    info = seg_info(stage_idx, 0, e)
                    if info is not None:
                        best[stage_idx, e] = float(info["weighted_total"])
                    continue
                for prev_end in range(stage_idx - 1, e):
                    prev_total = float(best[stage_idx - 1, prev_end])
                    if not np.isfinite(prev_total):
                        continue
                    info = seg_info(stage_idx, prev_end + 1, e)
                    if info is None:
                        continue
                    total = prev_total + float(info["weighted_total"])
                    if total < float(best[stage_idx, e]):
                        best[stage_idx, e] = float(total)
                        back[stage_idx, e] = int(prev_end)

        final_end = T - 1
        if not np.isfinite(best[self.num_stages - 1, final_end]):
            raise RuntimeError(f"No feasible MAP segmentation found for demo {demo_idx}.")

        stage_ends = [final_end]
        cur_end = final_end
        for stage_idx in range(self.num_stages - 1, 0, -1):
            prev_end = int(back[stage_idx, cur_end])
            if prev_end < 0:
                raise RuntimeError(f"Broken MAP DP backpointer for demo {demo_idx}, stage {stage_idx}.")
            stage_ends.append(prev_end)
            cur_end = prev_end
        stage_ends = sorted(int(x) for x in stage_ends)
        bounds = self._segment_bounds_from_stage_ends(stage_ends)
        cost_infos = [seg_info(stage_idx, s, e) for stage_idx, (s, e) in enumerate(bounds)]
        local_stage_params = [
            self._stage_feature_free_params(int(demo_idx), int(s), int(e))
            for s, e in bounds
        ]
        return {
            "cutpoints": [int(x) for x in stage_ends[:-1]],
            "stage_ends": [int(x) for x in stage_ends],
            "stage_params": local_stage_params,
            "constraint": float(sum(info["constraint"] for info in cost_infos)),
            "short_segment_penalty": 0.0,
            "progress": float(sum(info["progress"] for info in cost_infos)),
            "param_consensus": 0.0,
            "activation_consensus": 0.0,
            "total": float(best[self.num_stages - 1, final_end]),
        }

    def _majority_activation_signature_from_infos(self, selected_infos):
        if not selected_infos:
            return np.zeros((self.num_stages, self.num_features), dtype=float)
        signatures = []
        for info in selected_infos:
            rows = []
            for stage_params in info["stage_params"]:
                sig = np.zeros(self.num_features, dtype=float)
                for feat_idx in range(self.num_features):
                    mode = self._kind_to_mode(self._stage_feature_kind(stage_params, feat_idx))
                    if mode == "eq":
                        sig[feat_idx] = 1.0
                    elif mode == "lb":
                        sig[feat_idx] = -1.0
                    elif mode == "ub":
                        sig[feat_idx] = 2.0
                rows.append(sig)
            signatures.append(np.stack(rows, axis=0))
        stacked = np.stack(signatures, axis=0)
        out = np.zeros((self.num_stages, self.num_features), dtype=float)
        for stage_idx in range(self.num_stages):
            for feat_idx in range(self.num_features):
                vals = stacked[:, stage_idx, feat_idx]
                codes = [0.0, 1.0, -1.0, 2.0]
                counts = {code: int(np.sum(np.isclose(vals, code))) for code in codes}
                best_count = max(counts.values())
                best_codes = [code for code in codes if counts[code] == best_count]
                out[stage_idx, feat_idx] = 0.0 if 0.0 in best_codes else float(best_codes[0])
        return out

    def _shared_from_selected(self, selected_infos, shared_activation_mask=None):
        shared_stage_subgoals = []
        for stage_idx in range(self.num_stages):
            stage_subgoals = np.asarray(
                [info["stage_params"][stage_idx].subgoal for info in selected_infos],
                dtype=float,
            )
            shared_stage_subgoals.append(_geometric_median(stage_subgoals))

        shared_param_vectors = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        shared_param_kinds = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        shared_active = np.zeros((self.num_stages, self.num_features), dtype=float)
        shared_mode_costs = [[{} for _ in range(self.num_features)] for _ in range(self.num_stages)]
        shared_mode_votes = [[{} for _ in range(self.num_features)] for _ in range(self.num_stages)]

        for stage_idx in range(self.num_stages):
            for feat_idx in range(self.num_features):
                if feat_idx in self.force_inactive_feature_indices:
                    continue
                mode, vec, kind, costs = self._pooled_mode_mstep(selected_infos, int(stage_idx), int(feat_idx))
                shared_mode_costs[stage_idx][feat_idx] = {str(k): float(v) for k, v in costs.items()}
                shared_mode_votes[stage_idx][feat_idx] = dict(self._map_last_mode_vote_diagnostic)
                if mode == "inactive":
                    continue
                if vec is None or kind is None:
                    continue
                shared_active[stage_idx, feat_idx] = 1.0
                shared_param_kinds[stage_idx][feat_idx] = str(kind)
                shared_param_vectors[stage_idx][feat_idx] = np.asarray(vec, dtype=float)
        self.map_shared_mode_costs_ = shared_mode_costs
        self.map_shared_mode_votes_ = shared_mode_votes
        progress_kappas = self._fit_map_progress_kappas(selected_infos)
        return shared_stage_subgoals, shared_param_vectors, shared_param_kinds, shared_active, progress_kappas

    def _apply_map_shared_state(
        self,
        shared_stage_subgoals,
        shared_param_vectors,
        shared_param_kinds,
        shared_active,
        progress_kappas,
    ):
        self._apply_shared_state(shared_stage_subgoals, shared_param_vectors, shared_param_kinds)
        progress_kappas = np.asarray(progress_kappas, dtype=float).reshape(-1)
        if progress_kappas.size != self.num_stages:
            raise ValueError(
                f"MAP progress state must contain {self.num_stages} kappas, got {progress_kappas.size}."
            )
        if not np.all(np.isfinite(progress_kappas)) or np.any(progress_kappas < 0.0):
            raise ValueError("MAP progress kappas must be finite and nonnegative.")
        self.map_progress_kappas_ = progress_kappas.copy()
        self.shared_feature_score_mean = np.asarray(shared_active, dtype=float)
        self.r = np.rint(self.shared_feature_score_mean).astype(int)
        self.shared_activation_proto = np.asarray(self.shared_feature_score_mean, dtype=float).copy()
        signature = np.zeros_like(self.shared_feature_score_mean, dtype=float)
        for stage_idx in range(self.num_stages):
            for feat_idx in range(self.num_features):
                mode = self._kind_to_mode(shared_param_kinds[stage_idx][feat_idx])
                if mode == "eq":
                    signature[stage_idx, feat_idx] = 1.0
                elif mode == "lb":
                    signature[stage_idx, feat_idx] = -1.0
                elif mode == "ub":
                    signature[stage_idx, feat_idx] = 2.0
        self.shared_activation_signature_mean = signature
        self._map_shared_cache_version += 1
        self._map_shared_cost_cache.clear()
        self._map_shared_eq_prefix_cache.clear()

    def _record_iteration_state(self, iteration: int, selected_infos: Sequence[dict], total_loss: float):
        self.param_consensus_lambda_hist.append(0.0)
        self.activation_consensus_lambda_hist.append(0.0)
        self.stage_ends_ = [list(info["stage_ends"]) for info in selected_infos]
        self.current_stage_params_per_demo = [list(info["stage_params"]) for info in selected_infos]
        self.current_demo_cost_breakdown = [
            {
                "constraint": float(info["constraint"]),
                "short_segment_penalty": 0.0,
                "progress": float(info["progress"]),
                "param_consensus": 0.0,
                "activation_consensus": 0.0,
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
        self.demo_activation_matrices_ = [arr.astype(float) for arr in self.demo_r_matrices_]
        self.demo_activation_signature_matrices_ = [
            self._activation_signature_matrix_from_stage_params(info["stage_params"])
            for info in selected_infos
        ]
        if self.demo_activation_matrices_:
            self.demo_activation_history.append(np.stack(self.demo_activation_matrices_, axis=0).astype(float))
        if self.shared_activation_proto is not None:
            self.activation_proto_history.append(np.asarray(self.shared_activation_proto, dtype=float).copy())
        self.activation_rate_history.append(np.asarray(self._compute_current_activation_rate_matrix(), dtype=float))
        self.segmentation_history.append([list(item) for item in self.stage_ends_])
        total_constraint = float(np.sum([info["constraint"] for info in selected_infos]))
        total_progress = float(np.sum([info["progress"] for info in selected_infos]))
        self.loss_constraint.append(total_constraint)
        self.loss_short_segment_penalty.append(0.0)
        self.loss_progress.append(total_progress)
        self.loss_param_consensus.append(0.0)
        self.loss_activation_consensus.append(0.0)
        self.loss_total.append(float(total_loss))
        self.map_progress_kappa_history_.append(np.asarray(self.map_progress_kappas_, dtype=float).copy())
        self.posthoc_activation_summary_ = self._compute_posthoc_activation_summary()

        gammas = _hard_gammas_from_stage_ends([len(X) for X in self.demos], self.stage_ends_, self.num_stages)
        metrics = self.eval_fn(self, gammas, None) if self.eval_fn is not None else {}
        for name, value in metrics.items():
            if np.isscalar(value):
                self.metrics_hist.setdefault(name, []).append(float(value))
        if self.verbose:
            extras = {
                "stage_ends": self.stage_ends_,
                "active": int(np.sum(self.r)),
                "progress_kappa": np.round(self.map_progress_kappas_, 4).tolist(),
            }
            print(
                format_training_log(
                    "MAP",
                    int(iteration),
                    losses={
                        "total": float(total_loss),
                        "constraint": total_constraint,
                        "progress": total_progress,
                    },
                    metrics=metrics,
                    extras=extras,
                )
            )
        # MAP plots are emitted once from the final shared prototype state.

    def _create_demo_executor(self):
        worker_count = int(self.map_demo_num_workers)
        if worker_count <= 1 or len(self.demos) <= 1:
            return None
        if "fork" not in mp.get_all_start_methods():
            if self.verbose:
                print("[MAP] demo multiprocessing requires the fork start method; using one worker.", flush=True)
            return None
        if self.verbose:
            print(
                f"[MAP] demo multiprocessing enabled: {len(self.demos)} demos, {worker_count} workers",
                flush=True,
            )
        return ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("fork"),
            initializer=_map_demo_worker_init,
            initargs=(self,),
        )

    def _segment_all_demos(self, phase: str, *, executor=None, shared_state=None):
        if phase not in {"free", "shared"}:
            raise ValueError(f"Unknown MAP demo segmentation phase: {phase!r}.")
        if executor is None:
            cost_info_fn = self._free_interval_cost_info if phase == "free" else self._shared_interval_cost_info
            return [
                self._best_segmentation_by_interval_cost(
                    demo_idx=demo_idx,
                    cost_info_fn=cost_info_fn,
                    fixed_cutpoints_by_stage=self._fixed_cutpoint_map_for_demo(demo_idx),
                )
                for demo_idx in range(len(self.demos))
            ]

        results = [None for _ in self.demos]
        future_to_demo = {
            executor.submit(_map_demo_worker_run, demo_idx, phase, shared_state): demo_idx
            for demo_idx in range(len(self.demos))
        }
        for future in as_completed(future_to_demo):
            expected_demo_idx = int(future_to_demo[future])
            demo_idx, result = future.result()
            if int(demo_idx) != expected_demo_idx:
                raise RuntimeError(
                    f"MAP demo worker returned demo {demo_idx}, expected {expected_demo_idx}."
                )
            results[expected_demo_idx] = result
        if any(result is None for result in results):
            raise RuntimeError("MAP demo multiprocessing returned incomplete segmentation results.")
        return results

    def fit_fixed_segments(self, stage_ends_per_demo, *, verbose=True):
        if len(stage_ends_per_demo) != len(self.demos):
            raise ValueError("stage_ends_per_demo must match the number of demos.")

        self.verbose = bool(verbose)
        selected_infos = []
        for demo_idx, (X, stage_ends) in enumerate(zip(self.demos, stage_ends_per_demo)):
            ends = np.asarray(stage_ends, dtype=int).reshape(-1)
            if ends.size != self.num_stages:
                raise ValueError(
                    f"Each fixed segmentation must contain {self.num_stages} stage ends, got {ends.size}."
                )
            if int(ends[-1]) != int(len(X) - 1):
                raise ValueError("The final fixed stage end must equal the final demo index.")
            if np.any(ends < 0) or np.any(np.diff(ends) <= 0):
                raise ValueError("Fixed stage ends must be nonnegative and strictly increasing.")

            normalized_ends = [int(value) for value in ends.tolist()]
            bounds = self._segment_bounds_from_stage_ends(normalized_ends)
            stage_params = [
                self._stage_feature_free_params(int(demo_idx), int(start), int(end))
                for start, end in bounds
            ]
            constraint_cost = float(
                sum(float(np.sum(params.feature_constraint_costs)) for params in stage_params)
            )
            progress_cost = float(
                sum(
                    self._map_progress_cost(
                        X,
                        int(start),
                        int(end),
                        np.asarray(params.subgoal, dtype=float),
                        int(stage_idx),
                    )
                    for stage_idx, ((start, end), params) in enumerate(zip(bounds, stage_params))
                )
            )
            total_cost = float(constraint_cost + progress_cost)
            selected_infos.append(
                {
                    "cutpoints": normalized_ends[:-1],
                    "stage_ends": normalized_ends,
                    "stage_params": stage_params,
                    "constraint": constraint_cost,
                    "short_segment_penalty": 0.0,
                    "progress": progress_cost,
                    "param_consensus": 0.0,
                    "activation_consensus": 0.0,
                    "total": total_cost,
                }
            )

        shared_state = self._shared_from_selected(selected_infos)
        self._apply_map_shared_state(*shared_state)
        for demo_idx, info in enumerate(selected_infos):
            bounds = self._segment_bounds_from_stage_ends(info["stage_ends"])
            progress_cost = float(
                sum(
                    self._map_progress_cost(
                        self.demos[int(demo_idx)],
                        int(start),
                        int(end),
                        info["stage_params"][int(stage_idx)].subgoal,
                        int(stage_idx),
                    )
                    for stage_idx, (start, end) in enumerate(bounds)
                )
            )
            info["progress"] = progress_cost
            info["total"] = float(info["constraint"] + progress_cost)
        self._record_iteration_state(
            0,
            selected_infos,
            float(sum(info["total"] for info in selected_infos)),
        )
        return _hard_gammas_from_stage_ends(
            [len(X) for X in self.demos],
            self.stage_ends_,
            self.num_stages,
        )

    def fit(self, max_iter=30, verbose=True):
        self.verbose = bool(verbose)
        self.stage_ends_ = []
        self.shared_r_mean = None
        self.shared_feature_score_mean = None
        self.shared_activation_signature_mean = None
        self.shared_activation_proto = None
        self.demo_activation_matrices_ = []
        self.demo_activation_signature_matrices_ = []
        self.demo_activation_history = []
        self.activation_proto_history = []
        self.stage_subgoals_hist = []
        self.g1_hist = []
        self.g2_hist = []
        self.loss_total = []
        self.loss_constraint = []
        self.loss_short_segment_penalty = []
        self.loss_progress = []
        self.loss_param_consensus = []
        self.loss_activation_consensus = []
        self.metrics_hist = {}
        self.segmentation_history = []
        self.activation_rate_history = []
        self.param_consensus_lambda_hist = []
        self.activation_consensus_lambda_hist = []
        self._map_free_segment_cache.clear()
        self._map_local_mode_cache.clear()
        self._map_inactive_fit_cache.clear()
        self._map_interval_stats_cache.clear()
        self._map_shared_cost_cache.clear()
        self._map_shared_eq_prefix_cache.clear()
        self._map_shared_cache_version = 0
        self.map_shared_mode_costs_ = []
        self.map_shared_mode_votes_ = []
        self._map_last_mode_vote_diagnostic = {}
        initial_kappa = 0.0 if self.map_progress_kappa is None else float(self.map_progress_kappa)
        self.map_progress_kappas_ = np.full(self.num_stages, initial_kappa, dtype=float)
        self.map_progress_kappa_history_ = []

        executor = self._create_demo_executor()
        try:
            selected_infos = self._segment_all_demos("free", executor=executor)
            (
                shared_stage_subgoals,
                shared_param_vectors,
                shared_param_kinds,
                shared_active,
                progress_kappas,
            ) = self._shared_from_selected(selected_infos)
            self._apply_map_shared_state(
                shared_stage_subgoals,
                shared_param_vectors,
                shared_param_kinds,
                shared_active,
                progress_kappas,
            )
            self._record_iteration_state(-1, selected_infos, float(sum(info["total"] for info in selected_infos)))

            prev_signature = (
                [list(info["stage_ends"]) for info in selected_infos],
                [[None if v is None else tuple(np.round(np.asarray(v, dtype=float), 8)) for v in row] for row in self.shared_param_vectors],
                [[None if k is None else str(k) for k in row] for row in self.shared_param_kinds],
                tuple(np.round(np.asarray(self.map_progress_kappas_, dtype=float), 8)),
            )

            for iteration in range(int(max_iter)):
                self.current_param_consensus_lambda = 0.0
                self.current_activation_consensus_lambda = 0.0
                shared_state = (
                    shared_stage_subgoals,
                    shared_param_vectors,
                    shared_param_kinds,
                    shared_active,
                    progress_kappas,
                )
                selected_infos = self._segment_all_demos(
                    "shared",
                    executor=executor,
                    shared_state=shared_state,
                )
                (
                    shared_stage_subgoals,
                    shared_param_vectors,
                    shared_param_kinds,
                    shared_active,
                    progress_kappas,
                ) = self._shared_from_selected(selected_infos)
                self._apply_map_shared_state(
                    shared_stage_subgoals,
                    shared_param_vectors,
                    shared_param_kinds,
                    shared_active,
                    progress_kappas,
                )

                total_loss = float(sum(info["total"] for info in selected_infos))
                self._record_iteration_state(iteration, selected_infos, total_loss)
                signature = (
                    [list(info["stage_ends"]) for info in selected_infos],
                    [[None if v is None else tuple(np.round(np.asarray(v, dtype=float), 8)) for v in row] for row in self.shared_param_vectors],
                    [[None if k is None else str(k) for k in row] for row in self.shared_param_kinds],
                    tuple(np.round(np.asarray(self.map_progress_kappas_, dtype=float), 8)),
                )
                if signature == prev_signature:
                    if self.verbose:
                        print(f"[MAP] converged on stable stage_ends/prototypes at iter {iteration + 1:03d}")
                    break
                if self.loss_total and len(self.loss_total) >= 2:
                    delta = abs(float(self.loss_total[-2]) - float(self.loss_total[-1]))
                    if delta <= float(self.map_convergence_tol):
                        if self.verbose:
                            print(f"[MAP] converged on objective delta {delta:.3e} at iter {iteration + 1:03d}")
                        break
                prev_signature = signature
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        if not self.disable_plots:
            try:
                final_plot_iter = max(len(self.loss_total) - 1, 0)
                self._plot_map_final_pooled_diagnostics(final_plot_iter, selected_infos)
                plot_map_final_outputs(self, final_plot_iter)
            except Exception as exc:
                if self.verbose:
                    print(f"[MAP] final plots skipped: {exc}")
        return _hard_gammas_from_stage_ends([len(X) for X in self.demos], self.stage_ends_, self.num_stages)

    def _stage_params_activation_signature(self, stage_params):
        signature = np.zeros(self.num_features, dtype=float)
        active_mask = getattr(stage_params, "active_mask", None)
        if active_mask is None:
            return signature
        for feat_idx in range(self.num_features):
            if not int(np.asarray(active_mask, dtype=int)[feat_idx]):
                continue
            mode = self._kind_to_mode(self._stage_feature_kind(stage_params, feat_idx))
            if mode == "eq":
                signature[feat_idx] = 1.0
            elif mode == "lb":
                signature[feat_idx] = -1.0
            elif mode == "ub":
                signature[feat_idx] = 2.0
        return signature
