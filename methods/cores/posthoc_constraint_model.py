from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.models import (
    GaussianModel,
    MarginExpLowerEmission,
    MarginExpLowerLeftHNEmission,
    MarginExpUpperEmission,
    MarginExpUpperRightHNEmission,
    StudentTModel,
    ZeroMeanGaussianModel,
)


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
            out.append(name_to_column[str(value)])
        else:
            out.append(id_to_column[int(value)])
    return out


@dataclass
class _FeatureSpec:
    name: str
    column_idx: int
    raw_id: int


class FixedTauConstraintModel:
    def __init__(
        self,
        demos,
        env,
        true_taus=None,
        true_cutpoints=None,
        num_stages=2,
        tau_init=None,
        stage_ends_init=None,
        g2_init=None,
        fixed_feature_mask=None,
        selected_raw_feature_ids=None,
        feature_model_types=None,
        feat_weight=1.0,
        prog_weight=1.0,
        trans_weight=0.0,
        constraint_core_trim=0,
        plot_dir="outputs/plots",
        plot_every=None,
        eval_fn=None,
    ):
        self.demos = [np.asarray(X, dtype=float) for X in demos]
        self.env = env
        self.num_stages = int(num_stages)
        self.true_cutpoints = self._normalize_true_cutpoints(true_taus=true_taus, true_cutpoints=true_cutpoints)
        self.true_taus = [
            None if cuts is None or len(cuts) != 1 else int(cuts[0])
            for cuts in self.true_cutpoints
        ]
        self.stage_ends_ = self._normalize_stage_ends(stage_ends_init=stage_ends_init, tau_init=tau_init)
        self.sigma_irrel = 1.0
        self.feat_weight = float(feat_weight)
        self.prog_weight = float(prog_weight)
        self.trans_weight = float(trans_weight)
        self.constraint_core_trim = max(int(constraint_core_trim), 0)
        self.plot_dir = plot_dir
        self.plot_every = plot_every
        self.eval_fn = eval_fn
        self.g2_init = None if g2_init is None else np.asarray(g2_init, dtype=float)
        self.selected_raw_feature_ids = None if selected_raw_feature_ids is None else list(selected_raw_feature_ids)
        self.prog_kappa1 = 0.0
        self.prog_kappa2 = 0.0
        self.loss_loglik = []
        self.loss_feat = []
        self.loss_prog = []
        self.loss_trans = []
        self.metrics_hist = {}
        self.g1_hist = []
        self.g2_hist = []
        self.stage_subgoals_hist = []

        raw_features = [np.asarray(self.env.compute_all_features_matrix(X), dtype=float) for X in self.demos]
        self.raw_feature_specs = _feature_schema(self.env) or [
            {"id": i, "column_idx": i, "name": f"f{i}"} for i in range(raw_features[0].shape[1])
        ]
        self.selected_feature_columns = [int(i) for i in _resolve_selected_feature_columns(self.env, selected_raw_feature_ids)]
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

        self.feature_specs = []
        self.raw_id_to_column_idx = {}
        self.raw_id_to_local_idx = {}
        selected_column_to_local = {int(col): i for i, col in enumerate(self.selected_feature_columns)}
        for spec_idx, spec in enumerate(self.raw_feature_specs):
            raw_id = int(spec.get("id", spec_idx))
            col_idx = int(spec.get("column_idx", spec_idx))
            self.raw_id_to_column_idx[raw_id] = col_idx
            if col_idx in selected_column_to_local:
                self.raw_id_to_local_idx[raw_id] = int(selected_column_to_local[col_idx])
                self.feature_specs.append(
                    {"name": str(spec.get("name", f"f{selected_column_to_local[col_idx]}")), "column_idx": col_idx, "raw_id": raw_id}
                )

        self.feature_model_types = self._normalize_feature_model_types(feature_model_types)
        self.feature_models = [[self._make_model_from_kind(kind) for kind in self.feature_model_types] for _ in range(self.num_stages)]
        self.shared_param_vectors = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        self.shared_param_kinds = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        self.demo_param_vectors_ = []
        self.demo_stage_subgoals_ = []
        self.shared_stage_subgoals = []
        self.r = np.ones((self.num_stages, self.num_features), dtype=int)
        if fixed_feature_mask is not None:
            mask = np.asarray(fixed_feature_mask, dtype=int)
            if mask.shape != self.r.shape:
                raise ValueError(f"fixed_feature_mask must have shape {self.r.shape}.")
            self.r = mask.copy()
        self.state_dim = int(self.demos[0].shape[1])
        self._initialize_stage_subgoals()

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

    def _normalize_stage_ends(self, stage_ends_init=None, tau_init=None):
        if stage_ends_init is not None:
            if len(stage_ends_init) != len(self.demos):
                raise ValueError("stage_ends_init must match number of demos.")
            out = []
            for X, ends in zip(self.demos, stage_ends_init):
                arr = np.asarray(ends, dtype=int).reshape(-1)
                if arr.size != self.num_stages:
                    raise ValueError(f"Each stage_ends entry must have length {self.num_stages}.")
                arr = arr.copy()
                arr[-1] = len(X) - 1
                out.append(arr.astype(int).tolist())
            return out
        if tau_init is not None:
            if self.num_stages != 2:
                raise ValueError("tau_init is only supported for 2-stage fixed constraints.")
            return [[int(t), len(X) - 1] for X, t in zip(self.demos, tau_init)]
        return [[int(len(X) - 1)] for X in self.demos]

    def _initialize_stage_subgoals(self):
        points_per_stage = [[] for _ in range(self.num_stages)]
        for X, stage_ends in zip(self.demos, self.stage_ends_):
            for stage_idx, end in enumerate(stage_ends):
                points_per_stage[stage_idx].append(np.asarray(X[int(end)], dtype=float))
        self.stage_subgoals = []
        for stage_idx, pts in enumerate(points_per_stage):
            if stage_idx == self.num_stages - 1 and self.g2_init is not None:
                sg = np.asarray(self.g2_init, dtype=float).copy()
            else:
                sg = _geometric_median(np.stack(pts, axis=0))
            self.stage_subgoals.append(sg)
        self.shared_stage_subgoals = [np.asarray(x, dtype=float).copy() for x in self.stage_subgoals]
        self.stage_subgoals_hist.append([np.asarray(x, dtype=float).copy() for x in self.stage_subgoals])
        self.g1 = self.stage_subgoals[0].copy()
        self.g2 = self.stage_subgoals[1].copy() if self.num_stages >= 2 else self.stage_subgoals[0].copy()
        self.g1_hist.append(np.asarray(self.g1, dtype=float).copy())
        self.g2_hist.append(np.asarray(self.g2, dtype=float).copy())

    def _normalize_feature_model_types(self, types):
        if types is None:
            return ["gaussian"] * int(self.num_features)
        if isinstance(types, dict):
            resolved = []
            for spec in self.feature_specs:
                name = spec["name"]
                raw_id = spec["raw_id"]
                if name in types:
                    resolved.append(types[name])
                elif raw_id in types:
                    resolved.append(types[raw_id])
                else:
                    resolved.append("gaussian")
            return resolved
        if len(types) != int(self.num_features):
            raise ValueError("feature_model_types must match selected features.")
        return list(types)

    def _make_model_from_kind(self, kind):
        kind = str(kind).lower()
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
        if kind in {"gauss", "gaussian"}:
            return GaussianModel(mu=float(np.mean(xs)), sigma=float(max(np.std(xs), 1e-6)))
        if kind in {"student_t", "studentt", "t"}:
            model = StudentTModel(mu=0.0, sigma=max(float(np.std(xs)), 1e-6))
            model.m_step_update([xs])
            model._update_interval()
            return model
        if kind in {"zero_gauss", "zero_gaussian"}:
            return ZeroMeanGaussianModel(sigma=float(max(np.sqrt(np.mean(xs * xs)), 1e-6)))
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

    def _summary_to_vector(self, kind, summary):
        kind = str(kind).lower()
        if kind in {"gauss", "gaussian"}:
            return np.asarray(
                [float(summary["mu"]), float(np.log(max(float(summary["sigma"]), 1e-12)))],
                dtype=float,
            )
        if kind in {"student_t", "studentt", "t"}:
            return np.asarray(
                [float(summary["mu"]), float(np.log(max(float(summary["sigma"]), 1e-12)))],
                dtype=float,
            )
        if kind in {"zero_gauss", "zero_gaussian"}:
            return np.asarray(
                [0.0, float(np.log(max(float(summary["sigma"]), 1e-12)))],
                dtype=float,
            )
        if kind in {"margin_exp_lower", "marginexp", "margin_exp"}:
            return np.asarray(
                [float(summary["b"]), float(np.log(max(float(summary["lam"]), 1e-12)))],
                dtype=float,
            )
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
            return np.asarray(
                [float(summary["b"]), float(np.log(max(float(summary["lam"]), 1e-12)))],
                dtype=float,
            )
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

    def _model_to_vector(self, kind, model):
        return self._summary_to_vector(kind, model.get_summary())

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

    def _features_for_demo_matrix(self, X):
        demo_idx = self._demo_index(X)
        if demo_idx is not None:
            return self.standardized_features[demo_idx]
        F_raw = np.asarray(self.env.compute_all_features_matrix(X), dtype=float)
        F_sel = F_raw[:, self.selected_feature_columns]
        mean_sel = self.feat_mean[self.selected_feature_columns]
        std_sel = self.feat_std[self.selected_feature_columns]
        return (F_sel - mean_sel[None, :]) / std_sel[None, :]

    def _demo_index(self, X):
        return next((i for i, demo in enumerate(self.demos) if demo is X), None)

    @staticmethod
    def _segment_bounds_from_stage_ends(stage_ends):
        bounds = []
        start = 0
        for end in stage_ends:
            end_i = int(end)
            bounds.append((int(start), end_i))
            start = end_i + 1
        return bounds

    def _normalize_stage_ends_for_demo(self, stage_ends, T):
        arr = np.asarray(stage_ends, dtype=int).reshape(-1)
        if arr.size == 0:
            arr = np.asarray([int(T) - 1], dtype=int)
        arr = np.sort(arr)
        arr = np.clip(arr, 0, max(int(T) - 1, 0))
        if arr.size < self.num_stages:
            pad = np.full(self.num_stages - arr.size, int(T) - 1, dtype=int)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.size > self.num_stages:
            arr = arr[: self.num_stages]
        arr[-1] = int(T) - 1
        return arr.astype(int).tolist()

    def _segment_core_bounds(self, s, e):
        s_i = int(s)
        e_i = int(e)
        trim = int(self.constraint_core_trim)
        if trim <= 0:
            return s_i, e_i
        core_s = min(s_i + trim, e_i)
        core_e = max(core_s, e_i - trim)
        return int(core_s), int(core_e)

    def _core_mask_for_demo_stage(self, demo_idx, stage_idx):
        T = int(len(self.demos[demo_idx]))
        mask = np.zeros(T, dtype=bool)
        bounds = self._segment_bounds_from_stage_ends(self.stage_ends_[demo_idx])
        if int(stage_idx) >= len(bounds):
            return mask
        s, e = bounds[int(stage_idx)]
        if int(s) > int(e):
            return mask
        core_s, core_e = self._segment_core_bounds(s, e)
        mask[core_s : core_e + 1] = True
        return mask

    def _log_irrelevant(self, phi):
        phi = np.asarray(phi, dtype=float)
        sig = float(self.sigma_irrel)
        sig2 = sig * sig + 1e-12
        c = -0.5 * np.log(2.0 * np.pi * sig2)
        return c - 0.5 * (phi * phi) / sig2

    def _mstep_update_features(self, gammas):
        demo_param_vectors = []
        self.shared_param_vectors = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        self.shared_param_kinds = [[None for _ in range(self.num_features)] for _ in range(self.num_stages)]
        for stage_idx in range(self.num_stages):
            for feat_idx, kind in enumerate(self.feature_model_types):
                if self.r[stage_idx, feat_idx] != 1:
                    continue
                local_vectors = []
                for demo_idx, (F, gamma) in enumerate(zip(self.standardized_features, gammas)):
                    mask = np.asarray(gamma[:, stage_idx] > 0.5, dtype=bool)
                    mask &= self._core_mask_for_demo_stage(demo_idx, stage_idx)
                    if np.any(mask):
                        model = self._fit_local_model(kind, np.asarray(F[mask, feat_idx], dtype=float))
                        vec = self._model_to_vector(kind, model)
                        local_vectors.append((demo_idx, vec))
                for demo_idx, vec in local_vectors:
                    while demo_idx >= len(demo_param_vectors):
                        demo_param_vectors.append([[None for _ in range(self.num_features)] for _ in range(self.num_stages)])
                    demo_param_vectors[demo_idx][stage_idx][feat_idx] = np.asarray(vec, dtype=float)
                if local_vectors:
                    stacked = np.stack([vec for _, vec in local_vectors], axis=0)
                    shared_vec = np.median(stacked, axis=0)
                    self.shared_param_vectors[stage_idx][feat_idx] = np.asarray(shared_vec, dtype=float)
                    self.shared_param_kinds[stage_idx][feat_idx] = str(kind).lower()
                    self.feature_models[stage_idx][feat_idx] = self._vector_to_model(kind, shared_vec)
        while len(demo_param_vectors) < len(self.demos):
            demo_param_vectors.append([[None for _ in range(self.num_features)] for _ in range(self.num_stages)])
        self.demo_param_vectors_ = demo_param_vectors

    def _mstep_update_goals(self, gammas, xis_list, aux_list):
        points_per_stage = [[] for _ in range(self.num_stages)]
        stage_ends = []
        for X, gamma in zip(self.demos, gammas):
            z = np.argmax(gamma, axis=1)
            cuts = np.where(np.diff(z) != 0)[0].astype(int)
            ends = self._normalize_stage_ends_for_demo(cuts.tolist() + [len(X) - 1], len(X))
            stage_ends.append([int(x) for x in ends])
            for stage_idx, end in enumerate(ends):
                points_per_stage[stage_idx].append(np.asarray(X[int(end)], dtype=float))
        self.stage_ends_ = stage_ends
        self.demo_stage_subgoals_ = [[np.asarray(X[int(end)], dtype=float).copy() for end in ends] for X, ends in zip(self.demos, stage_ends)]
        self.stage_subgoals = [_geometric_median(np.stack(pts, axis=0)) for pts in points_per_stage]
        self.shared_stage_subgoals = [np.asarray(x, dtype=float).copy() for x in self.stage_subgoals]
        self.stage_subgoals_hist.append([np.asarray(x, dtype=float).copy() for x in self.stage_subgoals])
        self.g1 = self.stage_subgoals[0].copy()
        self.g2 = self.stage_subgoals[1].copy() if self.num_stages >= 2 else self.stage_subgoals[0].copy()
        self.g1_hist.append(np.asarray(self.g1, dtype=float).copy())
        self.g2_hist.append(np.asarray(self.g2, dtype=float).copy())

    def _emission_loglik(self, X, return_parts=False):
        F = self._features_for_demo_matrix(X)
        demo_idx = self._demo_index(X)
        T = len(X)
        ll_feat = np.zeros((T, self.num_stages), dtype=float)
        for stage_idx in range(self.num_stages):
            active = [m for m in range(self.num_features) if self.r[stage_idx, m] == 1]
            if not active:
                continue
            parts = []
            for feat_idx in active:
                parts.append(np.asarray(self.feature_models[stage_idx][feat_idx].logpdf(F[:, feat_idx]), dtype=float))
            stage_ll = self.feat_weight * np.mean(np.stack(parts, axis=1), axis=1)
            if demo_idx is None:
                ll_feat[:, stage_idx] = stage_ll
            else:
                core_mask = self._core_mask_for_demo_stage(demo_idx, stage_idx)
                ll_feat[core_mask, stage_idx] = stage_ll[core_mask]
        ll_prog = np.zeros((T, self.num_stages), dtype=float)
        ll_emit = ll_feat + ll_prog
        if return_parts:
            return ll_emit, ll_feat, ll_prog
        return ll_emit

    def _transition_logprob(self, X, return_aux=False):
        T = len(X)
        logA = np.zeros((max(T - 1, 0), self.num_stages, self.num_stages), dtype=float)
        if return_aux:
            return logA, None
        return logA
