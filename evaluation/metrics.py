import numpy as np

from methods.base import compute_cutpoint_metrics
from methods.common.tau_init import extract_taus_hat


_EQUALITY_MODEL_TYPES = {"gauss", "gaussian", "student_t", "studentt", "t", "gauss_zero", "zero_gaussian"}


def _get_env_feature_schema(env):
    if hasattr(env, "get_feature_schema"):
        return list(env.get_feature_schema())
    schema = getattr(env, "feature_schema", None)
    return list(schema) if schema is not None else []


def _get_constraint_specs(env):
    if hasattr(env, "get_constraint_specs"):
        return list(env.get_constraint_specs())
    specs = getattr(env, "constraint_specs", None)
    return list(specs) if specs is not None else []


def _raw_feature_id_from_name(env, feature_name):
    schema = _get_env_feature_schema(env)
    for i, spec in enumerate(schema):
        if str(spec.get("name", f"f{i}")) == str(feature_name):
            return int(spec.get("id", i))
    return None


def _selected_feature_index(learner, raw_feature_id):
    if raw_feature_id is None:
        return None

    raw_id_to_local_idx = getattr(learner, "raw_id_to_local_idx", None)
    if isinstance(raw_id_to_local_idx, dict) and int(raw_feature_id) in raw_id_to_local_idx:
        return int(raw_id_to_local_idx[int(raw_feature_id)])

    selected_raw_feature_ids = getattr(learner, "selected_raw_feature_ids", None)
    if selected_raw_feature_ids is None:
        return int(raw_feature_id)
    try:
        return int(list(selected_raw_feature_ids).index(int(raw_feature_id)))
    except ValueError:
        return None


def _raw_feature_column_index(learner, raw_feature_id):
    if raw_feature_id is None:
        return None

    raw_id_to_column_idx = getattr(learner, "raw_id_to_column_idx", None)
    if isinstance(raw_id_to_column_idx, dict) and int(raw_feature_id) in raw_id_to_column_idx:
        return int(raw_id_to_column_idx[int(raw_feature_id)])
    return int(raw_feature_id)


def _normalize_true_cutpoints(learner):
    true_cutpoints = getattr(learner, "true_cutpoints", None)
    if true_cutpoints is not None:
        return true_cutpoints

    true_taus = getattr(learner, "true_taus", None)
    if true_taus is None:
        return None
    return [None if tau is None else np.asarray([int(tau)], dtype=int) for tau in true_taus]


def _predicted_cutpoints(learner, gammas, xis_list):
    stage_ends = getattr(learner, "stage_ends_", None)
    if stage_ends is not None:
        return [np.asarray(ends[:-1], dtype=int) for ends in stage_ends]

    if gammas is not None:
        cutpoints = []
        for gamma in gammas:
            labels = np.argmax(np.asarray(gamma, dtype=float), axis=1).astype(int)
            cutpoints.append(np.where(np.diff(labels) != 0)[0].astype(int))
        return cutpoints

    if gammas is not None or xis_list is not None:
        taus_hat = extract_taus_hat(gammas, xis_list)
        if taus_hat is not None:
            return [np.asarray([int(tau)], dtype=int) for tau in taus_hat]
    return None


def _compute_segmentation_metrics(learner, gammas, xis_list):
    true_cutpoints = _normalize_true_cutpoints(learner)
    cutpoints_hat = _predicted_cutpoints(learner, gammas, xis_list)
    if true_cutpoints is None or cutpoints_hat is None:
        return {}
    return compute_cutpoint_metrics(cutpoints_hat, true_cutpoints, learner.demos)


def _shared_stage_subgoals(learner):
    stage_subgoals = getattr(learner, "stage_subgoals", None)
    if stage_subgoals is not None:
        return [np.asarray(x, dtype=float) for x in stage_subgoals]

    g1 = getattr(learner, "g1", None)
    g2 = getattr(learner, "g2", None)
    if g1 is not None and g2 is not None:
        return [np.asarray(g1, dtype=float), np.asarray(g2, dtype=float)]
    return None


def _compute_subgoal_metrics(learner):
    true_cutpoints = _normalize_true_cutpoints(learner)
    pred_subgoals = _shared_stage_subgoals(learner)
    if true_cutpoints is None or pred_subgoals is None:
        return {"MeanStageSubgoalError": np.nan}

    errs = []
    num_stages = len(pred_subgoals)
    for X, cuts in zip(learner.demos, true_cutpoints):
        if cuts is None:
            continue
        stage_ends = list(np.asarray(cuts, dtype=int).reshape(-1)) + [len(X) - 1]
        if len(stage_ends) != num_stages:
            continue
        for stage_idx, end in enumerate(stage_ends):
            errs.append(float(np.linalg.norm(np.asarray(pred_subgoals[stage_idx], dtype=float) - np.asarray(X[int(end)], dtype=float))))
    return {"MeanStageSubgoalError": float(np.mean(errs)) if errs else np.nan}


def _is_auto_feature_mode(learner):
    if str(getattr(learner, "feature_activation_mode", "")).lower() == "score":
        return True
    return bool(getattr(learner, "auto_feature_select", False))


def _predicted_constraint_active_mask(learner):
    num_stages = int(getattr(learner, "num_stages", 0))
    num_features = int(getattr(learner, "num_features", 0))
    mask = np.ones((num_stages, num_features), dtype=bool)

    shared_activation = getattr(learner, "shared_feature_score_mean", None)
    if shared_activation is not None:
        shared_arr = np.asarray(shared_activation, dtype=float)
        if shared_arr.shape == mask.shape:
            return np.rint(shared_arr).astype(bool)

    r = getattr(learner, "r", None)
    if r is not None:
        r_arr = np.asarray(r, dtype=int)
        if r_arr.shape == mask.shape:
            return r_arr.astype(bool)
    return mask


def _constraint_truth_matrices(learner):
    specs = _get_constraint_specs(learner.env)
    oracle = getattr(learner.env, "true_constraints", None)
    num_stages = int(getattr(learner, "num_stages", 0))
    num_features = int(getattr(learner, "num_features", 0))

    true_active = np.zeros((num_stages, num_features), dtype=bool)
    target_matrix = np.full((num_stages, num_features), np.nan, dtype=float)
    semantics = np.full((num_stages, num_features), "", dtype=object)

    if not specs or not isinstance(oracle, dict):
        return true_active, target_matrix, semantics

    for spec in specs:
        raw_feature_id = _raw_feature_id_from_name(learner.env, spec["feature_name"])
        local_idx = _selected_feature_index(learner, raw_feature_id)
        stage_idx = int(spec["stage"])
        if local_idx is None or stage_idx < 0 or stage_idx >= num_stages:
            continue
        oracle_key = str(spec["oracle_key"])
        true_active[stage_idx, local_idx] = True
        semantics[stage_idx, local_idx] = str(spec.get("semantics", ""))
        if oracle_key in oracle:
            target_matrix[stage_idx, local_idx] = float(oracle[oracle_key])

    return true_active, target_matrix, semantics


def _selected_feature_names(learner):
    names = []
    selected_columns = list(getattr(learner, "selected_feature_columns", list(range(int(getattr(learner, "num_features", 0))))))
    raw_specs = list(getattr(learner, "raw_feature_specs", []))
    for local_idx, column_idx in enumerate(selected_columns):
        name = None
        for spec_idx, spec in enumerate(raw_specs):
            if int(spec.get("column_idx", spec_idx)) == int(column_idx):
                name = str(spec.get("name", f"f{local_idx}"))
                break
        names.append(name or f"f{local_idx}")
    return names


def _segment_bounds_from_stage_ends(stage_ends):
    bounds = []
    start = 0
    for end in stage_ends:
        end_i = int(end)
        bounds.append((int(start), end_i))
        start = end_i + 1
    return bounds


def _raw_column_for_local_feature(learner, local_feature_idx):
    raw_feature_id = None
    if hasattr(learner, "feature_specs") and local_feature_idx < len(learner.feature_specs):
        raw_feature_id = int(learner.feature_specs[local_feature_idx]["raw_id"])
    if raw_feature_id is not None:
        raw_column_idx = _raw_feature_column_index(learner, raw_feature_id)
        if raw_column_idx is not None:
            return int(raw_column_idx)
    if hasattr(learner, "selected_feature_columns"):
        selected_columns = list(getattr(learner, "selected_feature_columns", []))
        if local_feature_idx < len(selected_columns):
            return int(selected_columns[local_feature_idx])
    return int(local_feature_idx)


def _constraint_feature_scales_raw(learner) -> np.ndarray:
    num_features = int(getattr(learner, "num_features", 0))
    scales = np.full(num_features, 1.0, dtype=float)
    demos = list(getattr(learner, "demos", []) or [])
    if num_features <= 0 or not demos:
        return scales

    raw_mats = []
    env = getattr(learner, "env", None)
    if env is not None and hasattr(env, "compute_all_features_matrix"):
        for X in demos:
            raw_mats.append(np.asarray(env.compute_all_features_matrix(X), dtype=float))
    else:
        standardized_features = list(getattr(learner, "standardized_features", []) or [])
        feat_mean = np.asarray(getattr(learner, "feat_mean", np.zeros(num_features)), dtype=float)
        feat_std = np.asarray(getattr(learner, "feat_std", np.ones(num_features)), dtype=float)
        for Fz in standardized_features:
            Fz = np.asarray(Fz, dtype=float)
            raw_mats.append(Fz * feat_std[None, : Fz.shape[1]] + feat_mean[None, : Fz.shape[1]])

    if not raw_mats:
        return scales

    for feat_idx in range(num_features):
        raw_col = _raw_column_for_local_feature(learner, feat_idx)
        vals = []
        for mat in raw_mats:
            if mat.ndim != 2 or raw_col >= mat.shape[1]:
                continue
            col = np.asarray(mat[:, raw_col], dtype=float).reshape(-1)
            finite = col[np.isfinite(col)]
            if finite.size > 0:
                vals.append(finite)
        if not vals:
            scales[feat_idx] = 1.0
            continue
        all_vals = np.concatenate(vals, axis=0)
        q05 = float(np.quantile(all_vals, 0.05))
        q95 = float(np.quantile(all_vals, 0.95))
        central_range = float(q95 - q05)
        std = float(np.std(all_vals))
        scale = central_range if central_range > 1e-8 else std
        scales[feat_idx] = float(max(scale, 1e-6))
    return scales


def _compute_shared_constraint_value_matrix(learner):
    num_stages = int(getattr(learner, "num_stages", 0))
    num_features = int(getattr(learner, "num_features", 0))
    learned_value_matrix = np.full((num_stages, num_features), np.nan, dtype=float)
    for stage_idx in range(num_stages):
        for feat_idx in range(num_features):
            learned_value_matrix[stage_idx, feat_idx] = _estimate_constraint_value_raw(learner, stage_idx, feat_idx)
    return learned_value_matrix


def _estimate_constraint_value_raw(learner, stage_idx, local_feature_idx):
    shared_param_vectors = getattr(learner, "shared_param_vectors", None)
    feature_model_types = getattr(learner, "feature_model_types", None)
    shared_vec = None
    model_type = ""
    try:
        if shared_param_vectors is not None:
            shared_vec = shared_param_vectors[stage_idx][local_feature_idx]
        if feature_model_types is not None and local_feature_idx < len(feature_model_types):
            model_type = str(feature_model_types[local_feature_idx]).lower()
    except Exception:
        shared_vec = None

    z_value = np.nan
    if shared_vec is not None:
        vec = np.asarray(shared_vec, dtype=float).reshape(-1)
        if vec.size > 0:
            z_value = float(vec[0])

    if not np.isfinite(z_value):
        try:
            model = learner.feature_models[stage_idx][local_feature_idx]
        except Exception:
            return np.nan
        summary = model.get_summary()
        model_type = str(summary.get("type", model_type)).lower()
        if model_type in _EQUALITY_MODEL_TYPES and "mu" in summary:
            z_value = float(summary["mu"])
        elif model_type in {
            "margin_exp_lower",
            "margin_exp_lower_left_hn",
            "margin_exp_upper",
            "margin_exp_upper_right_hn",
        } and "b" in summary:
            z_value = float(summary["b"])
        elif "mu" in summary:
            z_value = float(summary["mu"])
        elif "b" in summary:
            z_value = float(summary["b"])
        else:
            return np.nan

    raw_column_idx = _raw_column_for_local_feature(learner, local_feature_idx)
    if raw_column_idx is None:
        return np.nan

    raw_mean = float(np.asarray(learner.feat_mean, dtype=float)[raw_column_idx])
    raw_std = float(np.asarray(learner.feat_std, dtype=float)[raw_column_idx])

    def to_raw(z_value):
        return float(z_value) * raw_std + raw_mean

    return to_raw(z_value)


def _compute_constraint_metrics(learner):
    num_stages = int(getattr(learner, "num_stages", 0))
    num_features = int(getattr(learner, "num_features", 0))
    if num_stages <= 0 or num_features <= 0:
        return {}

    true_active, target_matrix, semantics = _constraint_truth_matrices(learner)
    predicted_active = _predicted_constraint_active_mask(learner)
    auto_mode = _is_auto_feature_mode(learner)

    per_demo_values = None
    learned_value_matrix = np.full((num_stages, num_features), np.nan, dtype=float)
    if learner.__class__.__name__ == "FCHMM":
        learned_value_matrix = _compute_shared_constraint_value_matrix(learner)

    feature_scales = _constraint_feature_scales_raw(learner)
    raw_error_matrix = np.full((num_stages, num_features), np.nan, dtype=float)
    error_matrix = np.full((num_stages, num_features), np.nan, dtype=float)
    for stage_idx in range(num_stages):
        for feat_idx in range(num_features):
            if not true_active[stage_idx, feat_idx]:
                continue

            if not predicted_active[stage_idx, feat_idx]:
                continue

            if learner.__class__.__name__ == "FCHMM":
                estimate = learned_value_matrix[stage_idx, feat_idx]
            else:
                estimate = _estimate_constraint_value_raw(learner, stage_idx, feat_idx)
            target = target_matrix[stage_idx, feat_idx]
            if np.isfinite(estimate) and np.isfinite(target):
                raw_error = float(abs(estimate - target))
                raw_error_matrix[stage_idx, feat_idx] = raw_error
                scale = float(feature_scales[feat_idx]) if feat_idx < len(feature_scales) else 1.0
                error_matrix[stage_idx, feat_idx] = raw_error / max(scale, 1e-6)

    finite_vals = error_matrix[np.isfinite(error_matrix)]
    raw_finite_vals = raw_error_matrix[np.isfinite(raw_error_matrix)]
    metrics = {
        "MeanConstraintError": float(np.mean(finite_vals)) if finite_vals.size > 0 else np.nan,
        "MeanConstraintErrorRaw": float(np.mean(raw_finite_vals)) if raw_finite_vals.size > 0 else np.nan,
        "ConstraintErrorMatrix": error_matrix.tolist(),
        "ConstraintErrorMatrixRaw": raw_error_matrix.tolist(),
        "ConstraintTrueActiveMask": true_active.astype(int).tolist(),
        "ConstraintPredictedActiveMask": predicted_active.astype(int).tolist(),
        "ConstraintTargetMatrix": target_matrix.tolist(),
        "ConstraintSemanticsMatrix": semantics.tolist(),
        "ConstraintFeatureNames": _selected_feature_names(learner),
        "ConstraintFeatureScales": feature_scales.tolist(),
    }
    if learner.__class__.__name__ == "FCHMM":
        metrics["ConstraintLearnedValueMatrix"] = learned_value_matrix.tolist()
        metrics["ConstraintLearnedValuePerDemo"] = per_demo_values.tolist() if per_demo_values is not None else []
    return metrics


def evaluate_model_metrics(learner, gammas, xis_list):
    metrics = {}
    metrics.update(_compute_segmentation_metrics(learner, gammas, xis_list))
    metrics.update(_compute_subgoal_metrics(learner))

    constraint_metrics = _compute_constraint_metrics(learner)
    if constraint_metrics:
        metrics.update(constraint_metrics)
    return metrics


__all__ = ["evaluate_model_metrics"]
