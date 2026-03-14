# evaluation/metrics.py
import numpy as np
from methods.common.tau_init import extract_taus_hat


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


def _feature_is_selected_for_stage(learner, stage_idx, local_feature_idx):
    if local_feature_idx is None:
        return False
    r = getattr(learner, "r", None)
    if r is None:
        return True
    try:
        return bool(r[stage_idx, local_feature_idx])
    except Exception:
        return False


def _compute_segmentation_metrics(learner, taus_hat):
    mae_list, nmae_list = [], []
    if learner.true_taus is not None:
        for tau_hat, tau_true, X in zip(taus_hat, learner.true_taus, learner.demos):
            if tau_true is None:
                continue
            tau_true = int(tau_true)
            err = abs(int(tau_hat) - tau_true)
            mae_list.append(float(err))
            nmae_list.append(float(err / max(len(X), 1)))
    return {
        "MAE_tau": float(np.mean(mae_list)) if mae_list else np.nan,
        "NMAE_tau": float(np.mean(nmae_list)) if nmae_list else np.nan,
    }


def _compute_goal_metrics(learner):
    if hasattr(learner.env, "subgoal") and hasattr(learner.env, "goal"):
        g1_true = np.asarray(learner.env.subgoal, float)
        g2_true = np.asarray(learner.env.goal, float)
        return {
            "e_g1": float(np.linalg.norm(np.asarray(learner.g1, float) - g1_true)),
            "e_g2": float(np.linalg.norm(np.asarray(learner.g2, float) - g2_true)),
        }
    return {"e_g1": np.nan, "e_g2": np.nan}


def _estimate_constraint_value(learner, spec):
    feature_name = spec["feature_name"]
    stage_idx = int(spec["stage"])
    semantics = str(spec["semantics"])
    estimator = str(spec.get("estimator", "bound"))

    raw_feature_id = _raw_feature_id_from_name(learner.env, feature_name)
    local_idx = _selected_feature_index(learner, raw_feature_id)
    if raw_feature_id is None or local_idx is None:
        return np.nan
    if not _feature_is_selected_for_stage(learner, stage_idx, local_idx):
        return np.nan

    model = learner.feature_models[stage_idx][local_idx]
    summary = model.get_summary()
    model_type = str(summary.get("type", "base"))

    raw_column_idx = _raw_feature_column_index(learner, raw_feature_id)
    if raw_column_idx is None:
        return np.nan

    raw_mean = float(learner.feat_mean[raw_column_idx])
    raw_std = float(learner.feat_std[raw_column_idx])

    def to_raw(z_value):
        return float(z_value) * raw_std + raw_mean

    if semantics == "lower_bound":
        if hasattr(model, "L"):
            return to_raw(model.L)
        return np.nan

    if semantics == "upper_bound":
        if hasattr(model, "U"):
            return to_raw(model.U)
        return np.nan

    if semantics == "target_value":
        if estimator == "mean" and "mu" in summary:
            return to_raw(summary["mu"])
        if estimator == "center":
            if hasattr(model, "L") and hasattr(model, "U"):
                return 0.5 * (to_raw(model.L) + to_raw(model.U))
            return np.nan
        if estimator == "lower" and hasattr(model, "L"):
            return to_raw(model.L)
        if estimator == "upper" and hasattr(model, "U"):
            return to_raw(model.U)
        if model_type in ("gauss", "gauss_zero") and "mu" in summary:
            return to_raw(summary["mu"])
        return np.nan

    return np.nan


def _compute_constraint_metrics(learner):
    metrics = {}
    specs = _get_constraint_specs(learner.env)
    oracle = getattr(learner.env, "true_constraints", None)
    if not specs or not isinstance(oracle, dict):
        return metrics

    for spec in specs:
        metric_name = str(spec["metric"])
        oracle_key = str(spec["oracle_key"])
        if oracle_key not in oracle:
            metrics[metric_name] = np.nan
            continue

        estimate = _estimate_constraint_value(learner, spec)
        target = float(oracle[oracle_key])
        metrics[metric_name] = float(abs(estimate - target)) if np.isfinite(estimate) else np.nan
    return metrics


def eval_goalhmm_auto(learner, gammas, xis_list):
    taus_hat = extract_taus_hat(gammas, xis_list)

    metrics = {}
    metrics.update(_compute_segmentation_metrics(learner, taus_hat))
    metrics.update(_compute_goal_metrics(learner))

    constraint_metrics = _compute_constraint_metrics(learner)
    if constraint_metrics:
        metrics.update(constraint_metrics)
    return metrics


def obs_avoid_metrics(learner, taus_hat):
    return eval_goalhmm_auto(
        learner,
        [np.eye(2)[np.r_[np.zeros(t + 1, dtype=int), np.ones(len(X) - t - 1, dtype=int)]] for X, t in zip(learner.demos, taus_hat)],
        None,
    )


def sine_corridor_metrics(learner, taus_hat):
    return eval_goalhmm_auto(
        learner,
        [np.eye(2)[np.r_[np.zeros(t + 1, dtype=int), np.ones(len(X) - t - 1, dtype=int)]] for X, t in zip(learner.demos, taus_hat)],
        None,
    )


__all__ = ["eval_goalhmm_auto", "obs_avoid_metrics", "sine_corridor_metrics"]
