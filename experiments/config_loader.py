from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


MAP_POSTHOC_PARAMETER_KEYS = (
    "selected_raw_feature_ids",
    "force_inactive_feature_ids",
    "constraint_core_trim",
    "map_mstep_boundary_trim",
    "truncated_z_soft_boundary_scale",
    "truncated_z_observation_noise_scale",
    "truncated_z_half_t_scale_quantile",
    "map_eq_sigma",
    "map_c_bg",
    "map_c_ineq",
    "map_eq_distribution",
    "map_inactive_distribution",
    "map_nu_eq",
    "map_nu_inactive",
    "map_nu_ineq",
    "map_boundary_quantile",
    "map_activation_prior",
    "map_active_mode_prior",
    "map_vote_prior_scope",
    "map_refit_winning_voters",
)

MAP_JOINT_METHOD_NAMES = frozenset(
    {"map", "map_pooled", "map_balanced_pooled", "map_balanced_vote"}
)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_experiment_config(
    dataset_config_path: str | Path,
    method_config_path: str | Path,
) -> Dict[str, Any]:
    dataset_cfg = load_json(dataset_config_path)
    method_cfg = load_json(method_config_path)
    return {
        "dataset": dataset_cfg,
        "method": method_cfg,
    }


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def resolve_dataset_method_override(
    method_name: str,
    dataset_method_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    method_name = str(method_name)
    if method_name not in MAP_JOINT_METHOD_NAMES or method_name == "map":
        return dict(dataset_method_overrides.get(method_name, {}))
    return deep_merge(
        dict(dataset_method_overrides.get("map", {})),
        dict(dataset_method_overrides.get(method_name, {})),
    )


def inherit_map_posthoc_parameters(
    method_name: str,
    method_cfg: Dict[str, Any],
    map_method_cfg: Dict[str, Any],
    dataset_method_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(method_cfg)
    if method_name == "fchmm" or not isinstance(merged.get("posthoc_constraint"), dict):
        return merged

    effective_map_cfg = deep_merge(
        dict(map_method_cfg),
        dict(dataset_method_overrides.get("map", {})),
    )
    inherited = {
        key: effective_map_cfg[key]
        for key in MAP_POSTHOC_PARAMETER_KEYS
        if key in effective_map_cfg
    }
    merged["posthoc_constraint"] = deep_merge(
        dict(merged["posthoc_constraint"]),
        inherited,
    )
    return merged
