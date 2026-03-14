from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


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
