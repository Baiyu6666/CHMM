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
