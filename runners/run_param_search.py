from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.artifacts import apply_run_plot_dirs, default_method_seed, resolve_run_dir, save_run_artifacts, write_json
from experiments.config_loader import deep_merge, load_json
from experiments.unified_experiment import run_experiment
from methods import JOINT_METHODS


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _split_int_csv(text: str) -> list[int]:
    return [int(item) for item in _split_csv(text)]


def _load_env_config(config_root: Path, dataset_name: str) -> dict[str, Any]:
    path = config_root / "envs" / f"{dataset_name}.json"
    cfg = dict(load_json(path))
    cfg.pop("name", None)
    return cfg


def _load_method_config(config_root: Path, method_name: str) -> dict[str, Any]:
    path = config_root / "methods" / f"{method_name}.json"
    cfg = dict(load_json(path))
    cfg.pop("name", None)
    return cfg


def _set_nested_value(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        raise ValueError("Override path must be non-empty.")
    cur = cfg
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def _grid_from_payload(payload: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Grid config must be a non-empty JSON object mapping parameter paths to candidate lists.")
    keys: list[str] = []
    value_lists: list[list[Any]] = []
    for key, values in payload.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Grid values for '{key}' must be a non-empty JSON list.")
        keys.append(str(key))
        value_lists.append(list(values))
    return keys, value_lists


def _combo_overrides(keys: list[str], combo_values: tuple[Any, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in zip(keys, combo_values):
        _set_nested_value(out, key, value)
    return out


def _extract_metrics(method_name: str, result: dict[str, Any]) -> dict[str, Any]:
    if method_name in JOINT_METHODS:
        return dict(result["joint_result"]["metrics"])
    return dict(result["constraints"]["metrics"])


def _extract_scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if np.isscalar(value):
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value_f):
                out[str(key)] = value_f
    return out


def _apply_method_seed(method_name: str, method_cfg: dict[str, Any], method_seed: int) -> dict[str, Any]:
    cfg = deepcopy(method_cfg)
    if method_name in JOINT_METHODS or method_name == "fchmm":
        cfg["seed"] = int(method_seed)
        return cfg

    segmenter_cfg = dict(cfg.get("segmenter", {}))
    segmenter_cfg["seed"] = int(method_seed)
    cfg["segmenter"] = segmenter_cfg
    return cfg


def _combo_param_record(keys: list[str], values: tuple[Any, ...]) -> dict[str, Any]:
    return {key: value for key, value in zip(keys, values)}


def _aggregate_combo(rows: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_runs": len(rows),
        "params": dict(rows[0]["params"]),
        "combo_id": int(rows[0]["combo_id"]),
    }
    for metric_key in metric_keys:
        vals = np.asarray([row["metrics"].get(metric_key, np.nan) for row in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        summary[f"{metric_key}_mean"] = float(np.mean(vals)) if vals.size > 0 else math.nan
        summary[f"{metric_key}_std"] = float(np.std(vals, ddof=0)) if vals.size > 0 else math.nan
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    param_keys = sorted({key for row in rows for key in row.get("params", {}).keys()})
    metric_keys = sorted({key for row in rows for key in row.get("metrics", {}).keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["combo_id", "dataset", "method", "dataset_seed", "method_seed", *param_keys, *metric_keys])
        for row in rows:
            writer.writerow(
                [
                    int(row["combo_id"]),
                    str(row["dataset"]),
                    str(row["method"]),
                    int(row["dataset_seed"]),
                    int(row["method_seed"]),
                    *[row.get("params", {}).get(key, "") for key in param_keys],
                    *[row.get("metrics", {}).get(key, "") for key in metric_keys],
                ]
            )
    return path


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    param_keys = sorted({key for row in rows for key in row.get("params", {}).keys()})
    metric_keys = sorted({key for row in rows if key.endswith("_mean") or key.endswith("_std")})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["combo_id", "num_runs", *param_keys, *metric_keys])
        for row in rows:
            writer.writerow(
                [
                    int(row["combo_id"]),
                    int(row["num_runs"]),
                    *[row.get("params", {}).get(key, "") for key in param_keys],
                    *[row.get(key, "") for key in metric_keys],
                ]
            )
    return path


def run_param_search(
    *,
    dataset_name: str,
    method_name: str,
    grid_payload: dict[str, Any],
    dataset_seed: int,
    method_seeds: list[int],
    config_root: str | Path = "configs",
    output_root: str | Path = "outputs/param_search",
    save_runs: bool = False,
) -> dict[str, Any]:
    config_root = Path(config_root)
    if not config_root.is_absolute():
        config_root = PROJECT_ROOT / config_root
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    env_cfg = _load_env_config(config_root, dataset_name)
    dataset_method_overrides = dict(env_cfg.pop("method_overrides", {}))
    dataset_kwargs = dict(env_cfg)
    dataset_kwargs["seed"] = int(dataset_seed)

    base_method_cfg = _load_method_config(config_root, method_name)
    base_method_cfg = deep_merge(base_method_cfg, dataset_method_overrides.get(method_name, {}))

    grid_keys, grid_values = _grid_from_payload(grid_payload)
    combos = list(itertools.product(*grid_values))
    metric_keys = ["MeanConstraintError", "MeanAbsCutpointError"]

    search_dir = output_root / method_name / dataset_name
    search_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []

    for combo_id, combo in enumerate(combos):
        combo_params = _combo_param_record(grid_keys, combo)
        combo_override = _combo_overrides(grid_keys, combo)
        combo_method_cfg = deep_merge(base_method_cfg, combo_override)
        combo_dir = search_dir / f"combo_{combo_id:04d}"
        combo_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ParamSearch] combo {combo_id + 1}/{len(combos)} params={json.dumps(combo_params, ensure_ascii=False)}")
        for method_seed in method_seeds:
            seeded_cfg = _apply_method_seed(method_name, combo_method_cfg, int(method_seed))
            run_dir = combo_dir / f"method_seed_{int(method_seed):03d}"
            seeded_cfg = apply_run_plot_dirs(method_name, seeded_cfg, run_dir)
            result = run_experiment(
                dataset_name=dataset_name,
                method_name=method_name,
                dataset_kwargs=dataset_kwargs,
                method_kwargs=seeded_cfg,
            )
            metrics = _extract_scalar_metrics(_extract_metrics(method_name, result))
            run_rows.append(
                {
                    "combo_id": int(combo_id),
                    "dataset": dataset_name,
                    "method": method_name,
                    "dataset_seed": int(dataset_seed),
                    "method_seed": int(method_seed),
                    "params": combo_params,
                    "metrics": metrics,
                }
            )
            if save_runs:
                save_run_artifacts(
                    run_dir=run_dir,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    dataset_kwargs=dataset_kwargs,
                    method_kwargs=seeded_cfg,
                    result=result,
                )

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in run_rows:
        grouped.setdefault(int(row["combo_id"]), []).append(row)
    combo_summary = [_aggregate_combo(grouped[combo_id], metric_keys) for combo_id in sorted(grouped)]
    combo_summary.sort(key=lambda row: (float(row.get("MeanConstraintError_mean", math.inf)), int(row["combo_id"])))

    best = combo_summary[0] if combo_summary else None
    summary_payload = {
        "dataset": dataset_name,
        "method": method_name,
        "dataset_seed": int(dataset_seed),
        "method_seeds": [int(x) for x in method_seeds],
        "grid": grid_payload,
        "metric_keys": metric_keys,
        "best": best,
        "results": combo_summary,
    }
    write_json(search_dir / "search_summary.json", summary_payload)
    write_json(search_dir / "search_runs.json", {"results": run_rows})
    _write_summary_csv(search_dir / "search_summary.csv", combo_summary)
    _write_csv(search_dir / "search_runs.csv", run_rows)
    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="Grid search method parameters with repeated method seeds.")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--method", required=True, type=str)
    parser.add_argument("--grid-config", required=True, type=str, help="JSON file mapping dotted parameter paths to candidate lists.")
    parser.add_argument("--dataset-seed", type=int, default=None)
    parser.add_argument("--method-seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--config-root", type=str, default="configs")
    parser.add_argument("--output-root", type=str, default="outputs/param_search")
    parser.add_argument("--save-runs", action="store_true", help="Also save full per-run artifacts under each combo directory.")
    args = parser.parse_args()

    config_root = Path(args.config_root)
    if not config_root.is_absolute():
        config_root = PROJECT_ROOT / config_root
    env_cfg = _load_env_config(config_root, args.dataset)
    dataset_seed = int(env_cfg.get("seed", 0) if args.dataset_seed is None else args.dataset_seed)

    grid_path = Path(args.grid_config)
    if not grid_path.is_absolute():
        grid_path = PROJECT_ROOT / grid_path
    grid_payload = load_json(grid_path)

    summary = run_param_search(
        dataset_name=args.dataset,
        method_name=args.method,
        grid_payload=grid_payload,
        dataset_seed=dataset_seed,
        method_seeds=_split_int_csv(args.method_seeds),
        config_root=config_root,
        output_root=args.output_root,
        save_runs=bool(args.save_runs),
    )
    best = summary.get("best")
    if best is not None:
        print("[Best]")
        print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
