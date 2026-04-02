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

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.artifacts import apply_run_plot_dirs, save_run_artifacts, write_json
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


def _append_scalar_if_finite(out: dict[str, float], key: str, value: Any) -> None:
    if value is None:
        return
    if not np.isscalar(value):
        return
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(value_f):
        out[str(key)] = value_f


def _extract_objective_scalars(method_name: str, result: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}

    segmentation = result.get("segmentation", {})
    seg_model = segmentation.model if hasattr(segmentation, "model") else None
    seg_extras = segmentation.extras if hasattr(segmentation, "extras") and isinstance(segmentation.extras, dict) else {}
    constraint_model = result.get("constraints", {}).get("model")

    if method_name == "cluster" and seg_model is not None:
        history = getattr(seg_model, "objective_history_", None)
        if history:
            _append_scalar_if_finite(out, "SegmentationObjectiveFinal", history[-1])
    elif method_name == "arhsmm":
        history = (seg_extras.get("segmentation_history") or {}).get("loglik")
        if history:
            _append_scalar_if_finite(out, "SegmentationLogLikelihoodFinal", history[-1])
    elif method_name in {"fchmm", "hmm"} and constraint_model is not None:
        history = getattr(constraint_model, "loss_loglik", None)
        if history:
            _append_scalar_if_finite(out, "TrainingLogLikelihoodFinal", history[-1])

    if constraint_model is not None:
        _append_scalar_if_finite(
            out,
            "PosthocObjectiveFinal",
            getattr(constraint_model, "posthoc_total_objective_", None),
        )
        _append_scalar_if_finite(
            out,
            "PosthocFeatureObjectiveFinal",
            getattr(constraint_model, "posthoc_feature_objective_", None),
        )

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


def _disable_plots_in_cfg(method_name: str, method_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(method_cfg)
    if method_name in JOINT_METHODS or method_name == "fchmm":
        cfg["disable_plots"] = True
        cfg["plot_every"] = None
        return cfg

    cfg["disable_plots"] = True
    segmenter_cfg = dict(cfg.get("segmenter", {}))
    segmenter_cfg["plot_every"] = None
    cfg["segmenter"] = segmenter_cfg
    constraint_cfg = dict(cfg.get("posthoc_constraint", {}))
    constraint_cfg["disable_plots"] = True
    cfg["posthoc_constraint"] = constraint_cfg
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
    metric_keys = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if str(key).endswith("_mean") or str(key).endswith("_std")
        }
    )
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


def _plot_search_summary(rows: list[dict[str, Any]], save_path: Path, *, top_k: int = 15) -> Path | None:
    if plt is None or not rows:
        return None

    top_rows = list(rows[: max(int(top_k), 1)])
    labels = []
    means = []
    stds = []
    for row in top_rows:
        params = dict(row.get("params", {}))
        label = ", ".join(f"{key}={params[key]}" for key in sorted(params))
        labels.append(label)
        means.append(float(row.get("MeanConstraintError_mean", math.nan)))
        stds.append(float(row.get("MeanConstraintError_std", math.nan)))

    x = np.arange(len(top_rows), dtype=float)
    fig_h = max(3.0, 0.45 * len(top_rows) + 1.4)
    fig, ax = plt.subplots(figsize=(9.5, fig_h), constrained_layout=False)
    ax.barh(
        x,
        means,
        xerr=stds,
        color="#4C78A8",
        alpha=0.82,
        edgecolor="#4C78A8",
        error_kw={"ecolor": "#1F2937", "elinewidth": 1.0, "capsize": 2.0},
    )
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MeanConstraintError", fontsize=10)
    ax.set_title("Top parameter combinations", fontsize=11, pad=6)
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout(pad=0.5)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _pick_metric_key(rows: list[dict[str, Any]], preferred: list[str]) -> str | None:
    if not rows:
        return None
    available = {str(key) for row in rows for key in row.get("metrics", {}).keys()}
    for key in preferred:
        if key in available:
            return key
    return None


def _plot_metric_vs_constraint(
    rows: list[dict[str, Any]],
    save_path: Path,
    *,
    preferred_metric_keys: list[str],
    title: str,
) -> Path | None:
    if plt is None or not rows:
        return None

    metric_key = _pick_metric_key(rows, preferred_metric_keys)
    if metric_key is None:
        return None

    xs = np.asarray([row.get("metrics", {}).get(metric_key, np.nan) for row in rows], dtype=float)
    ys = np.asarray([row.get("metrics", {}).get("MeanConstraintError", np.nan) for row in rows], dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[valid]
    ys = ys[valid]
    if xs.size < 2:
        return None

    fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=False)
    ax.scatter(xs, ys, s=34, color="#4C78A8", alpha=0.85, edgecolors="none")

    if xs.size >= 2 and not np.allclose(xs, xs[0]):
        slope, intercept = np.polyfit(xs, ys, deg=1)
        x_line = np.linspace(float(np.min(xs)), float(np.max(xs)), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#E45756", linewidth=1.8, alpha=0.95)

    corr = float(np.corrcoef(xs, ys)[0, 1]) if xs.size >= 2 else math.nan
    ax.set_xlabel(metric_key, fontsize=10)
    ax.set_ylabel("MeanConstraintError", fontsize=10)
    ax.set_title(f"{title} (r={corr:.3f})", fontsize=11, pad=6)
    ax.grid(alpha=0.22)
    fig.tight_layout(pad=0.6)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _plot_objective_vs_constraint(rows: list[dict[str, Any]], save_path: Path) -> Path | None:
    return _plot_metric_vs_constraint(
        rows,
        save_path,
        preferred_metric_keys=[
            "SegmentationObjectiveFinal",
            "SegmentationLogLikelihoodFinal",
            "TrainingLogLikelihoodFinal",
            "PosthocObjectiveFinal",
            "PosthocFeatureObjectiveFinal",
        ],
        title="Per-run objective vs constraint error",
    )


def _plot_posthoc_vs_constraint(rows: list[dict[str, Any]], save_path: Path) -> Path | None:
    return _plot_metric_vs_constraint(
        rows,
        save_path,
        preferred_metric_keys=[
            "PosthocObjectiveFinal",
            "PosthocFeatureObjectiveFinal",
            "TrainingLogLikelihoodFinal",
        ],
        title="Per-run posthoc/training likelihood vs constraint error",
    )


def _plot_cutpoint_vs_constraint(rows: list[dict[str, Any]], save_path: Path) -> Path | None:
    return _plot_metric_vs_constraint(
        rows,
        save_path,
        preferred_metric_keys=[
            "MeanAbsCutpointError",
        ],
        title="Per-run cutpoint error vs constraint error",
    )


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
    search_dir = output_root / method_name / dataset_name
    search_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []

    for combo_id, combo in enumerate(combos):
        combo_params = _combo_param_record(grid_keys, combo)
        combo_override = _combo_overrides(grid_keys, combo)
        combo_method_cfg = deep_merge(base_method_cfg, combo_override)
        combo_dir = search_dir / f"combo_{combo_id:04d}"

        print(f"[ParamSearch] combo {combo_id + 1}/{len(combos)} params={json.dumps(combo_params, ensure_ascii=False)}")
        for method_seed in method_seeds:
            seeded_cfg = _apply_method_seed(method_name, combo_method_cfg, int(method_seed))
            seeded_cfg = _disable_plots_in_cfg(method_name, seeded_cfg)
            run_dir = combo_dir / f"method_seed_{int(method_seed):03d}" if save_runs else (search_dir / "_disabled_plots")
            seeded_cfg = apply_run_plot_dirs(method_name, seeded_cfg, run_dir)
            result = run_experiment(
                dataset_name=dataset_name,
                method_name=method_name,
                dataset_kwargs=dataset_kwargs,
                method_kwargs=seeded_cfg,
            )
            metrics = _extract_scalar_metrics(_extract_metrics(method_name, result))
            metrics.update(_extract_objective_scalars(method_name, result))
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
                combo_dir.mkdir(parents=True, exist_ok=True)
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
    metric_keys = sorted({key for row in run_rows for key in row.get("metrics", {}).keys()})
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
    _plot_search_summary(combo_summary, search_dir / "search_summary_top.png")
    _plot_objective_vs_constraint(run_rows, search_dir / "search_objective_vs_constraint.png")
    _plot_posthoc_vs_constraint(run_rows, search_dir / "search_posthoc_vs_constraint.png")
    _plot_cutpoint_vs_constraint(run_rows, search_dir / "search_cutpoint_vs_constraint.png")
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
