from __future__ import annotations

import argparse
import csv
import sys
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

from experiments.config_loader import load_json
from experiments.artifacts import (
    apply_run_plot_dirs,
    default_method_seed,
    _extract_objectives,
    resolve_run_dir,
    save_run_artifacts,
    write_json,
)
from experiments.unified_experiment import run_experiment
from visualization.io import save_figure
from visualization.plot4panel import plot_demos_goals_snapshot


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _method_config_name(method_name: str) -> str:
    return str(method_name)


def _load_env_config(config_root: Path, dataset_name: str) -> dict[str, Any]:
    path = config_root / "envs" / f"{dataset_name}.json"
    cfg = load_json(path)
    cfg = dict(cfg)
    cfg.pop("name", None)
    return cfg


def _load_method_config(config_root: Path, method_name: str) -> dict[str, Any]:
    path = config_root / "methods" / f"{_method_config_name(method_name)}.json"
    if not path.exists():
        available = sorted(p.stem for p in (config_root / "methods").glob("*.json"))
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Use one of: {', '.join(available)}"
        )
    cfg = load_json(path)
    cfg = dict(cfg)
    cfg.pop("name", None)
    return cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _extract_metrics(method_name: str, result: dict[str, Any]) -> dict[str, Any]:
    if method_name in {"swcl"}:
        return dict(result["joint_result"]["metrics"])
    return dict(result["constraints"]["metrics"])


def _extract_plot_record(method_name: str, result: dict[str, Any], method_seed: int) -> dict[str, Any]:
    if method_name == "swcl":
        learner = result["joint_result"]["model"]
        gammas = result["joint_result"]["gammas"]
    elif method_name == "fchmm":
        segmentation = result["segmentation"]
        learner = segmentation.model
        gammas = segmentation.extras.get("gammas")
    else:
        learner = result["constraints"]["model"]
        gammas = result["constraints"]["gammas"]
    return {
        "seed": int(method_seed),
        "learner": learner,
        "gammas": gammas,
    }


def _goal_grid_taus(record: dict[str, Any]):
    learner = record["learner"]
    gammas = record["gammas"]
    stage_ends = getattr(learner, "stage_ends_", None)
    if stage_ends is not None:
        out = []
        for ends in stage_ends:
            ends_arr = np.asarray(ends, dtype=int).reshape(-1)
            if ends_arr.size <= 1:
                out.append(int(ends_arr[0]))
            else:
                out.append([int(x) for x in ends_arr[:-1].tolist()])
        return out
    if getattr(learner, "taus_hat", None) is not None:
        return learner.taus_hat
    return [None for _ in gammas]


def _plot_goal_grid(records: list[dict[str, Any]], *, dataset_name: str, method_name: str, outdir: Path) -> Path | None:
    if plt is None or not records:
        return None
    runs_per_row = 4
    n = len(records)
    nrows = int(np.ceil(max(n, 1) / runs_per_row))
    fig, axes = plt.subplots(
        nrows,
        runs_per_row,
        figsize=(2.7 * runs_per_row, 2.1 * nrows),
        squeeze=False,
        constrained_layout=False,
    )
    for idx, record in enumerate(records):
        row = idx // runs_per_row
        col = idx % runs_per_row
        ax = axes[row, col]
        learner = record["learner"]
        gammas = record["gammas"]
        taus = _goal_grid_taus(record)
        plot_demos_goals_snapshot(
            ax,
            learner,
            taus,
            gammas,
            title=f"seed={record['seed']}",
            show_legend=(idx == 0),
        )
        ax.title.set_fontsize(7)
        if col != 0:
            ax.set_ylabel("")
        if row != nrows - 1:
            ax.set_xlabel("")
    for idx in range(n, nrows * runs_per_row):
        axes[idx // runs_per_row, idx % runs_per_row].axis("off")
    fig.suptitle(f"{method_name}: {dataset_name} runs", fontsize=11)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.05, wspace=0.16, hspace=0.18)
    return save_figure(fig, outdir / f"{dataset_name}_{method_name}_goals.png", dpi=180)


def _clear_png_files(path: Path) -> None:
    if not path.exists():
        return
    for child in path.glob("*.png"):
        if child.is_file():
            child.unlink()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    metric_keys = sorted({k for row in rows for k in row["metrics"].keys()})
    objective_keys = sorted({k for row in rows for k in row.get("objectives", {}).keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dataset", "dataset_seed", "method_seed", *metric_keys, *objective_keys])
        for row in rows:
            writer.writerow(
                [
                    row["method"],
                    row["dataset"],
                    row["dataset_seed"],
                    row["method_seed"],
                    *[row["metrics"].get(key, "") for key in metric_keys],
                    *[row.get("objectives", {}).get(key, "") for key in objective_keys],
                ]
            )


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("method")),
        str(row.get("dataset")),
        int(row.get("dataset_seed", 0)),
        int(row.get("method_seed", 0)),
    )


def _merge_with_existing_summary(path: Path, new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    existing_rows: list[dict[str, Any]] = []
    existing_methods: list[str] = []
    existing_datasets: list[str] = []
    existing_method_seeds: list[int] = []
    existing_dataset_seed = None

    if path.exists():
        existing = load_json(path)
        existing_rows = list(existing.get("results", []))
        existing_methods = [str(x) for x in existing.get("methods", [])]
        existing_datasets = [str(x) for x in existing.get("datasets", [])]
        existing_method_seeds = [int(x) for x in existing.get("method_seeds", [])]
        if "dataset_seed" in existing:
            existing_dataset_seed = int(existing["dataset_seed"])

    merged_by_key = {_row_key(row): row for row in existing_rows}
    for row in new_rows:
        merged_by_key[_row_key(row)] = row

    merged_rows = sorted(
        merged_by_key.values(),
        key=lambda row: (
            str(row.get("dataset")),
            str(row.get("method")),
            int(row.get("dataset_seed", 0)),
            int(row.get("method_seed", 0)),
        ),
    )
    merged_methods = sorted({*existing_methods, *[str(row["method"]) for row in merged_rows]})
    merged_datasets = sorted({*existing_datasets, *[str(row["dataset"]) for row in merged_rows]})
    merged_method_seeds = sorted({*existing_method_seeds, *[int(row["method_seed"]) for row in merged_rows]})

    dataset_seeds = sorted({int(row.get("dataset_seed", 0)) for row in merged_rows})
    dataset_seed_value: int | list[int]
    if len(dataset_seeds) == 1:
        dataset_seed_value = dataset_seeds[0]
    else:
        dataset_seed_value = dataset_seeds

    if existing_dataset_seed is not None and isinstance(dataset_seed_value, int):
        dataset_seed_value = int(dataset_seed_value)

    return {
        "methods": merged_methods,
        "datasets": merged_datasets,
        "dataset_seed": dataset_seed_value,
        "method_seeds": merged_method_seeds,
        "results": merged_rows,
    }


def run_benchmark(
    methods: list[str],
    datasets: list[str],
    method_seeds: list[int],
    dataset_seed: int = 0,
    config_root: str | Path = "configs",
    outdir: str | Path = "outputs/benchmark",
    dataset_overrides: dict[str, dict[str, Any]] | None = None,
    method_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config_root = Path(config_root)
    if not config_root.is_absolute():
        config_root = PROJECT_ROOT / config_root

    outdir = Path(outdir)
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir

    dataset_overrides = dataset_overrides or {}
    method_overrides = method_overrides or {}

    rows: list[dict[str, Any]] = []
    goal_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for method_name in methods:
        base_method_cfg = _load_method_config(config_root, method_name)
        external_method_override = method_overrides.get(method_name, {})
        for dataset_name in datasets:
            base_dataset_cfg = _load_env_config(config_root, dataset_name)
            dataset_method_overrides = dict(base_dataset_cfg.pop("method_overrides", {}))
            env_method_override = dataset_method_overrides.get(method_name, {})
            external_dataset_override = dataset_overrides.get(dataset_name, {})
            base_dataset_cfg.setdefault("seed", int(dataset_seed))
            for method_seed in method_seeds:
                dataset_cfg = dict(base_dataset_cfg)
                run_method_cfg = dict(base_method_cfg)
                dataset_cfg = _deep_merge(dataset_cfg, external_dataset_override)
                run_method_cfg = _deep_merge(run_method_cfg, env_method_override)
                run_method_cfg = _deep_merge(run_method_cfg, external_method_override)
                if method_name in {"swcl", "fchmm"}:
                    run_method_cfg["seed"] = int(method_seed)
                else:
                    segmenter_cfg = dict(run_method_cfg.get("segmenter", {}))
                    segmenter_cfg["seed"] = int(method_seed)
                    run_method_cfg["segmenter"] = segmenter_cfg
                run_dir = resolve_run_dir(
                    method_name=method_name,
                    dataset_name=dataset_name,
                    dataset_seed=int(dataset_cfg.get("seed", dataset_seed)),
                    method_seed=default_method_seed(method_name, run_method_cfg),
                    output_root=PROJECT_ROOT / "outputs",
                )
                _clear_png_files(run_dir)
                run_method_cfg = apply_run_plot_dirs(
                    method_name,
                    run_method_cfg,
                    run_dir,
                )
                result = run_experiment(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    dataset_kwargs=dataset_cfg,
                    method_kwargs=run_method_cfg,
                )
                save_run_artifacts(
                    run_dir=run_dir,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    dataset_kwargs=dataset_cfg,
                    method_kwargs=run_method_cfg,
                    result=result,
                    env_config_path=config_root / "envs" / f"{dataset_name}.json",
                    method_config_path=config_root / "methods" / f"{_method_config_name(method_name)}.json",
                )
                rows.append(
                    {
                        "method": method_name,
                        "dataset": dataset_name,
                        "dataset_seed": int(dataset_seed),
                        "method_seed": int(method_seed),
                        "metrics": _extract_metrics(method_name, result),
                        "objectives": _extract_objectives(method_name, result),
                    }
                )
                goal_records.setdefault((dataset_name, method_name), []).append(
                    _extract_plot_record(method_name, result, method_seed)
                )

    for (dataset_name, method_name), records in goal_records.items():
        _plot_goal_grid(records, dataset_name=dataset_name, method_name=method_name, outdir=outdir)

    summary = _merge_with_existing_summary(outdir / "benchmark_results.json", rows)
    write_json(outdir / "benchmark_results.json", summary)
    _write_csv(outdir / "benchmark_results.csv", list(summary["results"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run benchmark over methods, datasets, and seeds.")
    parser.add_argument("--methods", default="swcl")
    parser.add_argument("--datasets", default="S3ObsAvoid")
    parser.add_argument("--method-seeds", default="0")
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--outdir", default="outputs/benchmark")
    args = parser.parse_args()

    methods = _split_csv(args.methods)
    datasets = _split_csv(args.datasets)
    method_seed_text = args.method_seeds if args.seeds is None else args.seeds
    method_seeds = [int(seed) for seed in _split_csv(method_seed_text)]

    summary = run_benchmark(
        methods=methods,
        datasets=datasets,
        method_seeds=method_seeds,
        dataset_seed=args.dataset_seed,
        config_root=args.config_root,
        outdir=args.outdir,
    )
    print(
        f"Completed {len(summary['results'])} runs "
        f"for methods={methods}, datasets={datasets}, dataset_seed={args.dataset_seed}, method_seeds={method_seeds}."
    )


if __name__ == "__main__":
    main()
