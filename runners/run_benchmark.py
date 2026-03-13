from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config_loader import load_json
from experiments.unified_experiment import run_experiment


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _load_env_config(config_root: Path, dataset_name: str) -> dict[str, Any]:
    path = config_root / "envs" / f"{dataset_name}.json"
    cfg = load_json(path)
    cfg = dict(cfg)
    cfg.pop("name", None)
    return cfg


def _load_method_config(config_root: Path, method_name: str) -> dict[str, Any]:
    path = config_root / "methods" / f"{method_name}.json"
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


def _default_plot_dir(outdir: Path, method_name: str, dataset_name: str, seed: int) -> str:
    return str(outdir.parent / "plots" / method_name / dataset_name)


def _extract_metrics(method_name: str, result: dict[str, Any]) -> dict[str, Any]:
    if method_name == "segcons":
        return dict(result["joint_result"]["metrics"])
    return dict(result["constraints"]["metrics"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    metric_keys = sorted({k for row in rows for k in row["metrics"].keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dataset", "seed", *metric_keys])
        for row in rows:
            writer.writerow(
                [
                    row["method"],
                    row["dataset"],
                    row["seed"],
                    *[row["metrics"].get(key, "") for key in metric_keys],
                ]
            )


def run_benchmark(
    methods: list[str],
    datasets: list[str],
    seeds: list[int],
    config_root: str | Path = "configs",
    outdir: str | Path = "outputs/experiments",
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
    for method_name in methods:
        method_cfg = _load_method_config(config_root, method_name)
        method_cfg.update(method_overrides.get(method_name, {}))
        for dataset_name in datasets:
            base_dataset_cfg = _load_env_config(config_root, dataset_name)
            base_dataset_cfg.update(dataset_overrides.get(dataset_name, {}))
            for seed in seeds:
                dataset_cfg = dict(base_dataset_cfg)
                dataset_cfg["seed"] = seed
                run_method_cfg = dict(method_cfg)
                plot_dir = _default_plot_dir(outdir, method_name, dataset_name, seed)
                if method_name == "segcons":
                    run_method_cfg.setdefault("plot_dir", plot_dir)
                else:
                    segmenter_cfg = dict(run_method_cfg.get("segmenter", {}))
                    constraint_cfg = dict(run_method_cfg.get("constraints", {}))
                    segmenter_cfg.setdefault("plot_dir", plot_dir)
                    constraint_cfg.setdefault("plot_dir", plot_dir)
                    run_method_cfg["segmenter"] = segmenter_cfg
                    run_method_cfg["constraints"] = constraint_cfg
                result = run_experiment(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    dataset_kwargs=dataset_cfg,
                    method_kwargs=run_method_cfg,
                )
                rows.append(
                    {
                        "method": method_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        "metrics": _extract_metrics(method_name, result),
                    }
                )

    summary = {
        "methods": methods,
        "datasets": datasets,
        "seeds": seeds,
        "results": rows,
    }
    _write_json(outdir / "benchmark_results.json", summary)
    _write_csv(outdir / "benchmark_results.csv", rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run CHMM benchmark over methods, datasets, and seeds.")
    parser.add_argument("--methods", default="segcons,gmmhmm,hdphmm,arhmm,changepoint")
    parser.add_argument("--datasets", default="2DObsAvoid")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--outdir", default="outputs/experiments")
    args = parser.parse_args()

    methods = _split_csv(args.methods)
    datasets = _split_csv(args.datasets)
    seeds = [int(seed) for seed in _split_csv(args.seeds)]

    summary = run_benchmark(
        methods=methods,
        datasets=datasets,
        seeds=seeds,
        config_root=args.config_root,
        outdir=args.outdir,
    )
    print(
        f"Completed {len(summary['results'])} runs "
        f"for methods={methods}, datasets={datasets}, seeds={seeds}."
    )


if __name__ == "__main__":
    main()
