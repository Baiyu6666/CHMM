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


def _default_plot_dir(outdir: Path, method_name: str, dataset_name: str, method_seed: int) -> str:
    return str(outdir.parent / "plots" / method_name / dataset_name)


def _should_replace_plot_dir(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text in {"", "outputs/plots"}


def _apply_runtime_plot_defaults(
    method_name: str,
    method_cfg: dict[str, Any],
    plot_dir: str,
    method_seed: int,
) -> dict[str, Any]:
    cfg = dict(method_cfg)
    if method_name in {"segcons", "ccp"}:
        if _should_replace_plot_dir(cfg.get("plot_dir")):
            cfg["plot_dir"] = plot_dir
        cfg.setdefault("seed", int(method_seed))
        return cfg

    segmenter_cfg = dict(cfg.get("segmenter", {}))
    constraint_cfg = dict(cfg.get("constraints", {}))
    if _should_replace_plot_dir(segmenter_cfg.get("plot_dir")):
        segmenter_cfg["plot_dir"] = plot_dir
    if _should_replace_plot_dir(constraint_cfg.get("plot_dir")):
        constraint_cfg["plot_dir"] = plot_dir
    segmenter_cfg.setdefault("seed", int(method_seed))
    cfg["segmenter"] = segmenter_cfg
    cfg["constraints"] = constraint_cfg
    return cfg


def _extract_metrics(method_name: str, result: dict[str, Any]) -> dict[str, Any]:
    if method_name in {"segcons", "ccp"}:
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
        writer.writerow(["method", "dataset", "dataset_seed", "method_seed", *metric_keys])
        for row in rows:
            writer.writerow(
                [
                    row["method"],
                    row["dataset"],
                    row["dataset_seed"],
                    row["method_seed"],
                    *[row["metrics"].get(key, "") for key in metric_keys],
                ]
            )


def run_benchmark(
    methods: list[str],
    datasets: list[str],
    method_seeds: list[int],
    dataset_seed: int = 0,
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
                plot_dir = _default_plot_dir(outdir, method_name, dataset_name, method_seed)
                dataset_cfg = _deep_merge(dataset_cfg, external_dataset_override)
                run_method_cfg = _deep_merge(run_method_cfg, env_method_override)
                run_method_cfg = _apply_runtime_plot_defaults(
                    method_name, run_method_cfg, plot_dir, method_seed
                )
                run_method_cfg = _deep_merge(run_method_cfg, external_method_override)
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
                        "dataset_seed": int(dataset_seed),
                        "method_seed": int(method_seed),
                        "metrics": _extract_metrics(method_name, result),
                    }
                )

    summary = {
        "methods": methods,
        "datasets": datasets,
        "dataset_seed": int(dataset_seed),
        "method_seeds": method_seeds,
        "results": rows,
    }
    _write_json(outdir / "benchmark_results.json", summary)
    _write_csv(outdir / "benchmark_results.csv", rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run CHMM benchmark over methods, datasets, and seeds.")
    parser.add_argument("--methods", default="segcons")
    parser.add_argument("--datasets", default="2DObsAvoid")
    parser.add_argument("--method-seeds", default="0")
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--outdir", default="outputs/experiments")
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
