from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config_loader import deep_merge, load_experiment_config
from experiments.artifacts import (
    apply_run_plot_dirs,
    default_method_seed,
    resolve_plot_dir,
    resolve_run_dir,
    save_run_artifacts,
)
from experiments.unified_experiment import run_experiment
from methods import JOINT_METHODS


def _clear_png_files(path_like: object) -> None:
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return
    for child in path.glob("*.png"):
        if child.is_file():
            child.unlink()


def main():
    parser = argparse.ArgumentParser(description="Run one experiment from environment/method config files.")
    parser.add_argument("--env-config", required=True, type=str)
    parser.add_argument("--method-config", required=True, type=str)
    parser.add_argument("--dataset-seed", type=int, default=None)
    parser.add_argument("--method-seed", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--output-root", type=str, default="outputs")
    args = parser.parse_args()

    cfg = load_experiment_config(args.env_config, args.method_config)
    dataset_cfg = dict(cfg["dataset"])
    method_cfg = dict(cfg["method"])

    dataset_name = dataset_cfg.pop("name")
    method_name = method_cfg.pop("name")
    dataset_method_overrides = dict(dataset_cfg.pop("method_overrides", {}))
    method_cfg = deep_merge(method_cfg, dataset_method_overrides.get(method_name, {}))

    if args.dataset_seed is not None:
        dataset_cfg["seed"] = int(args.dataset_seed)
    dataset_seed = int(dataset_cfg.get("seed", 0))
    if method_name in JOINT_METHODS:
        method_cfg.setdefault("seed", 0 if args.method_seed is None else int(args.method_seed))
        if args.method_seed is not None:
            method_cfg["seed"] = int(args.method_seed)
        if args.max_iter is not None:
            method_cfg["max_iter"] = int(args.max_iter)
    elif method_name == "fchmm":
        method_cfg.setdefault("seed", 0 if args.method_seed is None else int(args.method_seed))
        if args.method_seed is not None:
            method_cfg["seed"] = int(args.method_seed)
        if args.max_iter is not None:
            method_cfg["max_iter"] = int(args.max_iter)
    else:
        segmenter_cfg = dict(method_cfg.get("segmenter", {}))
        posthoc_key = "posthoc" if method_name == "hmm" else "constraints"
        constraint_cfg = dict(method_cfg.get(posthoc_key, {}))
        segmenter_cfg.setdefault("seed", 0 if args.method_seed is None else int(args.method_seed))
        if args.method_seed is not None:
            segmenter_cfg["seed"] = int(args.method_seed)
        if args.max_iter is not None:
            segmenter_cfg["max_iter"] = int(args.max_iter)
        method_cfg["segmenter"] = segmenter_cfg
        method_cfg[posthoc_key] = constraint_cfg

    method_seed = default_method_seed(method_name, method_cfg)
    run_dir = resolve_run_dir(
        method_name=method_name,
        dataset_name=dataset_name,
        dataset_seed=dataset_seed,
        method_seed=method_seed,
        output_root=args.output_root,
    )
    plot_dir = resolve_plot_dir(run_dir)
    method_cfg = apply_run_plot_dirs(method_name, method_cfg, plot_dir)
    _clear_png_files(plot_dir)

    results = run_experiment(
        dataset_name=dataset_name,
        method_name=method_name,
        dataset_kwargs=dataset_cfg,
        method_kwargs=method_cfg,
    )
    save_run_artifacts(
        run_dir=run_dir,
        dataset_name=dataset_name,
        method_name=method_name,
        dataset_kwargs=dataset_cfg,
        method_kwargs=method_cfg,
        result=results,
        env_config_path=args.env_config,
        method_config_path=args.method_config,
    )
    print(f"[Saved] {run_dir}")

    if method_name in JOINT_METHODS:
        print(results["joint_result"]["metrics"])
    else:
        print(results["constraints"]["metrics"])


if __name__ == "__main__":
    main()
