from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config_loader import deep_merge, load_experiment_config
from experiments.unified_experiment import run_experiment


def _default_plot_dir(method_name: str, dataset_name: str, seed: int) -> str:
    return str(PROJECT_ROOT / "outputs" / "plots" / method_name / dataset_name)


def main():
    parser = argparse.ArgumentParser(description="Run one experiment from environment/method config files.")
    parser.add_argument("--env-config", required=True, type=str)
    parser.add_argument("--method-config", required=True, type=str)
    parser.add_argument("--dataset-seed", type=int, default=None)
    parser.add_argument("--method-seed", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
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
    plot_dir = _default_plot_dir(method_name, dataset_name, dataset_seed)
    if method_name == "segcons":
        method_cfg.setdefault("plot_dir", plot_dir)
        method_cfg.setdefault("seed", 0 if args.method_seed is None else int(args.method_seed))
        if args.method_seed is not None:
            method_cfg["seed"] = int(args.method_seed)
        if args.max_iter is not None:
            method_cfg["max_iter"] = int(args.max_iter)
    else:
        segmenter_cfg = dict(method_cfg.get("segmenter", {}))
        constraint_cfg = dict(method_cfg.get("constraints", {}))
        segmenter_cfg.setdefault("plot_dir", plot_dir)
        segmenter_cfg.setdefault("seed", 0 if args.method_seed is None else int(args.method_seed))
        if args.method_seed is not None:
            segmenter_cfg["seed"] = int(args.method_seed)
        if args.max_iter is not None:
            segmenter_cfg["max_iter"] = int(args.max_iter)
        constraint_cfg.setdefault("plot_dir", plot_dir)
        method_cfg["segmenter"] = segmenter_cfg
        method_cfg["constraints"] = constraint_cfg

    results = run_experiment(
        dataset_name=dataset_name,
        method_name=method_name,
        dataset_kwargs=dataset_cfg,
        method_kwargs=method_cfg,
    )

    if method_name == "segcons":
        print(results["joint_result"]["metrics"])
    else:
        print(results["constraints"]["metrics"])


if __name__ == "__main__":
    main()
