from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import load_env
from experiments.config_loader import deep_merge, load_json
from methods import ALL_METHODS, JOINT_METHODS, SEQUENTIAL_METHODS
from pipelines import JointPipeline, SequentialPipeline


def _default_plot_dir(method_name: str, dataset_name: str) -> str:
    return str(PROJECT_ROOT / "outputs" / "plots" / method_name / dataset_name)


def _should_replace_plot_dir(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text in {"", "outputs/plots"}


def _load_env_config(dataset_name: str) -> Dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "envs" / f"{dataset_name}.json"
    cfg = dict(load_json(path))
    cfg.pop("name", None)
    return cfg


def _load_method_config(method_name: str) -> Dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "methods" / f"{method_name}.json"
    cfg = dict(load_json(path))
    cfg.pop("name", None)
    return cfg


def run_experiment(
    dataset_name: str,
    method_name: str,
    dataset_kwargs: Dict[str, Any] | None = None,
    method_kwargs: Dict[str, Any] | None = None,
):
    dataset_kwargs = dataset_kwargs or {}
    method_kwargs = method_kwargs or {}
    method_kwargs = dict(method_kwargs)
    dataset = load_env(dataset_name, **dataset_kwargs)
    default_plot_dir = _default_plot_dir(method_name, dataset_name)

    if method_name in JOINT_METHODS:
        if _should_replace_plot_dir(method_kwargs.get("plot_dir")):
            method_kwargs["plot_dir"] = default_plot_dir
        pipeline = JointPipeline(method_name=method_name, kwargs=method_kwargs)
        return pipeline.run(dataset)

    if method_name in SEQUENTIAL_METHODS:
        segmenter_kwargs = dict(method_kwargs.get("segmenter", {}))
        constraint_kwargs = dict(method_kwargs.get("constraints", {}))
        if _should_replace_plot_dir(segmenter_kwargs.get("plot_dir")):
            segmenter_kwargs["plot_dir"] = default_plot_dir
        if _should_replace_plot_dir(constraint_kwargs.get("plot_dir")):
            constraint_kwargs["plot_dir"] = default_plot_dir
        pipeline = SequentialPipeline(
            segmenter_name=method_name,
            segmenter_kwargs=segmenter_kwargs,
            constraint_kwargs=constraint_kwargs,
        )
        return pipeline.run(dataset)

    raise ValueError(
        f"Unknown method '{method_name}'. "
        f"Available: {list(ALL_METHODS)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Unified CHMM experiment entrypoint.")
    parser.add_argument("--dataset", type=str, default="2DObsAvoid")
    parser.add_argument("--method", type=str, default="scdp")
    parser.add_argument("--n-demos", type=int, default=10)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=None)
    args = parser.parse_args()

    dataset_cfg = _load_env_config(args.dataset)
    dataset_method_overrides = dict(dataset_cfg.pop("method_overrides", {}))
    dataset_kwargs = dataset_cfg
    dataset_kwargs = deep_merge(dataset_kwargs, {"n_demos": args.n_demos, "seed": args.dataset_seed})
    method_kwargs = _load_method_config(args.method)
    method_kwargs = deep_merge(method_kwargs, dataset_method_overrides.get(args.method, {}))
    if args.method in JOINT_METHODS:
        override_kwargs = {"verbose": True, "seed": args.method_seed}
        if args.max_iter is not None:
            override_kwargs["max_iter"] = args.max_iter
        method_kwargs = deep_merge(method_kwargs, override_kwargs)
    else:
        segmenter_override = {"verbose": True, "seed": args.method_seed}
        if args.max_iter is not None:
            segmenter_override["max_iter"] = args.max_iter
        method_kwargs = deep_merge(
            method_kwargs,
            {
                "segmenter": segmenter_override,
                "constraints": {"refine_steps": 5},
            },
        )

    results = run_experiment(
        dataset_name=args.dataset,
        method_name=args.method,
        dataset_kwargs=dataset_kwargs,
        method_kwargs=method_kwargs,
    )

    if args.method in JOINT_METHODS:
        print(results["joint_result"]["metrics"])
    else:
        print(results["constraints"]["metrics"])


if __name__ == "__main__":
    main()
