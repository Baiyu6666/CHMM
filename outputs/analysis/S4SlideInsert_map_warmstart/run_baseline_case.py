from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config_loader import deep_merge, load_json
from experiments.unified_experiment import run_experiment


OUTPUT_DIR = Path(__file__).resolve().parent


def load_configs(method_name: str):
    dataset_config = dict(load_json(ROOT / "configs/envs/S4SlideInsert.json"))
    dataset_config.pop("name", None)
    method_overrides = dict(dataset_config.pop("method_overrides", {}))
    method_config = dict(load_json(ROOT / f"configs/methods/{method_name}.json"))
    method_config.pop("name", None)
    method_config = deep_merge(method_config, method_overrides[method_name])
    if method_name == "fchmm":
        method_config.update({"seed": 0, "verbose": False, "plot_every": None, "disable_plots": True})
    else:
        method_config["segmenter"] = deep_merge(
            method_config.get("segmenter", {}),
            {"seed": 0, "verbose": False, "disable_plots": True},
        )
        method_config["posthoc_constraint"] = deep_merge(
            method_config.get("posthoc_constraint", {}),
            {"verbose": False, "disable_plots": True},
        )
    return dataset_config, method_config


def stage_ends_from_cutpoints(cutpoints, demos):
    return [list(map(int, cuts)) + [len(demo) - 1] for cuts, demo in zip(cutpoints, demos)]


def summarize(method_name: str, label: str, result, initial_stage_ends, elapsed_sec: float):
    segmentation = result["segmentation"]
    final_cutpoints = [list(map(int, row)) for row in segmentation.cutpoints]
    initial_cutpoints = [list(map(int, row[:-1])) for row in initial_stage_ends]
    true_cutpoints = [list(map(int, row)) for row in result["dataset"].true_cutpoints]
    map_cutpoints = json.loads((OUTPUT_DIR / "map_cutpoints.json").read_text())["cutpoints"]
    metrics = result["constraints"]["metrics"]

    def cutpoint_difference(left, right):
        shifts = [abs(int(a) - int(b)) for x, y in zip(left, right) for a, b in zip(x, y)]
        return {
            "moved_coordinates": int(sum(value != 0 for value in shifts)),
            "total_coordinates": int(len(shifts)),
            "total_abs_shift": int(sum(shifts)),
            "mean_abs_shift": float(np.mean(shifts)) if shifts else 0.0,
            "max_abs_shift": int(max(shifts)) if shifts else 0,
            "unchanged_demos": int(sum(x == y for x, y in zip(left, right))),
        }

    true_semantics = np.asarray(metrics.get("ConstraintSemanticsMatrix", []), dtype=object)
    learned_semantics = np.asarray(metrics.get("ConstraintLearnedSemanticsMatrix", []), dtype=object)
    semantic_matches = None
    if true_semantics.shape == learned_semantics.shape and true_semantics.size:
        semantic_matches = int(np.sum(true_semantics == learned_semantics))

    return {
        "method": method_name,
        "label": label,
        "elapsed_sec": float(elapsed_sec),
        "initial_cutpoints": initial_cutpoints,
        "final_cutpoints": final_cutpoints,
        "true_cutpoints": true_cutpoints,
        "movement_from_initial": cutpoint_difference(final_cutpoints, initial_cutpoints),
        "difference_from_map": cutpoint_difference(final_cutpoints, map_cutpoints),
        "difference_from_true": cutpoint_difference(final_cutpoints, true_cutpoints),
        "constraint_metrics": {
            "MeanConstraintError": metrics.get("MeanConstraintError"),
            "MeanConstraintErrorRaw": metrics.get("MeanConstraintErrorRaw"),
            "semantic_exact_cells": semantic_matches,
            "semantic_total_cells": int(true_semantics.size),
            "ConstraintPredictedActiveMask": metrics.get("ConstraintPredictedActiveMask"),
            "ConstraintLearnedSemanticsMatrix": metrics.get("ConstraintLearnedSemanticsMatrix"),
        },
        "segmentation_metrics": {
            key: metrics.get(key)
            for key in ("MeanAbsCutpointError", "CutpointExactMatchRate", "MeanAbsTauError", "TauAccuracy")
            if key in metrics
        },
    }


def run_case(method_name: str, warm_start: bool, map_cutpoints):
    dataset_config, method_config = load_configs(method_name)
    restore = None
    if warm_start and method_name == "fchmm":
        from methods.cores.fchmm_core import FCHMM

        original = FCHMM._initial_stage_ends

        def initial_stage_ends(self, tau_init=None):
            if tau_init is None:
                return original(self, tau_init=tau_init)
            return stage_ends_from_cutpoints(tau_init, self.demos)

        FCHMM._initial_stage_ends = initial_stage_ends
        method_config["tau_init"] = map_cutpoints
        restore = lambda: setattr(FCHMM, "_initial_stage_ends", original)
    elif warm_start and method_name == "arhsmm":
        method_config["segmenter"]["tau_init"] = map_cutpoints
    elif warm_start and method_name == "cluster":
        import methods.backends.ordered_cluster as ordered_cluster

        original = ordered_cluster._init_stage_ends

        def initial_stage_ends(demos, num_stages, min_len, rng, mode):
            return stage_ends_from_cutpoints(map_cutpoints, demos)

        ordered_cluster._init_stage_ends = initial_stage_ends
        restore = lambda: setattr(ordered_cluster, "_init_stage_ends", original)

    try:
        started = time.perf_counter()
        result = run_experiment("S4SlideInsert", method_name, dataset_config, method_config)
        elapsed = time.perf_counter() - started
    finally:
        if restore is not None:
            restore()

    model = result["segmentation"].model
    if warm_start:
        initial_stage_ends = stage_ends_from_cutpoints(map_cutpoints, result["dataset"].demos)
    elif method_name == "fchmm":
        initial_stage_ends = model.initial_stage_ends_
    elif method_name == "arhsmm":
        from methods.backends.hmm import _resolve_stage_ends_init

        segmenter_config = method_config["segmenter"]
        initial_stage_ends = _resolve_stage_ends_init(
            result["dataset"].demos,
            num_stages=segmenter_config["n_stages"],
            min_duration=segmenter_config["min_duration"],
            tau_init=segmenter_config.get("tau_init"),
            tau_init_mode=segmenter_config["tau_init_mode"],
            env=result["dataset"].env,
            seed=segmenter_config["seed"],
            use_velocity=segmenter_config["use_velocity"],
            vel_weight=segmenter_config["vel_weight"],
            standardize=segmenter_config["standardize"],
            use_env_features=segmenter_config["use_env_features"],
            selected_raw_feature_ids=segmenter_config["selected_raw_feature_ids"],
        )
    else:
        initial_stage_ends = model.segmentation_history_[0]
    return summarize(method_name, "map_warm_start" if warm_start else "original", result, initial_stage_ends, elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("fchmm", "arhsmm", "cluster"))
    args = parser.parse_args()
    map_cutpoints = json.loads((OUTPUT_DIR / "map_cutpoints.json").read_text())["cutpoints"]
    reports = [
        run_case(args.method, False, map_cutpoints),
        run_case(args.method, True, map_cutpoints),
    ]
    output_path = OUTPUT_DIR / f"{args.method}_comparison.json"
    output_path.write_text(json.dumps(reports, indent=2, allow_nan=True), encoding="utf-8")
    for report in reports:
        print(
            report["method"],
            report["label"],
            "movement=", report["movement_from_initial"],
            "seg=", report["segmentation_metrics"],
            "constraint=", {
                key: report["constraint_metrics"][key]
                for key in ("MeanConstraintError", "MeanConstraintErrorRaw", "semantic_exact_cells")
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
