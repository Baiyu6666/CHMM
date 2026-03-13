from __future__ import annotations

import json
from pathlib import Path

from .unified_experiment import run_experiment


DATASET_NAME = "3DObsAvoid"
METHODS = ["segcons", "gmmhmm", "hdphmm", "arhmm", "changepoint"]
N_DEMOS = 10
SEED = 123
MAX_ITER = 30
SAVE_PATH = Path("outputs/experiments/exp_compare_3d_results.json")


def _method_kwargs(method_name: str):
    if method_name == "segcons":
        return {"max_iter": MAX_ITER, "verbose": True, "plot_every": None}
    return {
        "segmenter": {"max_iter": MAX_ITER, "verbose": True, "seed": SEED},
        "constraints": {"refine_steps": 5},
    }


def run_compare():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for method_name in METHODS:
        out = run_experiment(
            dataset_name=DATASET_NAME,
            method_name=method_name,
            dataset_kwargs={"n_demos": N_DEMOS, "seed": SEED},
            method_kwargs=_method_kwargs(method_name),
        )
        metrics = out["joint_result"]["metrics"] if method_name == "segcons" else out["constraints"]["metrics"]
        results[method_name] = metrics
        print(f"[{method_name}] {metrics}")

    SAVE_PATH.write_text(json.dumps(results, indent=2))
    print(f"[Saved] {SAVE_PATH}")
    return results


if __name__ == "__main__":
    run_compare()
