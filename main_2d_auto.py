from __future__ import annotations

from runners.run_benchmark import run_benchmark


def run_experiment_2d(
    n_demos: int = 10,
    dataset_seed: int = 42,
    method_seed: int = 0,
    max_iter: int = 40,
    run_baseline: bool = True,
    init_mode: str = "uniform_taus",
):
    methods = ["segcons", "cghmm"] if run_baseline else ["segcons"]
    return run_benchmark(
        methods=methods,
        datasets=["2DObsAvoid"],
        method_seeds=[method_seed],
        dataset_seed=dataset_seed,
        dataset_overrides={"2DObsAvoid": {"n_demos": n_demos}},
        method_overrides={
            "segcons": {
                "max_iter": max_iter,
                "tau_init_mode": init_mode,
                "seed": method_seed,
                "plot_every": max_iter,
            },
            "cghmm": {
                "segmenter": {
                    "max_iter": max_iter,
                    "verbose": True,
                    "seed": method_seed,
                    "plot_every": max_iter,
                },
                "constraints": {"refine_steps": 5, "auto_feature_select": False},
            },
        },
    )


if __name__ == "__main__":
    run_experiment_2d(
        n_demos=10,
        dataset_seed=2224519,
        method_seed=0,
        max_iter=45,
        run_baseline=True,
    )
