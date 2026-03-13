from __future__ import annotations

from runners.run_benchmark import run_benchmark


def run_experiment_2d(
    n_demos: int = 10,
    seed: int = 42,
    max_iter: int = 40,
    run_baseline: bool = True,
    init_mode: str = "heuristic",
):
    methods = ["segcons", "gmmhmm"] if run_baseline else ["segcons"]
    return run_benchmark(
        methods=methods,
        datasets=["2DObsAvoid"],
        seeds=[seed],
        dataset_overrides={"2DObsAvoid": {"n_demos": n_demos}},
        method_overrides={
            "segcons": {
                "max_iter": max_iter,
                "g1_init": init_mode,
                "feature_ids": [0, 1],
                "feature_types": ["margin_exp_lower", "gauss", "gauss", "gauss"],
                "plot_every": max_iter,
            },
            "gmmhmm": {
                "segmenter": {"max_iter": max_iter, "verbose": True, "seed": seed, "plot_every": max_iter},
                "constraints": {"refine_steps": 5},
            },
        },
    )


if __name__ == "__main__":
    run_experiment_2d(
        n_demos=10,
        seed=2224519,
        max_iter=45,
        run_baseline=True,
    )
