from __future__ import annotations

from runners.run_benchmark import run_benchmark


def run_experiment_sine(
    seed: int = 0,
    n_demos: int = 12,
    max_iter: int = 30,
):
    return run_benchmark(
        methods=["segcons"],
        datasets=["3DSineCorridor"],
        seeds=[seed],
        dataset_overrides={"3DSineCorridor": {"n_demos": n_demos}},
        method_overrides={"segcons": {"max_iter": max_iter, "plot_every": max_iter}},
    )


def main():
    run_experiment_sine(
        seed=426,
        n_demos=12,
        max_iter=60,
    )


if __name__ == "__main__":
    main()
