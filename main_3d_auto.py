from __future__ import annotations

from runners.run_benchmark import run_benchmark


def run_experiment(
    n_demos: int = 10,
    seed: int = 42,
    use_3d: bool = True,
    max_iter: int = 40,
    run_baseline: bool = False,
    render: bool = False,
):
    dataset_name = "3DObsAvoid" if use_3d else "2DObsAvoid"
    methods = ["segcons", "gmmhmm"] if run_baseline else ["segcons"]
    outputs = run_benchmark(
        methods=methods,
        datasets=[dataset_name],
        seeds=[seed],
        dataset_overrides={dataset_name: {"n_demos": n_demos}},
        method_overrides={
            "segcons": {"max_iter": max_iter, "plot_every": max_iter},
            "gmmhmm": {
                "segmenter": {"max_iter": max_iter, "verbose": True, "seed": seed, "plot_every": max_iter},
                "constraints": {"refine_steps": 5},
            },
        },
    )

    if render:
        print("Rendering is no longer handled inside main_3d_auto.py.")
        print("Use visualization/pybullet_renderer.py and planner/* on the learned outputs explicitly.")

    return outputs


if __name__ == "__main__":
    run_experiment(
        n_demos=10,
        seed=42,
        use_3d=True,
        max_iter=40,
        run_baseline=False,
        render=False,
    )
