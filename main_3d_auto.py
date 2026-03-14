from __future__ import annotations

from runners.run_benchmark import run_benchmark


def run_experiment(
    n_demos: int = 10,
    dataset_seed: int = 42,
    method_seed: int = 0,
    use_3d: bool = True,
    max_iter: int = 40,
    run_baseline: bool = False,
    render: bool = False,
):
    dataset_name = "3DObsAvoid" if use_3d else "2DObsAvoid"
    methods = ["segcons", "cghmm"] if run_baseline else ["segcons"]
    outputs = run_benchmark(
        methods=methods,
        datasets=[dataset_name],
        method_seeds=[method_seed],
        dataset_seed=dataset_seed,
        dataset_overrides={dataset_name: {"n_demos": n_demos}},
        method_overrides={
            "segcons": {"max_iter": max_iter, "plot_every": max_iter, "seed": method_seed},
            "cghmm": {
                "segmenter": {
                    "max_iter": max_iter,
                    "verbose": True,
                    "seed": method_seed,
                    "plot_every": max_iter,
                    "tau_init_mode": "uniform_taus",
                },
                "constraints": {"refine_steps": 5, "auto_feature_select": False},
            },
        },
    )

    if render:
        print("Rendering is no longer handled inside main_3d_auto.py.")
        print("Use visualization/pybullet_renderer.py and planner/trajectory_refinement.py explicitly.")

    return outputs


if __name__ == "__main__":
    run_experiment(
        n_demos=10,
        dataset_seed=42,
        method_seed=0,
        use_3d=True,
        max_iter=40,
        run_baseline=False,
        render=False,
    )
