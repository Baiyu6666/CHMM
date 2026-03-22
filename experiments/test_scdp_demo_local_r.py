from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.analyze_scdp_feature_stability import _plot_demo_feature_costs, plt
from envs.registry import load_env
from experiments.config_loader import deep_merge, load_experiment_config
from methods.wrappers.joint_scdp import JointSCDPMethod
from visualization.plot4panel import plot_demos_goals_snapshot


def _feature_names(learner) -> list[str]:
    names: list[str] = []
    for local_idx in range(learner.num_features):
        selected_col = int(learner.selected_feature_columns[local_idx])
        name = None
        for i, spec in enumerate(learner.raw_feature_specs):
            if int(spec.get("column_idx", i)) == selected_col:
                name = str(spec.get("name", f"f{local_idx}"))
                break
        names.append(name or f"f{local_idx}")
    return names


def _print_feature_stage_table(title: str, values, feature_names: list[str], stage_count: int) -> None:
    arr = np.asarray(values)
    stage_labels = [f"stage{i + 1}" for i in range(stage_count)]
    row_label = "feature"
    row_width = max(len(row_label), max(len(name) for name in feature_names))
    formatted = [[str(int(arr[i, j])) for j in range(arr.shape[1])] for i in range(arr.shape[0])]

    col_widths = []
    for j, label in enumerate(stage_labels):
        width = len(label)
        for i in range(len(feature_names)):
            width = max(width, len(formatted[i][j]))
        col_widths.append(width)

    print(title)
    header = row_label.ljust(row_width) + " | " + " | ".join(
        label.rjust(col_widths[j]) for j, label in enumerate(stage_labels)
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for i, feat_name in enumerate(feature_names):
        row = feat_name.ljust(row_width) + " | " + " | ".join(
            formatted[i][j].rjust(col_widths[j]) for j in range(len(stage_labels))
        )
        print(row)
    print()


def _build_eval_state(learner, taus: list[int]):
    stage_ends = []
    stage_params_per_demo = []
    demo_r_matrices = []
    for demo_idx, (X, tau) in enumerate(zip(learner.demos, taus)):
        tau = max(1, min(int(tau), len(X) - 2))
        demo_stage_ends = [tau, len(X) - 1]
        demo_stage_params = []
        demo_r = []
        for stage_idx, (s, e) in enumerate(((0, tau), (tau + 1, len(X) - 1))):
            stage_params, _, _ = learner._fit_segment_stage(demo_idx, stage_idx, s, e)
            demo_stage_params.append(stage_params)
            demo_r.append(np.asarray(stage_params.active_mask, dtype=int))
        stage_ends.append(demo_stage_ends)
        stage_params_per_demo.append(demo_stage_params)
        demo_r_matrices.append(np.stack(demo_r, axis=0))
    return stage_ends, stage_params_per_demo, demo_r_matrices


def _resolve_dataset_eval_taus(dataset) -> list[int]:
    true_cutpoints = getattr(dataset, "true_cutpoints", None)
    if true_cutpoints is not None:
        taus = []
        for cuts in true_cutpoints:
            if cuts is None:
                raise ValueError("ground_truth cutpoints requested, but a demo has no true_cutpoints.")
            arr = np.asarray(cuts, dtype=int).reshape(-1)
            if arr.size != 1:
                raise ValueError("This debug script only supports a single ground-truth cutpoint per demo.")
            taus.append(int(arr[0]))
        return taus
    if getattr(dataset, "true_taus", None):
        return [int(tau) for tau in dataset.true_taus]
    raise ValueError("ground_truth cutpoints requested, but dataset.true_cutpoints/true_taus is unavailable.")


def main():
    parser = argparse.ArgumentParser(description="Test SCDP with per-demo local R chosen by min(fitted, baseline) avg NLL.")
    parser.add_argument("--env-config", default="configs/envs/2DObsAvoid.json", type=str)
    parser.add_argument("--method-config", default="configs/methods/scdp.json", type=str)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--equality-dispersion-ratio-threshold", type=float, default=0.1)
    parser.add_argument("--plots-dir", type=str, default=None, help="Directory to save per-demo debug feature-cost plots.")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-demo debug plots.")
    parser.add_argument(
        "--cutpoints-source",
        type=str,
        default="predicted",
        choices=["predicted", "ground_truth"],
        help="Which cutpoints to use when recomputing per-demo local costs and R masks.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config(args.env_config, args.method_config)
    dataset_cfg = dict(cfg["dataset"])
    method_cfg = dict(cfg["method"])
    dataset_name = dataset_cfg.pop("name")
    method_name = method_cfg.pop("name")
    if method_name != "scdp":
        raise ValueError(f"This script only supports scdp, got '{method_name}'.")

    dataset_method_overrides = dict(dataset_cfg.pop("method_overrides", {}))
    method_cfg = deep_merge(method_cfg, dataset_method_overrides.get(method_name, {}))
    method_cfg["max_iter"] = int(args.max_iter)
    method_cfg["plot_every"] = None
    method_cfg["verbose"] = False
    method_cfg["feature_activation_mode"] = "score"
    method_cfg["equality_dispersion_ratio_threshold"] = float(args.equality_dispersion_ratio_threshold)

    dataset = load_env(dataset_name, **dataset_cfg)
    result = JointSCDPMethod(kwargs=method_cfg).fit(dataset)
    learner = result["model"]

    if args.cutpoints_source == "ground_truth":
        eval_taus = _resolve_dataset_eval_taus(dataset)
    else:
        eval_taus = [int(tau) for tau in result["taus_hat"]]

    eval_stage_ends, eval_stage_params_per_demo, eval_demo_r_matrices = _build_eval_state(learner, eval_taus)

    out = {
        "dataset": dataset_name,
        "feature_names": _feature_names(learner),
        "taus_hat": result["taus_hat"],
        "taus_eval": eval_taus,
        "cutpoints_source": args.cutpoints_source,
        "demo_r_matrices": [r.tolist() for r in eval_demo_r_matrices],
    }

    plot_outputs = []
    overview_plot = None
    if not args.no_plots:
        if plt is None:
            out["plots_skipped"] = "matplotlib is not installed"
        else:
            if args.plots_dir is None:
                plot_dir = PROJECT_ROOT / "outputs" / "analysis" / "scdp_local_r_debug" / dataset_name
            else:
                plot_dir = Path(args.plots_dir)
                if not plot_dir.is_absolute():
                    plot_dir = PROJECT_ROOT / plot_dir
            plot_dir.mkdir(parents=True, exist_ok=True)
            gammas = []
            for X, tau in zip(learner.demos, eval_taus):
                T = len(X)
                gamma = [[1.0, 0.0] if t <= tau else [0.0, 1.0] for t in range(T)]
                gammas.append(gamma)
            original_stage_ends = learner.stage_ends_
            original_stage_params = learner.current_stage_params_per_demo
            try:
                learner.stage_ends_ = [list(x) for x in eval_stage_ends]
                learner.current_stage_params_per_demo = [list(x) for x in eval_stage_params_per_demo]
                fig = plt.figure(figsize=(6.2, 5.2))
                ax = fig.add_subplot(1, 1, 1, projection="3d") if learner.demos[0].shape[1] == 3 else fig.add_subplot(1, 1, 1)
                plot_demos_goals_snapshot(
                    ax=ax,
                    learner=learner,
                    taus=eval_taus,
                    gammas=gammas,
                    title=f"Final demos & goals ({args.cutpoints_source})",
                    show_legend=True,
                )
                fig.tight_layout()
                overview_path = plot_dir / "overview.png"
                fig.savefig(overview_path, dpi=180, bbox_inches="tight")
                plt.close(fig)
            finally:
                learner.stage_ends_ = original_stage_ends
                learner.current_stage_params_per_demo = original_stage_params
            overview_plot = str(overview_path)
            for demo_idx in range(len(learner.demos)):
                output_path = plot_dir / f"demo_{demo_idx:02d}.png"
                plot_outputs.append(
                    str(
                        _plot_demo_feature_costs(
                            dataset_name,
                            learner,
                            demo_idx,
                            output_path,
                            equality_dispersion_ratio_threshold=args.equality_dispersion_ratio_threshold,
                            stage_ends=eval_stage_ends[demo_idx],
                            stage_params=eval_stage_params_per_demo[demo_idx],
                        )
                    )
                )
            out["plot_outputs"] = plot_outputs
            out["overview_plot"] = overview_plot

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print("dataset", out["dataset"])
    print("feature_names", out["feature_names"])
    print("taus_hat", out["taus_hat"])
    print("cutpoints_source", out["cutpoints_source"])
    print("taus_eval", out["taus_eval"])
    if overview_plot:
        print("overview_plot", overview_plot)
    if plot_outputs:
        print("plot_dir", str(Path(plot_outputs[0]).parent))
    elif out.get("plots_skipped"):
        print("plots", out["plots_skipped"])
    print()
    for demo_idx, r in enumerate(out["demo_r_matrices"]):
        print(f"demo {demo_idx}")
        print(f"  tau = {out['taus_eval'][demo_idx]}")
        _print_feature_stage_table(
            "R",
            np.asarray(r, dtype=int).T,
            out["feature_names"],
            stage_count=len(r),
        )


if __name__ == "__main__":
    main()
