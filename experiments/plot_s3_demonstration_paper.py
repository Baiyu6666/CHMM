from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.registry import load_env
from visualization.swcl_4panel import plot_swcl_true_cutpoint_trajectory_paper


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot an S3 demonstration trajectory with true cutpoints for single-column paper figures."
    )
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/envs/S3ObsAvoid.json")
    parser.add_argument("--demo-idx", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/paper_figures/S3ObsAvoid")
    parser.add_argument("--formats", type=str, default="png,pdf")
    args = parser.parse_args()

    cfg = _load_json(args.env_config)
    env_name = str(cfg.pop("name"))
    cfg.pop("method_overrides", None)
    n_demos = max(int(cfg.get("n_demos", 1)), int(args.demo_idx) + 1)
    cfg["n_demos"] = n_demos

    bundle = load_env(env_name, **cfg)
    learner_like = SimpleNamespace(
        demos=bundle.demos,
        true_cutpoints=bundle.true_cutpoints,
        true_taus=None,
        env=bundle.env,
        plot_dir=args.output_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"s3_demonstration_true_cutpoints_demo_{int(args.demo_idx):02d}"
    for fmt in [x.strip().lower() for x in args.formats.split(",") if x.strip()]:
        out_path = args.output_dir / f"{stem}.{fmt}"
        plot_swcl_true_cutpoint_trajectory_paper(learner_like, demo_idx=args.demo_idx, save_path=out_path)
        print(out_path)


if __name__ == "__main__":
    main()
