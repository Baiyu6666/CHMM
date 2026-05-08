from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import load_env


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_s5_orbit_from_run(
    run_dir: Path,
    demo_idx: int = 0,
    yaws: tuple[float, ...] = (42.0, 205.0),
    output_path: Path | None = None,
) -> Path:
    run_dir = run_dir.resolve()
    metadata = _read_json(run_dir / "metadata.json")
    cfg = _read_json(run_dir / "config_snapshot.json")
    segmentation = _read_json(run_dir / "segmentation.json")

    dataset_name = str(metadata["dataset_name"])
    if dataset_name != "S5SphereInspect":
        raise ValueError(f"This renderer currently only supports S5SphereInspect, got '{dataset_name}'.")

    dataset_kwargs = dict(cfg.get("dataset_kwargs", {}))
    bundle = load_env(dataset_name, **dataset_kwargs)

    if len(yaws) < 2:
        raise ValueError("Expected at least two camera yaws: one for main view and one for inset view.")

    demo = np.asarray(bundle.demos[int(demo_idx)], dtype=float)
    true_cutpoints = [int(v) for v in segmentation["true_cutpoints"][int(demo_idx)]]
    learned_cutpoints = [int(v) for v in segmentation["predicted_cutpoints"][int(demo_idx)]]
    env = bundle.env
    scene_specs = list(bundle.meta.get("scene_specs", [])) if isinstance(getattr(bundle, "meta", None), dict) else []
    scene = dict(scene_specs[int(demo_idx)]) if int(demo_idx) < len(scene_specs) else env.sample_scene()
    tool_axis = env._lookup_cached_tool_axis_trace(demo)
    if tool_axis is None:
        tool_axis = env._estimate_tool_axis_from_geometry(demo)
    tool_axis = np.asarray(tool_axis, dtype=float)

    if output_path is None:
        output_path = run_dir / f"paper_pybullet_orbit_demo_{int(demo_idx):02d}.png"
    output_path = output_path.resolve()
    return env.render_episode(
        scene,
        demo,
        output_path,
        backend="pybullet",
        camera="paper_orbit",
        cutpoints=true_cutpoints,
        overlay_cutpoints=learned_cutpoints,
        tool_axis=tool_axis,
        main_yaw=float(yaws[0]),
        inset_yaw=float(yaws[1]),
        title=f"S5SphereInspect demo {int(demo_idx)}",
    )


def _parse_yaws(text: str) -> tuple[float, ...]:
    parts = [item.strip() for item in str(text).split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of camera yaws.")
    return tuple(float(v) for v in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an S5 PyBullet orbit figure from a saved SWCL run.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing metadata/config_snapshot/segmentation.")
    parser.add_argument("--demo-idx", type=int, default=0, help="Demo index to render.")
    parser.add_argument(
        "--camera-yaws",
        type=_parse_yaws,
        default=(42.0, 205.0),
        help="Comma-separated camera yaws for main and inset views.",
    )
    parser.add_argument("--output", default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    output_path = None if args.output is None else Path(args.output)
    saved = render_s5_orbit_from_run(
        run_dir=Path(args.run_dir),
        demo_idx=int(args.demo_idx),
        yaws=tuple(args.camera_yaws),
        output_path=output_path,
    )
    print(saved)


if __name__ == "__main__":
    main()
