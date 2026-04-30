from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DEFAULT_ENVS = ["S3ObsAvoid", "S4SlideInsert", "S5SphereInspect"]


def _stage_tag(env_name: str) -> str:
    match = re.match(r"^S(\d+)", str(env_name))
    if not match:
        raise ValueError(f"Cannot infer stage tag from env name: {env_name}")
    return f"S{match.group(1)}"


def _parse_demo_map(raw: str | None, env_names: list[str]) -> dict[str, int]:
    default_map = {env_name: 0 for env_name in env_names}
    if not raw:
        return default_map
    out = dict(default_map)
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid demo map item: {item!r}. Expected ENV:DEMO.")
        env_name, demo_idx = item.split(":", 1)
        env_name = env_name.strip()
        if env_name not in out:
            raise KeyError(f"Unknown env in demo map: {env_name}")
        out[env_name] = int(demo_idx)
    return out


def _latest_iter_file(run_dir: Path, pattern: str) -> Path:
    candidates = sorted(run_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern!r} in {run_dir}")

    def _iter_key(path: Path) -> int:
        match = re.search(r"_iter_(\d+)\.png$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=_iter_key)


def collect_swcl_paper_figures(
    outputs_root: Path,
    env_names: list[str],
    method_seed: int,
    demo_map: dict[str, int],
) -> list[tuple[Path, Path]]:
    out_dir = outputs_root / "swcl" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: list[tuple[Path, Path]] = []
    for env_name in env_names:
        stage_tag = _stage_tag(env_name)
        run_dir = outputs_root / "swcl" / env_name / f"method_seed_{method_seed:03d}"
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        activation_src = _latest_iter_file(run_dir, "paper_activation_rate_iter_*.png")
        activation_dst = out_dir / f"{stage_tag}_activation_rate.png"
        shutil.copy2(activation_src, activation_dst)
        copied.append((activation_src, activation_dst))

        demo_idx = int(demo_map[env_name])
        margin_src = _latest_iter_file(
            run_dir,
            f"paper_constraint_margin_demo_{demo_idx:02d}_iter_*.png",
        )
        margin_dst = out_dir / f"{stage_tag}_constraint_gain.png"
        shutil.copy2(margin_src, margin_dst)
        copied.append((margin_src, margin_dst))

        trace_src = _latest_iter_file(
            run_dir,
            f"paper_key_feature_traces_demo_{demo_idx:02d}_iter_*.png",
        )
        trace_dst = out_dir / f"{stage_tag}_feature_trace.png"
        shutil.copy2(trace_src, trace_dst)
        copied.append((trace_src, trace_dst))

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SWCL paper figures into outputs/swcl/paper_figures.")
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root outputs directory. Defaults to outputs.",
    )
    parser.add_argument(
        "--envs",
        default=",".join(DEFAULT_ENVS),
        help="Comma-separated env names. Defaults to S3ObsAvoid,S4SlideInsert,S5SphereInspect.",
    )
    parser.add_argument(
        "--method-seed",
        type=int,
        default=0,
        help="Method seed directory to read from. Defaults to 0.",
    )
    parser.add_argument(
        "--demo-map",
        default="",
        help="Comma-separated ENV:DEMO overrides for constraint margin, e.g. S3ObsAvoid:1,S4SlideInsert:3.",
    )
    args = parser.parse_args()

    env_names = [item.strip() for item in str(args.envs).split(",") if item.strip()]
    demo_map = _parse_demo_map(args.demo_map, env_names)
    copied = collect_swcl_paper_figures(
        outputs_root=Path(args.outputs_root),
        env_names=env_names,
        method_seed=int(args.method_seed),
        demo_map=demo_map,
    )
    for src, dst in copied:
        print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
