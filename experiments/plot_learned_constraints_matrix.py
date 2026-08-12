from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.learned_constraints_matrix import (
    plot_learned_constraints_matrix_paper,
    plot_true_vs_learned_constraints_matrix_paper,
)


def _infer_dataset_name(path: Path) -> str | None:
    text = str(path)
    for name in ("S3ObsAvoid", "S4SlideInsert", "S5SphereInspect"):
        if name in text:
            return name
    return None


def plot_learned_constraints(constraints_json: Path, output_path: Path) -> Path:
    payload = json.loads(constraints_json.read_text(encoding="utf-8"))
    out = plot_learned_constraints_matrix_paper(
        payload,
        save_path=output_path,
        dataset_name=_infer_dataset_name(constraints_json),
    )
    if out is None:
        raise RuntimeError("matplotlib is required to plot learned constraints.")
    return out


def plot_true_vs_learned_constraints(constraints_json: Path, output_path: Path) -> Path:
    payload = json.loads(constraints_json.read_text(encoding="utf-8"))
    out = plot_true_vs_learned_constraints_matrix_paper(
        payload,
        save_path=output_path,
        dataset_name=_infer_dataset_name(constraints_json),
    )
    if out is None:
        raise RuntimeError("matplotlib is required to plot learned constraints.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraints-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare", action="store_true", help="Plot true constraints above learned constraints.")
    args = parser.parse_args()
    if args.compare:
        out = plot_true_vs_learned_constraints(Path(args.constraints_json), Path(args.output))
    else:
        out = plot_learned_constraints(Path(args.constraints_json), Path(args.output))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
