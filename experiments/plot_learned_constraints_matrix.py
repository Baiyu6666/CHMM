from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.learned_constraints_matrix import plot_learned_constraints_matrix_paper


def plot_learned_constraints(constraints_json: Path, output_path: Path) -> Path:
    payload = json.loads(constraints_json.read_text(encoding="utf-8"))
    out = plot_learned_constraints_matrix_paper(payload, save_path=output_path)
    if out is None:
        raise RuntimeError("matplotlib is required to plot learned constraints.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraints-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = plot_learned_constraints(Path(args.constraints_json), Path(args.output))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
