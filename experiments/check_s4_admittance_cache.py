from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.s4.regression import verify_s4_cache


DEFAULT_CACHE = (
    PROJECT_ROOT
    / "envs"
    / "demo_cache"
    / "S4SlideInsert"
    / "seed_1342_fc7a384ee33073e7.npz"
)
DEFAULT_FIXTURE = PROJECT_ROOT / "envs" / "s4" / "fixtures" / "s4_admittance_v2_seed1342_fingerprint.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen S4 admittance demo cache.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args()
    fingerprint = verify_s4_cache(str(args.cache), str(args.fixture))
    print(
        f"S4 admittance cache OK: {fingerprint['semantic_sha256']} "
        f"({fingerprint['count']} demos, {fingerprint['array_count']} arrays)"
    )


if __name__ == "__main__":
    main()
