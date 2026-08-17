from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.s5.config import S5_SYNTHETIC_V20, cache_compatible_s5_loader_config
from envs.s5.dataset import _build_sphere_inspect_bundle
from envs.s5.regression import fingerprint_s5_analytic_bundle, verify_s5_v20_cache
from envs.s5.time_parameterization import FixedStepTimeParameterizer, gaussian_slowdown_weights


DEFAULT_CACHE = PROJECT_ROOT / "envs" / "demo_cache" / "S5SphereInspect" / "seed_127_0dd45ba56eadab53.npz"
DEFAULT_FIXTURE = PROJECT_ROOT / "envs" / "s5" / "fixtures" / "s5_v20_seed127_fingerprint.json"
DEFAULT_ANALYTIC_FIXTURE = PROJECT_ROOT / "envs" / "s5" / "fixtures" / "s5_v20_analytic_seed127_fingerprint.json"


def verify_time_parameterization_semantics() -> None:
    geometry = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    parameterizer = FixedStepTimeParameterizer(dt=0.1)
    result = parameterizer.parameterize_fixed_count(
        geometry,
        101,
        speed_intent=lambda count: gaussian_slowdown_weights(
            count,
            depth=0.25,
            center=0.5,
            width=0.08,
        ),
    )
    if not np.allclose(np.diff(result.timestamps), 0.1):
        raise AssertionError("S5 time parameterization must emit a fixed dt timestamp grid.")
    if not np.allclose(result.positions[[0, -1]], geometry[[0, -1]]):
        raise AssertionError("S5 time parameterization must preserve geometry endpoints.")
    if not np.allclose(result.positions[:, 1:], 0.0):
        raise AssertionError("S5 time parameterization must sample on the supplied geometry.")
    middle = len(result.reference_edge_speeds) // 2
    edge = max(1, len(result.reference_edge_speeds) // 10)
    if not result.reference_edge_speeds[middle] < result.reference_edge_speeds[edge]:
        raise AssertionError("A deliberate slowdown must reduce reference speed near its configured center.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen S5 v20 semantic dataset fingerprint.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--analytic-fixture", type=Path, default=DEFAULT_ANALYTIC_FIXTURE)
    parser.add_argument("--skip-analytic", action="store_true")
    args = parser.parse_args()
    verify_time_parameterization_semantics()
    print("S5 time-parameterization semantics OK")
    fingerprint = verify_s5_v20_cache(args.cache, args.fixture)
    print(
        f"S5 v20 regression fixture OK: {fingerprint['semantic_sha256']} "
        f"({fingerprint['count']} demos, {fingerprint['array_count']} arrays)"
    )
    if not args.skip_analytic:
        bundle = _build_sphere_inspect_bundle(
            task_name="S5SphereInspect",
            n_demos=10,
            seed=127,
            env_kwargs=cache_compatible_s5_loader_config(
                {
                    "cache_demos": False,
                    "rollout_backend": "analytic",
                    "observation_backend": "analytic_raw",
                }
            ),
            preset=S5_SYNTHETIC_V20,
        )
        actual_analytic = fingerprint_s5_analytic_bundle(bundle)
        expected_analytic = json.loads(args.analytic_fixture.read_text(encoding="utf-8"))
        if actual_analytic != expected_analytic:
            raise AssertionError(
                "S5 v20 analytic generator fingerprint mismatch: "
                f"expected {expected_analytic}, got {actual_analytic}"
            )
        print(
            f"S5 v20 analytic generator OK: {actual_analytic['semantic_sha256']} "
            f"({actual_analytic['count']} demos)"
        )


if __name__ == "__main__":
    main()
