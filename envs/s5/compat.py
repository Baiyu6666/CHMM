from __future__ import annotations


def sample_v20_reserved_length_scales(rng) -> tuple[float, float]:
    return float(rng.uniform(1.0, 1.0)), float(rng.uniform(1.0, 1.0))
