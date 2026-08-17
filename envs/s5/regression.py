from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FINGERPRINT_FORMAT_VERSION = 1
_METADATA_KEYS = ("dataset_manifest_json", "demo_metadata_json", "metadata_json", "scene_specs_json")


def _canonical_json_scalar(value: np.ndarray) -> bytes:
    parsed = json.loads(str(np.asarray(value).item()))
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def fingerprint_s5_v20_cache(cache_path: str | Path) -> dict[str, Any]:
    path = Path(cache_path)
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["dataset_manifest_json"].item()))
        count = int(np.asarray(data["count"]).item())
        array_digests: dict[str, str] = {}
        for key in sorted(data.files):
            if key == "count" or key in _METADATA_KEYS:
                continue
            array_digests[key] = _array_digest(data[key])

        metadata_digests = {
            key: hashlib.sha256(_canonical_json_scalar(data[key])).hexdigest()
            for key in _METADATA_KEYS
            if key in data
        }

    combined = hashlib.sha256()
    for key, value in sorted({**array_digests, **metadata_digests}.items()):
        combined.update(key.encode("utf-8"))
        combined.update(value.encode("ascii"))
    return {
        "format_version": FINGERPRINT_FORMAT_VERSION,
        "cache_version": int(manifest.get("cache_version", -1)),
        "count": count,
        "array_count": len(array_digests),
        "semantic_sha256": combined.hexdigest(),
        "array_digests": array_digests,
        "metadata_digests": metadata_digests,
    }


def verify_s5_v20_cache(cache_path: str | Path, fixture_path: str | Path) -> dict[str, Any]:
    actual = fingerprint_s5_v20_cache(cache_path)
    expected = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise AssertionError(
            "S5 v20 cache fingerprint mismatch: "
            f"{mismatches}"
        )
    return actual


def fingerprint_s5_analytic_bundle(bundle) -> dict[str, Any]:
    digest = hashlib.sha256()
    lengths = []
    tool_axis_traces = list(bundle.meta.get("tool_axis_traces", []))
    for trajectory, cutpoints, tool_axis in zip(bundle.demos, bundle.true_cutpoints, tool_axis_traces):
        lengths.append(int(len(trajectory)))
        for values in (trajectory, cutpoints, tool_axis):
            array = np.ascontiguousarray(np.asarray(values))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
    return {
        "cache_version": 20,
        "count": len(bundle.demos),
        "lengths": lengths,
        "seed": int(bundle.meta.get("seed", -1)),
        "semantic_sha256": digest.hexdigest(),
    }
