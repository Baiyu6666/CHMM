from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.registry import load_env


CONFIG_PATH = ROOT / "configs" / "envs" / "S4SlideInsert.json"
FIXTURE_PATH = ROOT / "envs" / "s4" / "fixtures" / "s4_seed1342_fingerprint.json"


def _update_array(digest: hashlib._Hash, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    dataset_name = config.pop("name")
    config.pop("method_overrides", None)
    bundle = load_env(dataset_name, **config)

    digest = hashlib.sha256()
    array_count = 0
    for index, (demo, cutpoints, labels) in enumerate(
        zip(bundle.demos, bundle.true_cutpoints, bundle.true_labels)
    ):
        arrays = (
            (f"demo_{index}", np.asarray(demo, dtype=np.float64)),
            (f"features_{index}", np.asarray(bundle.env.compute_all_features_matrix(demo), dtype=np.float64)),
            (f"cutpoints_{index}", np.asarray(cutpoints, dtype=np.int64)),
            (f"labels_{index}", np.asarray(labels, dtype=np.int64)),
        )
        for name, values in arrays:
            _update_array(digest, name, values)
            array_count += 1

    actual = {
        "task": dataset_name,
        "seed": int(config["seed"]),
        "n_demos": len(bundle.demos),
        "array_count": array_count,
        "demo_lengths": [len(demo) for demo in bundle.demos],
        "sha256": digest.hexdigest(),
    }
    if actual != fixture:
        raise AssertionError(
            "S4 regression fingerprint changed:\n"
            f"expected={json.dumps(fixture, indent=2)}\n"
            f"actual={json.dumps(actual, indent=2)}"
        )
    print(json.dumps(actual, indent=2))


if __name__ == "__main__":
    main()
