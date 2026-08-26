import json
import math
import os


ACTIVE_MODES = frozenset({"target_value", "lower_bound", "upper_bound"})


def _resolved_artifact_path(source, learned_root):
    root = os.path.realpath(str(learned_root))
    resolved = os.path.realpath(str(source))
    try:
        inside_root = os.path.commonpath((root, resolved)) == root
    except ValueError:
        inside_root = False
    if not inside_root or resolved == root:
        raise ValueError(
            "Learned constraint file is outside the configured artifact root"
        )
    return resolved


def configure_task_constraints(config, task_id, source, learned_root):
    """Apply one dense learned-constraint artifact to a loaded task config."""
    true_terms = [dict(value) for value in config["constraint_terms"]]
    source = str(source).strip()
    config["true_constraint_terms"] = true_terms
    config["planning_constraint_source"] = source or "true"
    if source in ("", "true"):
        return

    artifact_path = _resolved_artifact_path(source, learned_root)
    with open(artifact_path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("artifact_type") != "learned_stage_constraints":
        raise ValueError("Selected file is not a learned constraint artifact")
    if int(artifact.get("schema_version", -1)) != 2:
        raise ValueError("Unsupported learned constraint schema version")
    if str(artifact.get("task_id")) != task_id:
        raise ValueError(
            "Learned constraints are for {}, not {}".format(
                artifact.get("task_id"), task_id
            )
        )

    expected_task_frame = dict(config.get("task_frame", {}))
    artifact_task_frame = artifact.get("task_frame")
    if expected_task_frame:
        if not isinstance(artifact_task_frame, dict):
            raise ValueError("Learned constraints have no task-frame definition")
        if str(artifact_task_frame.get("frame_id")) != str(
            expected_task_frame.get("frame_id")
        ):
            raise ValueError("Learned constraint task frame does not match the planner")
        if str(artifact_task_frame.get("snapshot_policy")) != str(
            expected_task_frame.get("snapshot_policy")
        ):
            raise ValueError("Learned constraint frame snapshot policy does not match the planner")

    n_stages = len(config["stage_names"])
    if int(artifact.get("num_stages", -1)) != n_stages:
        raise ValueError("Learned constraint stage count does not match the task")

    schema = artifact.get("feature_schema")
    if not isinstance(schema, list) or not schema:
        raise ValueError("Learned constraint artifact has no candidate features")
    candidate_names = [str(value["name"]) for value in schema]
    if any(not value for value in candidate_names):
        raise ValueError("Learned constraint artifact has an empty feature name")
    candidate_features = set(candidate_names)
    if len(candidate_features) != len(candidate_names):
        raise ValueError("Learned constraint feature schema contains duplicates")

    expected_pairs = {
        (stage, feature_name)
        for stage in range(n_stages)
        for feature_name in candidate_features
    }
    pairs = artifact.get("feature_stage_modes")
    if not isinstance(pairs, list):
        raise ValueError("Learned constraint artifact has no feature-stage matrix")
    seen_pairs = set()
    planning_terms = []
    supported_features = set(config["feature_units"])
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("Learned constraint feature-stage entries must be objects")
        stage = int(pair["stage"])
        feature_name = str(pair["feature_name"])
        key = (stage, feature_name)
        if key in seen_pairs or key not in expected_pairs:
            raise ValueError("Learned constraint artifact has duplicate or invalid pairs")
        seen_pairs.add(key)

        mode = str(pair["mode"])
        if mode == "inactive":
            continue
        if mode not in ACTIVE_MODES:
            raise ValueError("Unsupported learned constraint mode {}".format(mode))
        if feature_name not in supported_features:
            raise ValueError(
                "Active learned feature {} is not supported by this planner".format(
                    feature_name
                )
            )
        value = float(pair["value"])
        scale = float(pair.get("scale", 1.0))
        weight = float(pair.get("weight", 1.0))
        if not math.isfinite(value):
            raise ValueError("Learned constraint value must be finite")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Learned constraint scale must be positive and finite")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("Learned constraint weight must be nonnegative and finite")
        planning_terms.append(
            {
                "feature_name": feature_name,
                "stage": stage,
                "semantics": mode,
                "value": value,
                "scale": scale,
                "weight": weight,
            }
        )
    if seen_pairs != expected_pairs:
        raise ValueError("Learned constraint artifact is missing feature-stage pairs")
    config["constraint_terms"] = planning_terms
