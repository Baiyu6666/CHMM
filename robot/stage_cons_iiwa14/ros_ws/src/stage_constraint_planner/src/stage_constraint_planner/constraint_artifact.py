import json
import math
import os


ACTIVE_MODES = frozenset({"target_value", "lower_bound", "upper_bound"})


def stage_zero_approach_clearance(config):
    """Use the selected task Stage 1 clearance for the pre-task Stage 0 approach."""
    matches = [
        term
        for term in config["constraint_terms"]
        if int(term["stage"]) == 0
        and str(term["feature_name"]) == "obs_dist"
        and str(term["semantics"]) == "lower_bound"
    ]
    if len(matches) != 1:
        raise ValueError(
            "Selected planning profile needs exactly one Stage 1 "
            "obstacle-clearance lower bound for the Stage 0 approach"
        )
    value = float(matches[0]["value"])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Stage 1 obstacle clearance must be finite and nonnegative")
    return value


def _task_fixed_term_parameters(true_terms, stage, feature_name):
    exact = [
        term
        for term in true_terms
        if int(term["stage"]) == int(stage)
        and str(term["feature_name"]) == str(feature_name)
    ]
    candidates = exact or [
        term
        for term in true_terms
        if str(term["feature_name"]) == str(feature_name)
    ]
    parameters = {
        (float(term["scale"]), float(term.get("weight", 1.0)))
        for term in candidates
    }
    if len(parameters) != 1:
        raise ValueError(
            "Task-fixed scale/weight is missing or ambiguous for stage {} feature {}".format(
                int(stage), str(feature_name)
            )
        )
    scale, weight = next(iter(parameters))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Task-fixed constraint scale must be positive and finite")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("Task-fixed constraint weight must be nonnegative and finite")
    return scale, weight


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


def configure_planning_profile(config, task_id, source, learned_root):
    """Apply learned modes, values, and endpoints to task-fixed planner settings."""
    true_terms = [dict(value) for value in config["constraint_terms"]]
    source = str(source).strip()
    config["true_constraint_terms"] = true_terms
    config["planning_constraint_source"] = source or "true"
    config["planning_profile_source"] = source or "true"
    if source in ("", "true"):
        return

    artifact_path = _resolved_artifact_path(source, learned_root)
    with open(artifact_path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("artifact_type") != "learned_stage_constraints":
        raise ValueError("Selected file is not a learned constraint artifact")
    if int(artifact.get("schema_version", -1)) != 5:
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

    expected_feature_definition = dict(config.get("feature_definition", {}))
    if dict(artifact.get("feature_definition", {})) != expected_feature_definition:
        raise ValueError("Learned feature definition does not match the planner")

    n_stages = len(config["stage_names"])
    if str(artifact.get("endpoint_coordinate_frame", "")) != str(
        config.get("endpoint_coordinate_frame", "")
    ):
        raise ValueError("Learned endpoint coordinate frame does not match the planner")
    endpoint_poses = artifact.get("stage_endpoint_poses_bar")
    if not isinstance(endpoint_poses, list) or len(endpoint_poses) != n_stages - 1:
        raise ValueError("Learned artifact has no matching aggregated endpoint poses")
    normalized_endpoint_poses = []
    for pose in endpoint_poses:
        if not isinstance(pose, list) or len(pose) != 7:
            raise ValueError("Learned endpoint poses must contain xyz+xyzw")
        values = [float(value) for value in pose]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Learned endpoint poses must be finite")
        quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
        if quaternion_norm <= 1e-9:
            raise ValueError("Learned endpoint quaternion must be nonzero")
        values[3:] = [value / quaternion_norm for value in values[3:]]
        normalized_endpoint_poses.append(values)
    config["stage_endpoint_poses_bar"] = normalized_endpoint_poses
    config["stage_endpoint_positions_bar"] = [pose[:3] for pose in normalized_endpoint_poses]
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
        if not math.isfinite(value):
            raise ValueError("Learned constraint value must be finite")
        scale, weight = _task_fixed_term_parameters(
            true_terms, stage, feature_name
        )
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
