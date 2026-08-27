from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares


def _unit(values, name="vector"):
    array = np.asarray(values, dtype=float)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(~np.isfinite(array)) or np.any(norm <= 1e-12):
        raise ValueError("{} must be finite and nonzero".format(name))
    return array / norm


def quaternion_to_matrix(quaternion):
    values = np.asarray(quaternion, dtype=float).reshape(4)
    values = _unit(values, "quaternion")
    x, y, z, w = values
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_quaternion(matrix):
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return _unit(quaternion, "rotation quaternion")


def _wrap_angle(values):
    values = np.asarray(values, dtype=float)
    return np.arctan2(np.sin(values), np.cos(values))


def transform_pose(pose, rotation, translation):
    """Apply a rigid coordinate-frame transform to an xyz+xyzw pose."""
    pose = np.asarray(pose, dtype=float).reshape(7)
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    translation = np.asarray(translation, dtype=float).reshape(3)
    if np.any(~np.isfinite(pose)):
        raise ValueError("pose must be finite")
    if np.any(~np.isfinite(rotation)) or np.any(~np.isfinite(translation)):
        raise ValueError("rigid transform must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("rigid transform rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError("rigid transform rotation must have determinant +1")

    transformed = np.empty(7, dtype=float)
    transformed[:3] = rotation @ pose[:3] + translation
    transformed[3:] = _matrix_to_quaternion(
        rotation @ quaternion_to_matrix(pose[3:])
    )
    return transformed


def canonicalize_bar_pose(
    bar_pose,
    obstacle_pose,
    table_normal,
    bar_axis_local,
    convention,
):
    """Resolve the 180-degree bar-axis ambiguity using the obstacle side."""
    bar_pose = np.asarray(bar_pose, dtype=float).reshape(7).copy()
    obstacle_pose = np.asarray(obstacle_pose, dtype=float).reshape(7)
    convention = str(convention)
    if convention in ("", "tracker_frame"):
        return bar_pose, False
    if convention not in ("away_from_obstacle", "toward_obstacle"):
        raise ValueError("Unsupported canonical_bar_axis {}".format(convention))

    table_normal = _unit(table_normal, "table normal").reshape(3)
    rotation = quaternion_to_matrix(bar_pose[3:7])
    bar_axis = rotation @ _unit(bar_axis_local, "bar axis local").reshape(3)
    bar_axis -= table_normal * float(bar_axis @ table_normal)
    bar_axis = _unit(bar_axis, "projected bar axis").reshape(3)
    toward_obstacle = obstacle_pose[:3] - bar_pose[:3]
    toward_obstacle -= table_normal * float(toward_obstacle @ table_normal)
    toward_obstacle = _unit(
        toward_obstacle, "bar-to-obstacle direction"
    ).reshape(3)
    points_toward_obstacle = float(bar_axis @ toward_obstacle) > 0.0
    should_flip = (
        points_toward_obstacle
        if convention == "away_from_obstacle"
        else not points_toward_obstacle
    )
    if not should_flip:
        return bar_pose, False

    # A half-turn about the table normal reverses both horizontal bar-frame
    # axes while preserving height. This reconstructs the demonstration frame
    # even when Motive exposes the same centered rigid body with opposite axes.
    half_turn = 2.0 * np.outer(table_normal, table_normal) - np.eye(3)
    bar_pose[3:] = _matrix_to_quaternion(half_turn @ rotation)
    return bar_pose, True


def build_bar_table_task_frame(
    bar_pose,
    obstacle_pose,
    table_surface_point,
    table_normal,
    bar_axis_local,
    convention,
):
    """Build one frozen task frame without using Motive's transverse axes.

    Motive contributes only the configured bar longitudinal axis.  That axis is
    projected into the calibrated table plane; table normal supplies task +Z,
    and +Y is reconstructed as Z cross X.  The origin is the tracked bar point
    projected onto the table so task-frame Z is distance above the table.
    """
    bar_pose = np.asarray(bar_pose, dtype=float).reshape(7)
    obstacle_pose = np.asarray(obstacle_pose, dtype=float).reshape(7)
    table_point = np.asarray(table_surface_point, dtype=float).reshape(3)
    if not all(
        np.all(np.isfinite(value))
        for value in (bar_pose, obstacle_pose, table_point)
    ):
        raise ValueError("Task-frame inputs must be finite")

    normal = _unit(table_normal, "table normal").reshape(3)
    motive_rotation = quaternion_to_matrix(bar_pose[3:7])
    axial = motive_rotation @ _unit(bar_axis_local, "bar axis local").reshape(3)
    axial -= normal * float(axial @ normal)
    axial = _unit(axial, "bar axis projected into table plane").reshape(3)

    convention = str(convention)
    flipped = False
    if convention not in ("", "tracker_frame"):
        if convention not in ("away_from_obstacle", "toward_obstacle"):
            raise ValueError("Unsupported canonical_bar_axis {}".format(convention))
        toward_obstacle = obstacle_pose[:3] - bar_pose[:3]
        toward_obstacle -= normal * float(toward_obstacle @ normal)
        toward_obstacle = _unit(
            toward_obstacle, "bar-to-obstacle direction projected into table plane"
        ).reshape(3)
        points_toward_obstacle = float(axial @ toward_obstacle) > 0.0
        flipped = (
            points_toward_obstacle
            if convention == "away_from_obstacle"
            else not points_toward_obstacle
        )
        if flipped:
            axial = -axial

    lateral = _unit(np.cross(normal, axial), "task lateral axis").reshape(3)
    rotation_world_from_task = np.column_stack((axial, lateral, normal))
    origin = bar_pose[:3] - normal * float((bar_pose[:3] - table_point) @ normal)
    return {
        "frame_id": "bar_table_task",
        "origin": origin,
        "rotation_world_from_task": rotation_world_from_task,
        "axial": axial,
        "lateral": lateral,
        "normal": normal,
        "bar_reference": bar_pose[:3].copy(),
        "axis_flipped": bool(flipped),
        "snapshot_policy": "frozen_per_task",
    }


def _interpolate_unit_vectors(first, second, phases):
    first = _unit(np.asarray(first, dtype=float).reshape(3), "first axis")
    second = _unit(np.asarray(second, dtype=float).reshape(3), "second axis")
    phases = np.asarray(phases, dtype=float).reshape(-1)
    dot = float(np.clip(first @ second, -1.0, 1.0))
    angle = math.acos(dot)
    if angle < 1e-8:
        return np.repeat(first[None, :], len(phases), axis=0)
    if math.pi - angle < 1e-6:
        reference = np.asarray([1.0, 0.0, 0.0])
        if abs(float(reference @ first)) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0])
        tangent = _unit(np.cross(first, reference), "axis interpolation tangent")
        return np.asarray(
            [math.cos(angle * phase) * first + math.sin(angle * phase) * tangent for phase in phases]
        )
    denominator = math.sin(angle)
    return np.asarray(
        [
            math.sin((1.0 - phase) * angle) / denominator * first
            + math.sin(phase * angle) / denominator * second
            for phase in phases
        ]
    )


def continuous_quaternions_from_axes(tool_axes, reference_quaternion):
    axes = _unit(np.asarray(tool_axes, dtype=float), "tool axes")
    reference_rotation = quaternion_to_matrix(reference_quaternion)
    reference_x = reference_rotation[:, 0]
    previous_quaternion = None
    quaternions = []
    for z_axis in axes:
        x_axis = reference_x - float(reference_x @ z_axis) * z_axis
        if np.linalg.norm(x_axis) <= 1e-8:
            fallback = np.asarray([1.0, 0.0, 0.0])
            if abs(float(fallback @ z_axis)) > 0.9:
                fallback = np.asarray([0.0, 1.0, 0.0])
            x_axis = fallback - float(fallback @ z_axis) * z_axis
        x_axis = _unit(x_axis, "tool basis x")
        y_axis = _unit(np.cross(z_axis, x_axis), "tool basis y")
        x_axis = _unit(np.cross(y_axis, z_axis), "tool basis x")
        quaternion = _matrix_to_quaternion(np.column_stack((x_axis, y_axis, z_axis)))
        if previous_quaternion is not None and float(previous_quaternion @ quaternion) < 0.0:
            quaternion = -quaternion
        quaternions.append(quaternion)
        previous_quaternion = quaternion
        reference_x = x_axis
    return np.asarray(quaternions, dtype=float)


def tool_yaw_from_quaternion(quaternion, task_frame):
    """Return Tool-X yaw in the frozen bar/table task frame."""
    rotation = quaternion_to_matrix(quaternion)
    normal = _unit(task_frame["normal"], "task normal").reshape(3)
    tool_x = rotation[:, 0]
    horizontal_x = tool_x - normal * float(tool_x @ normal)
    if np.linalg.norm(horizontal_x) <= 1e-8:
        raise ValueError("Tool-X cannot be projected into the table plane")
    horizontal_x = _unit(horizontal_x, "projected Tool-X").reshape(3)
    return float(
        math.atan2(
            float(horizontal_x @ task_frame["lateral"]),
            float(horizontal_x @ task_frame["axial"]),
        )
    )


def quaternions_from_axes_and_yaws(tool_axes, tool_yaws, task_frame):
    """Construct full orientations with an exact table-plane Tool-X heading."""
    axes = _unit(np.asarray(tool_axes, dtype=float), "tool axes")
    yaws = np.asarray(tool_yaws, dtype=float).reshape(-1)
    if len(axes) != len(yaws):
        raise ValueError("A tool yaw is required for every tool axis")
    axial = _unit(task_frame["axial"], "task axial axis").reshape(3)
    lateral = _unit(task_frame["lateral"], "task lateral axis").reshape(3)
    normal = _unit(task_frame["normal"], "task normal").reshape(3)
    quaternions = []
    previous = None
    for z_axis, yaw in zip(axes, yaws):
        heading = math.cos(float(yaw)) * axial + math.sin(float(yaw)) * lateral
        normal_component = float(normal @ z_axis)
        if abs(normal_component) > 1e-6:
            # Adding only a table-normal component keeps the horizontal
            # projection exactly on ``heading`` while making Tool-X ⟂ Tool-Z.
            x_axis = heading - normal * float(heading @ z_axis) / normal_component
        else:
            x_axis = heading - z_axis * float(heading @ z_axis)
        if np.linalg.norm(x_axis) <= 1e-8:
            raise ValueError("Tool-X yaw is singular for the requested Tool-Z axis")
        x_axis = _unit(x_axis, "tool basis x").reshape(3)
        y_axis = _unit(np.cross(z_axis, x_axis), "tool basis y").reshape(3)
        x_axis = _unit(np.cross(y_axis, z_axis), "tool basis x").reshape(3)
        quaternion = _matrix_to_quaternion(np.column_stack((x_axis, y_axis, z_axis)))
        if previous is not None and float(previous @ quaternion) < 0.0:
            quaternion = -quaternion
        quaternions.append(quaternion)
        previous = quaternion
    return np.asarray(quaternions, dtype=float)


class BarFeatureEvaluator:
    def __init__(self, config):
        self._table_point = np.asarray(config["table_surface_point"], dtype=float).reshape(3)
        self._table_normal = _unit(config["table_normal"], "table normal")
        self._bar_axis_local = _unit(config.get("bar_axis_local", [1.0, 0.0, 0.0]), "bar axis local")
        self._obstacle_radius = float(config["obstacle_radius"])
        self._bar_axial_offset_reference = float(
            config.get("bar_axial_offset_reference", 0.0)
        )

    def evaluate(self, positions, tool_axes, task_frame, obstacle_pose, tool_yaws=None):
        positions = np.asarray(positions, dtype=float)
        tool_axes = _unit(np.asarray(tool_axes, dtype=float), "tool axes")
        bar_axis = _unit(task_frame["axial"], "task axial axis")
        bar_lateral = _unit(task_frame["lateral"], "task lateral axis")
        task_origin = np.asarray(task_frame["origin"], dtype=float).reshape(3)
        obstacle_center = np.asarray(obstacle_pose, dtype=float).reshape(7)[:3]

        relative_obstacle = positions - obstacle_center[None, :]
        obstacle_radial = relative_obstacle - np.outer(
            relative_obstacle @ self._table_normal, self._table_normal
        )
        obstacle_clearance = np.linalg.norm(obstacle_radial, axis=1) - self._obstacle_radius
        surface_dist = (positions - self._table_point[None, :]) @ self._table_normal
        bar_lateral_offset = (positions - task_origin) @ bar_lateral
        bar_axial_offset = (
            (positions - task_origin) @ bar_axis
            - self._bar_axial_offset_reference
        )
        down_component = -(tool_axes @ self._table_normal)
        forward_component = tool_axes @ bar_axis
        tool_pitch = np.arctan2(down_component, forward_component)
        tool_plane_err = np.arcsin(np.clip(tool_axes @ bar_lateral, -1.0, 1.0))
        if tool_yaws is None:
            tool_yaw = np.zeros(len(positions), dtype=float)
        else:
            tool_yaw = _wrap_angle(np.asarray(tool_yaws, dtype=float).reshape(-1))
            if len(tool_yaw) != len(positions):
                raise ValueError("A tool yaw is required for every position")
        return {
            "obstacle_clearance": obstacle_clearance,
            "surface_dist": surface_dist,
            "bar_lateral_offset": bar_lateral_offset,
            "bar_axial_offset": bar_axial_offset,
            "tool_pitch": tool_pitch,
            "tool_plane_err": tool_plane_err,
            "tool_yaw": tool_yaw,
        }


class StageConstraintTrajectoryOptimizer:
    def __init__(
        self,
        config,
        *,
        control_spacing=0.045,
        output_spacing=0.005,
        output_axis_spacing=math.radians(2.0),
        min_control_points=6,
        max_control_points=10,
        max_nfev=80,
        multi_start=2,
    ):
        self._config = dict(config)
        self._endpoint_positions_bar = np.asarray(
            self._config["stage_endpoint_positions_bar"], dtype=float
        )
        self._endpoint_coordinate_frame = str(
            self._config.get("endpoint_coordinate_frame", "bar_tracker_frame")
        )
        self._canonical_bar_axis = str(
            self._config.get("canonical_bar_axis", "tracker_frame")
        )
        if (
            self._endpoint_positions_bar.ndim != 2
            or self._endpoint_positions_bar.shape[1] != 3
            or len(self._endpoint_positions_bar) < 1
        ):
            raise ValueError(
                "stage_endpoint_positions_bar must contain one or more XYZ points"
            )
        self._n_stages = len(self._endpoint_positions_bar) + 1
        self._constraint_terms = [dict(value) for value in self._config["constraint_terms"]]
        for term in self._constraint_terms:
            stage = int(term["stage"])
            if not 0 <= stage < self._n_stages:
                raise ValueError(
                    "Constraint stage {} is outside 0..{}".format(
                        stage, self._n_stages - 1
                    )
                )
        self._feature_evaluator = BarFeatureEvaluator(self._config)
        self._control_spacing = float(control_spacing)
        self._output_spacing = float(output_spacing)
        self._output_axis_spacing = float(output_axis_spacing)
        self._min_control_points = int(min_control_points)
        self._max_control_points = int(max_control_points)
        self._max_nfev = int(max_nfev)
        self._multi_start = int(max(multi_start, 1))
        self._weights = {
            "constraint": 3.0,
            "position_first": 0.12,
            "position_second": 0.65,
            "axis_first": 0.10,
            "axis_second": 0.55,
            "axis_norm": 0.05,
            "yaw_first": 0.10,
            "yaw_second": 0.55,
        }
        self._weights.update(dict(self._config.get("optimizer_weights", {})))
        self._position_scale = float(self._config.get("position_smooth_scale", 0.02))
        self._axis_scale = float(self._config.get("axis_smooth_scale", 0.20))
        self._yaw_scale = float(self._config.get("yaw_smooth_scale", 0.20))
        transition = dict(self._config["constraint_transition"])
        self._transition_fraction = float(transition["fraction"])
        self._transition_min_distance = float(transition["min_distance"])
        self._transition_max_distance = float(transition["max_distance"])
        settling = dict(self._config["constraint_settling"])
        self._settling_control_points = int(settling["control_points"])
        self._settling_max_progress = float(settling["max_progress_m"])
        self._settling_progress_weight = float(settling["progress_weight"])
        self._settling_smoothness_scale = float(settling["smoothness_scale"])
        self._settling_boundaries = self._constraint_change_boundaries()
        if (
            self._control_spacing <= 0.0
            or self._output_spacing <= 0.0
            or self._output_axis_spacing <= 0.0
        ):
            raise ValueError("Planner spacings must be positive")
        if self._min_control_points < 3 or self._max_control_points < self._min_control_points:
            raise ValueError("Invalid control-point limits")
        if not 0.0 < self._transition_fraction <= 1.0:
            raise ValueError("constraint_transition fraction must be in (0, 1]")
        if (
            self._transition_min_distance <= 0.0
            or self._transition_max_distance < self._transition_min_distance
        ):
            raise ValueError("Invalid constraint_transition distance limits")
        if self._settling_control_points < 1:
            raise ValueError("constraint_settling control_points must be positive")
        if self._settling_max_progress < 0.0:
            raise ValueError("constraint_settling max_progress_m must be nonnegative")
        if self._settling_progress_weight < 0.0:
            raise ValueError("constraint_settling progress_weight must be nonnegative")
        if not 0.0 <= self._settling_smoothness_scale <= 1.0:
            raise ValueError("constraint_settling smoothness_scale must be in [0, 1]")

    def _constraint_change_boundaries(self):
        """Select transitions that introduce or change next-stage constraints."""
        signatures = []
        for stage in range(self._n_stages):
            by_feature = {}
            for term in self._constraint_terms:
                if int(term["stage"]) != stage:
                    continue
                feature_name = str(term["feature_name"])
                by_feature.setdefault(feature_name, []).append(
                    (str(term["semantics"]), float(term["value"]))
                )
            signatures.append(
                {
                    feature_name: tuple(sorted(values))
                    for feature_name, values in by_feature.items()
                }
            )

        changed = []
        for stage in range(self._n_stages - 1):
            current = signatures[stage]
            following = signatures[stage + 1]
            changed.append(
                any(
                    feature_name not in current
                    or current[feature_name] != signature
                    for feature_name, signature in following.items()
                )
            )
        return np.asarray(changed, dtype=bool)

    def _world_endpoints(self, start_position, goal_position, bar_pose, task_frame=None):
        bar_pose = np.asarray(bar_pose, dtype=float).reshape(7)
        rotation = quaternion_to_matrix(bar_pose[3:7])
        if self._endpoint_coordinate_frame == "bar_tracker_frame":
            intermediate_offsets = self._endpoint_positions_bar @ rotation.T
            intermediate = bar_pose[:3][None, :] + intermediate_offsets
        elif self._endpoint_coordinate_frame == "bar_task_frame":
            table_normal = _unit(self._config["table_normal"], "table normal")
            bar_axis = rotation @ _unit(
                self._config.get("bar_axis_local", [1.0, 0.0, 0.0]),
                "bar axis local",
            )
            bar_axis -= table_normal * float(bar_axis @ table_normal)
            bar_axis = _unit(bar_axis, "projected bar axis")
            bar_lateral = _unit(np.cross(table_normal, bar_axis), "bar lateral")
            task_rotation = np.column_stack((bar_axis, bar_lateral, table_normal))
            intermediate_offsets = self._endpoint_positions_bar @ task_rotation.T
            intermediate = bar_pose[:3][None, :] + intermediate_offsets
        elif self._endpoint_coordinate_frame == "bar_table_task":
            if task_frame is None:
                raise ValueError("bar_table_task endpoints require a frozen task frame")
            task_rotation = np.asarray(
                task_frame["rotation_world_from_task"], dtype=float
            ).reshape(3, 3)
            task_origin = np.asarray(task_frame["origin"], dtype=float).reshape(3)
            intermediate = (
                task_origin[None, :]
                + self._endpoint_positions_bar @ task_rotation.T
            )
        else:
            raise ValueError(
                "Unsupported endpoint_coordinate_frame {}".format(
                    self._endpoint_coordinate_frame
                )
            )
        return np.vstack(
            [
                np.asarray(start_position, dtype=float).reshape(3),
                intermediate,
                np.asarray(goal_position, dtype=float).reshape(3),
            ]
        )

    def _initial_trajectory(self, endpoints, start_axis, goal_axis, start_yaw, goal_yaw):
        positions = []
        labels = []
        endpoint_indices = []
        for stage_index in range(self._n_stages):
            distance = float(np.linalg.norm(endpoints[stage_index + 1] - endpoints[stage_index]))
            count = int(np.clip(
                int(math.ceil(distance / self._control_spacing)) + 1,
                self._min_control_points,
                self._max_control_points,
            ))
            phases = np.linspace(0.0, 1.0, count)
            if stage_index > 0 and self._settling_boundaries[stage_index - 1]:
                settle_count = min(self._settling_control_points, count - 2)
                settle_progress = min(
                    self._settling_max_progress,
                    0.25 * distance,
                )
                settle_phase = settle_progress / distance if distance > 1e-12 else 0.0
                phases[:settle_count + 1] = np.linspace(
                    0.0, settle_phase, settle_count + 1
                )
                phases[settle_count + 1:] = np.linspace(
                    settle_phase,
                    1.0,
                    count - settle_count,
                )[1:]
            block = (
                endpoints[stage_index][None, :]
                + phases[:, None]
                * (endpoints[stage_index + 1] - endpoints[stage_index])[None, :]
            )
            if stage_index > 0:
                block = block[1:]
            positions.extend(block)
            labels.extend([stage_index] * len(block))
            endpoint_indices.append(len(positions) - 1)
        positions = np.asarray(positions, dtype=float)
        labels = np.asarray(labels, dtype=int)
        for stage_index, endpoint_index in enumerate(endpoint_indices[:-1]):
            labels[endpoint_index] = stage_index + 1
        axes = _interpolate_unit_vectors(
            start_axis,
            goal_axis,
            np.linspace(0.0, 1.0, len(positions)),
        )
        yaw_delta = float(_wrap_angle(goal_yaw - start_yaw))
        yaws = np.linspace(start_yaw, start_yaw + yaw_delta, len(positions))
        return positions, axes, yaws, labels, np.asarray(endpoint_indices, dtype=int)

    def _settling_windows(self, endpoints, endpoint_indices):
        windows = []
        for boundary_index, enabled in enumerate(self._settling_boundaries):
            if not enabled:
                continue
            boundary = int(endpoint_indices[boundary_index])
            next_boundary = int(endpoint_indices[boundary_index + 1])
            end = min(
                boundary + self._settling_control_points,
                next_boundary - 1,
            )
            if end <= boundary:
                continue
            direction = _unit(
                endpoints[boundary_index + 2] - endpoints[boundary_index + 1],
                "settling progress direction",
            ).reshape(3)
            progress_limit = min(
                self._settling_max_progress,
                0.25
                * float(
                    np.linalg.norm(
                        endpoints[boundary_index + 2]
                        - endpoints[boundary_index + 1]
                    )
                ),
            )
            indices = np.arange(boundary + 1, end + 1, dtype=int)
            windows.append(
                {
                    "boundary_index": int(boundary_index),
                    "boundary": boundary,
                    "end": end,
                    "indices": indices,
                    "origin": np.asarray(
                        endpoints[boundary_index + 1], dtype=float
                    ).copy(),
                    "direction": direction,
                    "progress_targets": np.linspace(
                        progress_limit / len(indices),
                        progress_limit,
                        len(indices),
                    ),
                }
            )
        return windows

    def _shape_weights(self, length, settling_windows):
        first = np.ones(int(length) - 1, dtype=float)
        second = np.ones(max(int(length) - 2, 0), dtype=float)
        for window in settling_windows:
            boundary = int(window["boundary"])
            end = int(window["end"])
            first[boundary:end] = 0.0
            center_start = max(boundary - 1, 0)
            center_end = min(end + 1, len(second))
            second[center_start:center_end] = self._settling_smoothness_scale
        return first, second

    def _transition_distances(self, endpoints):
        next_stage_lengths = np.linalg.norm(np.diff(endpoints, axis=0), axis=1)[1:]
        requested = np.clip(
            self._transition_fraction * next_stage_lengths,
            self._transition_min_distance,
            self._transition_max_distance,
        )
        return np.minimum(requested, next_stage_lengths)

    @staticmethod
    def _smoothstep5(values):
        values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        # The polynomial can overshoot 1.0 by a few ulps near the upper
        # endpoint.  Those values become negative complementary stage weights
        # and then NaNs when the residual takes their square root.
        smoothed = values ** 3 * (values * (values * 6.0 - 15.0) + 10.0)
        return np.clip(smoothed, 0.0, 1.0)

    @staticmethod
    def _feature_active_mask(
        stage_labels,
        constraint_terms,
        feature_name,
        stage_constraint_weights=None,
    ):
        active_stages = {
            int(term["stage"])
            for term in constraint_terms
            if str(term["feature_name"]) == str(feature_name)
        }
        if stage_constraint_weights is not None:
            weights = np.asarray(stage_constraint_weights, dtype=float)
            if weights.ndim != 2 or weights.shape[0] != len(stage_labels):
                raise ValueError(
                    "Stage constraint weights must match the output samples"
                )
            if not active_stages:
                return np.zeros(len(stage_labels), dtype=bool)
            active_weight = np.sum(
                weights[:, sorted(active_stages)], axis=1
            )
            # A transition out of an inactive stage is intentionally soft in
            # the optimizer.  Do not turn that blend into a hard executor
            # equality until the active-stage weight has fully ramped to one.
            return active_weight >= 1.0 - 1e-9
        return np.isin(np.asarray(stage_labels, dtype=int), sorted(active_stages))

    def _stage_constraint_weights(
        self,
        positions,
        endpoint_indices,
        transition_distances,
        settling_windows=None,
    ):
        positions = np.asarray(positions, dtype=float)
        endpoint_indices = np.asarray(endpoint_indices, dtype=int)
        transition_distances = np.asarray(transition_distances, dtype=float)
        weights = np.zeros((len(positions), self._n_stages), dtype=float)
        stage_start = 0
        for stage_index, stage_end in enumerate(endpoint_indices):
            stage_end = int(stage_end)
            weights[stage_start:stage_end + 1, stage_index] = 1.0
            stage_start = stage_end + 1

        settling_by_boundary = {
            int(window["boundary_index"]): window
            for window in (settling_windows or [])
        }
        transition_ends = []
        for stage_index, transition_distance in enumerate(transition_distances):
            boundary = int(endpoint_indices[stage_index])
            next_boundary = int(endpoint_indices[stage_index + 1])
            block = positions[boundary:next_boundary + 1]
            cumulative = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(block, axis=0), axis=1)))
            )
            settling = settling_by_boundary.get(stage_index)
            if settling is not None:
                relative_end = min(
                    int(settling["end"]) - boundary,
                    len(cumulative) - 1,
                )
                alpha = np.ones_like(cumulative)
                alpha[:relative_end + 1] = self._smoothstep5(
                    np.linspace(0.0, 1.0, relative_end + 1)
                )
            elif transition_distance <= 1e-12:
                alpha = np.ones_like(cumulative)
                alpha[0] = 0.0
                relative_end = 0
            else:
                alpha = self._smoothstep5(cumulative / transition_distance)
                relative_end = int(
                    np.searchsorted(cumulative, transition_distance, side="left")
                )
            weights[boundary:next_boundary + 1, stage_index] = 1.0 - alpha
            weights[boundary:next_boundary + 1, stage_index + 1] = alpha
            transition_ends.append(boundary + min(relative_end, len(cumulative) - 1))
        return weights, np.asarray(transition_ends, dtype=int)

    @staticmethod
    def _free_position_indices(length, endpoint_indices):
        fixed = {0}
        fixed.update(int(value) for value in endpoint_indices)
        return np.asarray([index for index in range(int(length)) if index not in fixed], dtype=int)

    @staticmethod
    def _pack(
        positions,
        raw_axes,
        yaws,
        free_positions,
        free_axes,
        free_yaws,
    ):
        return np.concatenate(
            [
                positions[free_positions].reshape(-1),
                raw_axes[free_axes].reshape(-1),
                yaws[free_yaws].reshape(-1),
            ]
        )

    @staticmethod
    def _unpack(
        values,
        position_template,
        axis_template,
        yaw_template,
        free_positions,
        free_axes,
        free_yaws,
    ):
        values = np.asarray(values, dtype=float)
        position_count = 3 * len(free_positions)
        positions = np.asarray(position_template, dtype=float).copy()
        positions[free_positions] = values[:position_count].reshape((-1, 3))
        axis_count = 3 * len(free_axes)
        raw_axes = np.asarray(axis_template, dtype=float).copy()
        raw_axes[free_axes] = values[
            position_count:position_count + axis_count
        ].reshape((-1, 3))
        axes = _unit(raw_axes, "optimized tool axes")
        yaws = np.asarray(yaw_template, dtype=float).copy()
        yaws[free_yaws] = values[position_count + axis_count:]
        return positions, raw_axes, axes, yaws

    def _residual(
        self,
        values,
        position_template,
        axis_template,
        yaw_template,
        free_positions,
        free_axes,
        free_yaws,
        stage_weights,
        target_steps,
        first_shape_weights,
        second_shape_weights,
        settling_windows,
        task_frame,
        obstacle_pose,
    ):
        positions, raw_axes, axes, yaws = self._unpack(
            values,
            position_template,
            axis_template,
            yaw_template,
            free_positions,
            free_axes,
            free_yaws,
        )
        residuals = []
        position_delta = np.diff(positions, axis=0)
        position_step = np.linalg.norm(position_delta, axis=1)
        residuals.append(
            math.sqrt(self._weights["position_first"])
            * np.sqrt(first_shape_weights)
            * (position_step - target_steps)
            / self._position_scale
        )
        if len(positions) > 2:
            residuals.append(
                math.sqrt(self._weights["position_second"])
                * np.repeat(np.sqrt(second_shape_weights), 3)
                * (positions[:-2] - 2.0 * positions[1:-1] + positions[2:]).reshape(-1)
                / self._position_scale
            )
            residuals.append(
                math.sqrt(self._weights["axis_second"])
                * np.repeat(np.sqrt(second_shape_weights), 3)
                * (axes[:-2] - 2.0 * axes[1:-1] + axes[2:]).reshape(-1)
                / self._axis_scale
            )
        residuals.append(
            math.sqrt(self._weights["axis_first"])
            * np.diff(axes, axis=0).reshape(-1)
            / self._axis_scale
        )
        residuals.append(
            math.sqrt(self._weights["axis_norm"])
            * (np.linalg.norm(raw_axes, axis=1) - 1.0)
        )
        yaw_delta = np.diff(yaws)
        residuals.append(
            math.sqrt(self._weights["yaw_first"])
            * yaw_delta
            / self._yaw_scale
        )
        if len(yaws) > 2:
            residuals.append(
                math.sqrt(self._weights["yaw_second"])
                * np.sqrt(second_shape_weights)
                * np.diff(yaws, n=2)
                / self._yaw_scale
            )

        if self._settling_progress_weight > 0.0:
            for window in settling_windows:
                progress = (
                    positions[window["indices"]] - window["origin"][None, :]
                ) @ window["direction"]
                residuals.append(
                    math.sqrt(self._settling_progress_weight)
                    * (progress - window["progress_targets"])
                    / self._position_scale
                )

        features = self._feature_evaluator.evaluate(
            positions, axes, task_frame, obstacle_pose, yaws
        )
        for term in self._constraint_terms:
            stage = int(term["stage"])
            values_stage = np.asarray(features[str(term["feature_name"])], dtype=float)
            target = float(term["value"])
            semantics = str(term["semantics"])
            if semantics == "target_value":
                violation = values_stage - target
                if str(term["feature_name"]) == "tool_yaw":
                    violation = _wrap_angle(violation)
            elif semantics == "upper_bound":
                violation = np.maximum(values_stage - target, 0.0)
            elif semantics == "lower_bound":
                violation = np.maximum(target - values_stage, 0.0)
            else:
                raise ValueError("Unsupported constraint semantics {}".format(semantics))
            scale = max(float(term.get("scale", 1.0)), 1e-8)
            weight = self._weights["constraint"] * float(term.get("weight", 1.0))
            residuals.append(
                np.sqrt(weight * stage_weights[:, stage]) * violation / scale
            )
        return np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in residuals])

    def _densify_orientation_changes(self, positions, axes, yaws):
        output_positions = [positions[0]]
        output_axes = [axes[0]]
        output_yaws = [yaws[0]]
        for index in range(1, len(positions)):
            axis_angle = math.acos(float(np.clip(axes[index - 1] @ axes[index], -1.0, 1.0)))
            yaw_delta = float(_wrap_angle(yaws[index] - yaws[index - 1]))
            angle = max(axis_angle, abs(yaw_delta))
            subdivisions = max(1, int(math.ceil(angle / self._output_axis_spacing)))
            phases = np.linspace(0.0, 1.0, subdivisions + 1)[1:]
            interpolated_positions = (
                (1.0 - phases[:, None]) * positions[index - 1][None, :]
                + phases[:, None] * positions[index][None, :]
            )
            interpolated_axes = _interpolate_unit_vectors(
                axes[index - 1], axes[index], phases
            )
            output_positions.extend(interpolated_positions)
            output_axes.extend(interpolated_axes)
            output_yaws.extend(yaws[index - 1] + phases * yaw_delta)
        return (
            np.asarray(output_positions, dtype=float),
            np.asarray(output_axes, dtype=float),
            np.asarray(output_yaws, dtype=float),
        )

    def _resample(self, positions, axes, yaws, endpoint_indices):
        output_positions = []
        output_axes = []
        output_yaws = []
        output_labels = []
        boundaries = []
        start_index = 0
        for stage_index, end_index in enumerate(endpoint_indices):
            block_positions = positions[start_index:int(end_index) + 1]
            block_axes = axes[start_index:int(end_index) + 1]
            block_yaws = yaws[start_index:int(end_index) + 1]
            edge = np.linalg.norm(np.diff(block_positions, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(edge)))
            length = float(cumulative[-1])
            count = max(2, int(math.ceil(length / self._output_spacing)) + 1)
            targets = np.linspace(0.0, length, count)
            if length <= 1e-12:
                sampled_positions = np.repeat(block_positions[:1], count, axis=0)
                sampled_axes = np.repeat(block_axes[:1], count, axis=0)
                sampled_yaws = np.repeat(block_yaws[:1], count)
            else:
                sampled_positions = np.column_stack(
                    [np.interp(targets, cumulative, block_positions[:, dim]) for dim in range(3)]
                )
                sampled_axes = np.column_stack(
                    [np.interp(targets, cumulative, block_axes[:, dim]) for dim in range(3)]
                )
                sampled_axes = _unit(sampled_axes, "resampled tool axes")
                sampled_yaws = np.interp(targets, cumulative, block_yaws)
            sampled_positions, sampled_axes, sampled_yaws = self._densify_orientation_changes(
                sampled_positions, sampled_axes, sampled_yaws
            )
            if stage_index > 0:
                sampled_positions = sampled_positions[1:]
                sampled_axes = sampled_axes[1:]
                sampled_yaws = sampled_yaws[1:]
            output_positions.extend(sampled_positions)
            output_axes.extend(sampled_axes)
            output_yaws.extend(sampled_yaws)
            output_labels.extend([stage_index] * len(sampled_positions))
            boundaries.append(len(output_positions) - 1)
            start_index = int(end_index)
        labels = np.asarray(output_labels, dtype=int)
        for stage_index, boundary in enumerate(boundaries[:-1]):
            labels[int(boundary)] = stage_index + 1
        return (
            np.asarray(output_positions, dtype=float),
            _unit(np.asarray(output_axes, dtype=float), "resampled tool axes"),
            np.asarray(output_yaws, dtype=float),
            labels,
            np.asarray(boundaries, dtype=int),
        )

    def plan(self, start_pose, goal_pose, bar_pose, obstacle_pose, seed=0):
        start_pose = np.asarray(start_pose, dtype=float).reshape(7)
        goal_pose = np.asarray(goal_pose, dtype=float).reshape(7)
        bar_pose = np.asarray(bar_pose, dtype=float).reshape(7)
        obstacle_pose = np.asarray(obstacle_pose, dtype=float).reshape(7)
        if not all(np.all(np.isfinite(value)) for value in (start_pose, goal_pose, bar_pose, obstacle_pose)):
            raise ValueError("Planner inputs must be finite")
        # This is the only task-frame construction for the plan.  Every
        # endpoint, residual and reported feature below reuses this immutable
        # snapshot even while live OptiTrack callbacks continue in the node.
        task_frame = build_bar_table_task_frame(
            bar_pose,
            obstacle_pose,
            self._config["table_surface_point"],
            self._config["table_normal"],
            self._config.get("bar_axis_local", [1.0, 0.0, 0.0]),
            self._canonical_bar_axis,
        )
        bar_axis_flipped = bool(task_frame["axis_flipped"])
        endpoints = self._world_endpoints(
            start_pose[:3], goal_pose[:3], bar_pose, task_frame=task_frame
        )
        start_axis = quaternion_to_matrix(start_pose[3:7])[:, 2]
        goal_axis = quaternion_to_matrix(goal_pose[3:7])[:, 2]
        start_yaw = tool_yaw_from_quaternion(start_pose[3:7], task_frame)
        goal_yaw = tool_yaw_from_quaternion(goal_pose[3:7], task_frame)
        position_template, axis_template, yaw_template, labels, endpoint_indices = self._initial_trajectory(
            endpoints, start_axis, goal_axis, start_yaw, goal_yaw
        )
        settling_windows = self._settling_windows(endpoints, endpoint_indices)
        first_shape_weights, second_shape_weights = self._shape_weights(
            len(position_template), settling_windows
        )
        transition_distances = self._transition_distances(endpoints)
        stage_weights, _transition_ends = self._stage_constraint_weights(
            position_template,
            endpoint_indices,
            transition_distances,
            settling_windows,
        )
        free_positions = self._free_position_indices(len(position_template), endpoint_indices)
        free_axes = np.arange(1, len(position_template) - 1, dtype=int)
        free_yaws = np.arange(1, len(position_template) - 1, dtype=int)
        target_steps = np.linalg.norm(np.diff(position_template, axis=0), axis=1)
        best = None
        for attempt in range(self._multi_start):
            rng = np.random.RandomState(int(seed) + 104729 * attempt)
            positions_initial = position_template.copy()
            axes_initial = axis_template.copy()
            yaws_initial = yaw_template.copy()
            if attempt > 0:
                positions_initial[free_positions] += rng.normal(
                    scale=0.004 * attempt, size=(len(free_positions), 3)
                )
                axes_initial[free_axes] = _unit(
                    axes_initial[free_axes]
                    + rng.normal(
                        scale=0.06 * attempt,
                        size=(len(free_axes), 3),
                    ),
                    "initial tool axes",
                )
                yaws_initial[free_yaws] += rng.normal(
                    scale=0.04 * attempt, size=len(free_yaws)
                )
            initial = self._pack(
                positions_initial,
                axes_initial,
                yaws_initial,
                free_positions,
                free_axes,
                free_yaws,
            )
            result = least_squares(
                self._residual,
                initial,
                args=(
                    position_template,
                    axis_template,
                    yaw_template,
                    free_positions,
                    free_axes,
                    free_yaws,
                    stage_weights,
                    target_steps,
                    first_shape_weights,
                    second_shape_weights,
                    settling_windows,
                    task_frame,
                    obstacle_pose,
                ),
                method="trf",
                loss="linear",
                max_nfev=self._max_nfev,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
            )
            if not np.all(np.isfinite(result.x)):
                continue
            score = float(np.sum(np.square(result.fun)))
            if best is None or score < best[0]:
                best = (score, result)
        if best is None:
            raise RuntimeError("Stage constraint optimization did not produce a finite candidate")

        score, result = best
        positions, _raw_axes, axes, yaws = self._unpack(
            result.x,
            position_template,
            axis_template,
            yaw_template,
            free_positions,
            free_axes,
            free_yaws,
        )
        optimized_settling_distances = transition_distances.copy()
        for window in settling_windows:
            block = positions[int(window["boundary"]):int(window["end"]) + 1]
            optimized_settling_distances[int(window["boundary_index"])] = float(
                np.sum(np.linalg.norm(np.diff(block, axis=0), axis=1))
            )
        positions, axes, yaws, labels, boundaries = self._resample(
            positions, axes, yaws, endpoint_indices
        )
        output_stage_weights, transition_ends = self._stage_constraint_weights(
            positions, boundaries, optimized_settling_distances
        )
        features = self._feature_evaluator.evaluate(
            positions, axes, task_frame, obstacle_pose, yaws
        )
        quaternions = quaternions_from_axes_and_yaws(axes, yaws, task_frame)
        constraint_report = []
        for term in self._constraint_terms:
            stage = int(term["stage"])
            term_weights = output_stage_weights[:, stage]
            feature_values = np.asarray(features[str(term["feature_name"])])
            target = float(term["value"])
            semantics = str(term["semantics"])
            if semantics == "target_value":
                violation = np.abs(feature_values - target)
                if str(term["feature_name"]) == "tool_yaw":
                    violation = np.abs(_wrap_angle(feature_values - target))
            elif semantics == "upper_bound":
                violation = np.maximum(feature_values - target, 0.0)
            else:
                violation = np.maximum(target - feature_values, 0.0)
            constraint_report.append(
                {
                    "stage": stage,
                    "feature_name": str(term["feature_name"]),
                    "semantics": semantics,
                    "value": target,
                    "mean_violation": float(
                        np.sum(term_weights * violation) / np.sum(term_weights)
                    ) if np.sum(term_weights) > 1e-12 else 0.0,
                    "max_violation": float(
                        np.max(violation[term_weights >= 0.5])
                    ) if np.any(term_weights >= 0.5) else 0.0,
                }
            )
        transition_windows = [
            {
                "from_stage": int(stage_index),
                "to_stage": int(stage_index + 1),
                "start_index": int(boundaries[stage_index]),
                "end_index": int(transition_ends[stage_index]),
                "distance": float(optimized_settling_distances[stage_index]),
                "constraint_settling": bool(self._settling_boundaries[stage_index]),
            }
            for stage_index in range(self._n_stages - 1)
        ]
        return {
            "positions": positions,
            "tool_axes": axes,
            "tool_yaws": _wrap_angle(yaws),
            "tool_quaternions": quaternions,
            "stage_labels": labels,
            "tool_yaw_active": self._feature_active_mask(
                labels,
                self._constraint_terms,
                "tool_yaw",
                output_stage_weights,
            ),
            "stage_boundaries": boundaries,
            "stage_constraint_weights": output_stage_weights,
            "stage_transition_windows": transition_windows,
            "stage_endpoints_world": endpoints[1:].copy(),
            "bar_axis_flipped": bool(bar_axis_flipped),
            "task_frame": {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in task_frame.items()
            },
            "features": {
                name: np.asarray(values, dtype=float).copy()
                for name, values in features.items()
            },
            "objective": score,
            "solver_success": bool(result.success),
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "solver_evaluations": int(result.nfev),
            "constraint_report": constraint_report,
        }
