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


class BarFeatureEvaluator:
    def __init__(self, config):
        self._table_point = np.asarray(config["table_surface_point"], dtype=float).reshape(3)
        self._table_normal = _unit(config["table_normal"], "table normal")
        self._bar_axis_local = _unit(config.get("bar_axis_local", [1.0, 0.0, 0.0]), "bar axis local")
        self._obstacle_radius = float(config["obstacle_radius"])
        self._bar_axial_offset_reference = float(
            config.get("bar_axial_offset_reference", 0.0)
        )

    def evaluate(self, positions, tool_axes, bar_pose, obstacle_pose):
        positions = np.asarray(positions, dtype=float)
        tool_axes = _unit(np.asarray(tool_axes, dtype=float), "tool axes")
        bar_rotation = quaternion_to_matrix(np.asarray(bar_pose, dtype=float)[3:7])
        bar_axis = bar_rotation @ self._bar_axis_local
        bar_axis -= self._table_normal * float(bar_axis @ self._table_normal)
        bar_axis = _unit(bar_axis, "bar axis")
        bar_lateral = _unit(np.cross(self._table_normal, bar_axis), "bar lateral")
        obstacle_center = np.asarray(obstacle_pose, dtype=float).reshape(7)[:3]

        relative_obstacle = positions - obstacle_center[None, :]
        obstacle_radial = relative_obstacle - np.outer(
            relative_obstacle @ self._table_normal, self._table_normal
        )
        obstacle_clearance = np.linalg.norm(obstacle_radial, axis=1) - self._obstacle_radius
        surface_dist = (positions - self._table_point[None, :]) @ self._table_normal
        bar_lateral_offset = (positions - np.asarray(bar_pose, dtype=float)[:3]) @ bar_lateral
        bar_axial_offset = (
            (positions - np.asarray(bar_pose, dtype=float)[:3]) @ bar_axis
            - self._bar_axial_offset_reference
        )
        down_component = -(tool_axes @ self._table_normal)
        forward_component = tool_axes @ bar_axis
        tool_pitch = np.arctan2(down_component, forward_component)
        tool_plane_err = np.arcsin(np.clip(tool_axes @ bar_lateral, -1.0, 1.0))
        return {
            "obstacle_clearance": obstacle_clearance,
            "surface_dist": surface_dist,
            "bar_lateral_offset": bar_lateral_offset,
            "bar_axial_offset": bar_axial_offset,
            "tool_pitch": tool_pitch,
            "tool_plane_err": tool_plane_err,
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
        }
        self._weights.update(dict(self._config.get("optimizer_weights", {})))
        self._position_scale = float(self._config.get("position_smooth_scale", 0.02))
        self._axis_scale = float(self._config.get("axis_smooth_scale", 0.20))
        transition = dict(self._config["constraint_transition"])
        self._transition_fraction = float(transition["fraction"])
        self._transition_min_distance = float(transition["min_distance"])
        self._transition_max_distance = float(transition["max_distance"])
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

    def _world_endpoints(self, start_position, goal_position, bar_pose):
        bar_pose = np.asarray(bar_pose, dtype=float).reshape(7)
        rotation = quaternion_to_matrix(bar_pose[3:7])
        if self._endpoint_coordinate_frame == "bar_tracker_frame":
            intermediate_offsets = self._endpoint_positions_bar @ rotation.T
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
        else:
            raise ValueError(
                "Unsupported endpoint_coordinate_frame {}".format(
                    self._endpoint_coordinate_frame
                )
            )
        intermediate = bar_pose[:3][None, :] + intermediate_offsets
        return np.vstack(
            [
                np.asarray(start_position, dtype=float).reshape(3),
                intermediate,
                np.asarray(goal_position, dtype=float).reshape(3),
            ]
        )

    def _initial_trajectory(self, endpoints, start_axis, goal_axis):
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
            block = np.linspace(endpoints[stage_index], endpoints[stage_index + 1], count)
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
        return positions, axes, labels, np.asarray(endpoint_indices, dtype=int)

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
        return values ** 3 * (values * (values * 6.0 - 15.0) + 10.0)

    def _stage_constraint_weights(self, positions, endpoint_indices, transition_distances):
        positions = np.asarray(positions, dtype=float)
        endpoint_indices = np.asarray(endpoint_indices, dtype=int)
        transition_distances = np.asarray(transition_distances, dtype=float)
        weights = np.zeros((len(positions), self._n_stages), dtype=float)
        stage_start = 0
        for stage_index, stage_end in enumerate(endpoint_indices):
            stage_end = int(stage_end)
            weights[stage_start:stage_end + 1, stage_index] = 1.0
            stage_start = stage_end + 1

        transition_ends = []
        for stage_index, transition_distance in enumerate(transition_distances):
            boundary = int(endpoint_indices[stage_index])
            next_boundary = int(endpoint_indices[stage_index + 1])
            block = positions[boundary:next_boundary + 1]
            cumulative = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(block, axis=0), axis=1)))
            )
            if transition_distance <= 1e-12:
                alpha = np.ones_like(cumulative)
                alpha[0] = 0.0
            else:
                alpha = self._smoothstep5(cumulative / transition_distance)
            weights[boundary:next_boundary + 1, stage_index] = 1.0 - alpha
            weights[boundary:next_boundary + 1, stage_index + 1] = alpha
            relative_end = int(np.searchsorted(cumulative, transition_distance, side="left"))
            transition_ends.append(boundary + min(relative_end, len(cumulative) - 1))
        return weights, np.asarray(transition_ends, dtype=int)

    @staticmethod
    def _free_position_indices(length, endpoint_indices):
        fixed = {0}
        fixed.update(int(value) for value in endpoint_indices)
        return np.asarray([index for index in range(int(length)) if index not in fixed], dtype=int)

    @staticmethod
    def _pack(positions, raw_axes, free_positions):
        return np.concatenate([positions[free_positions].reshape(-1), raw_axes.reshape(-1)])

    @staticmethod
    def _unpack(values, position_template, free_positions):
        values = np.asarray(values, dtype=float)
        position_count = 3 * len(free_positions)
        positions = np.asarray(position_template, dtype=float).copy()
        positions[free_positions] = values[:position_count].reshape((-1, 3))
        raw_axes = values[position_count:].reshape((-1, 3))
        axes = _unit(raw_axes, "optimized tool axes")
        return positions, raw_axes, axes

    def _residual(
        self,
        values,
        position_template,
        free_positions,
        stage_weights,
        target_steps,
        bar_pose,
        obstacle_pose,
    ):
        positions, raw_axes, axes = self._unpack(values, position_template, free_positions)
        residuals = []
        position_delta = np.diff(positions, axis=0)
        position_step = np.linalg.norm(position_delta, axis=1)
        residuals.append(
            math.sqrt(self._weights["position_first"])
            * (position_step - target_steps)
            / self._position_scale
        )
        if len(positions) > 2:
            residuals.append(
                math.sqrt(self._weights["position_second"])
                * (positions[:-2] - 2.0 * positions[1:-1] + positions[2:]).reshape(-1)
                / self._position_scale
            )
            residuals.append(
                math.sqrt(self._weights["axis_second"])
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

        features = self._feature_evaluator.evaluate(positions, axes, bar_pose, obstacle_pose)
        for term in self._constraint_terms:
            stage = int(term["stage"])
            values_stage = np.asarray(features[str(term["feature_name"])], dtype=float)
            target = float(term["value"])
            semantics = str(term["semantics"])
            if semantics == "target_value":
                violation = values_stage - target
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

    def _densify_axis_changes(self, positions, axes):
        output_positions = [positions[0]]
        output_axes = [axes[0]]
        for index in range(1, len(positions)):
            angle = math.acos(float(np.clip(axes[index - 1] @ axes[index], -1.0, 1.0)))
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
        return np.asarray(output_positions, dtype=float), np.asarray(output_axes, dtype=float)

    def _resample(self, positions, axes, endpoint_indices):
        output_positions = []
        output_axes = []
        output_labels = []
        boundaries = []
        start_index = 0
        for stage_index, end_index in enumerate(endpoint_indices):
            block_positions = positions[start_index:int(end_index) + 1]
            block_axes = axes[start_index:int(end_index) + 1]
            edge = np.linalg.norm(np.diff(block_positions, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(edge)))
            length = float(cumulative[-1])
            count = max(2, int(math.ceil(length / self._output_spacing)) + 1)
            targets = np.linspace(0.0, length, count)
            if length <= 1e-12:
                sampled_positions = np.repeat(block_positions[:1], count, axis=0)
                sampled_axes = np.repeat(block_axes[:1], count, axis=0)
            else:
                sampled_positions = np.column_stack(
                    [np.interp(targets, cumulative, block_positions[:, dim]) for dim in range(3)]
                )
                sampled_axes = np.column_stack(
                    [np.interp(targets, cumulative, block_axes[:, dim]) for dim in range(3)]
                )
                sampled_axes = _unit(sampled_axes, "resampled tool axes")
            sampled_positions, sampled_axes = self._densify_axis_changes(
                sampled_positions, sampled_axes
            )
            if stage_index > 0:
                sampled_positions = sampled_positions[1:]
                sampled_axes = sampled_axes[1:]
            output_positions.extend(sampled_positions)
            output_axes.extend(sampled_axes)
            output_labels.extend([stage_index] * len(sampled_positions))
            boundaries.append(len(output_positions) - 1)
            start_index = int(end_index)
        labels = np.asarray(output_labels, dtype=int)
        for stage_index, boundary in enumerate(boundaries[:-1]):
            labels[int(boundary)] = stage_index + 1
        return (
            np.asarray(output_positions, dtype=float),
            _unit(np.asarray(output_axes, dtype=float), "resampled tool axes"),
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
        endpoints = self._world_endpoints(start_pose[:3], goal_pose[:3], bar_pose)
        start_axis = quaternion_to_matrix(start_pose[3:7])[:, 2]
        goal_axis = quaternion_to_matrix(goal_pose[3:7])[:, 2]
        position_template, axis_template, labels, endpoint_indices = self._initial_trajectory(
            endpoints, start_axis, goal_axis
        )
        transition_distances = self._transition_distances(endpoints)
        stage_weights, _transition_ends = self._stage_constraint_weights(
            position_template, endpoint_indices, transition_distances
        )
        free_positions = self._free_position_indices(len(position_template), endpoint_indices)
        target_steps = np.linalg.norm(np.diff(position_template, axis=0), axis=1)
        best = None
        for attempt in range(self._multi_start):
            rng = np.random.RandomState(int(seed) + 104729 * attempt)
            positions_initial = position_template.copy()
            axes_initial = axis_template.copy()
            if attempt > 0:
                positions_initial[free_positions] += rng.normal(
                    scale=0.004 * attempt, size=(len(free_positions), 3)
                )
                axes_initial = _unit(
                    axes_initial + rng.normal(scale=0.06 * attempt, size=axes_initial.shape),
                    "initial tool axes",
                )
            initial = self._pack(positions_initial, axes_initial, free_positions)
            result = least_squares(
                self._residual,
                initial,
                args=(
                    position_template,
                    free_positions,
                    stage_weights,
                    target_steps,
                    bar_pose,
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
        positions, _raw_axes, axes = self._unpack(result.x, position_template, free_positions)
        positions, axes, labels, boundaries = self._resample(positions, axes, endpoint_indices)
        output_stage_weights, transition_ends = self._stage_constraint_weights(
            positions, boundaries, transition_distances
        )
        features = self._feature_evaluator.evaluate(positions, axes, bar_pose, obstacle_pose)
        constraint_report = []
        for term in self._constraint_terms:
            stage = int(term["stage"])
            term_weights = output_stage_weights[:, stage]
            feature_values = np.asarray(features[str(term["feature_name"])])
            target = float(term["value"])
            semantics = str(term["semantics"])
            if semantics == "target_value":
                violation = np.abs(feature_values - target)
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
                "distance": float(transition_distances[stage_index]),
            }
            for stage_index in range(self._n_stages - 1)
        ]
        return {
            "positions": positions,
            "tool_axes": axes,
            "stage_labels": labels,
            "stage_boundaries": boundaries,
            "stage_constraint_weights": output_stage_weights,
            "stage_transition_windows": transition_windows,
            "stage_endpoints_world": endpoints[1:].copy(),
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
