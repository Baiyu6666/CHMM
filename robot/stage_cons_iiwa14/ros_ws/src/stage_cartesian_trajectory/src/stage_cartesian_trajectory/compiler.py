"""Shared continuous IK and joint-trajectory time parameterization for iiwa14."""

from __future__ import annotations

import math

import numpy as np


class TrajectoryValidationError(RuntimeError):
    pass


class CartesianTrajectoryCompiler:
    """Compile position + Tool-Z Cartesian samples into two timed joint segments.

    The supplied PyBullet body is only used as a kinematic model. ``compile``
    restores it to ``start_q`` before returning, which also allows a live
    PyBullet simulation body to use this compiler without being teleported by
    the IK search.
    """

    def __init__(
        self,
        bullet_module,
        physics_client,
        robot_id,
        joint_indices,
        tip_index,
        lower_limits,
        upper_limits,
        velocity_limits,
        *,
        max_joint_step=0.15,
        velocity_scale=0.10,
        acceleration_limit=0.25,
        approach_speed=0.04,
        task_speed=0.025,
        approach_spacing=0.01,
        approach_axis_spacing=math.radians(2.0),
        approach_clearance_z=None,
        minimum_approach_z=0.20,
        position_tolerance=0.002,
        tool_z_tolerance=math.radians(2.0),
        minimum_singular_value=0.01,
        first_point_delay=0.5,
    ):
        self._bullet = bullet_module
        self._physics = physics_client
        self._robot = robot_id
        self._joint_indices = list(joint_indices)
        self._tip_index = int(tip_index)
        self._lower = np.asarray(lower_limits, dtype=float)
        self._upper = np.asarray(upper_limits, dtype=float)
        self._velocity_limits = np.asarray(velocity_limits, dtype=float)
        self._dof = len(self._joint_indices)
        if any(values.shape != (self._dof,) for values in (
            self._lower, self._upper, self._velocity_limits
        )):
            raise ValueError("Joint limits must match the number of controlled joints")
        if np.any(self._upper <= self._lower):
            raise ValueError("Every controlled joint must have finite ordered limits")

        self._max_joint_step = float(max_joint_step)
        self._velocity_scale = float(velocity_scale)
        self._acceleration_limit = float(acceleration_limit)
        self._approach_speed = float(approach_speed)
        self._task_speed = float(task_speed)
        self._approach_spacing = float(approach_spacing)
        self._approach_axis_spacing = float(approach_axis_spacing)
        self._approach_clearance_z = (
            None if approach_clearance_z is None else float(approach_clearance_z)
        )
        self._minimum_approach_z = float(minimum_approach_z)
        if not math.isfinite(self._minimum_approach_z):
            raise ValueError("minimum_approach_z must be finite")
        self._position_tolerance = float(position_tolerance)
        self._tool_z_tolerance = float(tool_z_tolerance)
        self._minimum_singular_value = float(minimum_singular_value)
        self._first_point_delay = float(first_point_delay)

    def _set_q(self, values):
        for index, value in zip(self._joint_indices, values):
            self._bullet.resetJointState(
                self._robot,
                index,
                float(value),
                targetVelocity=0.0,
                physicsClientId=self._physics,
            )

    def tip_state(self, values):
        self._set_q(values)
        state = self._bullet.getLinkState(
            self._robot,
            self._tip_index,
            computeForwardKinematics=True,
            physicsClientId=self._physics,
        )
        rotation = np.asarray(
            self._bullet.getMatrixFromQuaternion(state[5]), dtype=float
        ).reshape(3, 3)
        return np.asarray(state[4], dtype=float), rotation

    def tool_z_from_quaternion(self, quaternion):
        values = np.asarray(quaternion, dtype=float)
        if values.shape != (4,):
            raise TrajectoryValidationError("Quaternion must contain four values")
        norm = float(np.linalg.norm(values))
        if not np.all(np.isfinite(values)) or norm < 1e-9:
            raise TrajectoryValidationError("Path contains an invalid quaternion")
        rotation = np.asarray(
            self._bullet.getMatrixFromQuaternion((values / norm).tolist()), dtype=float
        ).reshape(3, 3)
        return rotation[:, 2]

    @staticmethod
    def _matrix_to_rpy(matrix):
        pitch = math.asin(max(-1.0, min(1.0, -float(matrix[2, 0]))))
        if abs(math.cos(pitch)) > 1e-8:
            roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
            yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
        else:
            roll = 0.0
            yaw = math.atan2(-float(matrix[0, 1]), float(matrix[1, 1]))
        return [roll, pitch, yaw]

    def _quaternion_from_basis(self, x_axis, z_axis):
        z_axis = np.asarray(z_axis, dtype=float)
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.asarray(x_axis, dtype=float)
        x_axis -= float(x_axis @ z_axis) * z_axis
        if np.linalg.norm(x_axis) < 1e-8:
            reference = np.array([1.0, 0.0, 0.0])
            if abs(float(reference @ z_axis)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            x_axis = reference - float(reference @ z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        matrix = np.column_stack((x_axis, y_axis, z_axis))
        return self._bullet.getQuaternionFromEuler(self._matrix_to_rpy(matrix))

    @staticmethod
    def _interpolate_axis(first, second, phases):
        first = np.asarray(first, dtype=float) / np.linalg.norm(first)
        second = np.asarray(second, dtype=float) / np.linalg.norm(second)
        dot = float(np.clip(first @ second, -1.0, 1.0))
        angle = math.acos(dot)
        if angle < 1e-8:
            return np.repeat(first[None, :], len(phases), axis=0)
        if math.pi - angle < 1e-5:
            reference = np.array([1.0, 0.0, 0.0])
            if abs(float(reference @ first)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            perpendicular = np.cross(first, reference)
            perpendicular /= np.linalg.norm(perpendicular)
            return np.asarray([
                math.cos(angle * phase) * first
                + math.sin(angle * phase) * perpendicular
                for phase in phases
            ])
        denominator = math.sin(angle)
        return np.asarray([
            math.sin((1.0 - phase) * angle) / denominator * first
            + math.sin(phase * angle) / denominator * second
            for phase in phases
        ])

    def _continuous_ik(self, positions, axes, seed, abort_requested, phase_name):
        previous = np.asarray(seed, dtype=float).copy()
        _, rotation = self.tip_state(previous)
        reference_x = rotation[:, 0]
        trajectory, position_errors, axis_errors = [], [], []
        ranges = self._upper - self._lower
        center = 0.5 * (self._lower + self._upper)
        for sample_index, (target_position, target_z) in enumerate(zip(positions, axes)):
            if abort_requested():
                raise TrajectoryValidationError("Trajectory compilation aborted")
            rest = 0.98 * previous + 0.02 * center
            rest = np.clip(rest, previous - 0.03, previous + 0.03)
            best = None
            for degrees in (0, 5, -5, 10, -10, 20, -20, 30, -30):
                spin = math.radians(degrees)
                tangent_x = reference_x - float(reference_x @ target_z) * target_z
                if np.linalg.norm(tangent_x) < 1e-8:
                    tangent_x = np.array([1.0, 0.0, 0.0])
                tangent_x /= np.linalg.norm(tangent_x)
                tangent_y = np.cross(target_z, tangent_x)
                spun_x = math.cos(spin) * tangent_x + math.sin(spin) * tangent_y
                quaternion = self._quaternion_from_basis(spun_x, target_z)
                self._set_q(previous)
                solution = self._bullet.calculateInverseKinematics(
                    self._robot,
                    self._tip_index,
                    target_position.tolist(),
                    quaternion,
                    lowerLimits=self._lower.tolist(),
                    upperLimits=self._upper.tolist(),
                    jointRanges=ranges.tolist(),
                    restPoses=rest.tolist(),
                    jointDamping=[0.03] * self._dof,
                    maxNumIterations=600,
                    residualThreshold=1e-8,
                    physicsClientId=self._physics,
                )
                candidate = np.asarray(solution[:self._dof], dtype=float)
                actual_position, actual_rotation = self.tip_state(candidate)
                position_error = float(np.linalg.norm(actual_position - target_position))
                axis_error = math.acos(float(np.clip(actual_rotation[:, 2] @ target_z, -1.0, 1.0)))
                step = float(np.max(np.abs(candidate - previous)))
                margin = float(np.min(np.minimum(candidate - self._lower, self._upper - candidate)))
                score = position_error * 1e6 + axis_error * 1e4 + step * 10.0 - margin
                if best is None or score < best[0]:
                    best = (score, candidate, actual_rotation, position_error, axis_error, step)
                if (
                    position_error <= self._position_tolerance
                    and axis_error <= self._tool_z_tolerance
                    and step <= self._max_joint_step
                    and margin >= 0.0
                ):
                    break
            _, candidate, actual_rotation, position_error, axis_error, step = best
            if position_error > self._position_tolerance or axis_error > self._tool_z_tolerance:
                raise TrajectoryValidationError(
                    "{} IK failed at sample {}: position error {:.4f} m, "
                    "Tool-Z error {:.2f} deg".format(
                        phase_name, sample_index, position_error, math.degrees(axis_error)
                    )
                )
            if step > self._max_joint_step:
                raise TrajectoryValidationError(
                    "{} IK branch jump at sample {}: {:.3f} rad exceeds {:.3f}".format(
                        phase_name, sample_index, step, self._max_joint_step
                    )
                )
            trajectory.append(candidate)
            position_errors.append(position_error)
            axis_errors.append(axis_error)
            previous = candidate
            reference_x = actual_rotation[:, 0]
        return np.asarray(trajectory), position_errors, axis_errors

    def _collision_and_singularity_checks(self, q_path):
        minimum_sv = math.inf
        for q in q_path:
            self._set_q(q)
            self._bullet.performCollisionDetection(physicsClientId=self._physics)
            for contact in self._bullet.getContactPoints(
                bodyA=self._robot,
                bodyB=self._robot,
                physicsClientId=self._physics,
            ):
                link_a, link_b, distance = int(contact[3]), int(contact[4]), float(contact[8])
                if abs(link_a - link_b) > 1 and distance < -0.001:
                    raise TrajectoryValidationError(
                        "Self-collision detected between links {} and {}".format(link_a, link_b)
                    )
            linear, angular = self._bullet.calculateJacobian(
                self._robot,
                self._tip_index,
                [0.0, 0.0, 0.0],
                q.tolist(),
                [0.0] * self._dof,
                [0.0] * self._dof,
                physicsClientId=self._physics,
            )
            singular_values = np.linalg.svd(
                np.vstack((linear, angular)), compute_uv=False
            )
            minimum_sv = min(minimum_sv, float(singular_values[-1]))
        if minimum_sv < self._minimum_singular_value:
            raise TrajectoryValidationError(
                "Trajectory approaches a singularity (minimum singular value {:.4f})".format(
                    minimum_sv
                )
            )
        return minimum_sv

    def time_parameterize(self, q_path, minimum_duration):
        q_path = np.asarray(q_path, dtype=float)
        if q_path.ndim != 2 or q_path.shape[1] != self._dof or len(q_path) < 2:
            raise TrajectoryValidationError(
                "A trajectory needs at least two samples with {} joints".format(self._dof)
            )
        joint_step = np.abs(np.diff(q_path, axis=0))
        velocity_limit = np.maximum(self._velocity_limits * self._velocity_scale, 0.05)
        segment_dt = np.maximum(
            0.05,
            np.max(joint_step / velocity_limit[None, :], axis=1),
        )
        raw_duration = float(np.sum(segment_dt))
        duration = max(float(minimum_duration), raw_duration, 0.5)
        relative_times = np.concatenate(([0.0], np.cumsum(segment_dt)))
        relative_times *= duration / relative_times[-1]
        for _ in range(5):
            times = self._first_point_delay + relative_times
            velocity = np.gradient(q_path, times, axis=0, edge_order=1)
            velocity[[0, -1]] = 0.0
            acceleration = np.gradient(velocity, times, axis=0, edge_order=1)
            acceleration[[0, -1]] = 0.0
            ratio_v = float(np.max(np.abs(velocity) / velocity_limit[None, :]))
            ratio_a = float(np.max(np.abs(acceleration) / self._acceleration_limit))
            scale = max(1.0, ratio_v, math.sqrt(ratio_a))
            if scale <= 1.00001:
                break
            relative_times *= scale * 1.01
        times = self._first_point_delay + relative_times
        velocity = np.gradient(q_path, times, axis=0, edge_order=1)
        velocity[[0, -1]] = 0.0
        acceleration = np.gradient(velocity, times, axis=0, edge_order=1)
        acceleration[[0, -1]] = 0.0
        return {
            "position": q_path,
            "velocity": velocity,
            "acceleration": acceleration,
            "time": times,
            "duration": float(times[-1]),
        }

    @staticmethod
    def sample_position(segment, elapsed):
        times = np.asarray(segment["time"], dtype=float)
        positions = np.asarray(segment["position"], dtype=float)
        elapsed = float(np.clip(elapsed, 0.0, times[-1]))
        return np.asarray([
            np.interp(elapsed, times, positions[:, joint])
            for joint in range(positions.shape[1])
        ])

    def _approach_samples(self, current_position, current_axis, target_position, target_axis):
        waypoints = [np.asarray(current_position, dtype=float)]
        target_position = np.asarray(target_position, dtype=float)
        transit_heights = [
            self._minimum_approach_z,
            float(current_position[2]),
            float(target_position[2]),
        ]
        if self._approach_clearance_z is not None:
            transit_heights.append(self._approach_clearance_z)
        transit_z = max(transit_heights)
        waypoints.extend(
            [
                np.asarray([current_position[0], current_position[1], transit_z]),
                np.asarray([target_position[0], target_position[1], transit_z]),
            ]
        )
        waypoints.append(target_position)
        compact = [waypoints[0]]
        for waypoint in waypoints[1:]:
            if float(np.linalg.norm(waypoint - compact[-1])) > 1e-9:
                compact.append(waypoint)
        if len(compact) == 1:
            return target_position[None, :], np.asarray(target_axis, dtype=float)[None, :], 0.0

        waypoints = np.asarray(compact, dtype=float)
        edge_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
        distance = float(cumulative[-1])
        axis_angle = math.acos(float(np.clip(current_axis @ target_axis, -1.0, 1.0)))
        sample_count = max(
            1,
            int(math.ceil(distance / self._approach_spacing)),
            int(math.ceil(axis_angle / self._approach_axis_spacing)),
        )
        targets = np.unique(
            np.concatenate((np.linspace(0.0, distance, sample_count + 1)[1:], cumulative[1:]))
        )
        positions = np.column_stack(
            [np.interp(targets, cumulative, waypoints[:, dim]) for dim in range(3)]
        )
        axes = self._interpolate_axis(current_axis, target_axis, targets / distance)
        return positions, axes, distance

    def compile(self, positions, tool_z_axes, start_q, abort_requested=None):
        positions = np.asarray(positions, dtype=float)
        axes = np.asarray(tool_z_axes, dtype=float)
        start_q = np.asarray(start_q, dtype=float)
        abort_requested = abort_requested or (lambda: False)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise TrajectoryValidationError("Cartesian path must contain at least two 3D positions")
        if axes.shape != positions.shape:
            raise TrajectoryValidationError("A Tool-Z axis is required for every Cartesian position")
        if start_q.shape != (self._dof,):
            raise TrajectoryValidationError("Start joint position has the wrong dimension")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(axes)):
            raise TrajectoryValidationError("Cartesian path contains non-finite values")
        axis_norm = np.linalg.norm(axes, axis=1)
        if np.any(axis_norm < 1e-9):
            raise TrajectoryValidationError("Cartesian path contains an invalid Tool-Z axis")
        axes = axes / axis_norm[:, None]

        try:
            current_position, current_rotation = self.tip_state(start_q)
            approach_positions, approach_axes, approach_distance = self._approach_samples(
                current_position, current_rotation[:, 2], positions[0], axes[0]
            )
            q_approach_tail, approach_position_error, approach_axis_error = self._continuous_ik(
                approach_positions, approach_axes, start_q, abort_requested, "Approach"
            )
            q_approach = np.vstack((start_q[None, :], q_approach_tail))

            q_task_tail, task_position_error, task_axis_error = self._continuous_ik(
                positions[1:], axes[1:], q_approach[-1], abort_requested, "Task"
            )
            q_task = np.vstack((q_approach[-1][None, :], q_task_tail))
            full_path = np.vstack((q_approach, q_task[1:]))
            if np.any(full_path < self._lower[None, :]) or np.any(full_path > self._upper[None, :]):
                raise TrajectoryValidationError("Joint position limit violation")
            maximum_step = float(np.max(np.abs(np.diff(full_path, axis=0))))
            if maximum_step > self._max_joint_step:
                raise TrajectoryValidationError(
                    "Joint continuity check failed at {:.3f} rad".format(maximum_step)
                )
            minimum_sv = self._collision_and_singularity_checks(full_path)

            task_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
            approach_minimum = approach_distance / max(self._approach_speed, 1e-4)
            task_minimum = task_length / max(self._task_speed, 1e-4)
            approach = self.time_parameterize(q_approach, approach_minimum)
            task = self.time_parameterize(q_task, task_minimum)
            return {
                "start": start_q.copy(),
                "approach": approach,
                "task": task,
                "metrics": {
                    "approach_points": len(q_approach),
                    "task_points": len(q_task),
                    "approach_duration_s": approach["duration"],
                    "task_duration_s": task["duration"],
                    "approach_distance_m": approach_distance,
                    "task_length_m": task_length,
                    "maximum_joint_step_rad": maximum_step,
                    "minimum_jacobian_singular_value": minimum_sv,
                    "maximum_ik_position_error_m": max(
                        approach_position_error + task_position_error
                    ),
                    "maximum_ik_tool_z_error_deg": math.degrees(
                        max(approach_axis_error + task_axis_error)
                    ),
                },
            }
        finally:
            self._set_q(start_q)
