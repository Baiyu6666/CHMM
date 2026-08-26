"""Shared continuous IK and joint-trajectory time parameterization for iiwa14."""

from __future__ import annotations

import math

import numpy as np


class TrajectoryValidationError(RuntimeError):
    pass


class CartesianTrajectoryCompiler:
    """Compile Cartesian samples into two timed joint segments.

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
        velocity_scale=0.20,
        acceleration_limit=1.00,
        approach_speed=0.06,
        task_speed=0.04,
        approach_spacing=0.01,
        approach_axis_spacing=math.radians(2.0),
        approach_joint_bridge_limit=3.00,
        approach_clearance_z=None,
        minimum_approach_z=0.20,
        position_tolerance=0.002,
        approach_position_tolerance=0.005,
        tool_z_tolerance=math.radians(2.0),
        tool_x_tolerance=math.radians(2.0),
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
        self.set_task_speeds(approach_speed, task_speed)
        self._approach_spacing = float(approach_spacing)
        self._approach_axis_spacing = float(approach_axis_spacing)
        self.set_approach_joint_bridge_limit(approach_joint_bridge_limit)
        self._approach_clearance_z = (
            None if approach_clearance_z is None else float(approach_clearance_z)
        )
        self._minimum_approach_z = float(minimum_approach_z)
        if not math.isfinite(self._minimum_approach_z):
            raise ValueError("minimum_approach_z must be finite")
        self._position_tolerance = float(position_tolerance)
        self._tool_z_tolerance = float(tool_z_tolerance)
        self._tool_x_tolerance = float(tool_x_tolerance)
        self._minimum_singular_value = float(minimum_singular_value)
        self._first_point_delay = float(first_point_delay)
        if (
            not math.isfinite(self._position_tolerance)
            or self._position_tolerance <= 0.0
            or not math.isfinite(self._tool_z_tolerance)
            or self._tool_z_tolerance <= 0.0
            or not math.isfinite(self._tool_x_tolerance)
            or self._tool_x_tolerance <= 0.0
        ):
            raise ValueError("IK tolerances must be positive and finite")
        self.set_approach_position_tolerance(approach_position_tolerance)

    def set_task_speeds(self, approach_speed, task_speed):
        """Update task-level speeds without changing hardware safety limits."""
        approach_speed = float(approach_speed)
        task_speed = float(task_speed)
        if (
            not math.isfinite(approach_speed)
            or not math.isfinite(task_speed)
            or approach_speed <= 0.0
            or task_speed <= 0.0
        ):
            raise ValueError("Task speeds must be positive and finite")
        self._approach_speed = approach_speed
        self._task_speed = task_speed

    def set_approach_position_tolerance(self, tolerance):
        """Update only the free-space move-to-start IK tolerance."""
        tolerance = float(tolerance)
        if (
            not math.isfinite(tolerance)
            or tolerance < self._position_tolerance
        ):
            raise ValueError(
                "Approach position tolerance must be finite and no tighter "
                "than the task tolerance"
            )
        self._approach_position_tolerance = tolerance

    def set_approach_joint_bridge_limit(self, limit):
        """Set the largest Stage-0 IK branch change that may be interpolated."""
        limit = float(limit)
        if not math.isfinite(limit) or limit < self._max_joint_step:
            raise ValueError(
                "Approach joint bridge limit must be finite and no smaller "
                "than the standard joint step limit"
            )
        self._approach_joint_bridge_limit = limit

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

    def _self_collision_pair(self):
        self._bullet.performCollisionDetection(physicsClientId=self._physics)
        for contact in self._bullet.getContactPoints(
            bodyA=self._robot,
            bodyB=self._robot,
            physicsClientId=self._physics,
        ):
            link_a = int(contact[3])
            link_b = int(contact[4])
            if abs(link_a - link_b) > 1 and float(contact[8]) < -0.001:
                return link_a, link_b
        return None

    def tool_z_from_quaternion(self, quaternion):
        return self.tool_basis_from_quaternion(quaternion)[1]

    def tool_basis_from_quaternion(self, quaternion):
        values = np.asarray(quaternion, dtype=float)
        if values.shape != (4,):
            raise TrajectoryValidationError("Quaternion must contain four values")
        norm = float(np.linalg.norm(values))
        if not np.all(np.isfinite(values)) or norm < 1e-9:
            raise TrajectoryValidationError("Path contains an invalid quaternion")
        rotation = np.asarray(
            self._bullet.getMatrixFromQuaternion((values / norm).tolist()), dtype=float
        ).reshape(3, 3)
        return rotation[:, 0], rotation[:, 2]

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

    def _continuous_ik(
        self,
        positions,
        axes,
        seed,
        abort_requested,
        phase_name,
        position_tolerance=None,
        final_position_tolerance=None,
        max_joint_step=None,
        x_axes=None,
        x_active=None,
    ):
        position_tolerance = (
            self._position_tolerance
            if position_tolerance is None
            else float(position_tolerance)
        )
        final_position_tolerance = (
            position_tolerance
            if final_position_tolerance is None
            else float(final_position_tolerance)
        )
        if (
            not math.isfinite(position_tolerance)
            or position_tolerance <= 0.0
            or not math.isfinite(final_position_tolerance)
            or final_position_tolerance <= 0.0
        ):
            raise ValueError("IK position tolerances must be positive and finite")
        max_joint_step = (
            self._max_joint_step
            if max_joint_step is None
            else float(max_joint_step)
        )
        if not math.isfinite(max_joint_step) or max_joint_step <= 0.0:
            raise ValueError("IK joint step limit must be positive and finite")
        previous = np.asarray(seed, dtype=float).copy()
        _, rotation = self.tip_state(previous)
        reference_x = rotation[:, 0]
        if x_axes is not None:
            x_axes = np.asarray(x_axes, dtype=float)
            if x_axes.shape != np.asarray(axes).shape:
                raise TrajectoryValidationError(
                    "A Tool-X axis is required for every constrained orientation"
                )
            if x_active is None:
                x_active = np.ones(len(x_axes), dtype=bool)
            else:
                x_active = np.asarray(x_active, dtype=bool)
                if x_active.shape != (len(x_axes),):
                    raise TrajectoryValidationError(
                        "The Tool-X active mask must match the Cartesian samples"
                    )
        elif x_active is not None and np.any(np.asarray(x_active, dtype=bool)):
            raise TrajectoryValidationError(
                "Tool-X cannot be active without a Tool-X axis"
            )
        trajectory, position_errors, axis_errors, x_errors = [], [], [], []
        ranges = self._upper - self._lower
        center = 0.5 * (self._lower + self._upper)
        for sample_index, (target_position, target_z) in enumerate(zip(positions, axes)):
            preferred_x = None if x_axes is None else x_axes[sample_index]
            target_x = (
                preferred_x
                if preferred_x is not None and bool(x_active[sample_index])
                else None
            )
            sample_position_tolerance = (
                final_position_tolerance
                if sample_index == len(positions) - 1
                else position_tolerance
            )
            if abort_requested():
                raise TrajectoryValidationError("Trajectory compilation aborted")
            rest = 0.98 * previous + 0.02 * center
            rest = np.clip(rest, previous - 0.03, previous + 0.03)
            best = None
            accepted = None
            nearest_orientation = None
            rejected = None
            reference_tangent = reference_x - float(reference_x @ target_z) * target_z
            if np.linalg.norm(reference_tangent) < 1e-8:
                reference_tangent = np.eye(3)[int(np.argmin(np.abs(target_z)))]
                reference_tangent -= float(reference_tangent @ target_z) * target_z
            reference_tangent /= np.linalg.norm(reference_tangent)
            if target_x is not None:
                target_tangent = target_x - float(target_x @ target_z) * target_z
                target_tangent /= np.linalg.norm(target_tangent)
                orientation_candidates = [
                    math.cos(math.radians(degrees)) * target_tangent
                    + math.sin(math.radians(degrees)) * np.cross(
                        target_z, target_tangent
                    )
                    for degrees in (
                        0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5,
                        2.0, -2.0, 3.0, -3.0, 5.0, -5.0,
                        7.5, -7.5, 10.0, -10.0, 15.0, -15.0,
                        20.0, -20.0, 30.0, -30.0,
                    )
                ]
            elif preferred_x is not None:
                preferred_tangent = preferred_x - float(preferred_x @ target_z) * target_z
                preferred_tangent /= np.linalg.norm(preferred_tangent)
                yaw_delta = math.atan2(
                    float(target_z @ np.cross(reference_tangent, preferred_tangent)),
                    float(reference_tangent @ preferred_tangent),
                )
                # Follow the planner's nominal yaw when it is continuous.  If
                # that would jump IK branches, progressively retreat toward
                # the current physical Tool-X direction.  This keeps yaw free
                # in inactive stages while using them to approach the next
                # active stage smoothly.
                orientation_candidates = [
                    math.cos(fraction * yaw_delta) * reference_tangent
                    + math.sin(fraction * yaw_delta) * np.cross(
                        target_z, reference_tangent
                    )
                    for fraction in (1.0, 0.75, 0.5, 0.25, 0.0)
                ]
                orientation_candidates.extend(
                    math.cos(math.radians(degrees)) * reference_tangent
                    + math.sin(math.radians(degrees)) * np.cross(
                        target_z, reference_tangent
                    )
                    for degrees in (
                        5, -5, 10, -10, 20, -20, 30, -30, 45, -45,
                        60, -60, 90, -90, 120, -120, 150, -150, 180,
                    )
                )
            else:
                orientation_candidates = [
                    math.cos(math.radians(degrees)) * reference_tangent
                    + math.sin(math.radians(degrees)) * np.cross(
                        target_z, reference_tangent
                    )
                    for degrees in (
                        0, 5, -5, 10, -10, 20, -20, 30, -30, 45, -45,
                        60, -60, 90, -90, 120, -120, 150, -150, 180,
                    )
                ]
            for tangent_x in orientation_candidates:
                quaternion = self._quaternion_from_basis(tangent_x, target_z)
                rest_candidates = [
                    rest,
                    0.75 * previous + 0.25 * center,
                    center,
                ]
                for candidate_rest in rest_candidates:
                    self._set_q(previous)
                    solution = self._bullet.calculateInverseKinematics(
                        self._robot,
                        self._tip_index,
                        target_position.tolist(),
                        quaternion,
                        lowerLimits=self._lower.tolist(),
                        upperLimits=self._upper.tolist(),
                        jointRanges=ranges.tolist(),
                        restPoses=candidate_rest.tolist(),
                        jointDamping=[0.03] * self._dof,
                        maxNumIterations=600,
                        residualThreshold=1e-8,
                        physicsClientId=self._physics,
                    )
                    candidate = np.asarray(solution[:self._dof], dtype=float)
                    actual_position, actual_rotation = self.tip_state(candidate)
                    position_error = float(
                        np.linalg.norm(actual_position - target_position)
                    )
                    axis_error = math.acos(
                        float(np.clip(actual_rotation[:, 2] @ target_z, -1.0, 1.0))
                    )
                    preferred_x_error = (
                        0.0
                        if preferred_x is None
                        else math.acos(
                            float(
                                np.clip(
                                    actual_rotation[:, 0] @ preferred_x,
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                    x_error = preferred_x_error if target_x is not None else 0.0
                    step = float(np.max(np.abs(candidate - previous)))
                    margin = float(
                        np.min(
                            np.minimum(
                                candidate - self._lower,
                                self._upper - candidate,
                            )
                        )
                    )
                    collision_pair = self._self_collision_pair()
                    if margin < 0.0 or collision_pair is not None:
                        rejection = (
                            max(-margin, 0.0),
                            candidate,
                            margin,
                            collision_pair,
                            step,
                        )
                        if rejected is None or rejection[0] < rejected[0]:
                            rejected = rejection
                        continue
                    score = (
                        position_error * 1e6
                        + axis_error * 1e4
                        + x_error * 1e4
                        + (0.0 if target_x is not None else preferred_x_error * 2.0)
                        + step * 10.0
                        - margin
                    )
                    candidate_record = (
                        score,
                        candidate,
                        actual_rotation,
                        position_error,
                        axis_error,
                        x_error,
                        step,
                    )
                    if best is None or score < best[0]:
                        best = candidate_record
                    if (
                        nearest_orientation is None
                        and position_error <= sample_position_tolerance
                        and axis_error <= self._tool_z_tolerance
                        and step <= max_joint_step
                    ):
                        nearest_orientation = candidate_record
                    if (
                        position_error <= sample_position_tolerance
                        and axis_error <= self._tool_z_tolerance
                        and x_error <= self._tool_x_tolerance
                        and step <= max_joint_step
                    ):
                        accepted = candidate_record
                        break
                if accepted is not None:
                    break
            if best is None:
                rejection_suffix = ""
                if rejected is not None:
                    rejection_suffix = (
                        "; closest candidate margin {:.4f} rad, "
                        "collision {}, step {:.3f} rad"
                    ).format(rejected[2], rejected[3], rejected[4])
                raise TrajectoryValidationError(
                    "{} IK found no collision-free in-limit solution at sample {}/{}{}".format(
                        phase_name,
                        sample_index + 1,
                        len(positions),
                        rejection_suffix,
                    )
                )
            (
                _,
                candidate,
                actual_rotation,
                position_error,
                axis_error,
                x_error,
                step,
            ) = (
                accepted
                if accepted is not None
                else nearest_orientation
                if nearest_orientation is not None
                else best
            )
            if (
                position_error > sample_position_tolerance
                or axis_error > self._tool_z_tolerance
                or x_error > self._tool_x_tolerance
            ):
                tool_x_suffix = (
                    ""
                    if target_x is None
                    else ", Tool-X error {:.2f} deg".format(math.degrees(x_error))
                )
                raise TrajectoryValidationError(
                    "{} IK failed at sample {}/{}: position error {:.4f} m "
                    "(limit {:.4f} m), Tool-Z error {:.2f} deg{}".format(
                        phase_name,
                        sample_index + 1,
                        len(positions),
                        position_error,
                        sample_position_tolerance,
                        math.degrees(axis_error),
                        tool_x_suffix,
                    )
                )
            if step > max_joint_step:
                raise TrajectoryValidationError(
                    "{} IK branch jump at sample {}/{}: {:.3f} rad exceeds {:.3f}".format(
                        phase_name,
                        sample_index + 1,
                        len(positions),
                        step,
                        max_joint_step,
                    )
                )
            trajectory.append(candidate)
            position_errors.append(position_error)
            axis_errors.append(axis_error)
            x_errors.append(x_error)
            previous = candidate
            reference_x = actual_rotation[:, 0]
        return np.asarray(trajectory), position_errors, axis_errors, x_errors

    @staticmethod
    def _densify_joint_path(q_path, maximum_step):
        """Linearly bridge accepted Stage-0 IK changes before validation/timing."""
        q_path = np.asarray(q_path, dtype=float)
        dense = [q_path[0].copy()]
        for target in q_path[1:]:
            start = dense[-1]
            segment_count = max(
                1,
                int(math.ceil(float(np.max(np.abs(target - start))) / maximum_step)),
            )
            dense.extend(
                start + (target - start) * (index / float(segment_count))
                for index in range(1, segment_count + 1)
            )
        return np.asarray(dense)

    def _collision_and_singularity_checks(self, q_path, phase_name):
        minimum_sv = math.inf
        for sample_index, q in enumerate(q_path):
            self._set_q(q)
            collision_pair = self._self_collision_pair()
            if collision_pair is not None:
                raise TrajectoryValidationError(
                    "{} self-collision at joint sample {}/{} between links {} and {}".format(
                        phase_name,
                        sample_index + 1,
                        len(q_path),
                        collision_pair[0],
                        collision_pair[1],
                    )
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
                "{} approaches a singularity (minimum singular value {:.4f})".format(
                    phase_name, minimum_sv
                )
            )
        return minimum_sv

    def _approach_workspace_checks(self, q_path):
        tcp_positions = []
        for q in q_path:
            position, _ = self.tip_state(q)
            tcp_positions.append(np.asarray(position, dtype=float))
        tcp_positions = np.asarray(tcp_positions)
        minimum_z = float(np.min(tcp_positions[:, 2]))
        initial_position = tcp_positions[0]
        initial_z = float(initial_position[2])
        if initial_z >= self._minimum_approach_z - 1e-6:
            if minimum_z >= self._minimum_approach_z - 1e-6:
                return minimum_z
            raise TrajectoryValidationError(
                "Stage-0 joint bridge drops TCP to {:.4f} m, below the safe "
                "minimum {:.4f} m".format(
                    minimum_z, self._minimum_approach_z
                )
            )

        # A TCP that is already below the normal transit floor must be allowed
        # to recover vertically. It may not descend further, sweep laterally at
        # low height, or fall below the floor again after reaching it.
        if minimum_z < initial_z - 0.002:
            raise TrajectoryValidationError(
                "Stage-0 vertical recovery drops TCP from {:.4f} m to {:.4f} m".format(
                    initial_z, minimum_z
                )
            )
        recovered = np.flatnonzero(
            tcp_positions[:, 2] >= self._minimum_approach_z - 1e-6
        )
        if len(recovered) == 0:
            raise TrajectoryValidationError(
                "Stage-0 vertical recovery never reaches the safe minimum {:.4f} m".format(
                    self._minimum_approach_z
                )
            )
        recovery_end = int(recovered[0])
        lateral_motion = np.linalg.norm(
            tcp_positions[: recovery_end + 1, :2] - initial_position[None, :2],
            axis=1,
        )
        lateral_limit = max(0.01, 2.0 * self._approach_position_tolerance)
        maximum_lateral = float(np.max(lateral_motion))
        if maximum_lateral > lateral_limit:
            raise TrajectoryValidationError(
                "Stage-0 vertical recovery moves TCP laterally {:.4f} m before "
                "reaching the safe height (limit {:.4f} m)".format(
                    maximum_lateral, lateral_limit
                )
            )
        if float(np.min(tcp_positions[recovery_end:, 2])) < self._minimum_approach_z - 1e-6:
            raise TrajectoryValidationError(
                "Stage-0 TCP falls below the safe minimum again after vertical recovery"
            )
        return minimum_z

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

    def _approach_samples(
        self,
        current_position,
        current_axis,
        target_position,
        target_axis,
        current_x=None,
        target_x=None,
    ):
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
                target_position,
            ]
        )
        compact = [waypoints[0]]
        for waypoint in waypoints[1:]:
            if float(np.linalg.norm(waypoint - compact[-1])) > 1e-9:
                compact.append(waypoint)
        if len(compact) == 1:
            x_axes = (
                None
                if target_x is None
                else np.asarray(target_x, dtype=float).reshape(1, 3)
            )
            return (
                target_position[None, :],
                np.asarray(target_axis, dtype=float)[None, :],
                x_axes,
                0.0,
            )

        waypoints = np.asarray(compact, dtype=float)
        edge_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
        distance = float(cumulative[-1])
        current_axis = np.asarray(current_axis, dtype=float)
        target_axis = np.asarray(target_axis, dtype=float)
        axis_angle = math.acos(float(np.clip(current_axis @ target_axis, -1.0, 1.0)))
        sample_count = max(
            1,
            int(math.ceil(distance / self._approach_spacing)),
            int(math.ceil(axis_angle / self._approach_axis_spacing)),
        )
        targets = np.unique(
            np.concatenate(
                (np.linspace(0.0, distance, sample_count + 1)[1:], cumulative[1:])
            )
        )
        positions = np.column_stack(
            [np.interp(targets, cumulative, waypoints[:, dim]) for dim in range(3)]
        )
        axes = self._interpolate_axis(current_axis, target_axis, targets / distance)
        x_axes = None
        if target_x is not None:
            if current_x is None:
                raise TrajectoryValidationError(
                    "Current Tool-X is required for a full-orientation approach"
                )
            x_axes = self._interpolate_axis(current_x, target_x, targets / distance)
            orthogonal = []
            for x_axis, z_axis in zip(x_axes, axes):
                x_axis = x_axis - float(x_axis @ z_axis) * z_axis
                if np.linalg.norm(x_axis) <= 1e-8:
                    raise TrajectoryValidationError(
                        "Full-orientation approach crosses a Tool-X singularity"
                    )
                orthogonal.append(x_axis / np.linalg.norm(x_axis))
            x_axes = np.asarray(orthogonal)
        return positions, axes, x_axes, distance

    def compile(
        self,
        positions,
        tool_z_axes,
        start_q,
        abort_requested=None,
        tool_x_axes=None,
        tool_x_active=None,
    ):
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
        x_axes = None
        x_active = None
        if tool_x_axes is not None:
            x_axes = np.asarray(tool_x_axes, dtype=float)
            if x_axes.shape != positions.shape or not np.all(np.isfinite(x_axes)):
                raise TrajectoryValidationError(
                    "A finite Tool-X axis is required for every Cartesian position"
                )
            orthogonal = []
            for x_axis, z_axis in zip(x_axes, axes):
                x_axis = x_axis - float(x_axis @ z_axis) * z_axis
                if np.linalg.norm(x_axis) <= 1e-8:
                    raise TrajectoryValidationError(
                        "Cartesian path contains a Tool-X axis parallel to Tool-Z"
                    )
                orthogonal.append(x_axis / np.linalg.norm(x_axis))
            x_axes = np.asarray(orthogonal)
            if tool_x_active is None:
                x_active = np.ones(len(x_axes), dtype=bool)
            else:
                x_active = np.asarray(tool_x_active, dtype=bool)
                if x_active.shape != (len(x_axes),):
                    raise TrajectoryValidationError(
                        "The Tool-X active mask must match the Cartesian path"
                    )
        elif tool_x_active is not None and np.any(
            np.asarray(tool_x_active, dtype=bool)
        ):
            raise TrajectoryValidationError(
                "Tool-X cannot be active without a Tool-X path"
            )

        try:
            current_position, current_rotation = self.tip_state(start_q)
            approach_positions, approach_axes, approach_x_axes, approach_distance = self._approach_samples(
                current_position,
                current_rotation[:, 2],
                positions[0],
                axes[0],
                current_x=(
                    None if x_axes is None else current_rotation[:, 0]
                ),
                target_x=None if x_axes is None else x_axes[0],
            )
            (
                q_approach_tail,
                approach_position_error,
                approach_axis_error,
                approach_x_error,
            ) = self._continuous_ik(
                approach_positions,
                approach_axes,
                start_q,
                abort_requested,
                "Approach",
                position_tolerance=self._approach_position_tolerance,
                final_position_tolerance=self._position_tolerance,
                max_joint_step=self._approach_joint_bridge_limit,
                x_axes=approach_x_axes,
                x_active=(
                    None
                    if approach_x_axes is None
                    else np.zeros(len(approach_x_axes), dtype=bool)
                ),
            )
            q_approach = np.vstack((start_q[None, :], q_approach_tail))
            q_approach = self._densify_joint_path(
                q_approach, self._max_joint_step
            )
            minimum_approach_z = self._approach_workspace_checks(q_approach)

            (
                q_task_tail,
                task_position_error,
                task_axis_error,
                task_x_error,
            ) = self._continuous_ik(
                positions[1:],
                axes[1:],
                q_approach[-1],
                abort_requested,
                "Task",
                x_axes=None if x_axes is None else x_axes[1:],
                x_active=None if x_active is None else x_active[1:],
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
            minimum_sv = min(
                self._collision_and_singularity_checks(q_approach, "Approach"),
                self._collision_and_singularity_checks(q_task, "Task"),
            )

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
                    "minimum_approach_tcp_z_m": minimum_approach_z,
                    "maximum_ik_position_error_m": max(
                        approach_position_error + task_position_error
                    ),
                    "maximum_ik_tool_z_error_deg": math.degrees(
                        max(approach_axis_error + task_axis_error)
                    ),
                    "maximum_ik_tool_x_error_deg": math.degrees(
                        max(approach_x_error + task_x_error)
                    ),
                },
            }
        finally:
            self._set_q(start_q)
