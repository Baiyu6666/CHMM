"""Shared continuous IK and joint-trajectory time parameterization for iiwa14."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import make_interp_spline


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
                # PyBullet's damped IK can converge repeatedly to the same
                # colliding or high-residual iiwa redundancy branch even when
                # another in-limit branch exists.  Keep the fast local seeds
                # first, then add deterministic, symmetric nullspace-biased
                # rest poses.  These are only search seeds: every returned
                # solution still has to pass pose, continuity, limit and
                # collision checks below.
                alternating = np.where(
                    np.arange(self._dof) % 2 == 0, 1.0, -1.0
                )
                sparse = np.zeros(self._dof, dtype=float)
                sparse[::2] = alternating[::2]
                for pattern in (alternating, sparse):
                    offset = 0.15 * ranges * pattern
                    rest_candidates.extend(
                        (
                            np.clip(previous + offset, self._lower, self._upper),
                            np.clip(previous - offset, self._lower, self._upper),
                        )
                    )
                rest_candidates.append(
                    np.clip(2.0 * center - previous, self._lower, self._upper)
                )
                low_discrepancy_steps = np.asarray(
                    [
                        0.6180339887,
                        0.4142135624,
                        0.7320508076,
                        0.2679491924,
                        0.5772156649,
                        0.3535533906,
                        0.7861513778,
                    ][: self._dof],
                    dtype=float,
                )
                for seed_index in range(1, 9):
                    fractions = np.mod(
                        float(seed_index) * low_discrepancy_steps,
                        1.0,
                    )
                    # Stay away from the hard joint limits while covering
                    # distinct iiwa elbow/swivel basins deterministically.
                    fractions = 0.10 + 0.80 * fractions
                    rest_candidates.append(
                        self._lower + fractions * ranges
                    )
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

    def _collision_checks(self, q_path, phase_name):
        """Reject self-collision without applying Cartesian singularity gates."""
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

    def _approach_workspace_checks(self, q_path, enforce_transit_floor=True):
        tcp_positions = []
        for q in q_path:
            position, _ = self.tip_state(q)
            tcp_positions.append(np.asarray(position, dtype=float))
        tcp_positions = np.asarray(tcp_positions)
        minimum_z = float(np.min(tcp_positions[:, 2]))
        if not enforce_transit_floor:
            return minimum_z
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

    @staticmethod
    def _approach_circle_geometry(obstacle):
        try:
            center = np.asarray(obstacle["center"], dtype=float).reshape(3)
            normal = np.asarray(obstacle["table_normal"], dtype=float).reshape(3)
            radius = float(obstacle["radius"])
            clearance = float(obstacle["clearance"])
            margin = float(obstacle.get("margin", 0.0))
        except (KeyError, TypeError, ValueError) as error:
            raise TrajectoryValidationError(
                "Stage-0 obstacle geometry is incomplete or invalid"
            ) from error
        normal_norm = float(np.linalg.norm(normal))
        if (
            str(obstacle.get("type", "circle")) != "circle"
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(normal))
            or normal_norm <= 1e-12
            or not all(math.isfinite(value) for value in (radius, clearance, margin))
            or radius <= 0.0
            or clearance < 0.0
            or margin < 0.0
        ):
            raise TrajectoryValidationError(
                "Stage-0 obstacle geometry must be finite with non-negative clearance"
            )
        return center, normal / normal_norm, radius, clearance, margin

    @staticmethod
    def _approach_capsule_geometry(obstacle):
        try:
            endpoints = np.asarray(obstacle["endpoints"], dtype=float).reshape(2, 3)
            normal = np.asarray(obstacle["table_normal"], dtype=float).reshape(3)
            radius = float(obstacle["radius"])
            clearance = float(obstacle["clearance"])
            margin = float(obstacle.get("margin", 0.0))
        except (KeyError, TypeError, ValueError) as error:
            raise TrajectoryValidationError(
                "Stage-0 capsule geometry is incomplete or invalid"
            ) from error
        normal_norm = float(np.linalg.norm(normal))
        if (
            str(obstacle.get("type")) != "capsule"
            or not np.all(np.isfinite(endpoints))
            or not np.all(np.isfinite(normal))
            or normal_norm <= 1e-12
            or not all(math.isfinite(value) for value in (radius, clearance, margin))
            or radius <= 0.0
            or clearance < 0.0
            or margin < 0.0
        ):
            raise TrajectoryValidationError(
                "Stage-0 capsule geometry must be finite with non-negative clearance"
            )
        normal = normal / normal_norm
        delta = endpoints[1] - endpoints[0]
        planar_delta = delta - float(delta @ normal) * normal
        if float(np.linalg.norm(planar_delta)) <= 1e-9:
            raise TrajectoryValidationError(
                "Stage-0 capsule endpoints must be distinct in the table plane"
            )
        return endpoints, normal, radius, clearance, margin

    @staticmethod
    def _point_segment_distances(points, start, end):
        points = np.asarray(points, dtype=float)
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        delta = end - start
        denominator = float(delta @ delta)
        if denominator <= 1e-18:
            return np.linalg.norm(points - start, axis=-1)
        ratios = np.clip(((points - start) @ delta) / denominator, 0.0, 1.0)
        closest = start + ratios[..., None] * delta
        return np.linalg.norm(points - closest, axis=-1)

    @classmethod
    def _capsule_planar_distances(cls, positions, endpoints, normal):
        positions = np.asarray(positions, dtype=float)
        origin = endpoints[0]
        relative = positions - origin
        planar_positions = relative - (relative @ normal)[..., None] * normal
        endpoint_delta = endpoints[1] - origin
        planar_endpoint = endpoint_delta - float(endpoint_delta @ normal) * normal
        return cls._point_segment_distances(
            planar_positions, np.zeros(3), planar_endpoint
        )

    def _approach_obstacle_checks(self, q_path, obstacle):
        if obstacle is None:
            return None
        obstacle_type = str(obstacle.get("type", "circle"))
        positions = []
        for q in np.asarray(q_path, dtype=float):
            positions.append(np.asarray(self.tip_state(q)[0], dtype=float))
        positions = np.asarray(positions, dtype=float)
        if obstacle_type == "circle":
            center, normal, radius, clearance, _margin = self._approach_circle_geometry(
                obstacle
            )
            relative = positions - center
            planar = relative - (relative @ normal)[:, None] * normal
            surface_distances = np.linalg.norm(planar, axis=1)
        elif obstacle_type == "capsule":
            endpoints, normal, radius, clearance, _margin = (
                self._approach_capsule_geometry(obstacle)
            )
            surface_distances = self._capsule_planar_distances(
                positions, endpoints, normal
            )
        else:
            raise TrajectoryValidationError(
                "Stage-0 obstacle type must be circle or capsule"
            )
        minimum_clearance = float(np.min(surface_distances)) - radius
        if minimum_clearance < -1e-6:
            raise TrajectoryValidationError(
                "Approach joint interpolation enters the physical obstacle"
            )
        required_radius = radius + clearance
        safe = np.flatnonzero(surface_distances >= required_radius - 1e-6)
        if len(safe) == 0:
            raise TrajectoryValidationError(
                "Approach joint interpolation never reaches obstacle clearance"
            )
        first_safe = int(safe[0])
        if np.any(np.diff(surface_distances[: first_safe + 1]) < -1e-6):
            raise TrajectoryValidationError(
                "Approach moves closer to the obstacle before clearing its envelope"
            )
        after_egress = surface_distances[first_safe:]
        minimum_after_egress = float(np.min(after_egress))
        if minimum_after_egress < required_radius - 1e-6:
            violation_index = first_safe + int(np.argmin(after_egress))
            raise TrajectoryValidationError(
                "Approach joint interpolation violates obstacle clearance after egress "
                "at sample {}/{}: {:.4f} m clearance, requires {:.4f} m".format(
                    violation_index + 1,
                    len(surface_distances),
                    minimum_after_egress - radius,
                    clearance,
                )
            )
        return minimum_clearance

    @staticmethod
    def _segment_point_distance(start, end, point):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        point = np.asarray(point, dtype=float)
        delta = end - start
        denominator = float(delta @ delta)
        if denominator <= 1e-18:
            return float(np.linalg.norm(start - point))
        ratio = float(np.clip(((point - start) @ delta) / denominator, 0.0, 1.0))
        return float(np.linalg.norm(start + ratio * delta - point))

    def _obstacle_avoiding_approach_waypoints(
        self,
        current_position,
        target_position,
        obstacle,
        *,
        use_long_arc=False,
    ):
        if str(obstacle.get("type", "circle")) == "capsule":
            return self._obstacle_avoiding_capsule_waypoints(
                current_position,
                target_position,
                obstacle,
                use_long_arc=use_long_arc,
            )
        current_position = np.asarray(current_position, dtype=float).reshape(3)
        target_position = np.asarray(target_position, dtype=float).reshape(3)
        center, normal, radius, clearance, margin = self._approach_circle_geometry(
            obstacle
        )
        reference = np.asarray([1.0, 0.0, 0.0])
        if abs(float(reference @ normal)) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0])
        plane_x = np.cross(normal, reference)
        plane_x /= np.linalg.norm(plane_x)
        plane_y = np.cross(normal, plane_x)

        def planar_coordinates(position):
            relative = np.asarray(position, dtype=float) - center
            return np.asarray([relative @ plane_x, relative @ plane_y], dtype=float)

        current_planar = planar_coordinates(current_position)
        target_planar = planar_coordinates(target_position)
        required_radius = radius + clearance
        chord_compensation = (
            self._approach_spacing * self._approach_spacing
            / max(8.0 * required_radius, 1e-9)
        )
        bypass_radius = max(
            required_radius + margin,
            required_radius + chord_compensation + 1e-6,
        )
        current_radius = float(np.linalg.norm(current_planar))
        target_radius = float(np.linalg.norm(target_planar))
        if current_radius <= radius + 1e-4:
            raise TrajectoryValidationError(
                "Stage-0 starts inside the tracked obstacle"
            )
        if target_radius < required_radius - 1e-6:
            raise TrajectoryValidationError(
                "Task start violates Stage-0 obstacle clearance by {:.4f} m".format(
                    required_radius - target_radius
                )
            )
        if self._segment_point_distance(
            current_planar, target_planar, np.zeros(2)
        ) >= required_radius - 1e-9:
            return np.vstack((current_position, target_position))

        current_angle = math.atan2(current_planar[1], current_planar[0])
        target_angle = math.atan2(target_planar[1], target_planar[0])
        angle_delta = math.atan2(
            math.sin(target_angle - current_angle),
            math.cos(target_angle - current_angle),
        )
        if use_long_arc and abs(angle_delta) > 1e-9:
            angle_delta -= math.copysign(2.0 * math.pi, angle_delta)
        arc_length = abs(angle_delta) * bypass_radius
        arc_segments = max(2, int(math.ceil(arc_length / self._approach_spacing)))
        arc_angles = np.linspace(
            current_angle, current_angle + angle_delta, arc_segments + 1
        )
        planar_waypoints = [current_planar]
        current_boundary = bypass_radius * np.asarray(
            [math.cos(current_angle), math.sin(current_angle)]
        )
        if np.linalg.norm(current_boundary - planar_waypoints[-1]) > 1e-9:
            planar_waypoints.append(current_boundary)
        for angle in arc_angles[1:]:
            planar_waypoints.append(
                bypass_radius * np.asarray([math.cos(angle), math.sin(angle)])
            )
        if np.linalg.norm(target_planar - planar_waypoints[-1]) > 1e-9:
            planar_waypoints.append(target_planar)

        planar_waypoints = np.asarray(planar_waypoints, dtype=float)
        planar_edges = np.linalg.norm(np.diff(planar_waypoints, axis=0), axis=1)
        planar_distance = float(np.sum(planar_edges))
        if planar_distance <= 1e-12:
            return np.vstack((current_position, target_position))
        progress = np.concatenate(([0.0], np.cumsum(planar_edges))) / planar_distance
        current_height = float((current_position - center) @ normal)
        target_height = float((target_position - center) @ normal)
        heights = current_height + progress * (target_height - current_height)
        return (
            center[None, :]
            + planar_waypoints[:, :1] * plane_x[None, :]
            + planar_waypoints[:, 1:] * plane_y[None, :]
            + heights[:, None] * normal[None, :]
        )

    @staticmethod
    def _segments_intersect_2d(first_start, first_end, second_start, second_end):
        def cross(first, second):
            return float(first[0] * second[1] - first[1] * second[0])

        first_delta = first_end - first_start
        second_delta = second_end - second_start
        denominator = cross(first_delta, second_delta)
        offset = second_start - first_start
        if abs(denominator) <= 1e-12:
            if abs(cross(offset, first_delta)) > 1e-12:
                return False
            axis = int(np.argmax(np.abs(first_delta)))
            if abs(first_delta[axis]) <= 1e-12:
                return bool(np.linalg.norm(first_start - second_start) <= 1e-12)
            first_bounds = sorted((first_start[axis], first_end[axis]))
            second_bounds = sorted((second_start[axis], second_end[axis]))
            return max(first_bounds[0], second_bounds[0]) <= min(
                first_bounds[1], second_bounds[1]
            ) + 1e-12
        first_ratio = cross(offset, second_delta) / denominator
        second_ratio = cross(offset, first_delta) / denominator
        return (
            -1e-12 <= first_ratio <= 1.0 + 1e-12
            and -1e-12 <= second_ratio <= 1.0 + 1e-12
        )

    @classmethod
    def _segment_segment_distance_2d(cls, first_start, first_end, second_start, second_end):
        if cls._segments_intersect_2d(
            first_start, first_end, second_start, second_end
        ):
            return 0.0
        return min(
            float(cls._point_segment_distances(first_start, second_start, second_end)),
            float(cls._point_segment_distances(first_end, second_start, second_end)),
            float(cls._point_segment_distances(second_start, first_start, first_end)),
            float(cls._point_segment_distances(second_end, first_start, first_end)),
        )

    def _obstacle_avoiding_capsule_waypoints(
        self,
        current_position,
        target_position,
        obstacle,
        *,
        use_long_arc=False,
    ):
        current_position = np.asarray(current_position, dtype=float).reshape(3)
        target_position = np.asarray(target_position, dtype=float).reshape(3)
        endpoints, normal, radius, clearance, margin = (
            self._approach_capsule_geometry(obstacle)
        )
        reference = np.asarray([1.0, 0.0, 0.0])
        if abs(float(reference @ normal)) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0])
        plane_x = np.cross(normal, reference)
        plane_x /= np.linalg.norm(plane_x)
        plane_y = np.cross(normal, plane_x)
        origin = endpoints[0]

        def planar_coordinates(position):
            relative = np.asarray(position, dtype=float) - origin
            return np.asarray([relative @ plane_x, relative @ plane_y], dtype=float)

        capsule_start = np.zeros(2)
        capsule_end = planar_coordinates(endpoints[1])
        current_planar = planar_coordinates(current_position)
        target_planar = planar_coordinates(target_position)
        required_radius = radius + clearance
        current_radius = float(
            self._point_segment_distances(
                current_planar, capsule_start, capsule_end
            )
        )
        target_radius = float(
            self._point_segment_distances(
                target_planar, capsule_start, capsule_end
            )
        )
        if current_radius <= radius + 1e-4:
            raise TrajectoryValidationError(
                "Stage-0 starts inside the tracked obstacle"
            )
        if target_radius < required_radius - 1e-6:
            raise TrajectoryValidationError(
                "Task start violates Stage-0 obstacle clearance by {:.4f} m".format(
                    required_radius - target_radius
                )
            )
        direct_distance = self._segment_segment_distance_2d(
            current_planar, target_planar, capsule_start, capsule_end
        )
        if (
            current_radius >= required_radius - 1e-9
            and direct_distance >= required_radius - 1e-9
        ):
            return np.vstack((current_position, target_position))

        chord_compensation = (
            self._approach_spacing * self._approach_spacing
            / max(8.0 * required_radius, 1e-9)
        )
        bypass_radius = max(
            required_radius + margin,
            required_radius + chord_compensation + 1e-6,
        )
        axis = capsule_end - capsule_start
        axis_length = float(np.linalg.norm(axis))
        axis /= axis_length
        side = np.asarray([-axis[1], axis[0]])
        side_segments = max(1, int(math.ceil(axis_length / self._approach_spacing)))
        cap_segments = max(
            2,
            int(math.ceil(math.pi * bypass_radius / self._approach_spacing)),
        )
        top = capsule_start + side * bypass_radius + np.linspace(
            0.0, 1.0, side_segments + 1
        )[:, None] * (capsule_end - capsule_start)
        end_angles = np.linspace(math.pi / 2.0, -math.pi / 2.0, cap_segments + 1)
        end_cap = capsule_end + bypass_radius * (
            np.cos(end_angles)[:, None] * axis
            + np.sin(end_angles)[:, None] * side
        )
        bottom = capsule_end - side * bypass_radius + np.linspace(
            0.0, 1.0, side_segments + 1
        )[:, None] * (capsule_start - capsule_end)
        start_angles = np.linspace(-math.pi / 2.0, -3.0 * math.pi / 2.0, cap_segments + 1)
        start_cap = capsule_start + bypass_radius * (
            np.cos(start_angles)[:, None] * axis
            + np.sin(start_angles)[:, None] * side
        )
        boundary = np.vstack((top, end_cap[1:], bottom[1:], start_cap[1:-1]))
        boundary_count = len(boundary)

        def connection_is_safe(point, boundary_point, allow_egress):
            if not allow_egress:
                return self._segment_segment_distance_2d(
                    point, boundary_point, capsule_start, capsule_end
                ) >= required_radius - 1e-9
            phases = np.linspace(0.0, 1.0, 33)
            samples = point + phases[:, None] * (boundary_point - point)
            distances = self._point_segment_distances(
                samples, capsule_start, capsule_end
            )
            return bool(
                np.min(distances) >= radius - 1e-9
                and np.all(np.diff(distances) >= -1e-9)
                and distances[-1] >= required_radius - 1e-9
            )

        start_indices = [
            index
            for index, point in enumerate(boundary)
            if connection_is_safe(
                current_planar,
                point,
                current_radius < required_radius - 1e-9,
            )
        ]
        target_indices = [
            index
            for index, point in enumerate(boundary)
            if connection_is_safe(target_planar, point, False)
        ]
        if not start_indices or not target_indices:
            raise TrajectoryValidationError(
                "Stage-0 could not connect the approach to the capsule boundary"
            )

        directional_routes = []
        for direction in (1, -1):
            best = None
            for start_index in start_indices:
                for target_index in target_indices:
                    indices = [start_index]
                    while indices[-1] != target_index:
                        indices.append((indices[-1] + direction) % boundary_count)
                        if len(indices) > boundary_count:
                            break
                    route = boundary[indices]
                    route_points = np.vstack((current_planar, route, target_planar))
                    length = float(
                        np.sum(np.linalg.norm(np.diff(route_points, axis=0), axis=1))
                    )
                    if best is None or length < best[0]:
                        best = (length, route_points)
            if best is not None:
                directional_routes.append(best)
        if not directional_routes:
            raise TrajectoryValidationError(
                "Stage-0 could not route around the capsule"
            )
        directional_routes.sort(key=lambda item: item[0])
        selected = directional_routes[-1 if use_long_arc and len(directional_routes) > 1 else 0]
        planar_waypoints = selected[1]
        planar_edges = np.linalg.norm(np.diff(planar_waypoints, axis=0), axis=1)
        planar_distance = float(np.sum(planar_edges))
        progress = np.concatenate(([0.0], np.cumsum(planar_edges))) / planar_distance
        current_height = float((current_position - origin) @ normal)
        target_height = float((target_position - origin) @ normal)
        heights = current_height + progress * (target_height - current_height)
        return (
            origin[None, :]
            + planar_waypoints[:, :1] * plane_x[None, :]
            + planar_waypoints[:, 1:] * plane_y[None, :]
            + heights[:, None] * normal[None, :]
        )

    @staticmethod
    def _smoothstep5(values):
        values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        return values ** 3 * (values * (values * 6.0 - 15.0) + 10.0)

    def task_segment_speed_scales(self, positions, stage_timing):
        """Return a smooth local speed multiplier for every task edge."""
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise TrajectoryValidationError(
                "Stage timing needs at least two Cartesian task positions"
            )
        if stage_timing is None:
            return np.ones(len(positions) - 1, dtype=float)
        try:
            boundaries = np.asarray(stage_timing["boundaries"], dtype=int)
            windows = np.asarray(stage_timing["transition_windows"], dtype=int)
            slow_scale = float(stage_timing["speed_scale"])
            ramp_before = float(stage_timing["ramp_before_m"])
            start_ramp = float(stage_timing["task_start_ramp_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise TrajectoryValidationError(
                "Stage timing metadata is incomplete or invalid"
            ) from error
        if (
            boundaries.ndim != 1
            or len(boundaries) < 1
            or boundaries[-1] != len(positions) - 1
            or boundaries[0] <= 0
            or np.any(np.diff(boundaries) <= 0)
            or windows.shape != (max(len(boundaries) - 1, 0), 2)
        ):
            raise TrajectoryValidationError(
                "Stage timing boundaries do not match the Cartesian task path"
            )
        for index, window in enumerate(windows):
            if (
                int(window[0]) != int(boundaries[index])
                or int(window[1]) < int(window[0])
                or int(window[1]) > int(boundaries[index + 1])
            ):
                raise TrajectoryValidationError(
                    "Stage transition window {} is outside its stages".format(index)
                )
        if (
            not all(
                math.isfinite(value)
                for value in (slow_scale, ramp_before, start_ramp)
            )
            or not 0.0 < slow_scale <= 1.0
            or ramp_before < 0.0
            or start_ramp < 0.0
        ):
            raise TrajectoryValidationError(
                "Stage transition timing values must be finite and non-negative"
            )

        edge_distance = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        distance = np.concatenate(([0.0], np.cumsum(edge_distance)))
        point_scale = np.ones(len(positions), dtype=float)
        if start_ramp > 1e-12:
            indices = np.flatnonzero(distance <= start_ramp + 1e-12)
            alpha = self._smoothstep5(distance[indices] / start_ramp)
            point_scale[indices] = np.minimum(
                point_scale[indices], slow_scale + (1.0 - slow_scale) * alpha
            )
        else:
            point_scale[0] = min(point_scale[0], slow_scale)

        for start_index, end_index in windows:
            start_index = int(start_index)
            end_index = int(end_index)
            boundary_distance = float(distance[start_index])
            if ramp_before > 1e-12:
                before = np.flatnonzero(
                    (distance >= boundary_distance - ramp_before - 1e-12)
                    & (distance <= boundary_distance + 1e-12)
                )
                alpha = self._smoothstep5(
                    (distance[before] - (boundary_distance - ramp_before))
                    / ramp_before
                )
                point_scale[before] = np.minimum(
                    point_scale[before], 1.0 - (1.0 - slow_scale) * alpha
                )
            else:
                point_scale[start_index] = min(
                    point_scale[start_index], slow_scale
                )
            transition_distance = float(distance[end_index] - boundary_distance)
            after = np.arange(start_index, end_index + 1, dtype=int)
            if transition_distance > 1e-12:
                alpha = self._smoothstep5(
                    (distance[after] - boundary_distance) / transition_distance
                )
                point_scale[after] = np.minimum(
                    point_scale[after], slow_scale + (1.0 - slow_scale) * alpha
                )
            else:
                point_scale[start_index] = min(
                    point_scale[start_index], slow_scale
                )
        return np.minimum(point_scale[:-1], point_scale[1:])

    @staticmethod
    def _dense_spline_samples(spline, times, maximum_step=0.005):
        samples = []
        segment_indices = []
        for index, (start, end) in enumerate(zip(times[:-1], times[1:])):
            count = max(3, int(math.ceil((end - start) / maximum_step)) + 1)
            segment_times = np.linspace(start, end, count, endpoint=False)
            samples.append(segment_times)
            segment_indices.extend([index] * len(segment_times))
        samples.append(np.asarray([times[-1]], dtype=float))
        segment_indices.append(len(times) - 2)
        sample_times = np.concatenate(samples)
        return (
            sample_times,
            np.asarray(segment_indices, dtype=int),
            np.asarray(spline(sample_times), dtype=float),
            np.asarray(spline(sample_times, 1), dtype=float),
            np.asarray(spline(sample_times, 2), dtype=float),
        )

    @staticmethod
    def _segment_peak(sample_values, segment_indices, segment_count):
        peaks = np.zeros(int(segment_count), dtype=float)
        np.maximum.at(peaks, segment_indices, np.asarray(sample_values, dtype=float))
        return peaks

    @staticmethod
    def _local_time_dilation(required_scale, support_radius=3):
        """Spread a limit violation only across the spline's local support."""
        required_scale = np.asarray(required_scale, dtype=float)
        growth = np.where(required_scale > 1.0, 1.01 * required_scale, 1.0)
        log_growth = np.log(growth)
        spread = log_growth.copy()
        radius = max(0, int(support_radius))
        for offset in range(1, radius + 1):
            influence = 1.0 - float(offset) / float(radius + 1)
            spread[:-offset] = np.maximum(
                spread[:-offset], influence * log_growth[offset:]
            )
            spread[offset:] = np.maximum(
                spread[offset:], influence * log_growth[:-offset]
            )
        return np.exp(spread)

    @staticmethod
    def _smooth_segment_durations(segment_dt, maximum_neighbor_ratio=2.0):
        """Avoid abrupt knot-time changes without globally scaling the path."""
        segment_dt = np.asarray(segment_dt, dtype=float).copy()
        ratio = float(maximum_neighbor_ratio)
        for _ in range(2):
            for index in range(1, len(segment_dt)):
                segment_dt[index] = max(
                    segment_dt[index], segment_dt[index - 1] / ratio
                )
            for index in range(len(segment_dt) - 2, -1, -1):
                segment_dt[index] = max(
                    segment_dt[index], segment_dt[index + 1] / ratio
                )
        return segment_dt

    def time_parameterize(
        self, q_path, minimum_duration, segment_speed_scales=None
    ):
        q_path = np.asarray(q_path, dtype=float)
        if q_path.ndim != 2 or q_path.shape[1] != self._dof or len(q_path) < 2:
            raise TrajectoryValidationError(
                "A trajectory needs at least two samples with {} joints".format(self._dof)
            )
        minimum_duration = float(minimum_duration)
        if not math.isfinite(minimum_duration) or minimum_duration < 0.0:
            raise TrajectoryValidationError(
                "Trajectory minimum duration must be finite and non-negative"
            )
        if segment_speed_scales is None:
            segment_speed_scales = np.ones(len(q_path) - 1, dtype=float)
        else:
            segment_speed_scales = np.asarray(
                segment_speed_scales, dtype=float
            )
        if (
            segment_speed_scales.shape != (len(q_path) - 1,)
            or not np.all(np.isfinite(segment_speed_scales))
            or np.any(segment_speed_scales <= 0.0)
            or np.any(segment_speed_scales > 1.0)
        ):
            raise TrajectoryValidationError(
                "Every trajectory edge needs a speed scale in (0, 1]"
            )
        joint_step = np.abs(np.diff(q_path, axis=0))
        velocity_limit = np.maximum(self._velocity_limits * self._velocity_scale, 0.05)
        local_velocity_limit = (
            segment_speed_scales[:, None] * velocity_limit[None, :]
        )
        segment_dt = np.maximum(
            0.05,
            np.max(joint_step / local_velocity_limit, axis=1),
        )
        raw_duration = float(np.sum(segment_dt))
        duration = max(float(minimum_duration), raw_duration, 0.5)
        segment_dt *= duration / np.sum(segment_dt)
        spline = None
        dense_position = None
        dense_velocity = None
        dense_acceleration = None
        ratio_v = math.inf
        ratio_a = math.inf
        timing_iterations = 0
        for iteration in range(24):
            relative_times = np.concatenate(([0.0], np.cumsum(segment_dt)))
            times = self._first_point_delay + relative_times
            spline = make_interp_spline(
                times,
                q_path,
                axis=0,
                k=5,
                bc_type=(
                    [
                        (1, np.zeros(self._dof, dtype=float)),
                        (2, np.zeros(self._dof, dtype=float)),
                    ],
                    [
                        (1, np.zeros(self._dof, dtype=float)),
                        (2, np.zeros(self._dof, dtype=float)),
                    ],
                ),
            )
            (
                _dense_times,
                dense_segments,
                dense_position,
                dense_velocity,
                dense_acceleration,
            ) = self._dense_spline_samples(spline, times)
            dense_velocity_limit = local_velocity_limit[dense_segments]
            ratio_v = float(
                np.max(np.abs(dense_velocity) / dense_velocity_limit)
            )
            ratio_a = float(
                np.max(np.abs(dense_acceleration) / self._acceleration_limit)
            )
            if max(ratio_v, ratio_a) <= 1.00001:
                break
            sample_velocity_ratio = np.max(
                np.abs(dense_velocity) / dense_velocity_limit, axis=1
            )
            sample_acceleration_ratio = np.max(
                np.abs(dense_acceleration) / self._acceleration_limit, axis=1
            )
            segment_velocity_ratio = self._segment_peak(
                sample_velocity_ratio,
                dense_segments,
                len(segment_dt),
            )
            segment_acceleration_ratio = self._segment_peak(
                sample_acceleration_ratio,
                dense_segments,
                len(segment_dt),
            )
            required_scale = np.maximum(
                segment_velocity_ratio,
                np.sqrt(segment_acceleration_ratio),
            )
            dilation = self._local_time_dilation(required_scale)
            segment_dt *= np.minimum(dilation, 2.5)
            segment_dt = self._smooth_segment_durations(segment_dt)
            timing_iterations = iteration + 1
        relative_times = np.concatenate(([0.0], np.cumsum(segment_dt)))
        times = self._first_point_delay + relative_times
        spline = make_interp_spline(
            times,
            q_path,
            axis=0,
            k=5,
            bc_type=(
                [
                    (1, np.zeros(self._dof, dtype=float)),
                    (2, np.zeros(self._dof, dtype=float)),
                ],
                [
                    (1, np.zeros(self._dof, dtype=float)),
                    (2, np.zeros(self._dof, dtype=float)),
                ],
            ),
        )
        velocity = np.asarray(spline(times, 1), dtype=float)
        acceleration = np.asarray(spline(times, 2), dtype=float)
        (
            _dense_times,
            dense_segments,
            dense_position,
            dense_velocity,
            dense_acceleration,
        ) = self._dense_spline_samples(spline, times)
        ratio_v = float(
            np.max(
                np.abs(dense_velocity)
                / local_velocity_limit[dense_segments]
            )
        )
        ratio_a = float(
            np.max(np.abs(dense_acceleration) / self._acceleration_limit)
        )
        if ratio_v > 1.001 or ratio_a > 1.001:
            raise TrajectoryValidationError(
                "Controller-interpolated trajectory exceeds joint limits "
                "(velocity {:.3f}, acceleration {:.3f})".format(
                    ratio_v, ratio_a
                )
            )
        if (
            np.any(dense_position < self._lower[None, :] - 1e-9)
            or np.any(dense_position > self._upper[None, :] + 1e-9)
        ):
            raise TrajectoryValidationError(
                "Controller interpolation exceeds a joint position limit"
            )
        midpoint_times = 0.5 * (times[:-1] + times[1:])
        midpoint_position = np.asarray(spline(midpoint_times), dtype=float)
        validation_position = np.empty(
            (2 * len(q_path) - 1, self._dof), dtype=float
        )
        validation_position[0::2] = q_path
        validation_position[1::2] = midpoint_position
        return {
            "position": q_path,
            "velocity": velocity,
            "acceleration": acceleration,
            "time": times,
            "duration": float(times[-1]),
            "timing_iterations": timing_iterations,
            "minimum_duration_s": duration + self._first_point_delay,
            "timing_overhead_s": float(times[-1])
            - (duration + self._first_point_delay),
            "segment_duration_s": segment_dt,
            "segment_speed_scales": segment_speed_scales,
            "maximum_interpolated_velocity_ratio": ratio_v,
            "maximum_interpolated_acceleration_ratio": ratio_a,
            "maximum_interpolated_velocity_rad_s": float(
                np.max(np.abs(dense_velocity))
            ),
            "maximum_interpolated_acceleration_rad_s2": float(
                np.max(np.abs(dense_acceleration))
            ),
            "_position_spline": spline,
            "_validation_position": validation_position,
        }

    @staticmethod
    def sample_position(segment, elapsed):
        times = np.asarray(segment["time"], dtype=float)
        positions = np.asarray(segment["position"], dtype=float)
        elapsed = float(np.clip(elapsed, 0.0, times[-1]))
        spline = segment.get("_position_spline")
        if spline is not None:
            return np.asarray(spline(max(elapsed, times[0])), dtype=float)
        return np.asarray([
            np.interp(elapsed, times, positions[:, joint])
            for joint in range(positions.shape[1])
        ])

    def _sample_approach_waypoints(
        self,
        waypoints,
        current_axis,
        target_axis,
        current_x=None,
        target_x=None,
        *,
        orientation_profile="blended",
        preserve_waypoints=False,
    ):
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) < 2:
            raise TrajectoryValidationError(
                "Approach waypoints must contain at least two 3D positions"
            )
        if orientation_profile not in {
            "blended",
            "translate_then_orient",
            "orient_then_translate",
        }:
            raise ValueError(
                "Unknown approach orientation profile {!r}".format(
                    orientation_profile
                )
            )

        edge_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
        distance = float(cumulative[-1])
        if distance <= 1e-12:
            spatial_positions = waypoints[-1][None, :]
            spatial_progress = np.ones(1, dtype=float)
        else:
            spatial_sample_count = max(
                1,
                int(math.ceil(distance / self._approach_spacing)),
            )
            spatial_targets = np.linspace(
                0.0, distance, spatial_sample_count + 1
            )[1:]
            if preserve_waypoints:
                spatial_targets = np.unique(
                    np.concatenate((spatial_targets, cumulative[1:]))
                )
            spatial_positions = np.column_stack(
                [
                    np.interp(
                        spatial_targets,
                        cumulative,
                        waypoints[:, dim],
                    )
                    for dim in range(3)
                ]
            )
            spatial_progress = spatial_targets / distance

        current_axis = np.asarray(current_axis, dtype=float)
        target_axis = np.asarray(target_axis, dtype=float)
        axis_angle = math.acos(
            float(np.clip(current_axis @ target_axis, -1.0, 1.0))
        )
        orientation_angle = axis_angle
        if target_x is not None:
            if current_x is None:
                raise TrajectoryValidationError(
                    "Current Tool-X is required for a full-orientation approach"
                )
            current_x = np.asarray(current_x, dtype=float)
            target_x = np.asarray(target_x, dtype=float)
            orientation_angle = max(
                orientation_angle,
                math.acos(
                    float(np.clip(current_x @ target_x, -1.0, 1.0))
                ),
            )
        orientation_sample_count = max(
            1,
            int(math.ceil(orientation_angle / self._approach_axis_spacing)),
        )

        if orientation_profile == "blended":
            sample_count = max(
                len(spatial_positions),
                orientation_sample_count,
            )
            if sample_count != len(spatial_positions) and distance > 1e-12:
                spatial_targets = np.linspace(
                    0.0, distance, sample_count + 1
                )[1:]
                if preserve_waypoints:
                    spatial_targets = np.unique(
                        np.concatenate((spatial_targets, cumulative[1:]))
                    )
                positions = np.column_stack(
                    [
                        np.interp(
                            spatial_targets,
                            cumulative,
                            waypoints[:, dim],
                        )
                        for dim in range(3)
                    ]
                )
                orientation_progress = spatial_targets / distance
            else:
                positions = spatial_positions
                orientation_progress = spatial_progress
        else:
            rotation_progress = np.linspace(
                0.0, 1.0, orientation_sample_count + 1
            )[1:]
            if orientation_profile == "translate_then_orient":
                positions = np.vstack(
                    (
                        spatial_positions,
                        np.repeat(
                            waypoints[-1][None, :],
                            orientation_sample_count,
                            axis=0,
                        ),
                    )
                )
                orientation_progress = np.concatenate(
                    (
                        np.zeros(len(spatial_positions), dtype=float),
                        rotation_progress,
                    )
                )
            else:
                positions = np.vstack(
                    (
                        np.repeat(
                            waypoints[0][None, :],
                            orientation_sample_count,
                            axis=0,
                        ),
                        spatial_positions,
                    )
                )
                orientation_progress = np.concatenate(
                    (
                        rotation_progress,
                        np.ones(len(spatial_positions), dtype=float),
                    )
                )

        axes = self._interpolate_axis(
            current_axis,
            target_axis,
            orientation_progress,
        )
        x_axes = None
        if target_x is not None:
            x_axes = self._interpolate_axis(
                current_x,
                target_x,
                orientation_progress,
            )
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

    def _approach_samples(
        self,
        current_position,
        current_axis,
        target_position,
        target_axis,
        current_x=None,
        target_x=None,
        approach_obstacle=None,
    ):
        target_position = np.asarray(target_position, dtype=float)
        if approach_obstacle is None:
            waypoints = [np.asarray(current_position, dtype=float)]
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
        else:
            waypoints = self._obstacle_avoiding_approach_waypoints(
                current_position, target_position, approach_obstacle
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

        return self._sample_approach_waypoints(
            np.asarray(compact, dtype=float),
            current_axis,
            target_axis,
            current_x=current_x,
            target_x=target_x,
            orientation_profile="blended",
            preserve_waypoints=approach_obstacle is None,
        )

    def _approach_sample_candidates(
        self,
        current_position,
        current_axis,
        target_position,
        target_axis,
        current_x=None,
        target_x=None,
        approach_obstacle=None,
    ):
        primary = self._approach_samples(
            current_position,
            current_axis,
            target_position,
            target_axis,
            current_x=current_x,
            target_x=target_x,
            approach_obstacle=approach_obstacle,
        )
        candidates = [("blended", *primary)]
        primary_route = np.vstack(
            (np.asarray(current_position, dtype=float), primary[0])
        )
        orientation_angle = math.acos(
            float(
                np.clip(
                    np.asarray(current_axis, dtype=float)
                    @ np.asarray(target_axis, dtype=float),
                    -1.0,
                    1.0,
                )
            )
        )
        if target_x is not None and current_x is not None:
            orientation_angle = max(
                orientation_angle,
                math.acos(
                    float(
                        np.clip(
                            np.asarray(current_x, dtype=float)
                            @ np.asarray(target_x, dtype=float),
                            -1.0,
                            1.0,
                        )
                    )
                ),
            )
        if orientation_angle > 1e-6:
            for profile in (
                "translate_then_orient",
                "orient_then_translate",
            ):
                candidate = self._sample_approach_waypoints(
                    primary_route,
                    current_axis,
                    target_axis,
                    current_x=current_x,
                    target_x=target_x,
                    orientation_profile=profile,
                    preserve_waypoints=False,
                )
                candidates.append((profile, *candidate))

        if approach_obstacle is not None:
            alternate_waypoints = self._obstacle_avoiding_approach_waypoints(
                current_position,
                target_position,
                approach_obstacle,
                use_long_arc=True,
            )
            primary_waypoints = self._obstacle_avoiding_approach_waypoints(
                current_position,
                target_position,
                approach_obstacle,
            )
            if (
                alternate_waypoints.shape != primary_waypoints.shape
                or not np.allclose(
                    alternate_waypoints,
                    primary_waypoints,
                    atol=1e-9,
                    rtol=0.0,
                )
            ):
                alternate = self._sample_approach_waypoints(
                    alternate_waypoints,
                    current_axis,
                    target_axis,
                    current_x=current_x,
                    target_x=target_x,
                    orientation_profile="blended",
                    preserve_waypoints=False,
                )
                candidates.append(("alternate_obstacle_arc", *alternate))
                if orientation_angle > 1e-6:
                    alternate_phased = self._sample_approach_waypoints(
                        alternate_waypoints,
                        current_axis,
                        target_axis,
                        current_x=current_x,
                        target_x=target_x,
                        orientation_profile="translate_then_orient",
                        preserve_waypoints=False,
                    )
                    candidates.append(
                        ("alternate_arc_translate_then_orient", *alternate_phased)
                    )
        return candidates

    def _home_vertical_recovery(self, start_q, abort_requested):
        """Keep the measured TCP pose fixed in XY/orientation while lifting."""
        current_position, current_rotation = self.tip_state(start_q)
        if current_position[2] >= self._minimum_approach_z - 1e-6:
            return None, start_q.copy(), 0.0, [], [], []

        # Aim one strict task-position tolerance above the floor. The floor
        # remains the safety boundary; this margin prevents an IK solution at
        # the tolerance edge from ending microscopically below it.
        recovery_z = self._minimum_approach_z + self._position_tolerance
        recovery_distance = float(recovery_z - current_position[2])
        sample_count = max(
            5,
            int(math.ceil(recovery_distance / self._approach_spacing)),
        )
        phases = np.linspace(0.0, 1.0, sample_count + 1)[1:]
        positions = np.repeat(current_position[None, :], sample_count, axis=0)
        positions[:, 2] = current_position[2] + phases * recovery_distance
        axes = np.repeat(current_rotation[None, :, 2], sample_count, axis=0)
        x_axes = np.repeat(current_rotation[None, :, 0], sample_count, axis=0)
        q_tail, position_errors, axis_errors, x_errors = self._continuous_ik(
            positions,
            axes,
            start_q,
            abort_requested,
            "Return Home vertical recovery",
            position_tolerance=self._position_tolerance,
            final_position_tolerance=self._position_tolerance,
            max_joint_step=self._max_joint_step,
            x_axes=x_axes,
            x_active=np.ones(sample_count, dtype=bool),
        )
        q_recovery = self._densify_joint_path(
            np.vstack((start_q[None, :], q_tail)), self._max_joint_step
        )
        minimum_duration = recovery_distance / max(self._approach_speed, 1e-4)
        recovery = self.time_parameterize(q_recovery, minimum_duration)
        return (
            recovery,
            q_recovery[-1].copy(),
            recovery_distance,
            position_errors,
            axis_errors,
            x_errors,
        )

    def compile_joint_home(
        self,
        start_q,
        target_q,
        abort_requested=None,
        approach_obstacle=None,
    ):
        """Compile a collision-checked joint move to a saved comfortable posture.

        Robot Home is a posture, not a Cartesian tracking task.  Solving a long
        lift/lateral Cartesian path with free yaw can select an unreachable IK
        branch even when both the current and saved Home postures are valid.
        If the measured TCP starts below the transit floor, first solve a
        fixed-XY, fixed-orientation Cartesian lift. Only after that recovery
        has completed may the executor start direct joint interpolation to
        Home. Both segments retain workspace, collision, velocity and
        acceleration validation.
        """
        start_q = np.asarray(start_q, dtype=float)
        target_q = np.asarray(target_q, dtype=float)
        abort_requested = abort_requested or (lambda: False)
        if start_q.shape != (self._dof,) or target_q.shape != (self._dof,):
            raise TrajectoryValidationError(
                "Robot Home needs {} measured and target joints".format(self._dof)
            )
        if not np.all(np.isfinite(start_q)) or not np.all(np.isfinite(target_q)):
            raise TrajectoryValidationError("Robot Home joints must be finite")
        if np.any(start_q < self._lower) or np.any(start_q > self._upper):
            raise TrajectoryValidationError("Current joints exceed a position limit")
        if np.any(target_q < self._lower) or np.any(target_q > self._upper):
            raise TrajectoryValidationError("Saved Robot Home exceeds a joint limit")
        if abort_requested():
            raise TrajectoryValidationError("Return Home compilation aborted")

        try:
            (
                recovery,
                recovered_q,
                recovery_distance,
                recovery_position_errors,
                recovery_axis_errors,
                recovery_x_errors,
            ) = self._home_vertical_recovery(start_q, abort_requested)
            q_path = self._densify_joint_path(
                np.vstack((recovered_q, target_q)), self._max_joint_step
            )
            tcp_positions = np.asarray(
                [self.tip_state(q)[0] for q in q_path], dtype=float
            )
            home_tcp_distance = float(
                np.sum(np.linalg.norm(np.diff(tcp_positions, axis=0), axis=1))
            )
            minimum_duration = home_tcp_distance / max(self._approach_speed, 1e-4)
            approach = self.time_parameterize(q_path, minimum_duration)
            if abort_requested():
                raise TrajectoryValidationError("Return Home compilation aborted")
            approach_validation = approach["_validation_position"]
            minimum_z = self._approach_workspace_checks(approach_validation)
            self._collision_checks(approach_validation, "Return Home")
            if recovery is not None:
                recovery_validation = recovery["_validation_position"]
                minimum_z = min(
                    minimum_z,
                    self._approach_workspace_checks(recovery_validation),
                )
                self._collision_and_singularity_checks(
                    recovery_validation, "Return Home vertical recovery"
                )
                obstacle_validation = np.vstack(
                    (recovery_validation, approach_validation[1:])
                )
            else:
                obstacle_validation = approach_validation
            minimum_obstacle_clearance = self._approach_obstacle_checks(
                obstacle_validation,
                approach_obstacle,
            )
            timed_segments = [approach] if recovery is None else [recovery, approach]
            return {
                "start": start_q.copy(),
                "recovery": recovery,
                "approach": approach,
                "metrics": {
                    "home_strategy": (
                        "joint_posture"
                        if recovery is None
                        else "vertical_recovery_then_joint_posture"
                    ),
                    "vertical_recovery_required": recovery is not None,
                    "vertical_recovery_distance_m": recovery_distance,
                    "vertical_recovery_duration_s": (
                        0.0 if recovery is None else recovery["duration"]
                    ),
                    "vertical_recovery_maximum_ik_position_error_m": (
                        0.0
                        if not recovery_position_errors
                        else max(recovery_position_errors)
                    ),
                    "vertical_recovery_maximum_ik_tool_z_error_deg": (
                        0.0
                        if not recovery_axis_errors
                        else math.degrees(max(recovery_axis_errors))
                    ),
                    "vertical_recovery_maximum_ik_tool_x_error_deg": (
                        0.0
                        if not recovery_x_errors
                        else math.degrees(max(recovery_x_errors))
                    ),
                    "approach_points": len(q_path),
                    "approach_duration_s": approach["duration"],
                    "approach_distance_m": recovery_distance + home_tcp_distance,
                    "maximum_joint_step_rad": max(
                        float(
                            np.max(
                                np.abs(np.diff(segment["position"], axis=0))
                            )
                        )
                        for segment in timed_segments
                    ),
                    "minimum_approach_tcp_z_m": minimum_z,
                    "minimum_home_obstacle_clearance_m": (
                        None
                        if minimum_obstacle_clearance is None
                        else minimum_obstacle_clearance
                    ),
                    "maximum_interpolated_joint_velocity_ratio": max(
                        segment["maximum_interpolated_velocity_ratio"]
                        for segment in timed_segments
                    ),
                    "maximum_interpolated_joint_acceleration_ratio": max(
                        segment["maximum_interpolated_acceleration_ratio"]
                        for segment in timed_segments
                    ),
                    "maximum_interpolated_joint_velocity_rad_s": max(
                        segment["maximum_interpolated_velocity_rad_s"]
                        for segment in timed_segments
                    ),
                    "maximum_interpolated_joint_acceleration_rad_s2": max(
                        segment["maximum_interpolated_acceleration_rad_s2"]
                        for segment in timed_segments
                    ),
                },
            }
        finally:
            self._set_q(start_q)

    def _compile_cartesian_candidate(
        self,
        *,
        route_name,
        approach_positions,
        approach_axes,
        approach_x_axes,
        approach_distance,
        positions,
        axes,
        x_axes,
        x_active,
        start_q,
        abort_requested,
        approach_obstacle,
        stage_timing,
    ):
        approach_x_active = None
        if approach_x_axes is not None:
            approach_x_active = np.zeros(len(approach_x_axes), dtype=bool)
            approach_x_active[-1] = True
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
            x_active=approach_x_active,
        )
        q_approach = np.vstack((start_q[None, :], q_approach_tail))
        q_approach = self._densify_joint_path(
            q_approach, self._max_joint_step
        )
        minimum_approach_z = self._approach_workspace_checks(
            q_approach,
            enforce_transit_floor=approach_obstacle is None,
        )

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
        if np.any(full_path < self._lower[None, :]) or np.any(
            full_path > self._upper[None, :]
        ):
            raise TrajectoryValidationError("Joint position limit violation")
        maximum_step = float(np.max(np.abs(np.diff(full_path, axis=0))))
        if maximum_step > self._max_joint_step:
            raise TrajectoryValidationError(
                "Joint continuity check failed at {:.3f} rad".format(
                    maximum_step
                )
            )
        task_edge_length = np.linalg.norm(
            np.diff(positions, axis=0), axis=1
        )
        task_length = float(np.sum(task_edge_length))
        task_speed_scales = self.task_segment_speed_scales(
            positions, stage_timing
        )
        approach_minimum = approach_distance / max(
            self._approach_speed, 1e-4
        )
        task_minimum = float(
            np.sum(
                task_edge_length
                / (
                    max(self._task_speed, 1e-4)
                    * task_speed_scales
                )
            )
        )
        approach = self.time_parameterize(q_approach, approach_minimum)
        task = self.time_parameterize(
            q_task,
            task_minimum,
            segment_speed_scales=task_speed_scales,
        )
        minimum_approach_obstacle_clearance = self._approach_obstacle_checks(
            approach["_validation_position"],
            approach_obstacle,
        )
        minimum_sv = min(
            self._collision_and_singularity_checks(
                approach["_validation_position"], "Approach"
            ),
            self._collision_and_singularity_checks(
                task["_validation_position"], "Task"
            ),
        )
        return {
            "start": start_q.copy(),
            "approach": approach,
            "task": task,
            "metrics": {
                "approach_route": str(route_name),
                "approach_points": len(q_approach),
                "task_points": len(q_task),
                "approach_duration_s": approach["duration"],
                "task_duration_s": task["duration"],
                "approach_distance_m": approach_distance,
                "task_length_m": task_length,
                "task_cartesian_minimum_duration_s": task_minimum,
                "approach_timing_overhead_s": approach["timing_overhead_s"],
                "task_timing_overhead_s": task["timing_overhead_s"],
                "approach_timing_iterations": approach["timing_iterations"],
                "task_timing_iterations": task["timing_iterations"],
                "task_minimum_speed_scale": float(
                    np.min(task_speed_scales)
                ),
                "maximum_joint_step_rad": maximum_step,
                "minimum_jacobian_singular_value": minimum_sv,
                "minimum_approach_tcp_z_m": minimum_approach_z,
                "minimum_approach_obstacle_clearance_m": (
                    minimum_approach_obstacle_clearance
                ),
                "maximum_ik_position_error_m": max(
                    approach_position_error + task_position_error
                ),
                "maximum_ik_tool_z_error_deg": math.degrees(
                    max(approach_axis_error + task_axis_error)
                ),
                "maximum_ik_tool_x_error_deg": math.degrees(
                    max(approach_x_error + task_x_error)
                ),
                "maximum_interpolated_joint_velocity_ratio": max(
                    approach["maximum_interpolated_velocity_ratio"],
                    task["maximum_interpolated_velocity_ratio"],
                ),
                "maximum_interpolated_joint_acceleration_ratio": max(
                    approach["maximum_interpolated_acceleration_ratio"],
                    task["maximum_interpolated_acceleration_ratio"],
                ),
                "maximum_interpolated_joint_velocity_rad_s": max(
                    approach["maximum_interpolated_velocity_rad_s"],
                    task["maximum_interpolated_velocity_rad_s"],
                ),
                "maximum_interpolated_joint_acceleration_rad_s2": max(
                    approach["maximum_interpolated_acceleration_rad_s2"],
                    task["maximum_interpolated_acceleration_rad_s2"],
                ),
            },
        }

    def compile(
        self,
        positions,
        tool_z_axes,
        start_q,
        abort_requested=None,
        tool_x_axes=None,
        tool_x_active=None,
        approach_obstacle=None,
        stage_timing=None,
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
                x_active = np.asarray(tool_x_active, dtype=bool).copy()
                if x_active.shape != (len(x_axes),):
                    raise TrajectoryValidationError(
                        "The Tool-X active mask must match the Cartesian path"
                    )
            # Start and goal are user-specified 6D boundary poses.  Learned yaw
            # activation only controls the interior of the task trajectory.
            x_active[[0, -1]] = True
        elif tool_x_active is not None and np.any(
            np.asarray(tool_x_active, dtype=bool)
        ):
            raise TrajectoryValidationError(
                "Tool-X cannot be active without a Tool-X path"
            )

        try:
            current_position, current_rotation = self.tip_state(start_q)
            candidates = self._approach_sample_candidates(
                current_position,
                current_rotation[:, 2],
                positions[0],
                axes[0],
                current_x=(
                    None if x_axes is None else current_rotation[:, 0]
                ),
                target_x=None if x_axes is None else x_axes[0],
                approach_obstacle=approach_obstacle,
            )
            failures = []
            for (
                route_name,
                approach_positions,
                approach_axes,
                approach_x_axes,
                approach_distance,
            ) in candidates:
                self._set_q(start_q)
                try:
                    return self._compile_cartesian_candidate(
                        route_name=route_name,
                        approach_positions=approach_positions,
                        approach_axes=approach_axes,
                        approach_x_axes=approach_x_axes,
                        approach_distance=approach_distance,
                        positions=positions,
                        axes=axes,
                        x_axes=x_axes,
                        x_active=x_active,
                        start_q=start_q,
                        abort_requested=abort_requested,
                        approach_obstacle=approach_obstacle,
                        stage_timing=stage_timing,
                    )
                except TrajectoryValidationError as error:
                    if abort_requested():
                        raise
                    failures.append((str(route_name), error))
            if len(failures) == 1:
                raise failures[0][1]
            details = "; ".join(
                "{}: {}".format(name, error)
                for name, error in failures
            )
            raise TrajectoryValidationError(
                "All {} safe approach candidates failed: {}".format(
                    len(failures), details
                )
            )
        finally:
            self._set_q(start_q)
