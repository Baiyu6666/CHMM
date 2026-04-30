import numpy as np

from .base import TaskBundle
from planner import optimize_trajectory, repair_trajectory_constraints, resample_polyline


class S3ObsAvoidEnv:
    """
    Three-stage 2D obstacle avoidance environment.

    Stage 1:
      - move around the obstacle while keeping safe clearance

    Stage 2:
      - move from stage 1 end to the entry point of a terminal arc
      - speed is concentrated around a stage-specific value with reduced fluctuation

    Stage 3:
      - follow a terminal circular arc into the goal
      - speed remains constrained at a lower stage-specific value
      - the equality feature is the distance to the arc center, with target value equal to the arc radius
    """

    def __init__(
        self,
        start=(-1.5, 0.0),
        stage1_end=(0.3, 0.0),
        stage2_end=(0.7, 0.0),
        obs_center=(-0.5, 0.0),
        obs_radius=0.3,
        stage1_aux_obstacle_offsets=((0.14, 0.18), (-0.26, -0.16)),
        stage1_aux_obstacle_radii=(0.15, 0.11),
        decoy_line_point=(0.15, 0.9),
        decoy_line_direction=(1.0, -0.35),
        clearance=0.1,
        stage1_speed_max=0.30,
        stage2_speed_max=0.055,
        stage3_speed_max=0.045,
        stage1_accel_max=0.06,
        stage2_accel_max=0.015,
        stage3_accel_max=0.015,
        terminal_arc_center_offset=(0.0, -0.2),
        terminal_arc_radius=0.2,
        terminal_arc_theta_start=0.5 * np.pi,
        terminal_arc_theta_end=-1.5 * np.pi,
        dt=0.5,
        noise_std=0.011,
    ):
        self.stage1_end = np.asarray(stage1_end, dtype=float)
        self.stage2_end = np.asarray(stage2_end, dtype=float)
        self.terminal_arc_center_offset = np.asarray(terminal_arc_center_offset, dtype=float)
        self.terminal_arc_center = self.stage2_end + self.terminal_arc_center_offset
        self.terminal_arc_radius = float(terminal_arc_radius)
        self.terminal_arc_theta_start = float(terminal_arc_theta_start)
        self.terminal_arc_theta_end = float(terminal_arc_theta_end)
        self.stage3_speed_max = float(stage3_speed_max)
        self.stage3_accel_max = float(stage3_accel_max)
        self.arc_entry = self._arc_point(self.terminal_arc_theta_start)
        if np.linalg.norm(self.arc_entry - self.stage2_end) > 1e-6:
            raise ValueError(
                "terminal arc parameters are inconsistent: stage2_end must coincide with the arc start. "
                "Adjust terminal_arc_center_offset / terminal_arc_radius / terminal_arc_theta_start."
            )
        self.stage3_end = self._arc_point(self.terminal_arc_theta_end)
        self.stage1_aux_obstacle_offsets = np.asarray(stage1_aux_obstacle_offsets, dtype=float).reshape(-1, 2)
        self.stage1_aux_obstacle_radii = np.asarray(stage1_aux_obstacle_radii, dtype=float).reshape(-1)
        self.decoy_line_point = np.asarray(decoy_line_point, dtype=float).reshape(2)
        self.decoy_line_direction = np.asarray(decoy_line_direction, dtype=float).reshape(2)
        self.decoy_line_direction = self.decoy_line_direction / max(np.linalg.norm(self.decoy_line_direction), 1e-12)
        if len(self.stage1_aux_obstacle_offsets) != len(self.stage1_aux_obstacle_radii):
            raise ValueError("stage1_aux_obstacle_offsets and stage1_aux_obstacle_radii must have the same length.")

        self.start = np.array(start, dtype=float)
        self.obs_center = np.array(obs_center, dtype=float)
        self.obs_radius = float(obs_radius)
        self.clearance = float(clearance)
        self.stage1_speed_max = float(stage1_speed_max)
        self.stage2_speed_max = float(stage2_speed_max)
        self.stage1_accel_max = float(stage1_accel_max)
        self.stage2_accel_max = float(stage2_accel_max)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.noise_vec = np.array([0.37, -0.58], dtype=float)
        self.noise_bias = 0.0

        self.stage1_aux_obstacle_centers = self.obs_center[None, :] + self.stage1_aux_obstacle_offsets
        self.subgoal = None
        self.goal = None
        self.hide_true_stage_end_markers = True
        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self._direct_true_constraints()
        self.constraint_specs = self.get_constraint_specs()

    def _direct_true_constraints(self):
        return {
            "d_safe": float(self.obs_radius + self.clearance),
            "v2_target": float(self.stage2_speed_max),
            "v3_target": float(self.stage3_speed_max),
            "arc_distance_target": float(self.terminal_arc_radius),
        }

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "obs_dist", "description": "Effective distance to the composite stage-1 obstacle"},
            {"id": 1, "name": "speed", "description": "2D speed magnitude"},
            {"id": 2, "name": "arc_dist", "description": "Distance to the terminal arc center"},
            {"id": 3, "name": "heading", "description": "Unwrapped planar heading angle derived from velocity"},
            {"id": 4, "name": "line_dist", "description": "Distance to an auxiliary line unrelated to the true task"},
            {"id": 5, "name": "noise", "description": "Deterministic auxiliary noise-like feature"},
        ]

    def get_true_constraints(self):
        return dict(self.true_constraints)

    def get_constraint_specs(self):
        return [
            {
                "feature_name": "obs_dist",
                "stage": 0,
                "semantics": "lower_bound",
                "oracle_key": "d_safe",
            },
            {
                "feature_name": "speed",
                "stage": 1,
                "semantics": "target_value",
                "oracle_key": "v2_target",
            },
            {
                "feature_name": "arc_dist",
                "stage": 2,
                "semantics": "target_value",
                "oracle_key": "arc_distance_target",
            },
            {
                "feature_name": "speed",
                "stage": 2,
                "semantics": "target_value",
                "oracle_key": "v3_target",
            },
        ]

    def _arc_points(self, theta_start: float, theta_end: float, direction: str, radius: float, n_arc: int):
        if direction == "up":
            delta = (theta_end - theta_start) % (2.0 * np.pi)
            if delta <= 1e-8:
                delta += 2.0 * np.pi
        else:
            delta = -((theta_start - theta_end) % (2.0 * np.pi))
            if abs(delta) <= 1e-8:
                delta -= 2.0 * np.pi
        thetas = theta_start + np.linspace(0.0, delta, max(int(n_arc), 8))
        return self.obs_center[None, :] + radius * np.c_[np.cos(thetas), np.sin(thetas)]

    def _smooth_process_noise(self, length: int, dim: int, scale: float, knots: int, rng):
        if int(length) <= 2 or float(scale) <= 0.0:
            return np.zeros((int(length), int(dim)), dtype=float)
        num_knots = max(int(knots), 2)
        knot_t = np.linspace(0.0, 1.0, num_knots)
        knot_vals = rng.randn(num_knots, int(dim)) * float(scale)
        knot_vals[0] = 0.0
        knot_vals[-1] = 0.0
        t = np.linspace(0.0, 1.0, int(length))
        noise = np.empty((int(length), int(dim)), dtype=float)
        for d in range(int(dim)):
            noise[:, d] = np.interp(t, knot_t, knot_vals[:, d])
        return noise

    def _arc_point(self, theta):
        theta = float(theta)
        return self.terminal_arc_center + self.terminal_arc_radius * np.array([np.cos(theta), np.sin(theta)], dtype=float)

    def _terminal_arc_points(self, num_points=24):
        thetas = np.linspace(self.terminal_arc_theta_start, self.terminal_arc_theta_end, max(int(num_points), 8))
        return self.terminal_arc_center[None, :] + self.terminal_arc_radius * np.c_[np.cos(thetas), np.sin(thetas)]

    def _nearest_terminal_arc_points(self, path, num_points=256):
        pts = np.asarray(path, dtype=float).reshape(-1, 2)
        arc_pts = self._terminal_arc_points(num_points=max(int(num_points), 32))
        d2 = np.sum((pts[:, None, :] - arc_pts[None, :, :]) ** 2, axis=2)
        nearest_idx = np.argmin(d2, axis=1)
        return arc_pts[nearest_idx]

    def _project_to_terminal_arc(self, path):
        return self._nearest_terminal_arc_points(path)

    def _arc_dist(self, traj):
        pts = np.asarray(traj, dtype=float)
        return np.linalg.norm(pts - self.terminal_arc_center[None, :], axis=1)

    def _stage1_obstacle_components(self):
        centers = [np.asarray(self.obs_center, dtype=float)]
        radii = [float(self.obs_radius)]
        for center, radius in zip(self.stage1_aux_obstacle_centers, self.stage1_aux_obstacle_radii):
            centers.append(np.asarray(center, dtype=float))
            radii.append(float(radius))
        return centers, radii

    def _stage1_effective_distance(self, traj):
        pts = np.asarray(traj, dtype=float).reshape(-1, 2)
        centers, radii = self._stage1_obstacle_components()
        surrogate = []
        for center, radius in zip(centers, radii):
            surrogate.append(np.linalg.norm(pts - center[None, :], axis=1) - float(radius) + float(self.obs_radius))
        return np.min(np.stack(surrogate, axis=0), axis=0)

    def _stage1_signed_clearance(self, traj):
        pts = np.asarray(traj, dtype=float).reshape(-1, 2)
        centers, radii = self._stage1_obstacle_components()
        clearances = []
        for center, radius in zip(centers, radii):
            clearances.append(np.linalg.norm(pts - center[None, :], axis=1) - float(radius))
        return np.min(np.stack(clearances, axis=0), axis=0)

    def _project_to_stage1_clearance_boundary(self, path):
        out = np.asarray(path, dtype=float).copy()
        centers, radii = self._stage1_obstacle_components()
        target_radii = [float(radius) + float(self.clearance) for radius in radii]
        for _ in range(3):
            for center, target_radius in zip(centers, target_radii):
                rel = out - center[None, :]
                norms = np.linalg.norm(rel, axis=1, keepdims=True)
                mask = norms[:, 0] < float(target_radius)
                if np.any(mask):
                    rel_mask = rel[mask]
                    norms_mask = norms[mask]
                    tiny = norms_mask[:, 0] < 1e-12
                    if np.any(tiny):
                        rel_mask[tiny] = np.array([1.0, 0.0], dtype=float)
                        norms_mask[tiny] = 1.0
                    rel_mask = rel_mask / np.maximum(norms_mask, 1e-12)
                    out[mask] = center[None, :] + float(target_radius) * rel_mask
        return out

    def _segment_stage1_effective_distance(self, a, b, num=64):
        u = np.linspace(0.0, 1.0, max(int(num), 8))
        pts = (1.0 - u)[:, None] * np.asarray(a, dtype=float)[None, :] + u[:, None] * np.asarray(b, dtype=float)[None, :]
        return float(np.min(self._stage1_effective_distance(pts)))

    def _decoy_dist(self, traj):
        pts = np.asarray(traj, dtype=float).reshape(-1, 2)
        rel = pts - self.decoy_line_point[None, :]
        normal = np.array([-self.decoy_line_direction[1], self.decoy_line_direction[0]], dtype=float)
        return np.abs(rel @ normal)

    @staticmethod
    def _segment_point_distance(a, b, p):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        p = np.asarray(p, dtype=float)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        closest = a + t * ab
        return float(np.linalg.norm(p - closest))

    def _add_terminal_arc_radial_jitter(self, path, jitter_scale, rng):
        pts = np.asarray(path, dtype=float).copy()
        if len(pts) <= 2 or jitter_scale <= 0.0:
            return pts
        rel = pts - self.terminal_arc_center[None, :]
        radii = np.linalg.norm(rel, axis=1, keepdims=True)
        unit = rel / np.maximum(radii, 1e-12)
        envelope = np.sin(np.linspace(0.0, np.pi, len(pts)))[:, None]
        radial_jitter = rng.randn(len(pts), 1) * float(jitter_scale) * envelope
        pts = pts + radial_jitter * unit
        pts[0] = path[0]
        pts[-1] = path[-1]
        return pts

    @staticmethod
    def _uniform_reparameterize_fixed_count(path, num_points):
        pts = np.asarray(path, dtype=float)
        n = int(num_points)
        if len(pts) <= 1 or n <= 1:
            return pts.copy()
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total_len = float(s[-1])
        if total_len <= 1e-12:
            out = np.repeat(pts[:1], n, axis=0)
            out[0] = pts[0]
            out[-1] = pts[-1]
            return out
        targets = np.linspace(0.0, total_len, n)
        out = np.empty((n, pts.shape[1]), dtype=float)
        out[0] = pts[0]
        out[-1] = pts[-1]
        j = 0
        for i, target in enumerate(targets[1:-1], start=1):
            while j + 1 < len(s) and s[j + 1] < target:
                j += 1
            frac = (target - s[j]) / max(s[j + 1] - s[j], 1e-12)
            out[i] = (1.0 - frac) * pts[j] + frac * pts[j + 1]
        return out

    def generate_demo(
        self,
        n1=24,
        direction=None,
        start_x_range=(-2., -0.5),
        start_y_range=(-0.5, 0.5),
        easy_start_prob=0.2,
        easy_start_x_range=(-0.95, -0.15),
        easy_start_y_range=(0.5, 0.95),
        direct_path_margin_scale=1.04,
        start_safe_scale=1.08,
        start_resample_tries=64,
        start_angle_jitter_deg=30.0,
        start_radius_jitter_frac=0.24,
        start_tangent_jitter_frac=0.24,
        subgoal_jitter_frac=0.28,
        subgoal_jitter_y_scale=1.9,
        arc_entry_jitter_scale=0.03,
        stage2_lateral_scale=0.011,
        stage3_arc_distance_noise_scale=0.028,
        process_noise_scale=0.03,
        process_noise_knots=7,
        stage1_speed_scale=1.0,
        stage2_speed_scale=1.0,
        stage3_speed_scale=1.0,
        stage1_speed_scale_jitter=0.18,
        stage2_speed_scale_jitter=0.14,
        stage3_speed_scale_jitter=0.14,
        rng=None,
    ):
        rng = np.random if rng is None else rng
        if direction is None:
            direction = rng.choice(["up", "down"])

        d_safe = float(self.true_constraints["d_safe"])
        seg_len = np.linalg.norm(self.stage3_end - self.stage1_end) + 1e-12
        direct_safe_radius = d_safe * max(float(direct_path_margin_scale), 1.0)
        stage1_speed_scale_local = float(stage1_speed_scale) * float(
            rng.uniform(max(0.3, 1.0 - stage1_speed_scale_jitter), 1.0 + stage1_speed_scale_jitter)
        )
        stage2_speed_scale_local = float(stage2_speed_scale) * float(
            rng.uniform(max(0.3, 1.0 - stage2_speed_scale_jitter), 1.0 + stage2_speed_scale_jitter)
        )
        stage3_speed_scale_local = float(stage3_speed_scale) * float(
            rng.uniform(max(0.3, 1.0 - stage3_speed_scale_jitter), 1.0 + stage3_speed_scale_jitter)
        )

        start_local = None
        use_direct_stage1 = False
        if float(easy_start_prob) > 0.0 and rng.rand() < float(easy_start_prob):
            x_lo, x_hi = sorted([float(easy_start_x_range[0]), float(easy_start_x_range[1])])
            y_lo, y_hi = sorted([float(easy_start_y_range[0]), float(easy_start_y_range[1])])
            start_sign = -1.0 if rng.rand() < 0.5 else 1.0
            for _ in range(max(int(start_resample_tries // 2), 8)):
                candidate = np.array(
                    [rng.uniform(x_lo, x_hi), start_sign * rng.uniform(y_lo, y_hi)],
                    dtype=float,
                )
                if self._stage1_effective_distance(candidate[None, :])[0] >= direct_safe_radius and self._segment_stage1_effective_distance(candidate, self.stage1_end) >= direct_safe_radius:
                    start_local = candidate
                    use_direct_stage1 = True
                    break
        if start_x_range is not None and start_y_range is not None:
            x_lo, x_hi = sorted([float(start_x_range[0]), float(start_x_range[1])])
            y_lo, y_hi = sorted([float(start_y_range[0]), float(start_y_range[1])])
            min_start_dist = d_safe * max(float(start_safe_scale), 1.0)
            if start_local is None:
                for _ in range(max(int(start_resample_tries), 1)):
                    candidate = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
                    if self._stage1_effective_distance(candidate[None, :])[0] >= min_start_dist:
                        start_local = candidate
                        break
                if start_local is None:
                    corners = np.array([[x_lo, y_lo], [x_lo, y_hi], [x_hi, y_lo], [x_hi, y_hi]], dtype=float)
                    distances = self._stage1_effective_distance(corners)
                    start_local = corners[int(np.argmax(distances))]

        if start_local is None:
            start_rel_base = self.start - self.obs_center
            start_radius_base = np.linalg.norm(start_rel_base)
            start_theta_base = np.arctan2(start_rel_base[1], start_rel_base[0])
            theta_jitter = np.deg2rad(float(start_angle_jitter_deg))
            start_theta = start_theta_base + rng.uniform(-theta_jitter, theta_jitter)
            start_radius = start_radius_base * (1.0 + rng.uniform(-start_radius_jitter_frac, start_radius_jitter_frac))
            start_radius = max(start_radius, d_safe * 1.15)
            start_dir = np.array([np.cos(start_theta), np.sin(start_theta)], dtype=float)
            tangent_dir = np.array([-start_dir[1], start_dir[0]], dtype=float)
            start_local = self.obs_center + start_radius * start_dir
            start_local = start_local + tangent_dir * (start_tangent_jitter_frac * seg_len * rng.uniform(-1.0, 1.0))

        subgoal_noise = rng.randn(2) * (subgoal_jitter_frac * seg_len)
        subgoal_noise[1] *= float(subgoal_jitter_y_scale)
        stage1_end_local = self.stage1_end + subgoal_noise
        arc_entry_theta_local = self.terminal_arc_theta_start + rng.uniform(
            -float(arc_entry_jitter_scale),
            float(arc_entry_jitter_scale),
        ) / max(self.terminal_arc_radius, 1e-12)
        arc_entry_local = self._arc_point(arc_entry_theta_local)

        start_rel = start_local - self.obs_center
        stage1_end_rel = stage1_end_local - self.obs_center
        theta_start = np.arctan2(start_rel[1], start_rel[0])
        theta_end = np.arctan2(stage1_end_rel[1], stage1_end_rel[0])
        start_anchor = self._project_to_stage1_clearance_boundary(
            self.obs_center[None, :] + d_safe * start_rel[None, :] / max(np.linalg.norm(start_rel), 1e-12)
        )[0]
        end_anchor = self._project_to_stage1_clearance_boundary(
            self.obs_center[None, :] + d_safe * stage1_end_rel[None, :] / max(np.linalg.norm(stage1_end_rel), 1e-12)
        )[0]

        if use_direct_stage1 and self._segment_stage1_effective_distance(start_local, stage1_end_local) >= direct_safe_radius:
            stage1_ctrl = np.vstack([start_local, stage1_end_local])
        else:
            arc = self._arc_points(theta_start, theta_end, direction, d_safe, n1)
            arc = self._project_to_stage1_clearance_boundary(arc)
            stage1_ctrl = np.vstack([start_local, start_anchor, arc[1:-1], end_anchor, stage1_end_local])
        stage1_ref = resample_polyline(stage1_ctrl, self.stage1_speed_max * stage1_speed_scale_local * self.dt)

        def stage1_projector(path):
            return self._project_to_stage1_clearance_boundary(path)

        stage1 = optimize_trajectory(
            stage1_ref,
            dt=self.dt,
            v_max=self.stage1_speed_max * stage1_speed_scale_local,
            a_max=self.stage1_accel_max,
            projector=stage1_projector,
        )

        stage2_vec = arc_entry_local - stage1_end_local
        stage2_len = np.linalg.norm(stage2_vec) + 1e-12
        tangent = stage2_vec / stage2_len
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        u2 = np.linspace(0.0, 1.0, 14)
        wobble = (
            stage2_lateral_scale
            * stage2_len
            * rng.uniform(0.5, 1.0)
            * np.sin(np.pi * u2)
            * np.sin(2.0 * np.pi * u2 + rng.uniform(0.0, 2.0 * np.pi))
        )
        stage2_ctrl = stage1_end_local[None, :] + u2[:, None] * stage2_vec[None, :] + wobble[:, None] * normal[None, :]
        stage2_ctrl[0] = stage1_end_local
        stage2_ctrl[-1] = arc_entry_local
        stage2_ref = resample_polyline(stage2_ctrl, self.stage2_speed_max * stage2_speed_scale_local * self.dt)
        stage2 = optimize_trajectory(
            stage2_ref,
            dt=self.dt,
            v_max=self.stage2_speed_max * stage2_speed_scale_local,
            a_max=self.stage2_accel_max,
            projector=None,
        )

        stage3_thetas = np.linspace(arc_entry_theta_local, self.terminal_arc_theta_end, 36)
        stage3_ctrl = self.terminal_arc_center[None, :] + self.terminal_arc_radius * np.c_[
            np.cos(stage3_thetas),
            np.sin(stage3_thetas),
        ]
        stage3_ctrl[0] = arc_entry_local
        stage3_ctrl[-1] = self.stage3_end
        stage3_ref = resample_polyline(stage3_ctrl, self.stage3_speed_max * stage3_speed_scale_local * self.dt)
        stage3 = optimize_trajectory(
            stage3_ref,
            dt=self.dt,
            v_max=self.stage3_speed_max * stage3_speed_scale_local,
            a_max=self.stage3_accel_max,
            projector=self._project_to_terminal_arc,
        )

        tau1 = int(len(stage1) - 1)
        tau2 = int(len(stage1) + len(stage2) - 2)
        traj = np.vstack([stage1, stage2[1:], stage3[1:]])

        traj = traj + self._smooth_process_noise(
            len(traj),
            traj.shape[1],
            process_noise_scale * seg_len,
            process_noise_knots,
            rng,
        )
        if self.noise_std > 0.0:
            traj = traj + rng.randn(*traj.shape) * (self.noise_std * 0.4)
            stage1_noisy = repair_trajectory_constraints(
                traj[: tau1 + 1],
                dt=self.dt,
                v_max=self.stage1_speed_max * stage1_speed_scale_local,
                a_max=self.stage1_accel_max,
                projector=stage1_projector,
            )
            stage2_noisy = repair_trajectory_constraints(
                traj[tau1 : tau2 + 1],
                dt=self.dt,
                v_max=self.stage2_speed_max * stage2_speed_scale_local,
                a_max=self.stage2_accel_max,
                projector=None,
                n_rounds=16,
            )
            stage2_uniform = self._uniform_reparameterize_fixed_count(stage2_noisy, len(stage2_noisy))
            stage2_noisy = 0.6 * stage2_noisy + 0.4 * stage2_uniform
            stage2_noisy[0] = traj[tau1]
            stage2_noisy[-1] = traj[tau2]
            stage2_noisy = repair_trajectory_constraints(
                stage2_noisy,
                dt=self.dt,
                v_max=self.stage2_speed_max * stage2_speed_scale_local,
                a_max=self.stage2_accel_max,
                projector=None,
                n_rounds=8,
            )
            stage3_noisy = repair_trajectory_constraints(
                traj[tau2:],
                dt=self.dt,
                v_max=self.stage3_speed_max * stage3_speed_scale_local,
                a_max=self.stage3_accel_max,
                projector=self._project_to_terminal_arc,
            )
            stage3_noisy = self._add_terminal_arc_radial_jitter(
                stage3_noisy,
                stage3_arc_distance_noise_scale * self.terminal_arc_radius,
                rng,
            )
            traj = np.vstack([stage1_noisy, stage2_noisy[1:], stage3_noisy[1:]])

        return traj, np.asarray([tau1, tau2], dtype=int)

    def compute_all_features_matrix(self, traj, feat_ids=None):
        traj = np.asarray(traj, float)
        T = traj.shape[0]

        d_main = self._stage1_effective_distance(traj)

        speeds_edge = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self.dt
        speeds = np.empty(T, dtype=float)
        heading = np.empty(T, dtype=float)
        if T > 1:
            delta = np.diff(traj, axis=0)
            speeds[0] = speeds_edge[0]
            speeds[1:] = speeds_edge
            heading_edge = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
            heading[0] = heading_edge[0]
            heading[1:] = heading_edge
        else:
            speeds[0] = 0.0
            heading[0] = 0.0

        d_arc = self._arc_dist(traj)

        d_decoy = self._decoy_dist(traj)

        t = np.linspace(0.0, 2.0 * np.pi, T)
        phase = float(np.dot(traj.mean(axis=0), self.noise_vec) + self.noise_bias)
        noise_feat = 0.38 * np.sin(5.0 * t + phase)

        F = np.stack([d_main, speeds, d_arc, heading, d_decoy, noise_feat], axis=1)
        return F if feat_ids is None else F[:, feat_ids]

    def compute_features_all(self, traj):
        F = self.compute_all_features_matrix(traj)
        return F[:, 0], F[:, 1]


def load_S3ObsAvoid(
    n_demos: int = 10,
    seed: int = 42,
    env_kwargs=None,
    demo_kwargs=None,
) -> TaskBundle:
    env_cfg = dict(env_kwargs or {})
    run_kwargs = dict(demo_kwargs or {})

    rng = np.random.RandomState(seed)
    env = S3ObsAvoidEnv(**env_cfg)
    demos, true_cutpoints = [], []
    for _ in range(n_demos):
        X, cuts = env.generate_demo(rng=rng, **run_kwargs)
        demos.append(X)
        true_cutpoints.append(np.asarray(cuts, dtype=int))
    return TaskBundle(
        name="S3ObsAvoid",
        demos=demos,
        env=env,
        true_taus=None,
        true_cutpoints=true_cutpoints,
        feature_schema=env.get_feature_schema(),
        true_constraints=env.get_true_constraints(),
        constraint_specs=env.get_constraint_specs(),
        meta={"seed": seed, "task_name": "S3ObsAvoid"},
    )
