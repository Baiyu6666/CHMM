from __future__ import annotations

import numpy as np

from .compat import sample_v20_reserved_length_scales
from .time_parameterization import (
    FixedStepTimeParameterizer,
    concatenate_stage_timestamps,
    gaussian_slowdown_weights,
    sample_polyline_at_distances,
    stabilize_tail_weights,
)


class S5DemoGeneratorMixin:
    def _sample_segment_lengths(self, rng):
        out = []
        for base, jitter in zip(self.seg_lengths, self.seg_length_jitter):
            delta = 0 if int(jitter) <= 0 else int(rng.randint(-int(jitter), int(jitter) + 1))
            out.append(max(int(base) + delta, 8))
        return tuple(out)

    def _orthonormal_frame(self, normal, rng):
        normal = self._unit(normal)
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(normal, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        t1 = np.cross(normal, ref)
        t1 = self._unit(t1)
        if float(np.linalg.norm(t1)) <= 1e-12:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
            t1 = self._unit(np.cross(normal, ref))
        phase = float(rng.uniform(-np.pi, np.pi))
        t1 = self._unit(np.cos(phase) * t1 + np.sin(phase) * np.cross(normal, t1))
        t2 = self._unit(np.cross(normal, t1))
        return normal, t1, t2

    def _normal_from_spherical(self, theta, phi):
        theta = float(theta)
        phi = float(phi)
        return self._unit(
            [
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
            ]
        )

    def _spherical_from_normal(self, normal):
        normal = self._unit(normal)
        theta = float(np.arctan2(normal[1], normal[0]))
        phi = float(np.arccos(np.clip(normal[2], -1.0, 1.0)))
        return theta, phi

    def _make_latitude_surface_path(
        self,
        theta_start,
        theta_end,
        phi,
        num_points,
        *,
        radius_offset=0.0,
        phi_bump=0.0,
    ):
        theta = np.linspace(float(theta_start), float(theta_end), int(num_points), endpoint=True)
        u = np.linspace(0.0, 1.0, int(num_points), endpoint=True)
        phi = float(phi) + float(phi_bump) * np.sin(np.pi * u)
        phi = np.clip(phi, 0.08 * np.pi, 0.48 * np.pi)
        normals = np.stack(
            [
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
            ],
            axis=1,
        )
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        radius = float(self.sphere_radius + float(radius_offset))
        return self.sphere_center[None, :] + radius * normals

    def _slerp_unit(self, u0, u1, num_points, endpoint=True):
        u0 = self._unit(u0)
        u1 = self._unit(u1)
        dots = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
        if dots > 0.9995:
            t = np.linspace(0.0, 1.0, int(num_points), endpoint=endpoint)
            out = (1.0 - t)[:, None] * u0[None, :] + t[:, None] * u1[None, :]
            return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        omega = float(np.arccos(dots))
        sin_omega = float(np.sin(omega))
        t = np.linspace(0.0, 1.0, int(num_points), endpoint=endpoint)
        out = (
            np.sin((1.0 - t) * omega)[:, None] * u0[None, :]
            + np.sin(t * omega)[:, None] * u1[None, :]
        ) / max(sin_omega, 1e-12)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    def _slerp_unit_fraction(self, axis_start, axis_end, fraction):
        axis_start = self._unit(axis_start)
        axis_end = self._unit(axis_end)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        dot = float(np.clip(np.dot(axis_start, axis_end), -1.0, 1.0))
        if dot > 0.9995:
            return self._unit((1.0 - fraction) * axis_start + fraction * axis_end)
        omega = float(np.arccos(dot))
        sin_omega = max(float(np.sin(omega)), 1e-12)
        return self._unit(
            (
                np.sin((1.0 - fraction) * omega) * axis_start
                + np.sin(fraction * omega) * axis_end
            )
            / sin_omega
        )

    def _make_spherical_shell_path(self, n_start, n_end, num_points, *, radius_offset=0.0, detour_angle=0.0):
        normals = self._slerp_unit(n_start, n_end, num_points, endpoint=True)
        detour_angle = float(max(detour_angle, 0.0))
        if detour_angle > 1e-8 and len(normals) > 2:
            axis = np.cross(self._unit(n_start), self._unit(n_end))
            if float(np.linalg.norm(axis)) <= 1e-8:
                _, _, detour_dir = self._orthonormal_frame(n_start, np.random.RandomState(0))
            else:
                axis = self._unit(axis)
                detour_dir = np.cross(axis, normals)
                detour_dir = detour_dir / np.maximum(np.linalg.norm(detour_dir, axis=1, keepdims=True), 1e-12)
            u = np.linspace(0.0, 1.0, len(normals), endpoint=True)
            bend = detour_angle * np.sin(np.pi * u)
            normals = (
                np.cos(bend)[:, None] * normals
                + np.sin(bend)[:, None] * detour_dir
            )
            normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        radius = float(self.sphere_radius + float(radius_offset))
        return self.sphere_center[None, :] + radius * normals

    def _build_stage3_transition(self, n_start, n_end, n_points, shell_offset, rng):
        n_points = int(max(n_points, 8))
        n_start = self._unit(n_start)
        n_end = self._unit(n_end)
        normals = self._slerp_unit(n_start, n_end, n_points, endpoint=True)
        if len(normals) > 2:
            phase = float(rng.uniform(-0.25 * np.pi, 0.25 * np.pi))
            u = np.linspace(0.0, 1.0, len(normals), endpoint=True)
            envelope = np.sin(np.pi * u) ** 1.05
            wiggle = 0.035 * np.sin(2.2 * np.pi * u + phase) * envelope
            ref = np.cross(n_start, n_end)
            if float(np.linalg.norm(ref)) <= 1e-8:
                _, _, ref = self._orthonormal_frame(n_start, rng)
            ref = self._unit(ref)
            detour = np.cross(ref[None, :], normals)
            detour = detour / np.maximum(np.linalg.norm(detour, axis=1, keepdims=True), 1e-12)
            normals = normals + wiggle[:, None] * detour
            normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

        u = np.linspace(0.0, 1.0, len(normals), endpoint=True)
        radial_progress = u
        radius = self.sphere_radius + float(shell_offset) * radial_progress
        return self.sphere_center[None, :] + radius[:, None] * normals

    def _time_parameterizer(self) -> FixedStepTimeParameterizer:
        return FixedStepTimeParameterizer(
            dt=self.dt,
            segment_count_slack=self.segment_count_slack,
        )

    @staticmethod
    def _make_deliberate_slowdown_weights(num_edges: int, *, depth, center, width) -> np.ndarray:
        return gaussian_slowdown_weights(
            num_edges,
            depth=depth,
            center=center,
            width=width,
        )

    def _sample_stage3_speed_intent_weights(self, num_edges: int, rng) -> np.ndarray:
        n = int(max(num_edges, 1))
        if n == 1:
            return np.ones(1, dtype=float)
        local_rng = np.random if rng is None else rng
        noise = np.asarray(local_rng.randn(n), dtype=float)
        noise = self._smooth_trace(noise, kernel_size=self.stage3_speed_jitter_kernel)
        noise = noise - float(np.mean(noise))
        scale = float(np.std(noise))
        if scale > 1e-8:
            noise = noise / scale
        jitter_std = float(max(self.stage3_speed_jitter_std, 0.0))
        clip = float(max(self.stage3_speed_jitter_clip, 0.0))
        weights = 1.0 + jitter_std * noise
        if clip > 0.0:
            weights = np.clip(weights, 1.0 - clip, 1.0 + clip)
        weights = self._smooth_trace(weights, kernel_size=3)
        mean = float(np.mean(weights))
        if mean > 1e-8:
            weights = weights / mean
        return np.clip(weights, 1e-6, None)

    def _stage4_speed_intent_weights(self, num_edges: int) -> np.ndarray:
        return stabilize_tail_weights(
            self._make_deliberate_slowdown_weights(
                num_edges,
                depth=self.stage4_speed_valley_depth,
                center=self.stage4_speed_valley_center,
                width=self.stage4_speed_valley_width,
            ),
            tail_len=2,
            floor_ratio=0.98,
        )

    def _time_parameterize_fixed_count(self, path, num_points: int, *, speed_intent=None):
        return self._time_parameterizer().parameterize_fixed_count(
            path,
            num_points,
            speed_intent=speed_intent,
        )

    def _project_to_shell(self, path, *, shell_offset):
        pts = np.asarray(path, dtype=float)
        rel = pts - self.sphere_center[None, :]
        rel_norm = np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
        shell_radius = float(self.sphere_radius + shell_offset)
        return self.sphere_center[None, :] + shell_radius * rel / rel_norm

    def _soften_stage3_radial_profile(self, path, *, shell_offset, blend=0.7):
        pts = np.asarray(path, dtype=float)
        if len(pts) <= 1:
            return pts.copy()
        rel = pts - self.sphere_center[None, :]
        rel_norm = np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
        normals = rel / rel_norm
        u = np.linspace(0.0, 1.0, len(pts), endpoint=True)
        current_radius = rel_norm.reshape(-1)
        linear_radius = self.sphere_radius + float(shell_offset) * u
        blend = float(np.clip(blend, 0.0, 1.0))
        radius = (1.0 - blend) * current_radius + blend * linear_radius
        return self.sphere_center[None, :] + radius[:, None] * normals

    def _regularize_stage3_transition_path(
        self,
        path,
        *,
        shell_offset,
        radial_blend=0.8,
        normal_kernel=5,
        dense_factor=6,
        speed_intent=None,
    ):
        pts = np.asarray(path, dtype=float)
        target_count = len(pts)
        if target_count <= 1:
            return pts.copy()
        if target_count <= 3:
            out = self._soften_stage3_radial_profile(pts, shell_offset=shell_offset, blend=radial_blend)
            return self._time_parameterize_fixed_count(
                out,
                target_count,
                speed_intent=speed_intent,
            ).positions

        rel = pts - self.sphere_center[None, :]
        rel_norm = np.maximum(np.linalg.norm(rel, axis=1), 1e-12)
        normals = rel / rel_norm[:, None]

        smooth_normals = np.stack(
            [self._smooth_trace(normals[:, dim], kernel_size=normal_kernel) for dim in range(normals.shape[1])],
            axis=1,
        )
        smooth_normals[0] = normals[0]
        smooth_normals[-1] = normals[-1]
        smooth_normals = smooth_normals / np.maximum(np.linalg.norm(smooth_normals, axis=1, keepdims=True), 1e-12)

        u = np.linspace(0.0, 1.0, target_count, endpoint=True)
        linear_radius = self.sphere_radius + float(shell_offset) * u
        blend = float(np.clip(radial_blend, 0.0, 1.0))
        radius = (1.0 - blend) * rel_norm + blend * linear_radius
        radius[0] = self.sphere_radius
        radius[-1] = self.sphere_radius + float(shell_offset)

        dense_count = max(int(target_count) * int(max(dense_factor, 1)), 32)
        dense_u = np.linspace(0.0, 1.0, dense_count, endpoint=True)
        dense_normals = np.stack(
            [np.interp(dense_u, u, smooth_normals[:, dim]) for dim in range(smooth_normals.shape[1])],
            axis=1,
        )
        dense_normals = dense_normals / np.maximum(np.linalg.norm(dense_normals, axis=1, keepdims=True), 1e-12)
        dense_radius = np.interp(dense_u, u, radius)
        dense_path = self.sphere_center[None, :] + dense_radius[:, None] * dense_normals
        return self._time_parameterize_fixed_count(
            dense_path,
            target_count,
            speed_intent=speed_intent,
        ).positions

    def _regularize_tail_spacing(self, path, *, tail_points: int = 5):
        pts = np.asarray(path, dtype=float).copy()
        tail_points = int(max(tail_points, 0))
        if len(pts) < max(tail_points, 3):
            return pts
        start = len(pts) - tail_points
        tail = pts[start:].copy()
        edges = np.linalg.norm(np.diff(tail, axis=0), axis=1)
        total = float(np.sum(edges))
        if total <= 1e-10:
            return pts
        dists = np.linspace(0.0, total, len(tail), endpoint=True)
        pts[start:] = sample_polyline_at_distances(tail, dists)
        return pts

    def _repair_stage4_departure_tail(self, path, departure_target, *, shell_offset, tail_points: int = 9):
        pts = np.asarray(path, dtype=float).copy()
        if len(pts) < 4:
            return pts
        tail_points = int(max(tail_points, 2))
        departure_target = np.asarray(departure_target, dtype=float).reshape(3)
        shell_radius = float(self.sphere_radius + shell_offset)

        rel = pts - self.sphere_center[None, :]
        normals = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
        pts = self.sphere_center[None, :] + shell_radius * normals
        end_normal = self._unit(normals[-1])
        dep = departure_target - pts[-1]
        dep_norm = float(np.linalg.norm(dep))
        if dep_norm <= 1e-10:
            return pts
        dep_u = dep / dep_norm

        check_start = max(0, len(pts) - tail_points)
        tail_edges = np.diff(pts[check_start:], axis=0)
        if len(tail_edges) > 0 and float(np.min(tail_edges @ dep_u)) >= -1e-6:
            return pts

        original_edge_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        nominal_step = float(np.median(original_edge_lengths)) if len(original_edge_lengths) else 0.0
        nominal_step = max(nominal_step, 1e-6)
        best = pts
        best_min_proj = -float("inf")
        first_start = max(1, len(pts) - tail_points)
        for start in range(first_start, 0, -1):
            candidate = pts.copy()
            anchor_normal = self._unit(normals[start - 1])
            repaired_normals = self._slerp_unit(anchor_normal, end_normal, len(pts) - start + 1, endpoint=True)
            candidate[start:] = self.sphere_center[None, :] + shell_radius * repaired_normals[1:]
            candidate[-1] = self.sphere_center + shell_radius * end_normal
            candidate_edges = np.diff(candidate[check_start:], axis=0)
            min_proj = float(np.min(candidate_edges @ dep_u)) if len(candidate_edges) else 0.0
            candidate_step = float(np.max(np.linalg.norm(np.diff(candidate[start - 1 :], axis=0), axis=1)))
            if min_proj > best_min_proj:
                best_min_proj = min_proj
                best = candidate
            if min_proj >= -1e-6:
                return candidate

        tangent = dep_u - float(np.dot(dep_u, end_normal)) * end_normal
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > 1e-10:
            tangent = tangent / tangent_norm
            for start in range(len(pts) - 2, first_start - 1, -1):
                candidate = pts.copy()
                anchor_normal = self._unit(normals[start - 1])
                span = float(np.dot(end_normal - anchor_normal, tangent))
                if span <= 1e-4:
                    continue
                span = float(np.clip(span, 0.02, 0.18))
                alphas = np.linspace(span, 0.0, len(pts) - start, endpoint=True)
                repaired_normals = end_normal[None, :] - alphas[:, None] * tangent[None, :]
                repaired_normals = repaired_normals / np.maximum(
                    np.linalg.norm(repaired_normals, axis=1, keepdims=True), 1e-12
                )
                candidate[start:] = self.sphere_center[None, :] + shell_radius * repaired_normals
                candidate[-1] = self.sphere_center + shell_radius * end_normal
                candidate_edges = np.diff(candidate[check_start:], axis=0)
                min_proj = float(np.min(candidate_edges @ dep_u)) if len(candidate_edges) else 0.0
                candidate_step = float(np.max(np.linalg.norm(np.diff(candidate[start - 1 :], axis=0), axis=1)))
                if min_proj > best_min_proj:
                    best_min_proj = min_proj
                    best = candidate
                if min_proj >= -1e-6 and candidate_step <= 3.0 * nominal_step:
                    return candidate
        return best

    def _time_parameterize_path(
        self,
        path,
        speed_limit,
        acceleration_limit,
        *,
        target_speed=None,
        nominal_count=None,
        enforce_motion_limits=True,
        speed_intent=None,
    ):
        return self._time_parameterizer().parameterize(
            path,
            speed_limit=speed_limit,
            acceleration_limit=acceleration_limit,
            target_speed=target_speed,
            nominal_count=nominal_count,
            enforce_motion_limits=enforce_motion_limits,
            speed_intent=speed_intent,
        )

    @staticmethod
    def _build_stage1_geometry(p_start, p_contact):
        mid = 0.35 * np.asarray(p_start, dtype=float) + 0.65 * np.asarray(p_contact, dtype=float)
        return np.vstack([p_start, mid, p_contact])

    def _stage1_speed_intent_weights(self, num_edges: int, *, speed_limit=None) -> np.ndarray:
        n = int(max(num_edges, 1))
        if n == 1:
            return np.ones(1, dtype=float)
        taper_fraction = float(np.clip(self.stage1_speed_taper_fraction, 0.0, 1.0))
        if taper_fraction <= 1e-8:
            return np.ones(n, dtype=float)
        if self.stage1_speed_taper_end_ratio is None:
            stage1_v = self.stage1_target_speed_ratio * float(
                self.stage1_speed_max if speed_limit is None else speed_limit
            )
            stage2_v = self.stage2_target_speed_ratio * float(self.stage2_speed_max)
            end_ratio = stage2_v / max(stage1_v, 1e-8)
            end_ratio = float(np.clip(end_ratio, 0.72, 0.92))
        else:
            end_ratio = float(np.clip(self.stage1_speed_taper_end_ratio, 0.45, 1.0))
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        start = 1.0 - taper_fraction
        alpha = np.clip((u - start) / max(taper_fraction, 1e-8), 0.0, 1.0)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        weights = 1.0 - (1.0 - end_ratio) * smooth
        return np.clip(weights, 0.45, None)

    def _build_departure_geometry(self, p_start, n_start, rng):
        normal, t1, t2 = self._orthonormal_frame(n_start, rng)
        depart_radius = self.sphere_radius + self.depart_offset * rng.uniform(0.85, 1.15)
        lateral = 0.10 * self.sphere_radius * rng.uniform(-1.0, 1.0)
        vertical = 0.08 * self.sphere_radius * rng.uniform(-1.0, 1.0)
        p_end = (
            self.sphere_center
            + depart_radius * normal
            + lateral * t1
            + vertical * t2
        )
        ctrl = np.vstack([p_start, 0.55 * p_start + 0.45 * p_end, p_end])
        return ctrl

    def _interpolate_unit_axes(self, axis_start, axis_end, num_points):
        return self._slerp_unit(axis_start, axis_end, num_points, endpoint=True)

    def _make_irregular_axis_transition(self, axis_start, axis_end, num_points, rng, max_tilt):
        base = self._interpolate_unit_axes(axis_start, axis_end, num_points)
        n = len(base)
        if n == 0:
            return base
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        env = np.sin(np.pi * u) ** 0.78
        tilt = (
            0.30
            + 0.46 * np.sin(3.4 * np.pi * u - 0.15 * np.pi)
            + 0.24 * np.sin(7.2 * np.pi * u + 0.33 * np.pi)
            + 0.16 * np.sin(11.4 * np.pi * u - 0.52 * np.pi)
        )
        tilt = env * np.abs(tilt)
        tilt = self._smooth_trace(tilt, kernel_size=3)
        tilt = np.clip(tilt * float(max_tilt), 0.0, float(max_tilt))

        tangent_phase = (
            float(rng.uniform(-np.pi, np.pi))
            + 1.10 * np.sin(4.2 * np.pi * u)
            + 0.75 * np.sin(8.6 * np.pi * u - 0.40)
            + 0.35 * np.sin(14.0 * np.pi * u + 0.25)
        )
        tangent_phase = tangent_phase + self._smooth_noise(rng, n, scale=0.22, kernel_size=3)

        out = np.empty_like(base)
        for i, axis in enumerate(base):
            axis = self._unit(axis)
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(axis, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            t1 = self._unit(np.cross(axis, ref))
            t2 = self._unit(np.cross(axis, t1))
            tangent = np.cos(tangent_phase[i]) * t1 + np.sin(tangent_phase[i]) * t2
            out[i] = self._unit(np.cos(tilt[i]) * axis + np.sin(tilt[i]) * tangent)
        out[0] = self._unit(axis_start)
        out[-1] = self._unit(axis_end)
        return out

    def _make_emitted_axis_transition(self, axis_start, axis_end, num_points, rng, max_tilt):
        num_points = int(num_points)
        if num_points <= 0:
            return np.empty((0, 3), dtype=float)
        if int(getattr(self, "preset_version", 23)) < 23:
            return self._make_irregular_axis_transition(
                axis_start,
                axis_end,
                num_points,
                rng=rng,
                max_tilt=max_tilt,
            )
        shared_boundary_trace = self._make_irregular_axis_transition(
            axis_start,
            axis_end,
            num_points,
            rng=rng,
            max_tilt=max_tilt,
        )
        if num_points == 1:
            return np.asarray(shared_boundary_trace, dtype=float)
        source_progress = np.linspace(0.0, 1.0, num_points, endpoint=True)
        emitted_progress = np.linspace(1.0 / float(num_points), 1.0, num_points, endpoint=True)
        emitted = np.empty_like(shared_boundary_trace)
        for sample_idx, progress in enumerate(emitted_progress):
            right_idx = min(int(np.searchsorted(source_progress, progress, side="right")), num_points - 1)
            left_idx = max(right_idx - 1, 0)
            span = max(float(source_progress[right_idx] - source_progress[left_idx]), 1e-12)
            fraction = float((progress - source_progress[left_idx]) / span)
            emitted[sample_idx] = self._slerp_unit_fraction(
                shared_boundary_trace[left_idx],
                shared_boundary_trace[right_idx],
                fraction,
            )
        return emitted

    def _v20_stage2_normal_error_angles(self, length: int, max_error: float) -> np.ndarray:
        length = int(length)
        angle_margin = self._make_stage_margin_profile(
            length,
            offset=0.03 * max_error,
            amplitude=0.52 * max_error,
            cycles=4.6,
            phase=0.0,
            noise_scale=0.0,
            rng=None,
            kernel_size=1,
        )
        angle = 1.00 * max_error - angle_margin
        return np.clip(angle, 0.48 * max_error, 0.99 * max_error)

    def _sample_stage2_normal_error_angles(self, length: int, rng, max_error: float):
        template = self._v20_stage2_normal_error_angles(length, max_error)
        if self.stage2_normal_error_policy == "fixed_periodic_v20":
            return template, {"policy": "fixed_periodic_v20"}

        progress = np.linspace(0.0, 1.0, int(length), endpoint=True)
        if self.stage2_normal_error_policy == "periodic_quantile_matched_v21":
            cycles_std = 0.35
            cycles = float(
                np.clip(
                    rng.normal(4.6, cycles_std),
                    4.6 - 2.0 * cycles_std,
                    4.6 + 2.0 * cycles_std,
                )
            )
            profile_phase = float(rng.uniform(-np.pi, np.pi))
            drift_phase = float(rng.uniform(-np.pi, np.pi))
            carrier = np.sin(2.0 * np.pi * cycles * progress + profile_phase)
            carrier = carrier + 0.18 * np.sin(1.5 * np.pi * progress + drift_phase)
            carrier_metadata = {
                "policy": "periodic_quantile_matched_v21",
                "cycles": cycles,
                "profile_phase": profile_phase,
                "drift_phase": drift_phase,
            }
        else:
            count_lo, count_hi = self.stage2_normal_error_control_point_count_range
            control_point_count = int(rng.randint(count_lo, count_hi + 1))
            control_progress = np.linspace(0.0, 1.0, control_point_count, endpoint=True)
            if control_point_count > 2:
                spacing = 1.0 / float(control_point_count - 1)
                jitter = rng.uniform(
                    -0.35 * spacing,
                    0.35 * spacing,
                    size=control_point_count - 2,
                )
                control_progress[1:-1] = np.sort(control_progress[1:-1] + jitter)
            control_values = np.asarray(rng.randn(control_point_count), dtype=float)
            carrier = np.interp(progress, control_progress, control_values)
            samples_per_span = float(max(int(length) - 1, 1)) / float(
                max(control_point_count - 1, 1)
            )
            smoothing_kernel = max(3, int(round(samples_per_span)))
            if smoothing_kernel % 2 == 0:
                smoothing_kernel += 1
            carrier = self._smooth_trace(carrier, kernel_size=smoothing_kernel)
            carrier_metadata = {
                "policy": "random_control_points_quantile_matched",
                "control_point_count": control_point_count,
                "control_point_progress": control_progress.tolist(),
                "control_point_values": control_values.tolist(),
                "smoothing_kernel": smoothing_kernel,
            }

        order = np.argsort(carrier, kind="mergesort")
        angle = np.empty_like(template)
        angle[order] = np.sort(template)

        depth_std = float(self.stage2_normal_error_depth_scale_std)
        depth_scale = float(
            np.clip(
                rng.normal(1.0, depth_std),
                1.0 - 2.0 * depth_std,
                1.0 + 2.0 * depth_std,
            )
        )
        bias_std = float(self.stage2_normal_error_bias_std)
        bias_ratio = float(
            np.clip(
                rng.normal(0.0, bias_std),
                -2.0 * bias_std,
                2.0 * bias_std,
            )
        )
        upper = 0.99 * max_error
        angle = upper + bias_ratio * max_error - depth_scale * (upper - angle)
        angle = np.clip(angle, 0.48 * max_error, upper)
        if len(angle) > 0:
            angle[0] = template[0]
            angle[-1] = template[-1]
        return angle, {
            **carrier_metadata,
            "depth_scale": depth_scale,
            "bias_ratio": bias_ratio,
        }

    def _make_aligned_axis_trace(self, normals, rng, max_error):
        normals = np.asarray(normals, dtype=float)
        n = len(normals)
        out = np.empty_like(normals)
        if n == 0:
            return out, {"policy": str(self.stage2_normal_error_policy)}
        max_error = float(max(max_error, 1e-4))
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        phase0 = float(rng.uniform(-np.pi, np.pi))
        phase_fraction = float(np.clip((phase0 + np.pi) / (2.0 * np.pi), 0.0, 1.0))
        profile_seed = int(phase_fraction * float(np.iinfo(np.uint32).max))
        profile_rng = np.random.RandomState(profile_seed)
        angle, profile_metadata = self._sample_stage2_normal_error_angles(n, profile_rng, max_error)

        tangent_phase = (
            phase0
            + 1.15 * np.sin(2.9 * np.pi * u + 0.18 * phase0)
            + 0.55 * np.sin(5.8 * np.pi * u - 0.33 * phase0)
        )
        tangent_phase = self._smooth_trace(tangent_phase, kernel_size=7)

        for i, normal in enumerate(normals):
            normal = self._unit(normal)
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(normal, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            t1 = self._unit(np.cross(normal, ref))
            t2 = self._unit(np.cross(normal, t1))
            tangent = np.cos(tangent_phase[i]) * t1 + np.sin(tangent_phase[i]) * t2
            out[i] = self._unit(np.cos(angle[i]) * normal + np.sin(angle[i]) * tangent)
        profile_metadata["tangent_phase"] = phase0
        profile_metadata["profile_seed"] = profile_seed
        return out, profile_metadata

    def _sample_axis_near_normal(self, normal, rng, max_error, min_fraction=0.35):
        normal = self._unit(normal)
        max_error = float(max(max_error, 0.0))
        if max_error <= 1e-8:
            return normal
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(normal, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        t1 = self._unit(np.cross(normal, ref))
        t2 = self._unit(np.cross(normal, t1))
        phase = float(rng.uniform(-np.pi, np.pi))
        tangent = self._unit(np.cos(phase) * t1 + np.sin(phase) * t2)
        lo = float(np.clip(min_fraction, 0.0, 1.0)) * max_error
        angle = float(rng.uniform(lo, max_error))
        return self._unit(np.cos(angle) * normal + np.sin(angle) * tangent)

    def _generate_tool_axis_trace(self, traj, stage_lengths, normals_stage2, normals_stage3, n_contact, rng):
        lengths = [int(x) for x in stage_lengths]
        if len(lengths) != 5:
            raise ValueError(f"S5 formal generator expects five stage lengths, got {len(lengths)}.")
        total = int(sum(stage_lengths))
        axis_start = self._unit(0.76 * self._unit(n_contact) + 0.24 * self._unit(rng.randn(3)))
        l1, l2, l3, l4, l5 = lengths
        stage1 = self._make_irregular_axis_transition(
            axis_start,
            n_contact,
            l1,
            rng=rng,
            max_tilt=0.38 * float(self.tool_align_max_stage2),
        )
        stage2, stage2_profile_metadata = self._make_aligned_axis_trace(
            normals_stage2,
            rng,
            max_error=self.tool_align_max_stage2,
        )
        mid_anchor = self._unit(0.72 * self._unit(normals_stage3[0]) + 0.28 * self._unit(rng.randn(3)))
        stage3 = self._make_emitted_axis_transition(
            stage2[-1],
            mid_anchor,
            l3,
            rng=rng,
            max_tilt=0.46 * float(self.tool_align_max_stage2),
        )
        free_axis = self._sample_axis_near_normal(
            normals_stage3[-1],
            rng,
            max_error=self.stage4_tool_normal_max_error,
            min_fraction=0.45,
        )
        stage4 = self._make_emitted_axis_transition(
            stage3[-1],
            free_axis,
            l4,
            rng=rng,
            max_tilt=0.92 * float(self.tool_align_max_stage2),
        )
        s5_start = int(l1 + l2 + l3 + l4)
        stage5_rel = np.asarray(traj[s5_start : s5_start + l5], dtype=float) - self.sphere_center[None, :]
        if len(stage5_rel) > 0:
            stage5_normals = stage5_rel / np.maximum(np.linalg.norm(stage5_rel, axis=1, keepdims=True), 1e-12)
            stage5_target_normal = stage5_normals[-1]
        else:
            stage5_target_normal = normals_stage3[-1]
        stage5_end = self._sample_axis_near_normal(
            stage5_target_normal,
            rng,
            max_error=self.stage5_tool_normal_max_error,
            min_fraction=0.25,
        )
        stage5 = self._make_emitted_axis_transition(
            stage4[-1],
            stage5_end,
            l5,
            rng=rng,
            max_tilt=0.36 * float(self.tool_align_max_stage2),
        )
        axis = np.vstack([stage1, stage2, stage3, stage4, stage5])
        if axis.shape[0] != total:
            axis = np.asarray(axis[:total], dtype=float)
        return axis, {
            "stage2_normal_error": stage2_profile_metadata,
            "stage_boundary_sampling": "drop_shared_orientation_start",
        }

    def generate_demo(self, rng=None, return_metadata=False, **kwargs):
        rng = np.random if rng is None else rng
        demo_index = kwargs.get("demo_index", None)
        l1, l2, l3, l4 = self._sample_segment_lengths(rng)

        phi0 = float(rng.uniform(self.contact_phi_range[0], self.contact_phi_range[1]))
        trace_angle = float(rng.uniform(self.stage2_trace_angle_range[0], self.stage2_trace_angle_range[1]))
        lateral_sign = -1.0 if float(rng.rand()) < 0.5 else 1.0
        sin_phi = max(float(np.sin(phi0)), 1e-6)
        delta_theta = min(float(trace_angle) / sin_phi, 0.92 * np.pi)
        theta_center = float(self.stage2_lateral_center_theta)
        bump_lo, bump_hi = self.stage2_lateral_phi_bump_range
        phi_bump = float(rng.uniform(bump_lo, bump_hi))
        theta0 = theta_center - 0.5 * lateral_sign * delta_theta
        theta1 = theta_center + 0.5 * lateral_sign * delta_theta
        n_contact = self._normal_from_spherical(theta0, phi0)
        n_trace_end = self._normal_from_spherical(theta1, phi0)
        normal0 = n_contact
        t1 = self._unit(
            [
                -lateral_sign * np.sin(theta0) * np.sin(phi0),
                lateral_sign * np.cos(theta0) * np.sin(phi0),
                0.0,
            ]
        )
        t2 = self._unit(np.cross(normal0, t1))
        stage2_latitude_path = (theta0, theta1, phi0, phi_bump)

        theta_trace_end, phi_trace_end = self._spherical_from_normal(n_trace_end)
        top_phi_lo, top_phi_hi = self.stage345_top_phi_range
        phi_cap = float(rng.uniform(min(top_phi_lo, top_phi_hi), max(top_phi_lo, top_phi_hi)))
        theta_pull = float(np.clip(self.stage345_top_theta_pull, 0.0, 1.0))
        theta_repos = (
            (1.0 - theta_pull) * theta_trace_end
            + theta_pull * float(self.stage2_lateral_center_theta)
            + float(rng.uniform(-self.stage345_top_theta_jitter, self.stage345_top_theta_jitter))
        )
        n_repos_end = self._normal_from_spherical(theta_repos, phi_cap)

        p_contact = self.sphere_center + self.sphere_radius * n_contact
        p_start = (
            self.sphere_center
            + (self.sphere_radius + self.approach_offset * rng.uniform(0.85, 1.15)) * n_contact
            + 0.18 * self.sphere_radius * rng.uniform(-1.0, 1.0) * t1
            + 0.12 * self.sphere_radius * rng.uniform(-1.0, 1.0) * t2
        )

        stage1_geometry = self._build_stage1_geometry(p_start, p_contact)
        stage1_timing = self._time_parameterize_path(
            stage1_geometry,
            speed_limit=self.stage1_speed_max,
            acceleration_limit=self.stage1_accel_max,
            target_speed=self.stage1_target_speed_ratio * float(self.stage1_speed_max),
            nominal_count=l1,
            speed_intent=lambda n: self._stage1_speed_intent_weights(
                n,
                speed_limit=self.stage1_speed_max,
            ),
        )
        stage1 = stage1_timing.positions
        stage2_length_scale, stage4_length_scale = sample_v20_reserved_length_scales(rng)
        stage2_raw = self._make_latitude_surface_path(
            stage2_latitude_path[0],
            stage2_latitude_path[1],
            stage2_latitude_path[2],
            max(int(4 * l2), 96),
            phi_bump=stage2_latitude_path[3],
        )
        stage2_timing = self._time_parameterize_path(
            stage2_raw,
            speed_limit=self.stage2_speed_max / stage2_length_scale,
            acceleration_limit=self.stage2_accel_max / stage2_length_scale,
            target_speed=self.stage2_target_speed_ratio * self.stage2_speed_max,
            nominal_count=l2,
            enforce_motion_limits=False,
            speed_intent=lambda n: self._make_deliberate_slowdown_weights(
                n,
                depth=self.stage2_speed_valley_depths,
                center=self.stage2_speed_valley_centers,
                width=self.stage2_speed_valley_widths,
            ),
        )
        stage2 = stage2_timing.positions
        shell_offset = float(self.true_constraints["surface_near_target"])
        shell_blend = float(rng.uniform(self.stage3_shell_blend_range[0], self.stage3_shell_blend_range[1]))
        n_shell_start = self._unit((1.0 - shell_blend) * n_trace_end + shell_blend * n_repos_end)
        stage3_raw = self._build_stage3_transition(
            n_trace_end,
            n_shell_start,
            max(int(5 * l3), 96),
            shell_offset=shell_offset,
            rng=rng,
        )
        stage3_timing = self._time_parameterize_path(
            stage3_raw,
            speed_limit=self.stage3_speed_max,
            acceleration_limit=self.stage3_accel_max,
            target_speed=self.stage3_target_speed_ratio * self.stage3_speed_max,
            nominal_count=l3,
            enforce_motion_limits=False,
        )
        stage3 = stage3_timing.positions
        stage3_intent_weights = self._sample_stage3_speed_intent_weights(max(len(stage3) - 1, 1), rng)
        stage3 = self._regularize_stage3_transition_path(
            stage3,
            shell_offset=shell_offset,
            radial_blend=0.8,
            speed_intent=lambda n, w=stage3_intent_weights: w,
        )
        stage4_raw = self._make_spherical_shell_path(
            n_shell_start,
            n_repos_end,
            max(int(5 * l4), 96),
            radius_offset=shell_offset,
            detour_angle=float(max(self.stage4_shell_detour_angle, 0.0)),
        )
        stage4_timing = self._time_parameterize_path(
            stage4_raw,
            speed_limit=self.stage2_speed_max / stage4_length_scale,
            acceleration_limit=self.stage3_accel_max / stage4_length_scale,
            target_speed=self.stage4_target_speed_ratio * self.stage2_speed_max,
            nominal_count=None,
            enforce_motion_limits=False,
            speed_intent=self._stage4_speed_intent_weights,
        )
        stage4 = stage4_timing.positions
        stage4 = self._regularize_tail_spacing(stage4, tail_points=5)
        stage5_geometry = self._build_departure_geometry(stage4[-1], n_repos_end, rng=rng)
        stage4 = self._repair_stage4_departure_tail(
            stage4,
            stage5_geometry[-1],
            shell_offset=shell_offset,
            tail_points=min(5, max(2, len(stage4) // 2)),
        )
        stage4_timing = self._time_parameterize_path(
            stage4,
            speed_limit=self.stage2_speed_max / stage4_length_scale,
            acceleration_limit=self.stage3_accel_max / stage4_length_scale,
            target_speed=self.stage4_target_speed_ratio * self.stage2_speed_max,
            nominal_count=None,
            enforce_motion_limits=False,
            speed_intent=self._stage4_speed_intent_weights,
        )
        stage4 = stage4_timing.positions
        stage4 = self._project_to_shell(stage4, shell_offset=shell_offset)
        stage5_timing = self._time_parameterize_path(
            stage5_geometry,
            speed_limit=self.stage4_speed_max,
            acceleration_limit=self.stage4_accel_max,
            target_speed=self.stage5_target_speed_ratio * self.stage4_speed_max,
            nominal_count=l4,
        )
        stage5 = stage5_timing.positions
        traj = np.vstack([stage1, stage2[1:], stage3[1:], stage4[1:], stage5[1:]])
        if self.noise_std > 0.0:
            noise = np.stack(
                [
                    self._smooth_noise(rng, len(traj), scale=self.noise_std, kernel_size=self.trajectory_noise_kernel)
                    for _ in range(traj.shape[1])
                ],
                axis=1,
            )
            stage_noise_scale = np.ones(len(traj), dtype=float)
            stage2_slice = slice(len(stage1), len(stage1) + len(stage2) - 1)
            stage4_slice = slice(
                len(stage1) + len(stage2) + len(stage3) - 2,
                len(stage1) + len(stage2) + len(stage3) + len(stage4) - 3,
            )
            stage_noise_scale[stage2_slice] *= float(self.stage2_noise_scale)
            stage_noise_scale[stage4_slice] *= float(self.stage4_noise_scale)
            traj = traj + noise * stage_noise_scale[:, None]
            radii = np.linalg.norm(traj - self.sphere_center[None, :], axis=1)
            contact_mask = slice(len(stage1), len(stage1) + len(stage2) - 1)
            safe = np.maximum(radii[contact_mask], 1e-12)
            traj[contact_mask] = self.sphere_center[None, :] + (
                self.sphere_radius * (traj[contact_mask] - self.sphere_center[None, :]) / safe[:, None]
            )
            shell_offset = float(self.true_constraints["surface_near_target"])
            stage3_slice_after_noise = slice(
                len(stage1) + len(stage2) - 2,
                len(stage1) + len(stage2) + len(stage3) - 2,
            )
            if (stage3_slice_after_noise.stop - stage3_slice_after_noise.start) >= 4:
                traj[stage3_slice_after_noise] = self._regularize_stage3_transition_path(
                    traj[stage3_slice_after_noise],
                    shell_offset=shell_offset,
                    radial_blend=0.9,
                    speed_intent=lambda n, w=stage3_intent_weights: w,
                )
            stage4_slice_after_noise = slice(
                len(stage1) + len(stage2) + len(stage3) - 2,
                len(stage1) + len(stage2) + len(stage3) + len(stage4) - 3,
            )
            if (stage4_slice_after_noise.stop - stage4_slice_after_noise.start) >= 5:
                stage4_tail_fixed = self._regularize_tail_spacing(
                    traj[stage4_slice_after_noise],
                    tail_points=5,
                )
                stage4_tail_fixed = self._project_to_shell(stage4_tail_fixed, shell_offset=shell_offset)
                departure_target = traj[min(stage4_slice_after_noise.stop, len(traj) - 1)]
                stage4_tail_fixed = self._repair_stage4_departure_tail(
                    stage4_tail_fixed,
                    departure_target,
                    shell_offset=shell_offset,
                    tail_points=min(5, max(2, len(stage4_tail_fixed) // 2)),
                )
                stage4_tail_fixed = self._time_parameterize_fixed_count(
                    stage4_tail_fixed,
                    len(stage4_tail_fixed),
                    speed_intent=self._stage4_speed_intent_weights,
                ).positions
                stage4_tail_fixed = self._project_to_shell(stage4_tail_fixed, shell_offset=shell_offset)
                traj[stage4_slice_after_noise] = stage4_tail_fixed

        true_cutpoints = np.asarray(
            [
                int(len(stage1) - 1),
                int(len(stage1) + len(stage2) - 2),
                int(len(stage1) + len(stage2) + len(stage3) - 3),
                int(len(stage1) + len(stage2) + len(stage3) + len(stage4) - 4),
            ],
            dtype=int,
        )
        stage_lengths = (len(stage1), len(stage2) - 1, len(stage3) - 1, len(stage4) - 1, len(stage5) - 1)
        stage2_slice = slice(len(stage1), len(stage1) + len(stage2) - 1)
        stage4_slice = slice(
            len(stage1) + len(stage2) + len(stage3) - 2,
            len(stage1) + len(stage2) + len(stage3) + len(stage4) - 3,
        )
        normals_stage2 = traj[stage2_slice] - self.sphere_center[None, :]
        normals_stage2 = normals_stage2 / np.maximum(np.linalg.norm(normals_stage2, axis=1, keepdims=True), 1e-12)
        normals_stage4 = traj[stage4_slice] - self.sphere_center[None, :]
        normals_stage4 = normals_stage4 / np.maximum(np.linalg.norm(normals_stage4, axis=1, keepdims=True), 1e-12)
        tool_axis, orientation_metadata = self._generate_tool_axis_trace(
            traj=traj,
            stage_lengths=stage_lengths,
            normals_stage2=normals_stage2,
            normals_stage3=normals_stage4,
            n_contact=n_contact,
            rng=rng,
        )
        if tool_axis.shape[0] != len(traj):
            axis_fixed = np.empty((len(traj), 3), dtype=float)
            axis_fixed[: min(len(tool_axis), len(traj))] = tool_axis[: min(len(tool_axis), len(traj))]
            if len(tool_axis) < len(traj):
                axis_fixed[len(tool_axis) :] = tool_axis[-1]
            tool_axis = axis_fixed
        reference_timestamps = concatenate_stage_timestamps(
            (stage1_timing, stage2_timing, stage3_timing, stage4_timing, stage5_timing)
        )
        if len(reference_timestamps) != len(traj):
            raise ValueError("S5 time parameterization must align timestamps with the emitted trajectory.")
        self.register_tool_axis_trace(traj, tool_axis)
        self.register_timestamp_trace(traj, reference_timestamps)
        generation_metadata = {
            "demo_index": None if demo_index is None else int(demo_index),
            "sampled_base_segment_lengths": [int(l1), int(l2), int(l3), int(l4)],
            "emitted_stage_lengths": [int(value) for value in stage_lengths],
            "true_cutpoints": np.asarray(true_cutpoints, dtype=int).tolist(),
            "contact": {
                "theta": float(theta0),
                "phi": float(phi0),
                "normal": np.asarray(n_contact, dtype=float).tolist(),
                "position": np.asarray(p_contact, dtype=float).tolist(),
            },
            "surface_trace": {
                "geodesic_angle": float(trace_angle),
                "end_theta": float(theta_trace_end),
                "end_phi": float(phi_trace_end),
                "end_normal": np.asarray(n_trace_end, dtype=float).tolist(),
                "latitude_template": {
                    "theta_start": float(stage2_latitude_path[0]),
                    "theta_end": float(stage2_latitude_path[1]),
                    "phi": float(stage2_latitude_path[2]),
                    "phi_bump": float(stage2_latitude_path[3]),
                },
            },
            "shell_target": {
                "theta": float(theta_repos),
                "phi": float(phi_cap),
                "normal": np.asarray(n_repos_end, dtype=float).tolist(),
                "offset": float(shell_offset),
                "reference_position": np.asarray(
                    self.sphere_center + (self.sphere_radius + shell_offset) * n_repos_end,
                    dtype=float,
                ).tolist(),
            },
            "reference_stage_end_positions": [
                np.asarray(traj[int(index)], dtype=float).tolist()
                for index in np.asarray(true_cutpoints, dtype=int)
            ] + [np.asarray(traj[-1], dtype=float).tolist()],
            "time_parameterization": {
                "method": "fixed_step_path_time_parameterization",
                "dt": float(self.dt),
                "stage2_length_scale": float(stage2_length_scale),
                "stage4_length_scale": float(stage4_length_scale),
                "stages": {
                    "stage1": stage1_timing.summary(),
                    "stage2": stage2_timing.summary(),
                    "stage3": stage3_timing.summary(),
                    "stage4": stage4_timing.summary(),
                    "stage5": stage5_timing.summary(),
                },
                "speed_intent": {
                    "stage1": "approach_taper",
                    "stage2": "deliberate_gaussian_slowdowns",
                    "stage3": "smooth_correlated_variation",
                    "stage4": "deliberate_gaussian_slowdown",
                    "stage5": "constant_departure_intent",
                },
            },
            "orientation_policy": orientation_metadata,
        }
        if return_metadata:
            return traj, true_cutpoints, generation_metadata
        return traj, true_cutpoints

    def generate_demos(self, n_demos=10, rng=None, **kwargs):
        rng = np.random if rng is None else rng
        demos = []
        true_cutpoints = []
        for _ in range(int(n_demos)):
            traj, cutpoints = self.generate_demo(rng=rng, **kwargs)
            demos.append(np.asarray(traj, dtype=float))
            true_cutpoints.append(np.asarray(cutpoints, dtype=int))
        return demos, true_cutpoints
