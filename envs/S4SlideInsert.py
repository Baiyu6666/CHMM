from __future__ import annotations

import numpy as np

from .base import TaskBundle


class S4SlideInsertEnv:
    def __init__(
        self,
        seg_lengths=(20, 8, 38, 12),
        start=(-1.4, 0.35),
        start_jitter=(0.10, 0.05),
        stage1_end=(-0.25, 0.014),
        stage2_end=(-0.20, 0.0),
        stage3_end=(0.70, 0.0),
        stage4_end=(1.00, 0.0),
        stage_end_jitter=((0.032, 0.012), (0.024, 0.006), (0.038, 0.006), (0.014, 0.003)),
        stage2_end_x_range=(-0.70, 0.0),
        stage2_end_z_range=(-0.010, 0.010),
        stage2_theta_end_range=(-0.030, 0.085),
        slot_x=1.0,
        slot_theta=0.0,
        theta_start=0.34,
        theta_stage1_end=0.12,
        theta_stage2_end=0.03,
        theta_stage3_end=0.01,
        theta_stage4_end=0.0,
        theta_end_jitter=(0.055, 0.032, 0.018, 0.010),
        theta_start_jitter=0.05,
        v1_target=0.060,
        v2_target=0.007,
        v3_target=0.045,
        v4_target=0.018,
        f_contact_min=0.40,
        f_slide_min=0.72,
        f_insert_min=1.00,
        orientation_error_max_stage3=0.06,
        orientation_error_max_stage4=0.04,
        transition_half_window: int = 1,
        noise_pos: float = 0.003,
        noise_misc: float = 0.02,
        seg_length_jitter=(5, 3, 6, 4),
        seg_length_scale_range=(0.84, 1.18),
        dt: float = 0.7,
    ):
        self.seg_lengths = tuple(int(x) for x in seg_lengths)
        self.start = np.asarray(start, dtype=float)
        self.start_jitter = np.asarray(start_jitter, dtype=float)
        self.stage1_end = np.asarray(stage1_end, dtype=float)
        self.stage2_end = np.asarray(stage2_end, dtype=float)
        self.stage3_end = np.asarray(stage3_end, dtype=float)
        self.stage4_end = np.asarray(stage4_end, dtype=float)
        self.stage_end_jitter = tuple(np.asarray(x, dtype=float) for x in stage_end_jitter)
        self.stage2_end_x_range = tuple(float(x) for x in stage2_end_x_range)
        self.stage2_end_z_range = tuple(float(x) for x in stage2_end_z_range)
        self.stage2_theta_end_range = tuple(float(x) for x in stage2_theta_end_range)
        self.slot_x = float(slot_x)
        self.slot_theta = float(slot_theta)
        self.theta_start = float(theta_start)
        self.theta_stage1_end = float(theta_stage1_end)
        self.theta_stage2_end = float(theta_stage2_end)
        self.theta_stage3_end = float(theta_stage3_end)
        self.theta_stage4_end = float(theta_stage4_end)
        self.theta_end_jitter = tuple(float(x) for x in theta_end_jitter)
        self.theta_start_jitter = float(theta_start_jitter)
        self.v1_target = float(v1_target)
        self.v2_target = float(v2_target)
        self.v3_target = float(v3_target)
        self.v4_target = float(v4_target)
        self.f_contact_min = float(f_contact_min)
        self.f_slide_min = float(f_slide_min)
        self.f_insert_min = float(f_insert_min)
        self.orientation_error_max_stage3 = float(orientation_error_max_stage3)
        self.orientation_error_max_stage4 = float(orientation_error_max_stage4)
        self.transition_half_window = int(transition_half_window)
        self.noise_pos = float(noise_pos)
        self.noise_misc = float(noise_misc)
        self.seg_length_jitter = tuple(int(x) for x in seg_length_jitter)
        self.seg_length_scale_range = tuple(float(x) for x in seg_length_scale_range)
        self.dt = float(dt)
        self.eval_tag = "S4SlideInsert"
        self.n_segments = 4
        self._cached_force_traces = {}
        self._cached_speed_traces = {}
        self.true_constraints = self.get_true_constraints()
        self.constraint_specs = self.get_constraint_specs()
        self.feature_schema = self.get_feature_schema()
        self.subgoal = np.array([self.stage2_end[0], self.stage2_end[1], self.theta_stage2_end], dtype=float)
        self.goal = np.array([self.stage4_end[0], self.stage4_end[1], self.theta_stage4_end], dtype=float)
        self.demo_subgoals = None
        self.demo_goals = None
        self.demo_stage_lengths = None

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "surface_distance", "description": "Absolute distance to the contact surface z=0"},
            {"id": 1, "name": "force", "description": "Contact force proxy"},
            {"id": 2, "name": "orientation_error", "description": "Absolute angle error between object and slot"},
            {"id": 3, "name": "speed", "description": "Planar speed magnitude"},
            {"id": 4, "name": "noise_aux", "description": "Auxiliary irrelevant feature"},
        ]

    def get_true_constraints(self):
        return {
            "surface_target": 0.0,
            "v2_target": float(self.v2_target),
            "v3_target": float(self.v3_target),
            "v4_target": float(self.v4_target),
            "f_contact_min": float(self.f_contact_min),
            "f_slide_min": float(self.f_slide_min),
            "f_insert_min": float(self.f_insert_min),
            "orientation_error_max_stage3": float(self.orientation_error_max_stage3),
            "orientation_error_max_stage4": float(self.orientation_error_max_stage4),
        }

    def get_constraint_specs(self):
        return [
            {"feature_name": "surface_distance", "stage": 1, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "speed", "stage": 1, "semantics": "target_value", "oracle_key": "v2_target"},
            {"feature_name": "force", "stage": 1, "semantics": "lower_bound", "oracle_key": "f_contact_min"},
            {"feature_name": "surface_distance", "stage": 2, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "speed", "stage": 2, "semantics": "target_value", "oracle_key": "v3_target"},
            {"feature_name": "force", "stage": 2, "semantics": "lower_bound", "oracle_key": "f_slide_min"},
            {
                "feature_name": "orientation_error",
                "stage": 2,
                "semantics": "upper_bound",
                "oracle_key": "orientation_error_max_stage3",
            },
            {"feature_name": "surface_distance", "stage": 3, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "speed", "stage": 3, "semantics": "target_value", "oracle_key": "v4_target"},
            {"feature_name": "force", "stage": 3, "semantics": "lower_bound", "oracle_key": "f_insert_min"},
            {
                "feature_name": "orientation_error",
                "stage": 3,
                "semantics": "upper_bound",
                "oracle_key": "orientation_error_max_stage4",
            },
        ]

    def _piecewise_segment(self, start, end, length, endpoint=False):
        x = np.linspace(float(start[0]), float(end[0]), int(length), endpoint=endpoint)
        z = np.linspace(float(start[1]), float(end[1]), int(length), endpoint=endpoint)
        return np.c_[x, z]

    @staticmethod
    def _smoothstep(u):
        u = np.asarray(u, dtype=float)
        return u * u * (3.0 - 2.0 * u)

    def _smooth_segment(self, start, end, length, endpoint=False):
        u = np.linspace(0.0, 1.0, int(length), endpoint=endpoint)
        s = self._smoothstep(u)
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        return start[None, :] + s[:, None] * (end - start)[None, :]

    @staticmethod
    def _path_length(path: np.ndarray) -> float:
        pts = np.asarray(path, dtype=float)
        if len(pts) <= 1:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    @staticmethod
    def _resample_fixed_count(path: np.ndarray, num_points: int) -> np.ndarray:
        pts = np.asarray(path, dtype=float)
        n = int(num_points)
        if len(pts) <= 1 or n <= 1:
            return pts.copy()
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total <= 1e-12:
            out = np.repeat(pts[:1], n, axis=0)
            out[0] = pts[0]
            out[-1] = pts[-1]
            return out
        targets = np.linspace(0.0, total, n)
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

    def _timewarp_path(
        self,
        path: np.ndarray,
        strength: float,
        rng: np.random.RandomState,
        cycles: float = 2.0,
    ) -> np.ndarray:
        pts = np.asarray(path, dtype=float)
        if len(pts) <= 2 or float(strength) <= 0.0:
            return pts.copy()
        s = np.linspace(0.0, 1.0, len(pts))
        phase = float(rng.uniform(-np.pi, np.pi))
        envelope = np.sin(np.pi * s) ** 1.1
        weights = 1.0 + float(strength) * envelope * np.sin(float(cycles) * np.pi * s + phase)
        weights = np.clip(weights, 0.35, None)
        targets = np.cumsum(weights)
        targets = (targets - targets[0]) / max(targets[-1] - targets[0], 1e-12)
        out = np.empty_like(pts)
        out[0] = pts[0]
        out[-1] = pts[-1]
        for d in range(pts.shape[1]):
            out[:, d] = np.interp(targets, s, pts[:, d])
        out[0] = pts[0]
        out[-1] = pts[-1]
        return out

    def _timewarp_decelerating_path(
        self,
        path: np.ndarray,
        strength: float,
        rng: np.random.RandomState,
        floor: float = 0.55,
    ) -> np.ndarray:
        pts = np.asarray(path, dtype=float)
        if len(pts) <= 2 or float(strength) <= 0.0:
            return pts.copy()
        s = np.linspace(0.0, 1.0, len(pts))
        exponent = float(rng.uniform(1.2, 1.8))
        front_bias = 1.0 - s**exponent
        ripple_phase = float(rng.uniform(-0.35 * np.pi, 0.35 * np.pi))
        ripple = 0.10 * np.sin(2.2 * np.pi * s + ripple_phase) * np.sin(np.pi * s) ** 1.3
        weights = 1.0 + float(strength) * front_bias + ripple
        weights = np.clip(weights, float(floor), None)
        targets = np.cumsum(weights)
        targets = (targets - targets[0]) / max(targets[-1] - targets[0], 1e-12)
        out = np.empty_like(pts)
        for d in range(pts.shape[1]):
            out[:, d] = np.interp(targets, s, pts[:, d])
        out[0] = pts[0]
        out[-1] = pts[-1]
        return out

    def _make_target_speed_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_points: int,
        target_speed: float,
        rng: np.random.RandomState,
        max_amp: float,
        cycles: float = 1.0,
        vertical_bias: float = 0.0,
    ) -> np.ndarray:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        n = int(num_points)
        desired_length = max(float(target_speed) * self.dt * max(n - 1, 1), float(np.linalg.norm(end - start)))
        u_hr = np.linspace(0.0, 1.0, 256)
        direction = end - start
        dist = float(np.linalg.norm(direction))
        if dist <= 1e-12:
            direction_unit = np.array([1.0, 0.0], dtype=float)
        else:
            direction_unit = direction / dist
        normal = np.array([-direction_unit[1], direction_unit[0]], dtype=float)
        phase = float(rng.uniform(-0.5 * np.pi, 0.5 * np.pi))
        sign = -1.0 if rng.rand() < 0.5 else 1.0
        envelope = np.sin(np.pi * u_hr)
        waveform = envelope * np.sin(float(cycles) * np.pi * u_hr + phase)

        def build_path(amp: float) -> np.ndarray:
            base = start[None, :] + u_hr[:, None] * (end - start)[None, :]
            offset = sign * float(amp) * waveform
            path = base + offset[:, None] * normal[None, :]
            if vertical_bias != 0.0:
                path[:, 1] += float(vertical_bias) * envelope**2
            path[0] = start
            path[-1] = end
            return path

        if self._path_length(build_path(0.0)) >= desired_length - 1e-6:
            return self._resample_fixed_count(build_path(0.0), n)

        lo = 0.0
        hi = max(float(max_amp), 1e-4)
        for _ in range(20):
            if self._path_length(build_path(hi)) >= desired_length:
                break
            hi *= 1.5
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if self._path_length(build_path(mid)) < desired_length:
                lo = mid
            else:
                hi = mid
        return self._resample_fixed_count(build_path(hi), n)

    def _make_surface_search_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_points: int,
        target_speed: float,
        rng: np.random.RandomState,
        max_x_amp: float,
        z_amp: float,
        cycles: float = 2.5,
    ) -> np.ndarray:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        n = int(num_points)
        desired_length = max(float(target_speed) * self.dt * max(n - 1, 1), float(np.linalg.norm(end - start)))
        u_hr = np.linspace(0.0, 1.0, 512)
        phase = float(rng.uniform(-0.35 * np.pi, 0.35 * np.pi))
        envelope = np.sin(np.pi * u_hr) ** 1.2
        speed_wave = np.sin(float(cycles) * np.pi * u_hr + phase)
        z_wave = np.sin((float(cycles) + 0.75) * np.pi * u_hr + 0.5 * phase)
        x_weights = np.clip(1.0 + 0.55 * envelope * speed_wave, 0.25, None)
        x_cum = np.cumsum(x_weights)
        x_progress = (x_cum - x_cum[0]) / max(x_cum[-1] - x_cum[0], 1e-12)

        def build_path(amp_z: float) -> np.ndarray:
            x = start[0] + x_progress * (end[0] - start[0])
            z = start[1] + u_hr * (end[1] - start[1]) + float(z_amp) * envelope * z_wave
            z += float(amp_z) * envelope * np.sin((float(cycles) + 1.5) * np.pi * u_hr + phase)
            path = np.c_[x, z]
            path[0] = start
            path[-1] = end
            return path

        if self._path_length(build_path(0.0)) >= desired_length - 1e-6:
            return self._resample_fixed_count(build_path(0.0), n)

        lo = 0.0
        hi = max(float(max_x_amp) * 0.25, 1e-4)
        for _ in range(20):
            if self._path_length(build_path(hi)) >= desired_length:
                break
            hi *= 1.5
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self._path_length(build_path(mid)) < desired_length:
                lo = mid
            else:
                hi = mid
        return self._resample_fixed_count(build_path(hi), n)

    @staticmethod
    def _smooth_noise(rng: np.random.RandomState, length: int, scale: float, kernel_size: int = 7) -> np.ndarray:
        raw = rng.randn(int(length)) * float(scale)
        kernel = np.ones(int(kernel_size), dtype=float) / float(kernel_size)
        return np.convolve(raw, kernel, mode="same")

    @staticmethod
    def _sample_margin_excess(
        rng: np.random.RandomState,
        length: int,
        scale: float,
        max_extra: float,
        near_boundary_prob: float = 0.82,
    ) -> np.ndarray:
        # Most samples stay close to the lower boundary, with a light exponential tail.
        base = rng.exponential(scale=float(scale), size=int(length))
        tail_mask = rng.rand(int(length)) > float(near_boundary_prob)
        if np.any(tail_mask):
            base[tail_mask] += rng.exponential(scale=1.8 * float(scale), size=int(tail_mask.sum()))
        return np.clip(base, 0.0, float(max_extra))

    @staticmethod
    def _smooth_trace(values: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        k = max(1, int(kernel_size))
        if k <= 1 or len(vals) == 0:
            return vals
        kernel = np.ones(k, dtype=float) / float(k)
        pad_left = k // 2
        pad_right = k - 1 - pad_left
        padded = np.pad(vals, (pad_left, pad_right), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _smooth_positive_trace(values: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        vals = np.clip(np.asarray(values, dtype=float), 0.0, None)
        return S4SlideInsertEnv._smooth_trace(vals, kernel_size=kernel_size)

    def _make_stage_force_margin_profile(
        self,
        stage_idx: int,
        z: np.ndarray,
        speed: np.ndarray,
        tangential_speed: np.ndarray,
        dz: np.ndarray,
        orientation_error: np.ndarray,
        contact_gate: np.ndarray,
        slide_progress: np.ndarray,
        insert_progress: np.ndarray,
        rng: np.random.RandomState | None,
        latents: dict | None,
    ) -> np.ndarray:
        n = int(len(z))
        if n <= 0:
            return np.zeros(0, dtype=float)

        u = np.linspace(0.0, 1.0, n, endpoint=True)
        if int(stage_idx) == 1:
            cycles = 2.40
            offset = 0.028
            amplitude = 0.070
            margin_cap = 0.115
        elif int(stage_idx) == 2:
            cycles = 3.20
            offset = 0.034
            amplitude = 0.082
            margin_cap = 0.135
        else:
            cycles = 2.80
            offset = 0.032
            amplitude = 0.076
            margin_cap = 0.125

        amp_scale = 1.0 if latents is None else float(latents.get("force_excess_scale", 1.0))
        amp_scale = float(np.clip(amp_scale, 0.9, 1.1))
        base_wave = np.sin(2.0 * np.pi * float(cycles) * u - 0.5 * np.pi)
        half_wave = np.maximum(base_wave, 0.0)
        margin = float(amplitude) * amp_scale * half_wave - float(offset)
        margin = self._smooth_trace(margin, kernel_size=3)
        margin_floor = -float(offset) - 0.01
        return np.clip(margin, float(margin_floor), float(margin_cap))

    @staticmethod
    def _sample_sparse_margin_excess(
        rng: np.random.RandomState,
        length: int,
        base_scale: float,
        base_cap: float,
        burst_scale: float,
        max_extra: float,
        max_bursts: int = 2,
        base_activation_prob: float = 0.16,
    ) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.zeros(0, dtype=float)
        base = np.zeros(n, dtype=float)
        base_mask = rng.rand(n) < float(base_activation_prob)
        if np.any(base_mask):
            base[base_mask] = np.clip(
                rng.exponential(scale=float(base_scale), size=int(base_mask.sum())),
                0.0,
                float(base_cap),
            )
        excess = base
        num_bursts = int(rng.randint(1, max(int(max_bursts), 1) + 1))
        for _ in range(num_bursts):
            center = int(rng.randint(0, n))
            half_width = int(rng.randint(1, 3))
            amp = min(float(rng.exponential(scale=float(burst_scale))), 0.6 * float(max_extra))
            left = max(0, center - half_width)
            right = min(n, center + half_width + 1)
            window = np.arange(left, right, dtype=float)
            envelope = 1.0 - np.abs(window - float(center)) / float(half_width + 1)
            excess[left:right] += amp * np.clip(envelope, 0.0, None)
        return np.clip(excess, 0.0, float(max_extra))

    @staticmethod
    def _blend_segment_boundary(values: np.ndarray, boundary: int, half_window: int = 2) -> np.ndarray:
        out = np.asarray(values, dtype=float).copy()
        left = max(0, int(boundary) - int(half_window))
        right = min(len(out) - 1, int(boundary) + int(half_window) + 1)
        if left < 1 or right >= len(out) - 1 or right - left < 2:
            return out

        p0 = out[left].copy()
        p1 = out[right].copy()
        span = float(right - left)
        # Match incoming and outgoing finite-difference slopes so the transition
        # smooths the kink without forcing a dip/spike at the boundary.
        m0 = (out[left] - out[left - 1]) * span
        m1 = (out[right + 1] - out[right]) * span

        u = np.linspace(0.0, 1.0, right - left + 1)
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        out[left:right + 1] = (
            h00[:, None] * p0
            + h10[:, None] * m0
            + h01[:, None] * p1
            + h11[:, None] * m1
        )
        return out

    def _sample_demo_latents(self, rng: np.random.RandomState):
        style = float(rng.uniform(0.85, 1.18))
        return {
            "style": style,
            "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            "force_excess_scale": float(rng.uniform(0.95, 1.12)),
            "force_bias": float(rng.uniform(-0.004, 0.008)),
            "precontact_force_mean": float(rng.uniform(0.0065, 0.0095)),
            "precontact_force_sigma": float(rng.uniform(0.0010, 0.0018)),
            "contact_force_coupling": float(rng.uniform(0.006, 0.016)),
            "slide_force_coupling": float(rng.uniform(0.010, 0.024)),
            "insert_force_coupling": float(rng.uniform(0.012, 0.028)),
            "micro_wobble": float(rng.uniform(0.004, 0.012)),
            "surface_wobble": float(rng.uniform(0.0015, 0.005)),
            "theta_wobble": float(rng.uniform(0.004, 0.014)),
        }

    def _sample_segment_lengths(self, rng: np.random.RandomState):
        global_scale = float(rng.uniform(*self.seg_length_scale_range))
        min_lengths = [max(6, int(np.floor(0.65 * base))) for base in self.seg_lengths]
        lengths = []
        for base, jitter, min_len in zip(self.seg_lengths, self.seg_length_jitter, min_lengths):
            local_scale = global_scale * float(rng.uniform(0.90, 1.12))
            length = int(round(float(base) * local_scale)) + int(rng.randint(-int(jitter), int(jitter) + 1))
            lengths.append(max(int(min_len), length))
        return tuple(int(x) for x in lengths)

    def _compute_force_signal(
        self,
        pos: np.ndarray,
        theta: np.ndarray,
        stage3_end_x: float,
        labels: np.ndarray,
        rng: np.random.RandomState,
        latents: dict,
    ) -> np.ndarray:
        pos = np.asarray(pos, dtype=float)
        theta = np.asarray(theta, dtype=float)
        labels = np.asarray(labels, dtype=int)
        T = len(pos)
        speed = np.zeros(T, dtype=float)
        tangential_speed = np.zeros(T, dtype=float)
        dz = np.zeros(T, dtype=float)
        if T > 1:
            delta = np.diff(pos, axis=0) / self.dt
            speed[1:] = np.linalg.norm(delta, axis=1)
            speed[0] = speed[1]
            tangential_speed[1:] = np.abs(delta[:, 0])
            tangential_speed[0] = tangential_speed[1]
            dz[1:] = np.abs(delta[:, 1])
            dz[0] = dz[1]
        orientation_error = np.abs(self._wrap_to_pi(theta - self.slot_theta))
        z = pos[:, 1]
        x = pos[:, 0]
        contact_gate = 1.0 / (1.0 + np.exp((z - 0.012) / 0.006))
        slide_progress = 1.0 / (1.0 + np.exp(-(x - self.stage2_end[0]) / 0.055))
        insert_progress = 1.0 / (1.0 + np.exp(-(x - float(stage3_end_x)) / 0.045))

        stage_lower_bounds = np.take(
            np.array([0.0, self.f_contact_min, self.f_slide_min, self.f_insert_min], dtype=float),
            labels,
        )
        force = stage_lower_bounds.copy()
        precontact_mask = labels == 0
        if np.any(precontact_mask):
            n0 = int(precontact_mask.sum())
            precontact_force = (
                latents["precontact_force_mean"]
                + self._smooth_noise(rng, n0, latents["precontact_force_sigma"], kernel_size=7)
            )
            precontact_force += 0.0007 * np.sin(np.linspace(0.0, 2.0 * np.pi, n0) + latents["phase"])
            force[precontact_mask] = np.clip(precontact_force, 0.0, 0.02)

        for stage_idx in (1, 2, 3):
            mask = labels == stage_idx
            if not np.any(mask):
                continue
            raw_stage_force = stage_lower_bounds[mask] + self._make_stage_force_margin_profile(
                stage_idx=stage_idx,
                z=z[mask],
                speed=speed[mask],
                tangential_speed=tangential_speed[mask],
                dz=dz[mask],
                orientation_error=orientation_error[mask],
                contact_gate=contact_gate[mask],
                slide_progress=slide_progress[mask],
                insert_progress=insert_progress[mask],
                rng=rng,
                latents=latents,
            )
            force[mask] = np.maximum(stage_lower_bounds[mask], raw_stage_force)

        for boundary in np.where(np.diff(labels) != 0)[0]:
            force = self._blend_segment_boundary(force[:, None], boundary=int(boundary), half_window=max(2, self.transition_half_window + 1)).ravel()

        if np.any(precontact_mask):
            pre_idx = np.where(precontact_mask)[0]
            force[pre_idx] += self._smooth_noise(rng, len(pre_idx), 0.003, kernel_size=11)
        constrained_mask = labels >= 1
        if np.any(constrained_mask):
            constrained_idx = np.where(constrained_mask)[0]
            force[constrained_idx] = np.asarray(force[constrained_idx], dtype=float)

        force[constrained_mask] = np.maximum(force[constrained_mask], stage_lower_bounds[constrained_mask])
        force_out = force.copy()
        force_out[~constrained_mask] = np.clip(force_out[~constrained_mask] + latents["force_bias"], 0.0, 0.02)
        return force_out

    def generate_demo(self, seed: int):
        rng = np.random.RandomState(seed)
        l1, l2, l3, l4 = self._sample_segment_lengths(rng)
        latents = self._sample_demo_latents(rng)
        start_local = self.start + rng.randn(2) * self.start_jitter
        start_local[0] = float(np.clip(start_local[0], -1.65, -1.12))
        start_local[1] = float(np.clip(start_local[1], 0.24, 0.46))

        stage2_end_local = np.array(
            [
                rng.uniform(*self.stage2_end_x_range),
                rng.uniform(*self.stage2_end_z_range),
            ],
            dtype=float,
        )
        stage1_end_local = self.stage1_end + rng.randn(2) * self.stage_end_jitter[0]
        stage4_end_local = self.stage4_end + rng.randn(2) * self.stage_end_jitter[3]

        stage1_end_local[0] = float(np.clip(stage2_end_local[0] - rng.uniform(0.004, 0.015), -0.82, -0.08))
        stage1_end_local[1] = float(np.clip(rng.uniform(0.004, 0.014), 0.003, 0.020))
        stage2_end_local[0] = float(np.clip(max(stage2_end_local[0], stage1_end_local[0] + 0.003), -0.70, 0.0))
        stage2_end_local[1] = float(np.clip(stage2_end_local[1], *self.stage2_end_z_range))
        stage4_end_local[0] = float(np.clip(stage4_end_local[0], 0.96, 1.03))
        stage4_end_local[1] = float(np.clip(stage4_end_local[1], -0.008, 0.008))
        stage3_end_local = self.stage3_end + rng.randn(2) * self.stage_end_jitter[2]
        stage3_end_local[0] = float(np.clip(max(stage2_end_local[0] + rng.uniform(0.88, 1.10), stage4_end_local[0] - rng.uniform(0.09, 0.13)), 0.78, 0.93))
        stage3_end_local[1] = float(np.clip(stage3_end_local[1], -0.010, 0.010))

        v1_demo = self.v1_target * rng.uniform(0.94, 1.06)
        v2_demo = self.v2_target * rng.uniform(0.98, 1.02)
        v3_demo = self.v3_target * rng.uniform(1.08, 1.14)
        v4_demo = self.v4_target * rng.uniform(0.98, 1.02)

        seg1 = self._make_target_speed_segment(
            start_local,
            stage1_end_local,
            l1 + 1,
            v1_demo,
            rng,
            max_amp=0.12,
            cycles=1.0,
            vertical_bias=0.05,
        )[:-1]

        seg2 = self._make_target_speed_segment(
            stage1_end_local,
            stage2_end_local,
            l2 + 1,
            v2_demo,
            rng,
            max_amp=0.008,
            cycles=1.0,
            vertical_bias=-0.0015,
        )[:-1]
        u2 = np.linspace(0.0, 1.0, l2, endpoint=False)
        seg2[:, 0] += 0.004 * latents["style"] * np.sin(2.0 * np.pi * u2 + latents["phase"]) * np.sin(np.pi * u2)
        seg2[:, 1] += (
            0.0025 * np.exp(-2.8 * u2) * np.sin(3.2 * np.pi * u2 + latents["phase"])
            - 0.0012 * np.sin(np.pi * u2) ** 2
        )
        seg2 = self._resample_fixed_count(seg2, l2)
        seg2[0] = stage1_end_local
        seg2[-1] = stage2_end_local

        seg3 = self._make_surface_search_segment(
            stage2_end_local,
            stage3_end_local,
            l3 + 1,
            v3_demo,
            rng,
            max_x_amp=0.13,
            z_amp=0.005,
            cycles=2.6,
        )[:-1]
        u3 = np.linspace(0.0, 1.0, l3, endpoint=False)
        seg3[:, 1] += 0.35 * latents["surface_wobble"] * np.sin(3.0 * np.pi * u3 + 0.5 * latents["phase"]) * np.sin(np.pi * u3)

        seg4 = self._make_target_speed_segment(
            stage3_end_local,
            stage4_end_local,
            l4,
            v4_demo,
            rng,
            max_amp=0.010,
            cycles=1.1,
            vertical_bias=0.0,
        )
        u4 = np.linspace(0.0, 1.0, l4, endpoint=True)
        seg4[:, 1] += 0.12 * latents["surface_wobble"] * np.sin(2.0 * np.pi * u4 + latents["phase"]) * np.sin(np.pi * u4)

        seg1 = self._timewarp_decelerating_path(seg1, strength=0.85, rng=rng, floor=0.46)
        seg2 = self._timewarp_path(seg2, strength=0.05, rng=rng, cycles=1.4)
        seg3 = self._timewarp_path(seg3, strength=0.10, rng=rng, cycles=2.0)
        seg4 = self._timewarp_path(seg4, strength=0.04, rng=rng, cycles=1.3)

        pos = np.vstack([seg1, seg2, seg3, seg4])
        labels = np.repeat(np.arange(4), [l1, l2, l3, l4])
        theta_start_local = self.theta_start + self.theta_start_jitter * rng.randn()
        theta_stage1_end = self.theta_stage1_end + self.theta_end_jitter[0] * rng.randn()
        theta_stage2_end = float(rng.uniform(*self.stage2_theta_end_range))
        theta_stage3_end = self.theta_stage3_end + self.theta_end_jitter[2] * rng.randn()
        theta_stage4_end = self.theta_stage4_end + self.theta_end_jitter[3] * rng.randn()

        theta1 = np.linspace(theta_start_local, theta_stage1_end, l1, endpoint=False)
        theta2 = np.linspace(theta_stage1_end, theta_stage2_end, l2, endpoint=False)
        theta3 = np.zeros(l3, dtype=float)
        theta4 = np.zeros(l4, dtype=float)
        sign3 = -1.0 if float(theta_stage2_end) < 0.0 else 1.0
        sign4 = sign3 if abs(float(theta_stage4_end)) < 1e-6 else (1.0 if float(theta_stage4_end) >= 0.0 else -1.0)
        if l3 > 0:
            u3_theta = np.linspace(0.0, 1.0, l3, endpoint=False)
            half_wave3 = np.maximum(np.sin(2.35 * np.pi * u3_theta - 0.5 * np.pi + 0.20 * latents["phase"]), 0.0)
            margin3 = 0.62 * self.orientation_error_max_stage3 * half_wave3 - 0.18 * self.orientation_error_max_stage3
            abs_theta3 = np.clip(self.orientation_error_max_stage3 - margin3, 0.0, 0.96 * self.orientation_error_max_stage3)
            theta3 = sign3 * self._smooth_trace(abs_theta3, kernel_size=3)
        if l4 > 0:
            u4_theta = np.linspace(0.0, 1.0, l4, endpoint=True)
            half_wave4 = np.maximum(np.sin(1.95 * np.pi * u4_theta - 0.5 * np.pi + 0.16 * latents["phase"]), 0.0)
            margin4 = 0.58 * self.orientation_error_max_stage4 * half_wave4 - 0.16 * self.orientation_error_max_stage4
            abs_theta4 = np.clip(self.orientation_error_max_stage4 - margin4, 0.0, 0.96 * self.orientation_error_max_stage4)
            theta4 = sign4 * self._smooth_trace(abs_theta4, kernel_size=3)
        theta = np.concatenate([theta1, theta2, theta3, theta4])
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 - 1, half_window=self.transition_half_window).ravel()
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 + l2 - 1, half_window=self.transition_half_window).ravel()
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 + l2 + l3 - 1, half_window=self.transition_half_window).ravel()

        pos_noise_scale_x = np.take(np.array([1.0, 0.22, 0.32, 0.18], dtype=float), labels)
        pos_noise_scale_z = np.take(np.array([1.0, 0.20, 0.28, 0.16], dtype=float), labels)
        pos[:, 0] += self._smooth_noise(rng, len(pos), 1.35 * self.noise_pos, kernel_size=11) * pos_noise_scale_x
        pos[:, 1] += self._smooth_noise(rng, len(pos), 0.95 * self.noise_pos, kernel_size=9) * pos_noise_scale_z
        stage1_floor = np.linspace(0.045, 0.012, l1)
        pos[:l1, 1] = np.maximum(pos[:l1, 1], stage1_floor)
        contact_smooth = self._smooth_noise(rng, len(pos), 0.0025, kernel_size=13)
        pos[l1:, 1] += 0.6 * contact_smooth[l1:]
        pos[l1:, 1] = np.clip(pos[l1:, 1], -0.018, 0.018)
        stage_bounds = [(0, l1), (l1, l1 + l2), (l1 + l2, l1 + l2 + l3), (l1 + l2 + l3, len(pos))]
        for stage_idx in (1, 2, 3):
            start_i, end_i = stage_bounds[stage_idx]
            pos[start_i:end_i] = self._resample_fixed_count(pos[start_i:end_i], end_i - start_i)
        for boundary in (l1 - 1, l1 + l2 - 1, l1 + l2 + l3 - 1):
            pos = self._blend_segment_boundary(pos, boundary=boundary, half_window=self.transition_half_window)
        theta_noise_scale = np.take(np.array([1.0, 0.30, 0.08, 0.05], dtype=float), labels)
        theta += self._smooth_noise(rng, len(theta), 0.28 * self.noise_misc, kernel_size=11) * theta_noise_scale
        theta += latents["theta_wobble"] * np.sin(np.linspace(0.0, 4.5 * np.pi, len(theta)) + latents["phase"]) * np.r_[
            np.linspace(0.3, 1.0, l1 + l2),
            np.linspace(0.18, 0.08, l3 + l4),
        ] * theta_noise_scale
        theta[l1 + l2:l1 + l2 + l3] = np.clip(
            theta[l1 + l2:l1 + l2 + l3],
            -0.95 * self.orientation_error_max_stage3,
            0.95 * self.orientation_error_max_stage3,
        )
        theta[l1 + l2 + l3:] = np.clip(
            theta[l1 + l2 + l3:],
            -0.95 * self.orientation_error_max_stage4,
            0.95 * self.orientation_error_max_stage4,
        )
        for boundary in (l1 - 1, l1 + l2 - 1, l1 + l2 + l3 - 1):
            theta = self._blend_segment_boundary(theta[:, None], boundary=boundary, half_window=self.transition_half_window).ravel()
        theta[l1 + l2:l1 + l2 + l3] = np.clip(
            theta[l1 + l2:l1 + l2 + l3],
            -0.98 * self.orientation_error_max_stage3,
            0.98 * self.orientation_error_max_stage3,
        )
        theta[l1 + l2 + l3:] = np.clip(
            theta[l1 + l2 + l3:],
            -0.98 * self.orientation_error_max_stage4,
            0.98 * self.orientation_error_max_stage4,
        )

        force = self._compute_force_signal(pos, theta, stage3_end_local[0], labels, rng, latents)
        speed_trace = np.zeros(len(pos), dtype=float)
        if len(pos) > 1:
            delta = np.diff(pos, axis=0) / self.dt
            speed_trace[1:] = np.linalg.norm(delta, axis=1)
            speed_trace[0] = speed_trace[1]
        stage_targets = {
            1: 0.96 * float(self.v2_target),
            2: 1.00 * float(self.v3_target),
            3: 1.00 * float(self.v4_target),
        }
        stage_amplitudes = {
            1: 0.10 * float(self.v2_target),
            2: 0.10 * float(self.v3_target),
            3: 0.08 * float(self.v4_target),
        }
        stage_noise = {
            1: 0.03 * float(self.v2_target),
            2: 0.04 * float(self.v3_target),
            3: 0.03 * float(self.v4_target),
        }
        stage_bounds = [(0, l1), (l1, l1 + l2), (l1 + l2, l1 + l2 + l3), (l1 + l2 + l3, len(pos))]
        for stage_idx, (start_i, end_i) in enumerate(stage_bounds[1:], start=1):
            n = int(end_i - start_i)
            if n <= 0:
                continue
            u = np.linspace(0.0, 1.0, n, endpoint=True)
            profile = (
                stage_targets[stage_idx]
                + stage_amplitudes[stage_idx]
                * np.sin(np.pi * u)
                * np.sin((1.00 + 0.10 * stage_idx) * np.pi * u + float(latents["phase"]) + 0.20 * stage_idx)
            )
            profile += self._smooth_noise(rng, n, stage_noise[stage_idx], kernel_size=5)
            speed_trace[start_i:end_i] = np.clip(profile, 0.0, None)

        return np.asarray(pos, dtype=float), np.asarray(theta, dtype=float), np.asarray(labels, dtype=int), force, speed_trace

    @staticmethod
    def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
        return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _traj_cache_key(traj: np.ndarray):
        arr = np.ascontiguousarray(np.asarray(traj, dtype=np.float64))
        return arr.shape, arr.tobytes()

    def register_force_trace(self, traj: np.ndarray, force: np.ndarray):
        self._cached_force_traces[self._traj_cache_key(traj)] = np.asarray(force, dtype=float).copy()

    def register_speed_trace(self, traj: np.ndarray, speed: np.ndarray):
        self._cached_speed_traces[self._traj_cache_key(traj)] = np.asarray(speed, dtype=float).copy()

    def _lookup_cached_force_trace(self, traj: np.ndarray):
        key = self._traj_cache_key(traj)
        force = self._cached_force_traces.get(key)
        if force is None:
            return None
        return np.asarray(force, dtype=float)

    def _lookup_cached_speed_trace(self, traj: np.ndarray):
        key = self._traj_cache_key(traj)
        speed = self._cached_speed_traces.get(key)
        if speed is None:
            return None
        return np.asarray(speed, dtype=float)

    def _estimate_force_from_state(self, traj: np.ndarray) -> np.ndarray:
        traj = np.asarray(traj, dtype=float)
        pos = traj[:, :2]
        theta = traj[:, 2]
        T = len(pos)
        speed = np.zeros(T, dtype=float)
        tangential_speed = np.zeros(T, dtype=float)
        dz = np.zeros(T, dtype=float)
        if T > 1:
            delta = np.diff(pos, axis=0) / self.dt
            speed[1:] = np.linalg.norm(delta, axis=1)
            speed[0] = speed[1]
            tangential_speed[1:] = np.abs(delta[:, 0])
            tangential_speed[0] = tangential_speed[1]
            dz[1:] = np.abs(delta[:, 1])
            dz[0] = dz[1]
        orientation_error = np.abs(self._wrap_to_pi(theta - self.slot_theta))
        z = pos[:, 1]
        x = pos[:, 0]
        contact_gate = 1.0 / (1.0 + np.exp((z - 0.012) / 0.006))
        slide_progress = 1.0 / (1.0 + np.exp(-(x - self.stage2_end[0]) / 0.055))
        insert_progress = 1.0 / (1.0 + np.exp(-(x - self.stage3_end[0]) / 0.045))

        contact_weight = np.clip(contact_gate * (1.0 - slide_progress), 0.0, 1.0)
        slide_weight = np.clip(contact_gate * slide_progress * (1.0 - insert_progress), 0.0, 1.0)
        insert_weight = np.clip(contact_gate * insert_progress, 0.0, 1.0)
        weight_sum = np.maximum(contact_weight + slide_weight + insert_weight, 1e-6)
        contact_weight = contact_weight / weight_sum
        slide_weight = slide_weight / weight_sum
        insert_weight = insert_weight / weight_sum
        precontact_gate = 1.0 / (1.0 + np.exp((x - self.stage1_end[0]) / 0.05))
        precontact_mean = 0.010
        base_lower = (
            contact_weight * self.f_contact_min
            + slide_weight * self.f_slide_min
            + insert_weight * self.f_insert_min
        )
        contact_margin = self._make_stage_force_margin_profile(
            stage_idx=1,
            z=z,
            speed=speed,
            tangential_speed=tangential_speed,
            dz=dz,
            orientation_error=orientation_error,
            contact_gate=contact_gate,
            slide_progress=slide_progress,
            insert_progress=insert_progress,
            rng=None,
            latents=None,
        )
        slide_margin = self._make_stage_force_margin_profile(
            stage_idx=2,
            z=z,
            speed=speed,
            tangential_speed=tangential_speed,
            dz=dz,
            orientation_error=orientation_error,
            contact_gate=contact_gate,
            slide_progress=slide_progress,
            insert_progress=insert_progress,
            rng=None,
            latents=None,
        )
        insert_margin = self._make_stage_force_margin_profile(
            stage_idx=3,
            z=z,
            speed=speed,
            tangential_speed=tangential_speed,
            dz=dz,
            orientation_error=orientation_error,
            contact_gate=contact_gate,
            slide_progress=slide_progress,
            insert_progress=insert_progress,
            rng=None,
            latents=None,
        )
        raw_force = (
            contact_weight * (self.f_contact_min + contact_margin)
            + slide_weight * (self.f_slide_min + slide_margin)
            + insert_weight * (self.f_insert_min + insert_margin)
        )
        force = precontact_gate * precontact_mean + (1.0 - precontact_gate) * np.maximum(base_lower, raw_force)
        force[precontact_gate > 0.5] = np.clip(force[precontact_gate > 0.5], 0.0, 0.03)
        return np.clip(force, 0.0, None)

    def compute_all_features_matrix(self, traj: np.ndarray, feat_ids=None) -> np.ndarray:
        traj = np.asarray(traj, dtype=float)
        T = traj.shape[0]
        speed_cached = self._lookup_cached_speed_trace(traj)
        speed = np.zeros(T, dtype=float) if speed_cached is None else np.asarray(speed_cached, dtype=float)
        if speed_cached is None and T > 1:
            speed_edge = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1) / self.dt
            speed[0] = speed_edge[0]
            speed[1:] = speed_edge
        surface_distance = np.abs(traj[:, 1])
        orientation_error = np.abs(self._wrap_to_pi(traj[:, 2] - self.slot_theta))
        if traj.shape[1] >= 4:
            force = np.asarray(traj[:, 3], dtype=float)
        else:
            force = self._lookup_cached_force_trace(traj)
            if force is None:
                force = self._estimate_force_from_state(traj)
        noise_aux = 0.35 * np.sin(0.19 * np.arange(T)) + 0.15 * np.cos(0.07 * np.arange(T))
        F = np.c_[surface_distance, force, orientation_error, speed, noise_aux]
        return F if feat_ids is None else F[:, feat_ids]


def load_S4SlideInsert(
    n_demos: int = 10,
    seed: int = 123,
    env_kwargs=None,
):
    env_cfg = {}
    if env_kwargs:
        env_cfg.update(env_kwargs)
    env = S4SlideInsertEnv(**env_cfg)

    demos = []
    labels = []
    for i in range(n_demos):
        pos, theta, z, force, speed = env.generate_demo(seed=seed + i)
        demo = np.c_[pos, theta]
        env.register_force_trace(demo, force)
        env.register_speed_trace(demo, speed)
        demos.append(np.asarray(demo, dtype=float))
        labels.append(np.asarray(z, dtype=int))

    cutpoints = [np.where(np.diff(z) != 0)[0].astype(int) for z in labels]
    env.demo_subgoals = [np.asarray(x[int(c[1]), :3], dtype=float).copy() for x, c in zip(demos, cutpoints)]
    env.demo_goals = [np.asarray(x[-1, :3], dtype=float).copy() for x in demos]
    env.demo_stage_lengths = [np.bincount(np.asarray(z, dtype=int), minlength=env.n_segments).astype(int) for z in labels]
    env.subgoal = np.mean(np.stack(env.demo_subgoals, axis=0), axis=0)
    env.goal = np.mean(np.stack(env.demo_goals, axis=0), axis=0)

    return TaskBundle(
        name="S4SlideInsert",
        demos=demos,
        env=env,
        true_taus=None,
        true_cutpoints=[np.asarray(c, dtype=int) for c in cutpoints],
        true_labels=labels,
        feature_schema=env.get_feature_schema(),
        true_constraints=env.get_true_constraints(),
        constraint_specs=env.get_constraint_specs(),
        meta={"seed": seed, "cutpoints": [c.tolist() for c in cutpoints], "task_name": "S4SlideInsert"},
    )
