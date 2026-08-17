from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from planner import optimize_trajectory, resample_polyline


SpeedIntent = Optional[Union[Callable[[int], np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class TimeParameterizedPath:
    positions: np.ndarray
    timestamps: np.ndarray
    sample_distances: np.ndarray
    intent_weights: np.ndarray
    intended_edge_speeds: np.ndarray
    reference_edge_speeds: np.ndarray
    target_speed: float
    speed_limit: float
    acceleration_limit: float
    motion_limits_enforced: bool

    def summary(self) -> dict:
        return {
            "method": "fixed_step_path_time_parameterization",
            "sample_count": int(len(self.positions)),
            "duration": float(self.timestamps[-1]) if len(self.timestamps) else 0.0,
            "target_speed": float(self.target_speed),
            "speed_limit": float(self.speed_limit),
            "acceleration_limit": float(self.acceleration_limit),
            "motion_limits_enforced": bool(self.motion_limits_enforced),
        }


def polyline_length(path) -> float:
    points = np.asarray(path, dtype=float)
    if len(points) <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def sample_polyline_at_distances(path, sample_distances):
    points = np.asarray(path, dtype=float)
    distances = np.asarray(sample_distances, dtype=float).reshape(-1)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    if len(points) == 1:
        return np.repeat(points, len(distances), axis=0)
    edges = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(edges)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(points[:1], len(distances), axis=0)
    clipped = np.clip(distances, 0.0, total)
    output = np.empty((len(clipped), points.shape[1]), dtype=float)
    for output_index, target in enumerate(clipped):
        edge_index = int(np.searchsorted(cumulative, target, side="right") - 1)
        edge_index = max(0, min(edge_index, len(points) - 2))
        span = float(cumulative[edge_index + 1] - cumulative[edge_index])
        alpha = 0.0 if span <= 1e-12 else float((target - cumulative[edge_index]) / span)
        output[output_index] = (1.0 - alpha) * points[edge_index] + alpha * points[edge_index + 1]
    return output


def smooth_speed_intent(values, kernel_size: int = 3) -> np.ndarray:
    intent = np.asarray(values, dtype=float)
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or len(intent) == 0:
        return intent
    kernel = np.ones(kernel_size, dtype=float) / float(kernel_size)
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    padded = np.pad(intent, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def gaussian_slowdown_weights(num_edges: int, *, depth, center, width) -> np.ndarray:
    num_edges = int(max(num_edges, 1))
    if num_edges == 1:
        return np.ones(1, dtype=float)
    progress = np.linspace(0.0, 1.0, num_edges, endpoint=True)
    depths = np.asarray(depth, dtype=float).reshape(-1)
    centers = np.asarray(center, dtype=float).reshape(-1)
    widths = np.asarray(width, dtype=float).reshape(-1)
    event_count = max(len(depths), len(centers), len(widths))
    if len(depths) == 1 and event_count > 1:
        depths = np.repeat(depths, event_count)
    if len(centers) == 1 and event_count > 1:
        centers = np.repeat(centers, event_count)
    if len(widths) == 1 and event_count > 1:
        widths = np.repeat(widths, event_count)
    if not (len(depths) == len(centers) == len(widths)):
        raise ValueError("depth, center, and width must broadcast to the same number of slowdown events.")

    slowdown = np.zeros_like(progress)
    for event_depth, event_center, event_width in zip(depths, centers, widths):
        event_center = float(np.clip(event_center, 0.08, 0.92))
        event_width = float(max(event_width, 0.015))
        event_depth = float(np.clip(event_depth, 0.0, 0.45))
        slowdown = slowdown + event_depth * np.exp(
            -0.5 * ((progress - event_center) / event_width) ** 2
        )
    weights = smooth_speed_intent(1.0 - slowdown, kernel_size=3)
    return np.clip(weights, 0.55, None)


def stabilize_tail_weights(weights, *, tail_len: int = 3, floor_ratio: float = 0.94) -> np.ndarray:
    intent = np.asarray(weights, dtype=float).copy()
    tail_len = int(max(tail_len, 0))
    if len(intent) <= 2 or tail_len <= 0:
        return intent
    start = max(0, len(intent) - tail_len)
    anchor_start = max(0, start - 3)
    anchor = (
        float(np.mean(intent[anchor_start:start]))
        if start > anchor_start
        else float(intent[start - 1])
    )
    floor = float(floor_ratio) * anchor
    intent[start:] = np.maximum(intent[start:], floor)
    return intent


def concatenate_stage_timestamps(stages) -> np.ndarray:
    parameterized_stages = list(stages)
    if not parameterized_stages:
        return np.zeros(0, dtype=float)
    parts = []
    offset = 0.0
    for stage in parameterized_stages:
        timestamps = np.asarray(stage.timestamps, dtype=float).reshape(-1)
        if len(timestamps) == 0:
            continue
        if not parts:
            parts.append(timestamps)
        else:
            parts.append(offset + timestamps[1:])
        offset += float(timestamps[-1])
    if not parts:
        return np.zeros(0, dtype=float)
    combined = np.concatenate(parts)
    if len(combined) <= 1:
        return combined
    fixed_dt = next(
        (
            float(stage.timestamps[1] - stage.timestamps[0])
            for stage in parameterized_stages
            if len(stage.timestamps) > 1
        ),
        None,
    )
    if fixed_dt is None:
        return combined
    return np.arange(len(combined), dtype=float) * fixed_dt


class FixedStepTimeParameterizer:
    def __init__(self, *, dt: float, segment_count_slack: float = 0.0):
        self.dt = float(dt)
        self.segment_count_slack = float(segment_count_slack)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")

    @staticmethod
    def _resolve_intent(speed_intent: SpeedIntent, num_edges: int) -> np.ndarray:
        if speed_intent is None:
            return np.ones(int(num_edges), dtype=float)
        values = speed_intent(int(num_edges)) if callable(speed_intent) else speed_intent
        weights = np.asarray(values, dtype=float).reshape(-1)
        if weights.size != int(num_edges):
            raise ValueError(f"speed intent produced {weights.size} values, expected {int(num_edges)}.")
        return np.clip(weights, 1e-6, None)

    def _result(
        self,
        *,
        positions,
        sample_distances,
        intent_weights,
        target_speed: float,
        speed_limit: float,
        acceleration_limit: float,
        motion_limits_enforced: bool,
    ) -> TimeParameterizedPath:
        positions = np.asarray(positions, dtype=float)
        timestamps = np.arange(len(positions), dtype=float) * self.dt
        sample_distances = np.asarray(sample_distances, dtype=float).reshape(-1)
        intent_weights = np.asarray(intent_weights, dtype=float).reshape(-1)
        if len(sample_distances) != len(positions):
            raise ValueError("sample distances must align with time-parameterized positions.")
        if len(intent_weights) != max(len(positions) - 1, 0):
            raise ValueError("speed-intent weights must align with time-parameterized edges.")
        intended_edge_speeds = np.diff(sample_distances) / self.dt if len(sample_distances) > 1 else np.zeros(0)
        reference_edge_speeds = (
            np.linalg.norm(np.diff(positions, axis=0), axis=1) / self.dt
            if len(positions) > 1
            else np.zeros(0, dtype=float)
        )
        return TimeParameterizedPath(
            positions=positions,
            timestamps=timestamps,
            sample_distances=sample_distances,
            intent_weights=intent_weights,
            intended_edge_speeds=intended_edge_speeds,
            reference_edge_speeds=reference_edge_speeds,
            target_speed=float(target_speed),
            speed_limit=float(speed_limit),
            acceleration_limit=float(acceleration_limit),
            motion_limits_enforced=bool(motion_limits_enforced),
        )

    def parameterize_fixed_count(
        self,
        path,
        num_points: int,
        *,
        speed_intent: SpeedIntent = None,
    ) -> TimeParameterizedPath:
        points = np.asarray(path, dtype=float)
        target_count = int(max(int(num_points), 2))
        if len(points) <= 1 or target_count <= 1:
            positions = points.copy()
            distances = np.zeros(len(positions), dtype=float)
            weights = np.ones(max(len(positions) - 1, 0), dtype=float)
            return self._result(
                positions=positions,
                sample_distances=distances,
                intent_weights=weights,
                target_speed=0.0,
                speed_limit=float("inf"),
                acceleration_limit=float("inf"),
                motion_limits_enforced=False,
            )

        path_length = polyline_length(points)
        num_edges = target_count - 1
        weights = self._resolve_intent(speed_intent, num_edges)
        if path_length <= 1e-10:
            positions = np.repeat(points[:1], target_count, axis=0)
            distances = np.zeros(target_count, dtype=float)
        elif speed_intent is None:
            distances = np.linspace(0.0, path_length, target_count, endpoint=True)
            positions = sample_polyline_at_distances(points, distances)
        else:
            step_lengths = path_length * (weights / np.sum(weights))
            distances = np.concatenate([[0.0], np.cumsum(step_lengths)])
            distances[-1] = path_length
            positions = sample_polyline_at_distances(points, distances)

        target_speed = path_length / max(num_edges * self.dt, 1e-12)
        return self._result(
            positions=positions,
            sample_distances=distances,
            intent_weights=weights,
            target_speed=target_speed,
            speed_limit=float("inf"),
            acceleration_limit=float("inf"),
            motion_limits_enforced=False,
        )

    def parameterize(
        self,
        path,
        *,
        speed_limit: float,
        acceleration_limit: float,
        target_speed: float | None = None,
        nominal_count: int | None = None,
        enforce_motion_limits: bool = True,
        speed_intent: SpeedIntent = None,
    ) -> TimeParameterizedPath:
        points = np.asarray(path, dtype=float)
        speed_limit = float(speed_limit)
        acceleration_limit = float(acceleration_limit)
        if target_speed is None:
            target_speed = 0.78 * speed_limit
        target_speed = float(np.clip(target_speed, 1e-4, max(1e-4, 0.995 * speed_limit)))

        if len(points) <= 2:
            distances = np.zeros(len(points), dtype=float)
            if len(points) > 1:
                distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            weights = np.ones(max(len(points) - 1, 0), dtype=float)
            return self._result(
                positions=points.copy(),
                sample_distances=distances,
                intent_weights=weights,
                target_speed=target_speed,
                speed_limit=speed_limit,
                acceleration_limit=acceleration_limit,
                motion_limits_enforced=False,
            )

        path_length = polyline_length(points)
        if path_length <= 1e-10:
            reference = points.copy()
            distances = np.zeros(len(reference), dtype=float)
            weights = np.ones(max(len(reference) - 1, 0), dtype=float)
        else:
            derived_count = int(np.ceil(path_length / max(target_speed * self.dt, 1e-6))) + 1
            if nominal_count is not None:
                nominal = max(int(nominal_count), 2)
                slack = max(self.segment_count_slack, 0.0)
                lower = max(2, int(np.floor((1.0 - slack) * nominal)))
                upper = max(lower + 1, int(np.ceil((1.0 + slack) * nominal)))
                target_count = int(np.clip(derived_count, lower, upper))
            else:
                target_count = max(2, int(derived_count))
            num_edges = max(target_count - 1, 1)
            weights = self._resolve_intent(speed_intent, num_edges)
            if speed_intent is None:
                max_step = path_length / max(num_edges, 1)
                reference = resample_polyline(points, max_step=max(max_step, 1e-6))
                distances = np.linspace(0.0, path_length, len(reference), endpoint=True)
                if len(reference) - 1 != len(weights):
                    weights = np.ones(max(len(reference) - 1, 0), dtype=float)
            else:
                step_lengths = path_length * (weights / np.sum(weights))
                distances = np.concatenate([[0.0], np.cumsum(step_lengths)])
                distances[-1] = path_length
                reference = sample_polyline_at_distances(points, distances)

        positions = np.asarray(reference, dtype=float)
        if enforce_motion_limits:
            positions = optimize_trajectory(
                positions,
                dt=self.dt,
                v_max=speed_limit,
                a_max=acceleration_limit,
                projector=None,
            )
        return self._result(
            positions=positions,
            sample_distances=distances,
            intent_weights=weights,
            target_speed=target_speed,
            speed_limit=speed_limit,
            acceleration_limit=acceleration_limit,
            motion_limits_enforced=enforce_motion_limits,
        )
