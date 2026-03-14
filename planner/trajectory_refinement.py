from __future__ import annotations

from typing import Callable

import numpy as np

"""
Unified planner entrypoint for the current codebase.

The project uses a single heuristic constrained-refinement stack for both:
- synthetic trajectory generation inside envs
- future planner-side testing and debugging

These routines do not solve a strict constrained optimization problem. They
combine reference tracking, smoothing, repeated projection, and velocity /
acceleration clipping to produce trajectories that approximately satisfy task
constraints.
"""


Projector = Callable[[np.ndarray], np.ndarray]


def resample_polyline(points: np.ndarray, max_step: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return pts.copy()

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total_len = float(np.sum(seg))
    if total_len <= 1e-12:
        return pts[[0]].copy()

    max_step = max(float(max_step), 1e-6)
    n_steps = max(2, int(np.ceil(total_len / max_step)) + 1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total_len, n_steps)

    out = []
    j = 0
    for target in targets:
        while j + 1 < len(s) and s[j + 1] < target:
            j += 1
        if j + 1 >= len(s):
            out.append(pts[-1])
            continue
        frac = (target - s[j]) / max(s[j + 1] - s[j], 1e-12)
        out.append((1.0 - frac) * pts[j] + frac * pts[j + 1])
    return np.asarray(out, dtype=float)


def _enforce_velocity_limit(path: np.ndarray, dt: float, v_max: float) -> np.ndarray:
    if len(path) <= 1:
        return path
    x = np.asarray(path, dtype=float)
    max_step = max(float(v_max) * max(float(dt), 1e-12), 1e-12)
    out = x.copy()
    out[0] = x[0]
    for i in range(1, len(out)):
        step = out[i] - out[i - 1]
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step:
            out[i] = out[i - 1] + (max_step / max(step_norm, 1e-12)) * step

    out[-1] = x[-1]
    for i in range(len(out) - 2, -1, -1):
        step = out[i] - out[i + 1]
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step:
            out[i] = out[i + 1] + (max_step / max(step_norm, 1e-12)) * step

    out[0] = x[0]
    for i in range(1, len(out)):
        step = out[i] - out[i - 1]
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step:
            out[i] = out[i - 1] + (max_step / max(step_norm, 1e-12)) * step
    return out


def _enforce_acceleration_limit(path: np.ndarray, dt: float, a_max: float, n_sweeps: int = 8) -> np.ndarray:
    if len(path) <= 2:
        return path
    dt = max(float(dt), 1e-12)
    out = np.asarray(path, dtype=float).copy()
    max_acc = float(a_max)
    for _ in range(max(int(n_sweeps), 1)):
        changed = False
        for t in range(1, len(out) - 1):
            acc_vec = (out[t + 1] - 2.0 * out[t] + out[t - 1]) / (dt * dt)
            acc_norm = float(np.linalg.norm(acc_vec))
            if acc_norm > max_acc:
                midpoint = 0.5 * (out[t - 1] + out[t + 1])
                alpha = min(1.0, (acc_norm - max_acc) / max(acc_norm, 1e-12))
                out[t] = (1.0 - alpha) * out[t] + alpha * midpoint
                changed = True
        out[0] = path[0]
        out[-1] = path[-1]
        if not changed:
            break
    return out


def max_speed(path: np.ndarray, dt: float) -> float:
    if len(path) <= 1:
        return 0.0
    vel = np.diff(np.asarray(path, dtype=float), axis=0) / max(float(dt), 1e-12)
    return float(np.max(np.linalg.norm(vel, axis=1)))


def max_acceleration(path: np.ndarray, dt: float) -> float:
    if len(path) <= 2:
        return 0.0
    x = np.asarray(path, dtype=float)
    acc = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / max(float(dt), 1e-12) ** 2
    return float(np.max(np.linalg.norm(acc, axis=1)))


def repair_trajectory_constraints(
    path: np.ndarray,
    dt: float,
    v_max: float,
    a_max: float,
    projector: Projector | None = None,
    n_rounds: int = 12,
) -> np.ndarray:
    """Heuristically repair a path to satisfy projector, speed, and acceleration limits."""
    out = np.asarray(path, dtype=float).copy()
    if len(out) <= 2:
        return out

    start = out[0].copy()
    goal = out[-1].copy()
    v_tol = max(float(v_max), 1e-12) * 1.001
    a_tol = max(float(a_max), 1e-12) * 1.01

    for _ in range(max(int(n_rounds), 1)):
        if projector is not None:
            out = projector(out)
        out[0] = start
        out[-1] = goal

        out = _enforce_velocity_limit(out, dt=dt, v_max=v_max)
        out[0] = start
        out[-1] = goal

        if projector is not None:
            out = projector(out)
        out[0] = start
        out[-1] = goal

        out = _enforce_acceleration_limit(out, dt=dt, a_max=a_max, n_sweeps=4)
        out[0] = start
        out[-1] = goal

        if projector is not None:
            out = projector(out)
        out[0] = start
        out[-1] = goal

        out = _enforce_velocity_limit(out, dt=dt, v_max=v_max)
        out[0] = start
        out[-1] = goal

        if max_speed(out, dt) <= v_tol and max_acceleration(out, dt) <= a_tol:
            break

    return out


def optimize_trajectory(
    reference: np.ndarray,
    dt: float,
    v_max: float,
    a_max: float,
    projector: Projector | None = None,
    n_iters: int = 80,
    anchor_weight: float = 0.35,
    smooth_weight: float = 0.45,
    step_size: float = 0.25,
) -> np.ndarray:
    """
    Refine a reference path with smoothing and repeated constraint repair.

    This is a heuristic constrained-refinement routine, not a globally optimal
    trajectory solver.
    """
    ref = np.asarray(reference, dtype=float)
    if len(ref) <= 2:
        return ref.copy()

    path = ref.copy()
    start = ref[0].copy()
    goal = ref[-1].copy()

    for _ in range(max(int(n_iters), 1)):
        grad_anchor = ref[1:-1] - path[1:-1]
        grad_smooth = path[:-2] + path[2:] - 2.0 * path[1:-1]
        path[1:-1] = path[1:-1] + step_size * (
            anchor_weight * grad_anchor + smooth_weight * grad_smooth
        )
        path[0] = start
        path[-1] = goal

        if projector is not None:
            path = projector(path)
            path[0] = start
            path[-1] = goal

        path = _enforce_velocity_limit(path, dt=dt, v_max=v_max)
        path[0] = start
        path[-1] = goal
        path = _enforce_acceleration_limit(path, dt=dt, a_max=a_max)
        path[0] = start
        path[-1] = goal

        if projector is not None:
            path = projector(path)
            path[0] = start
            path[-1] = goal

    return repair_trajectory_constraints(
        path,
        dt=dt,
        v_max=v_max,
        a_max=a_max,
        projector=projector,
        n_rounds=12,
    )

__all__ = [
    "resample_polyline",
    "repair_trajectory_constraints",
    "optimize_trajectory",
    "max_speed",
    "max_acceleration",
]
