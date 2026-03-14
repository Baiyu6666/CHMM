from __future__ import annotations

import numpy as np


def compute_per_demo_lastpoint_subgoals(X_full, Z_list, cl_dims=None):
    per_demo = []
    max_segments = 0
    for x, z in zip(X_full, Z_list):
        x = np.asarray(x, float)
        z = np.asarray(z, int)
        cut_idxs = np.where(np.diff(z) != 0)[0]
        endpoints = [x[idx] for idx in cut_idxs]
        vec = np.asarray(endpoints, float) if endpoints else np.zeros((0, x.shape[1]), dtype=float)
        if cl_dims is not None and vec.size > 0:
            vec = vec[:, cl_dims]
        per_demo.append(vec)
        max_segments = max(max_segments, len(vec))
    return per_demo, max_segments


def average_subgoals_from_per_demo(per_demo_vec, K_target=None):
    if K_target is None:
        K_target = max((len(v) for v in per_demo_vec), default=0)
    if K_target <= 0:
        return np.zeros((0, 0), dtype=float)
    dim = 0
    for vec in per_demo_vec:
        if len(vec) > 0:
            dim = vec.shape[1]
            break
    if dim == 0:
        return np.zeros((K_target, 0), dtype=float)
    out = []
    for k in range(K_target):
        pts = [vec[k] for vec in per_demo_vec if len(vec) > k]
        if pts:
            out.append(np.mean(np.stack(pts, axis=0), axis=0))
        else:
            out.append(np.full(dim, np.nan))
    return np.asarray(out, float)


def take_first2_for_plot(per_demo_vec):
    out = []
    for vec in per_demo_vec:
        arr = np.asarray(vec, float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out.append(arr[:, :2] if arr.size > 0 else np.zeros((0, 2), dtype=float))
    return out


def take_first2_array(arr):
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    return arr[:, :2]
