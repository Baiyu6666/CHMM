# methods/hmm_backend.py
from __future__ import annotations

import numpy as np
from ..base import compute_cutpoint_metrics, format_training_log
from ..common.tau_init import clip_tau_for_sequence, resolve_tau_init_for_demos

# ---------- module-level switch for left-to-right constraint ----------
LEFT_RIGHT_DEFAULT = True  #      -> (Bakis)  

def _left_right_mask(K: int):
    """English documentation omitted during cleanup."""
    M = np.zeros((K, K), dtype=float)
    for i in range(K):
        M[i, i] = 1.0
        if i + 1 < K:
            M[i, i + 1] = 1.0
    return M

# ---------- utils ----------
def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return s if keepdims else np.squeeze(s, axis=axis)

def _feature_schema(env):
    if hasattr(env, "get_feature_schema"):
        return list(env.get_feature_schema())
    if hasattr(env, "feature_schema") and env.feature_schema is not None:
        return list(env.feature_schema)
    return None


def _resolve_selected_feature_columns(env, selected_raw_feature_ids):
    if selected_raw_feature_ids is None:
        schema = _feature_schema(env)
        if schema is not None:
            return [int(spec.get("column_idx", i)) for i, spec in enumerate(schema)]
        raise ValueError("env feature schema is required when selected_raw_feature_ids is None.")

    schema = _feature_schema(env)
    name_to_column = {}
    id_to_column = {}
    if schema is not None:
        for i, spec in enumerate(schema):
            column_idx = int(spec.get("column_idx", i))
            name_to_column[str(spec.get("name", f"f{i}"))] = column_idx
            id_to_column[int(spec.get("id", i))] = column_idx

    out = []
    for value in selected_raw_feature_ids:
        if isinstance(value, str):
            if value not in name_to_column:
                raise KeyError(f"Unknown feature name '{value}'.")
            out.append(name_to_column[value])
        else:
            raw_id = int(value)
            if raw_id not in id_to_column:
                raise KeyError(f"Unknown raw feature id '{raw_id}'.")
            out.append(id_to_column[raw_id])
    return out



def _make_point_features_list(
    X_list,
    use_velocity=False,
    vel_weight=1.0,
    standardize=False,
    env=None,
    use_env_features=True,
    selected_raw_feature_ids=None,
):
    """
    English documentation omitted during cleanup.
      English documentation omitted during cleanup.
      English documentation omitted during cleanup.
    English documentation omitted during cleanup.
    """
    Ys = []
    for x in X_list:
        x = np.asarray(x, float)
        if not use_velocity:
            Ys.append(x)
        else:
            v = np.vstack([np.zeros((1, x.shape[1]), dtype=float), x[1:] - x[:-1]])
            y = np.hstack([x, vel_weight * v])
            Ys.append(y)
    if use_env_features:
        if env is None:
            raise ValueError("env must be provided when use_env_features=True.")
        feature_cols = _resolve_selected_feature_columns(env, selected_raw_feature_ids)
        Ys = [
            np.hstack([y, np.asarray(env.compute_all_features_matrix(x), float)[:, feature_cols]])
            for x, y in zip(X_list, Ys)
        ]
    if standardize:
        allY = np.vstack(Ys)
        mu = allY.mean(axis=0, keepdims=True)
        std = allY.std(axis=0, keepdims=True) + 1e-8
        Ys = [ (y - mu)/std for y in Ys ]
    return Ys

# ---------- BIC helpers ----------
def _count_free_trans_params(K: int, left_right: bool) -> int:
    """English documentation omitted during cleanup."""
    if not left_right:
        # English comment omitted during cleanup.
        return K * (K - 1)
    # English comment omitted during cleanup.
    # English comment omitted during cleanup.
    free = 0
    for i in range(K):
        allowed = 1 + (1 if i + 1 < K else 0)
        free += max(allowed - 1, 0)
    return free

def _num_params_standard(K: int, D: int, left_right: bool) -> int:
    """
    English documentation omitted during cleanup.
      English documentation omitted during cleanup.
      English documentation omitted during cleanup.
      English documentation omitted during cleanup.
    """
    p_pi = K - 1
    p_A  = _count_free_trans_params(K, left_right)
    p_em = K * (2 * D)
    return p_pi + p_A + p_em

def _num_params_ar(K: int, D: int, left_right: bool) -> int:
    """
    English documentation omitted during cleanup.
      English documentation omitted during cleanup.
      English documentation omitted during cleanup.
      English documentation omitted during cleanup.
      English documentation omitted during cleanup.
    """
    p_pi = K - 1
    p_A = 0 if left_right else _count_free_trans_params(K, left_right)
    p_init = K * (2 * D)
    p_W = K * (D * (1 + D))
    p_var = K * D
    p_dur = 2 * K
    return p_pi + p_A + p_init + p_W + p_var + p_dur

def _total_loglik_ar(model, X_list):
    """English documentation omitted during cleanup."""
    total = 0.0
    for x in X_list:
        if hasattr(model, "score_sequence"):
            total += float(model.score_sequence(x))
        else:
            _, _, logZ = model._fb(x)
            total += float(logZ)
    return total

def _uniform_stage_ends(T: int, num_stages: int, min_duration: int) -> list[int]:
    ends = np.linspace(0, T, num_stages + 1, dtype=int)[1:] - 1
    ends[-1] = T - 1
    for k in range(num_stages - 1):
        min_end = (k + 1) * min_duration - 1
        max_end = ends[k + 1] - min_duration
        ends[k] = int(np.clip(ends[k], min_end, max_end))
    return [int(v) for v in ends.tolist()]


def _stage_ends_to_labels(T: int, stage_ends: list[int]) -> np.ndarray:
    labels = np.zeros(int(T), dtype=int)
    start = 0
    for stage_idx, end in enumerate(stage_ends):
        end_i = int(end)
        labels[start : end_i + 1] = int(stage_idx)
        start = end_i + 1
    return labels


def _diag_logpdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(mean, dtype=float)
    var_arr = np.clip(np.asarray(var, dtype=float), 1e-12, None)
    return float(-0.5 * np.sum(np.log(2.0 * np.pi * var_arr) + (diff * diff) / var_arr))


class ARHSMM:
    def __init__(
        self,
        n_stages,
        n_dims,
        sticky=0.0,
        rng=None,
        min_covar=1e-4,
        ridge=1e-6,
        left_right: bool = LEFT_RIGHT_DEFAULT,
        min_duration: int = 1,
        max_duration: int | None = None,
        duration_weight: float = 1.0,
        duration_var_floor: float = 4.0,
    ):
        self.K = int(n_stages)
        self.D = int(n_dims)
        self.rng = rng or np.random.RandomState(0)
        self.sticky = float(sticky)
        self.min_covar = float(min_covar)
        self.ridge = float(ridge)
        self.left_right = bool(left_right)
        if not self.left_right:
            raise ValueError("ARHSMM currently requires left_right=True.")
        self.min_duration = max(int(min_duration), 1)
        self.max_duration = None if max_duration is None else max(int(max_duration), self.min_duration)
        self.duration_weight = float(duration_weight)
        self.duration_var_floor = max(float(duration_var_floor), 1e-6)

        self.pi = np.zeros(self.K, dtype=float)
        self.pi[0] = 1.0
        self.A = np.zeros((self.K, self.K), dtype=float)
        for i in range(self.K - 1):
            self.A[i, i + 1] = 1.0
        if self.K > 0:
            self.A[-1, -1] = 1.0

        self.init_mu = self.rng.randn(self.K, self.D)
        self.init_var = np.ones((self.K, self.D), dtype=float)
        self.mu = self.init_mu.copy()
        self.W = np.zeros((self.K, 1 + self.D, self.D), dtype=float)
        self.var = np.ones((self.K, self.D), dtype=float)
        self.duration_mean = np.full(self.K, float(self.min_duration), dtype=float)
        self.duration_var = np.full(self.K, self.duration_var_floor, dtype=float)
        self.stage_ends_ = None

    def _warm_start_stage_ends(self, X, tau_init):
        stage_ends = []
        if tau_init is not None and self.K == 2:
            tau_init = np.asarray(tau_init, dtype=int)
            for x, tau in zip(X, tau_init):
                t = clip_tau_for_sequence(x, tau)
                stage_ends.append([int(t), int(len(x) - 1)])
            return stage_ends
        for x in X:
            if len(x) < self.K * self.min_duration:
                raise ValueError(
                    f"Sequence length {len(x)} is too short for {self.K} stages with minimum duration {self.min_duration}."
                )
            stage_ends.append(_uniform_stage_ends(len(x), self.K, self.min_duration))
        return stage_ends

    def _fit_from_stage_ends(self, X, stage_ends_list):
        global_stack = np.concatenate([np.asarray(x, dtype=float) for x in X], axis=0)
        global_mean = np.mean(global_stack, axis=0)
        global_var = np.var(global_stack, axis=0) + self.min_covar

        start_sum = np.zeros((self.K, self.D), dtype=float)
        start_sq = np.zeros((self.K, self.D), dtype=float)
        start_count = np.zeros(self.K, dtype=float)
        point_sum = np.zeros((self.K, self.D), dtype=float)
        point_count = np.zeros(self.K, dtype=float)
        XtX = [np.zeros((1 + self.D, 1 + self.D), dtype=float) for _ in range(self.K)]
        XtY = [np.zeros((1 + self.D, self.D), dtype=float) for _ in range(self.K)]
        durations = [[] for _ in range(self.K)]

        for x, stage_ends in zip(X, stage_ends_list):
            start = 0
            for stage_idx, end in enumerate(stage_ends):
                end_i = int(end)
                seg = np.asarray(x[start : end_i + 1], dtype=float)
                if len(seg) == 0:
                    start = end_i + 1
                    continue
                start_sum[stage_idx] += seg[0]
                start_sq[stage_idx] += seg[0] ** 2
                start_count[stage_idx] += 1.0
                point_sum[stage_idx] += np.sum(seg, axis=0)
                point_count[stage_idx] += float(len(seg))
                durations[stage_idx].append(int(len(seg)))
                for t in range(1, len(seg)):
                    y = np.hstack([1.0, seg[t - 1]])
                    XtX[stage_idx] += np.outer(y, y)
                    XtY[stage_idx] += np.outer(y, seg[t])
                start = end_i + 1

        for stage_idx in range(self.K):
            if start_count[stage_idx] > 0:
                self.init_mu[stage_idx] = start_sum[stage_idx] / start_count[stage_idx]
                start_var = start_sq[stage_idx] / start_count[stage_idx] - self.init_mu[stage_idx] ** 2
                self.init_var[stage_idx] = np.clip(start_var, self.min_covar, None)
            else:
                self.init_mu[stage_idx] = global_mean
                self.init_var[stage_idx] = global_var

            if point_count[stage_idx] > 0:
                self.mu[stage_idx] = point_sum[stage_idx] / point_count[stage_idx]
            else:
                self.mu[stage_idx] = self.init_mu[stage_idx]

            XtXk = XtX[stage_idx] + self.ridge * np.eye(1 + self.D)
            if np.count_nonzero(XtX[stage_idx]) == 0:
                self.W[stage_idx] = 0.0
            else:
                try:
                    self.W[stage_idx] = np.linalg.solve(XtXk, XtY[stage_idx])
                except np.linalg.LinAlgError:
                    self.W[stage_idx] = np.linalg.lstsq(XtXk, XtY[stage_idx], rcond=None)[0]

            dur_arr = np.asarray(durations[stage_idx], dtype=float)
            if dur_arr.size > 0:
                self.duration_mean[stage_idx] = float(np.mean(dur_arr))
                self.duration_var[stage_idx] = float(
                    max(np.var(dur_arr) + 1.0, self.duration_var_floor)
                )
            else:
                self.duration_mean[stage_idx] = float(self.min_duration)
                self.duration_var[stage_idx] = float(self.duration_var_floor)

        res_ss = np.zeros((self.K, self.D), dtype=float)
        res_count = np.zeros(self.K, dtype=float)
        for x, stage_ends in zip(X, stage_ends_list):
            start = 0
            for stage_idx, end in enumerate(stage_ends):
                end_i = int(end)
                seg = np.asarray(x[start : end_i + 1], dtype=float)
                for t in range(1, len(seg)):
                    y = np.hstack([1.0, seg[t - 1]])
                    pred = y @ self.W[stage_idx]
                    diff = seg[t] - pred
                    res_ss[stage_idx] += diff ** 2
                    res_count[stage_idx] += 1.0
                start = end_i + 1
        for stage_idx in range(self.K):
            if res_count[stage_idx] > 0:
                self.var[stage_idx] = np.clip(res_ss[stage_idx] / res_count[stage_idx], self.min_covar, None)
            else:
                self.var[stage_idx] = self.init_var[stage_idx].copy()

    def _emission_tables(self, x):
        x = np.asarray(x, dtype=float)
        T = len(x)
        init_log = np.zeros((self.K, T), dtype=float)
        ar_log = np.zeros((self.K, T), dtype=float)
        for stage_idx in range(self.K):
            for t in range(T):
                init_log[stage_idx, t] = _diag_logpdf(x[t], self.init_mu[stage_idx], self.init_var[stage_idx])
            for t in range(1, T):
                y = np.hstack([1.0, x[t - 1]])
                mean = y @ self.W[stage_idx]
                ar_log[stage_idx, t] = _diag_logpdf(x[t], mean, self.var[stage_idx])
        ar_prefix = np.cumsum(ar_log, axis=1)
        return init_log, ar_prefix

    def _segment_loglik(self, init_log, ar_prefix, stage_idx: int, start: int, end: int) -> float:
        score = float(init_log[stage_idx, start])
        if end > start:
            score += float(ar_prefix[stage_idx, end] - ar_prefix[stage_idx, start])
        return score

    def _duration_logprob(self, stage_idx: int, duration: int) -> float:
        mean = max(float(self.duration_mean[stage_idx]), float(self.min_duration))
        var = max(float(self.duration_var[stage_idx]), self.duration_var_floor)
        diff = float(duration) - mean
        return float(self.duration_weight * (-0.5 * (np.log(2.0 * np.pi * var) + (diff * diff) / var)))

    def _decode(self, x):
        x = np.asarray(x, dtype=float)
        T = len(x)
        if T < self.K * self.min_duration:
            raise ValueError(
                f"Sequence length {T} is too short for {self.K} stages with minimum duration {self.min_duration}."
            )
        init_log, ar_prefix = self._emission_tables(x)
        max_duration = T if self.max_duration is None else min(int(self.max_duration), T)
        dp = np.full((self.K, T), -np.inf, dtype=float)
        prev = np.full((self.K, T), -1, dtype=int)

        for end in range(self.min_duration - 1, T - self.min_duration * (self.K - 1)):
            duration = end + 1
            if duration > max_duration:
                continue
            dp[0, end] = self._segment_loglik(init_log, ar_prefix, 0, 0, end) + self._duration_logprob(0, duration)

        for stage_idx in range(1, self.K):
            end_low = (stage_idx + 1) * self.min_duration - 1
            end_high = T - self.min_duration * (self.K - stage_idx - 1) - 1
            for end in range(end_low, end_high + 1):
                start_low = max(stage_idx * self.min_duration, end - max_duration + 1)
                start_high = end - self.min_duration + 1
                best = -np.inf
                best_prev = -1
                for start in range(start_low, start_high + 1):
                    duration = end - start + 1
                    prev_end = start - 1
                    if prev_end < 0 or not np.isfinite(dp[stage_idx - 1, prev_end]):
                        continue
                    cand = (
                        dp[stage_idx - 1, prev_end]
                        + self._segment_loglik(init_log, ar_prefix, stage_idx, start, end)
                        + self._duration_logprob(stage_idx, duration)
                    )
                    if cand > best:
                        best = float(cand)
                        best_prev = int(prev_end)
                dp[stage_idx, end] = best
                prev[stage_idx, end] = best_prev

        final_score = float(dp[self.K - 1, T - 1])
        if not np.isfinite(final_score):
            raise RuntimeError("ARHSMM decoding failed to find a feasible segmentation.")

        stage_ends = [T - 1]
        cur_end = T - 1
        for stage_idx in range(self.K - 1, 0, -1):
            prev_end = int(prev[stage_idx, cur_end])
            if prev_end < 0:
                raise RuntimeError("ARHSMM failed to backtrack stage boundaries.")
            stage_ends.append(prev_end)
            cur_end = prev_end
        stage_ends.reverse()
        labels = _stage_ends_to_labels(T, stage_ends)
        return labels, stage_ends, final_score

    def score_sequence(self, x):
        _, _, score = self._decode(x)
        return float(score)

    def fit(self, X, n_iter=30, verbose=True, true_taus=None, tau_init=None):
        stage_ends_list = self._warm_start_stage_ends(X, tau_init)
        hist_loglik = []
        hist_metrics = {"MeanAbsCutpointError": [], "CutpointExactMatchRate": []}
        stage_ends_history = []
        self.converged_ = False
        self.converged_iter_ = None
        Z = [_stage_ends_to_labels(len(x), ends) for x, ends in zip(X, stage_ends_list)]

        for it in range(n_iter):
            self._fit_from_stage_ends(X, stage_ends_list)
            decoded_ends = []
            decoded_labels = []
            total_score = 0.0
            for x in X:
                labels, stage_ends, score = self._decode(x)
                decoded_labels.append(labels)
                decoded_ends.append(stage_ends)
                total_score += float(score)
            converged = decoded_ends == stage_ends_list
            stage_ends_list = decoded_ends
            stage_ends_history.append([[int(v) for v in ends] for ends in stage_ends_list])
            self.stage_ends_ = [[int(v) for v in ends] for ends in stage_ends_list]
            Z = [np.asarray(z, dtype=int) for z in decoded_labels]

            avg_ll = total_score / max(len(X), 1)
            hist_loglik.append(float(avg_ll))
            iter_metrics = {}
            if true_taus is not None:
                cutpoints_cur = [np.where(np.diff(z.astype(int)) != 0)[0].astype(int) for z in Z]
                if self.K == 2:
                    true_cutpoints = [None if t is None else np.asarray([int(t)], dtype=int) for t in true_taus]
                else:
                    true_cutpoints = [None if t is None else np.asarray(t, dtype=int).reshape(-1) for t in true_taus]
                iter_metrics = compute_cutpoint_metrics(cutpoints_cur, true_cutpoints, X)
                for name in hist_metrics:
                    if name in iter_metrics:
                        hist_metrics[name].append(iter_metrics.get(name, np.nan))
            should_log = converged or ((it + 1) % 10 == 0) or (it == n_iter - 1)
            if verbose and should_log:
                print(format_training_log("ARHSMM", it, losses={"loss": avg_ll}, metrics=iter_metrics))
            if converged:
                self.converged_ = True
                self.converged_iter_ = int(it)
                if verbose:
                    print(f"[ARHSMM] converged on stable stage_ends at iter {it + 1:03d}")
                break

        hist = {"loglik": hist_loglik, "stage_ends": stage_ends_history}
        if self.converged_:
            hist["converged_iter"] = int(self.converged_iter_)
        nonempty_metrics = {k: v for k, v in hist_metrics.items() if v}
        if nonempty_metrics:
            hist.update(nonempty_metrics)
        return Z, hist

    def viterbi(self, x):
        labels, _, _ = self._decode(x)
        return labels

def segment_with_hmm(
    X_full,
    env=None,
    true_taus=None,
    method="ar",
    n_stages=2,
    sticky=10.0,
    n_iter=30,
    verbose=True,
    seg_dims=None,
    seed: int = 0,
    use_velocity: bool = False,
    vel_weight: float = 1.0,
    standardize: bool = False,
    use_env_features: bool = True,
    selected_raw_feature_ids=None,
    tau_init=None,
    tau_init_mode: str = "uniform_taus",
    left_right: bool | None = None,
    min_duration: int = 1,
    max_duration: int | None = None,
    duration_weight: float = 1.0,
    duration_var_floor: float = 4.0,
):
    if left_right is None:
        left_right = LEFT_RIGHT_DEFAULT
    else:
        left_right = bool(left_right)

    if method.lower() != "ar":
        raise ValueError(f"Only method='ar' is supported. Got method={method!r}.")

    # English comment omitted during cleanup.
    X_seg_raw = [ (x if seg_dims is None else x[:, seg_dims]) for x in X_full ]
    # English comment omitted during cleanup.
    X_seg = _make_point_features_list(
        X_seg_raw, use_velocity=use_velocity,
        vel_weight=vel_weight,
        standardize=standardize,
        env=env,
        use_env_features=use_env_features,
        selected_raw_feature_ids=selected_raw_feature_ids,
    )
    D = X_seg[0].shape[1]

    import numpy as _np
    _rng = _np.random.RandomState(int(seed))
    tau_init_resolved = resolve_tau_init_for_demos(
        X_full,
        tau_init=tau_init,
        tau_init_mode=tau_init_mode,
        env=env,
        seed=seed,
        use_velocity=use_velocity,
        vel_weight=vel_weight,
        standardize=standardize,
        use_env_features=use_env_features,
        selected_raw_feature_ids=selected_raw_feature_ids,
    )

    def _fit_for_K(K_try: int):
        mdl = ARHSMM(
            n_stages=K_try,
            n_dims=D,
            sticky=sticky,
            rng=_rng,
            left_right=left_right,
            min_duration=min_duration,
            max_duration=max_duration,
            duration_weight=duration_weight,
            duration_var_floor=duration_var_floor,
        )
        Z_list_, hist = mdl.fit(
            X_seg,
            n_iter=n_iter,
            verbose=verbose,
            true_taus=true_taus,
            tau_init=tau_init_resolved if int(K_try) == 2 else None,
        )
        return mdl, Z_list_, hist

    chosen_model = None
    Z_states_list = None
    seg_hist = {}
    if isinstance(n_stages, (list, tuple, range)):
        K_cands = list(n_stages)
        if len(K_cands) == 0:
            raise ValueError("Empty K candidate list for BIC.")
        best = {"bic": np.inf, "K": None, "mdl": None, "Z": None, "hist": None}
        N_data = sum(len(x) for x in X_seg)
        bic_table = []
        for K_try in K_cands:
            mdl, Z_tmp, hist = _fit_for_K(int(K_try))
            logL = _total_loglik_ar(mdl, X_seg)
            p = _num_params_ar(K_try, D, left_right)
            bic = -2.0 * logL + p * np.log(max(N_data, 1))
            if verbose:
                print(f"[BIC] method=ar K={K_try} | logL={logL:.3f} p={p} N={N_data} BIC={bic:.3f}")
            bic_table.append(
                {"K": int(K_try), "logL": float(logL), "p": int(p), "N": int(N_data), "BIC": float(bic)})
            if bic < best["bic"]:
                best.update({"bic": bic, "K": K_try, "mdl": mdl, "Z": Z_tmp, "hist": hist})
        chosen_model = best["mdl"]
        Z_states_list = best["Z"]
        seg_hist = best["hist"] if isinstance(best["hist"], dict) else {}
        bic_table.sort(key=lambda d: d["K"])
        seg_hist["bic_table"] = bic_table
        seg_hist["bic_selected_K"] = int(best["K"])
    else:
        mdl, Z_states_list, seg_hist = _fit_for_K(int(n_stages))
        chosen_model = mdl

    Z_list = [np.asarray(z, dtype=int) for z in Z_states_list]
    return Z_list, chosen_model, seg_hist
