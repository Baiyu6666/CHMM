from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import logsumexp

from ..base import compute_cutpoint_metrics, format_training_log
from .hmm import _resolve_selected_feature_columns, _resolve_stage_ends_init


def _build_observations(
    demos: list[np.ndarray],
    *,
    env,
    use_state: bool,
    use_velocity: bool,
    velocity_weight: float,
    use_env_features: bool,
    selected_raw_feature_ids,
    standardize: bool,
) -> list[np.ndarray]:
    feature_columns = (
        _resolve_selected_feature_columns(env, selected_raw_feature_ids)
        if use_env_features
        else []
    )
    observations = []
    for demo in demos:
        values = np.asarray(demo, dtype=float)
        parts = []
        if use_state:
            parts.append(values)
            if use_velocity:
                velocity = np.vstack(
                    [
                        np.zeros((1, values.shape[1]), dtype=float),
                        values[1:] - values[:-1],
                    ]
                )
                parts.append(float(velocity_weight) * velocity)
        if use_env_features:
            all_features = np.asarray(env.compute_all_features_matrix(values), dtype=float)
            parts.append(all_features[:, feature_columns])
        observations.append(np.hstack(parts))
    if standardize:
        stacked = np.vstack(observations)
        mean = np.mean(stacked, axis=0, keepdims=True)
        std = np.std(stacked, axis=0, keepdims=True) + 1e-8
        observations = [(values - mean) / std for values in observations]
    return observations


class DiagonalGMMHMM:
    def __init__(
        self,
        observations: list[np.ndarray],
        *,
        n_stages: int,
        n_components: int = 3,
        reg_covar: float = 1e-3,
        seed: int = 0,
        stage_ends_init: list[list[int]],
    ):
        self.observations = [np.asarray(values, dtype=float) for values in observations]
        self.num_stages = int(n_stages)
        self.n_components = int(n_components)
        self.reg_covar = max(float(reg_covar), 1e-9)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        if self.num_stages < 2:
            raise ValueError("GMM-HMM requires at least two stages.")
        if self.n_components < 1:
            raise ValueError("n_components must be at least one.")
        if not self.observations:
            raise ValueError("GMM-HMM requires at least one observation sequence.")
        self.n_features = int(self.observations[0].shape[1])
        if any(values.ndim != 2 or values.shape[1] != self.n_features for values in self.observations):
            raise ValueError("All GMM-HMM observation sequences must have the same feature dimension.")

        self.stage_ends_ = [[int(value) for value in ends] for ends in stage_ends_init]
        self.initial_stage_ends_ = [list(ends) for ends in self.stage_ends_]
        self.segmentation_history_ = [[list(ends) for ends in self.stage_ends_]]
        self.loss_loglik: list[float] = []
        self.objective_history_ = self.loss_loglik
        self.converged_ = False
        self.converged_iter_: int | None = None

        self.weights_ = np.full(
            (self.num_stages, self.n_components),
            1.0 / float(self.n_components),
            dtype=float,
        )
        self.means_ = np.zeros(
            (self.num_stages, self.n_components, self.n_features),
            dtype=float,
        )
        self.variances_ = np.ones_like(self.means_)
        self.transition_matrix_ = np.zeros((self.num_stages, self.num_stages), dtype=float)
        self._initialize_from_segments()

    def _stage_samples(self) -> list[np.ndarray]:
        samples: list[list[np.ndarray]] = [[] for _ in range(self.num_stages)]
        for values, stage_ends in zip(self.observations, self.stage_ends_):
            start = 0
            for stage_idx, end in enumerate(stage_ends):
                samples[stage_idx].append(values[start : int(end) + 1])
                start = int(end) + 1
        return [np.concatenate(parts, axis=0) for parts in samples]

    def _kmeans_plus_plus(self, values: np.ndarray) -> np.ndarray:
        n_samples = len(values)
        means = np.empty((self.n_components, self.n_features), dtype=float)
        means[0] = values[int(self.rng.randint(n_samples))]
        closest_sq = np.sum((values - means[0]) ** 2, axis=1)
        for component_idx in range(1, self.n_components):
            total = float(np.sum(closest_sq))
            if not np.isfinite(total) or total <= 1e-12:
                selected_idx = int(self.rng.randint(n_samples))
            else:
                selected_idx = int(self.rng.choice(n_samples, p=closest_sq / total))
            means[component_idx] = values[selected_idx]
            distance_sq = np.sum((values - means[component_idx]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, distance_sq)
        return means

    def _initialize_stage_gmm(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(values) == 0:
            raise ValueError("Cannot initialize a stage GMM without observations.")
        means = self._kmeans_plus_plus(values)
        base_variance = np.var(values, axis=0) + self.reg_covar
        variances = np.broadcast_to(base_variance, means.shape).copy()
        weights = np.full(self.n_components, 1.0 / float(self.n_components), dtype=float)

        for _ in range(10):
            log_components = self._diag_component_loglik(values, weights, means, variances)
            log_norm = logsumexp(log_components, axis=1, keepdims=True)
            responsibilities = np.exp(log_components - log_norm)
            counts = np.sum(responsibilities, axis=0) + 1e-8
            weights = counts / np.sum(counts)
            means = responsibilities.T @ values / counts[:, None]
            diff = values[:, None, :] - means[None, :, :]
            variances = (
                np.sum(responsibilities[:, :, None] * diff * diff, axis=0)
                / counts[:, None]
                + self.reg_covar
            )
        return weights, means, variances

    def _initialize_from_segments(self) -> None:
        for stage_idx, values in enumerate(self._stage_samples()):
            weights, means, variances = self._initialize_stage_gmm(values)
            self.weights_[stage_idx] = weights
            self.means_[stage_idx] = means
            self.variances_[stage_idx] = variances

        for stage_idx in range(self.num_stages - 1):
            durations = []
            for stage_ends in self.stage_ends_:
                start = 0 if stage_idx == 0 else int(stage_ends[stage_idx - 1]) + 1
                durations.append(int(stage_ends[stage_idx]) - start + 1)
            mean_duration = max(float(np.mean(durations)), 1.0)
            p_next = float(np.clip(1.0 / mean_duration, 1e-4, 1.0 - 1e-4))
            self.transition_matrix_[stage_idx, stage_idx] = 1.0 - p_next
            self.transition_matrix_[stage_idx, stage_idx + 1] = p_next
        self.transition_matrix_[-1, -1] = 1.0

    @staticmethod
    def _diag_component_loglik(
        values: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
    ) -> np.ndarray:
        safe_variances = np.clip(np.asarray(variances, dtype=float), 1e-12, None)
        diff = values[:, None, :] - means[None, :, :]
        gaussian = -0.5 * (
            values.shape[1] * np.log(2.0 * np.pi)
            + np.sum(np.log(safe_variances)[None, :, :] + diff * diff / safe_variances[None, :, :], axis=2)
        )
        return gaussian + np.log(np.clip(weights, 1e-12, None))[None, :]

    def _emission_loglik(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        component_loglik = np.empty(
            (len(values), self.num_stages, self.n_components),
            dtype=float,
        )
        for stage_idx in range(self.num_stages):
            component_loglik[:, stage_idx, :] = self._diag_component_loglik(
                values,
                self.weights_[stage_idx],
                self.means_[stage_idx],
                self.variances_[stage_idx],
            )
        return logsumexp(component_loglik, axis=2), component_loglik

    def _log_transition_matrix(self) -> np.ndarray:
        log_transition = np.full_like(self.transition_matrix_, -np.inf, dtype=float)
        positive = self.transition_matrix_ > 0.0
        log_transition[positive] = np.log(self.transition_matrix_[positive])
        return log_transition

    def _forward_backward(
        self,
        emission_loglik: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        n_steps = int(emission_loglik.shape[0])
        log_transition = self._log_transition_matrix()
        alpha = np.full((n_steps, self.num_stages), -np.inf, dtype=float)
        alpha[0, 0] = float(emission_loglik[0, 0])
        for time_idx in range(1, n_steps):
            alpha[time_idx] = emission_loglik[time_idx] + logsumexp(
                alpha[time_idx - 1, :, None] + log_transition,
                axis=0,
            )

        loglik = float(alpha[-1, -1])
        if not np.isfinite(loglik):
            raise RuntimeError("GMM-HMM forward pass could not reach the final ordered stage.")

        beta = np.full((n_steps, self.num_stages), -np.inf, dtype=float)
        beta[-1, -1] = 0.0
        for time_idx in range(n_steps - 2, -1, -1):
            beta[time_idx] = logsumexp(
                log_transition
                + emission_loglik[time_idx + 1][None, :]
                + beta[time_idx + 1][None, :],
                axis=1,
            )

        gamma = np.exp(alpha + beta - loglik)
        gamma /= np.clip(np.sum(gamma, axis=1, keepdims=True), 1e-300, None)
        xi = np.zeros((max(n_steps - 1, 0), self.num_stages, self.num_stages), dtype=float)
        for time_idx in range(n_steps - 1):
            log_xi = (
                alpha[time_idx, :, None]
                + log_transition
                + emission_loglik[time_idx + 1][None, :]
                + beta[time_idx + 1][None, :]
                - loglik
            )
            xi[time_idx] = np.exp(log_xi)
        return gamma, xi, loglik

    def _expectation(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], float]:
        gammas: list[np.ndarray] = []
        xis: list[np.ndarray] = []
        mixture_responsibilities: list[np.ndarray] = []
        total_loglik = 0.0
        for values in self.observations:
            emission_loglik, component_loglik = self._emission_loglik(values)
            gamma, xi, loglik = self._forward_backward(emission_loglik)
            component_given_stage = np.exp(component_loglik - emission_loglik[:, :, None])
            gammas.append(gamma)
            xis.append(xi)
            mixture_responsibilities.append(gamma[:, :, None] * component_given_stage)
            total_loglik += float(loglik)
        return gammas, xis, mixture_responsibilities, float(total_loglik)

    def _maximization(
        self,
        xis: list[np.ndarray],
        mixture_responsibilities: list[np.ndarray],
    ) -> None:
        transition_counts = np.sum(
            [np.sum(xi, axis=0) for xi in xis],
            axis=0,
        )
        for stage_idx in range(self.num_stages - 1):
            self_count = float(transition_counts[stage_idx, stage_idx]) + 1e-3
            next_count = float(transition_counts[stage_idx, stage_idx + 1]) + 1e-3
            total = self_count + next_count
            self.transition_matrix_[stage_idx, :] = 0.0
            self.transition_matrix_[stage_idx, stage_idx] = self_count / total
            self.transition_matrix_[stage_idx, stage_idx + 1] = next_count / total
        self.transition_matrix_[-1, :] = 0.0
        self.transition_matrix_[-1, -1] = 1.0

        counts = np.zeros((self.num_stages, self.n_components), dtype=float)
        first_moments = np.zeros_like(self.means_)
        for values, responsibilities in zip(self.observations, mixture_responsibilities):
            counts += np.sum(responsibilities, axis=0)
            first_moments += np.einsum("tkm,td->kmd", responsibilities, values)
        safe_counts = np.clip(counts, 1e-10, None)
        new_means = first_moments / safe_counts[:, :, None]

        second_moments = np.zeros_like(self.variances_)
        for values, responsibilities in zip(self.observations, mixture_responsibilities):
            diff = values[:, None, None, :] - new_means[None, :, :, :]
            second_moments += np.sum(
                responsibilities[:, :, :, None] * diff * diff,
                axis=0,
            )

        state_counts = np.sum(counts, axis=1, keepdims=True)
        self.weights_ = counts / np.clip(state_counts, 1e-10, None)
        self.weights_ = np.clip(self.weights_, 1e-8, None)
        self.weights_ /= np.sum(self.weights_, axis=1, keepdims=True)
        self.means_ = new_means
        self.variances_ = second_moments / safe_counts[:, :, None] + self.reg_covar

    def _viterbi(self, values: np.ndarray) -> tuple[np.ndarray, float]:
        emission_loglik, _ = self._emission_loglik(values)
        log_transition = self._log_transition_matrix()
        n_steps = len(values)
        scores = np.full((n_steps, self.num_stages), -np.inf, dtype=float)
        backpointers = np.full((n_steps, self.num_stages), -1, dtype=int)
        scores[0, 0] = emission_loglik[0, 0]
        for time_idx in range(1, n_steps):
            candidates = scores[time_idx - 1, :, None] + log_transition
            backpointers[time_idx] = np.argmax(candidates, axis=0)
            scores[time_idx] = emission_loglik[time_idx] + np.max(candidates, axis=0)

        labels = np.empty(n_steps, dtype=int)
        labels[-1] = self.num_stages - 1
        for time_idx in range(n_steps - 1, 0, -1):
            labels[time_idx - 1] = backpointers[time_idx, labels[time_idx]]
        if labels[0] != 0 or np.any(np.diff(labels) < 0) or len(np.unique(labels)) != self.num_stages:
            raise RuntimeError("GMM-HMM Viterbi path violated the ordered-stage topology.")
        return labels, float(scores[-1, -1])

    def fit(
        self,
        *,
        max_iter: int = 30,
        tol: float = 1e-4,
        verbose: bool = True,
        true_cutpoints: list[np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
        previous_loglik: float | None = None
        gammas: list[np.ndarray] = []
        for iteration in range(int(max_iter)):
            gammas, xis, mixture_responsibilities, _ = self._expectation()
            self._maximization(xis, mixture_responsibilities)
            gammas, _, _, total_loglik = self._expectation()
            self.loss_loglik.append(float(total_loglik))

            labels_and_scores = [self._viterbi(values) for values in self.observations]
            labels = [item[0] for item in labels_and_scores]
            self.stage_ends_ = [
                np.where(np.diff(label) != 0)[0].astype(int).tolist() + [len(label) - 1]
                for label in labels
            ]
            self.segmentation_history_.append([list(ends) for ends in self.stage_ends_])

            metrics = compute_cutpoint_metrics(
                [ends[:-1] for ends in self.stage_ends_],
                true_cutpoints,
                self.observations,
            ) if true_cutpoints is not None else {}
            improvement = np.inf if previous_loglik is None else total_loglik - previous_loglik
            converged = previous_loglik is not None and abs(improvement) <= float(tol) * (1.0 + abs(previous_loglik))
            should_log = converged or ((iteration + 1) % 10 == 0) or iteration == int(max_iter) - 1
            if verbose and should_log:
                print(
                    format_training_log(
                        "GMM-HMM",
                        iteration,
                        losses={"loglik": total_loglik},
                        metrics=metrics,
                    )
                )
            if converged:
                self.converged_ = True
                self.converged_iter_ = int(iteration)
                break
            previous_loglik = float(total_loglik)

        final_gammas, _, _, _ = self._expectation()
        final_labels = [self._viterbi(values)[0] for values in self.observations]
        self.stage_ends_ = [
            np.where(np.diff(label) != 0)[0].astype(int).tolist() + [len(label) - 1]
            for label in final_labels
        ]
        history: dict[str, Any] = {
            "loglik": list(self.loss_loglik),
            "stage_ends": list(self.segmentation_history_),
        }
        if self.converged_iter_ is not None:
            history["converged_iter"] = int(self.converged_iter_)
        return final_labels, final_gammas, history


def segment_with_gmm_hmm(
    demos: list[np.ndarray],
    *,
    env,
    true_cutpoints=None,
    n_stages: int,
    n_components: int = 3,
    reg_covar: float = 1e-3,
    max_iter: int = 30,
    tol: float = 1e-4,
    seed: int = 0,
    use_state: bool = True,
    use_velocity: bool = False,
    velocity_weight: float = 1.0,
    use_env_features: bool = True,
    selected_raw_feature_ids=None,
    standardize: bool = True,
    init_mode: str = "random_stage_ends",
    min_len: int = 3,
    verbose: bool = True,
) -> tuple[list[np.ndarray], DiagonalGMMHMM, dict[str, Any], list[np.ndarray]]:
    if not use_state and not use_env_features:
        raise ValueError("GMM-HMM requires state observations, environment features, or both.")
    observations = _build_observations(
        demos,
        env=env,
        use_state=bool(use_state),
        use_velocity=bool(use_velocity),
        velocity_weight=float(velocity_weight),
        use_env_features=bool(use_env_features),
        selected_raw_feature_ids=selected_raw_feature_ids,
        standardize=bool(standardize),
    )
    stage_ends_init = _resolve_stage_ends_init(
        demos,
        num_stages=int(n_stages),
        min_duration=int(min_len),
        tau_init=None,
        tau_init_mode=init_mode,
        env=env,
        seed=int(seed),
        use_velocity=bool(use_velocity and use_state),
        vel_weight=float(velocity_weight),
        standardize=bool(standardize),
        use_env_features=bool(use_env_features),
        selected_raw_feature_ids=selected_raw_feature_ids,
    )
    model = DiagonalGMMHMM(
        observations,
        n_stages=int(n_stages),
        n_components=int(n_components),
        reg_covar=float(reg_covar),
        seed=int(seed),
        stage_ends_init=stage_ends_init,
    )
    labels, gammas, history = model.fit(
        max_iter=int(max_iter),
        tol=float(tol),
        verbose=bool(verbose),
        true_cutpoints=true_cutpoints,
    )
    return labels, model, history, gammas
