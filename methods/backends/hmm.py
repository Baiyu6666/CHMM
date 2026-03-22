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
    p_A  = _count_free_trans_params(K, left_right)
    p_W  = K * (D * (1 + D))
    p_var= K * D
    return p_pi + p_A + p_W + p_var

def _total_loglik_ar(model, X_list):
    """English documentation omitted during cleanup."""
    total = 0.0
    for x in X_list:
        _, _, logZ = model._fb(x)
        total += float(logZ)
    return total

# English comment omitted during cleanup.
class ARHMM:
    def __init__(self, n_states, n_dims, sticky=0.0, rng=None, min_covar=1e-4, ridge=1e-6, left_right: bool = LEFT_RIGHT_DEFAULT):
        self.K=n_states; self.D=n_dims; self.rng=rng or np.random.RandomState(0)
        self.sticky=sticky; self.min_covar=min_covar; self.ridge=ridge
        self.left_right = bool(left_right)

        self.pi = np.ones(self.K)/self.K
        # English comment omitted during cleanup.
        if self.left_right:
            A = np.zeros((self.K, self.K))
            for i in range(self.K):
                A[i, i] = 0.9
                if i + 1 < self.K:
                    A[i, i + 1] = 0.1
            self.A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        else:
            A = 0.1*self.rng.rand(self.K,self.K)+0.9*np.eye(self.K)
            self.A = A / A.sum(axis=1, keepdims=True)

        self.W = np.zeros((self.K, 1+self.D, self.D))
        self.var = np.ones((self.K,self.D))
        self.mu = self.rng.randn(self.K,self.D)

    def _warm_start_from_taus(self, X, tau_init):
        if self.K != 2:
            raise ValueError("tau-based warm start for ARHMM currently requires n_states=2.")
        tau_init = np.asarray(tau_init, dtype=int)
        labels = []
        for x, tau in zip(X, tau_init):
            t = clip_tau_for_sequence(x, tau)
            z = np.zeros(len(x), dtype=int)
            z[t + 1 :] = 1
            labels.append(z)

        pi_counts = np.zeros(self.K)
        A_counts = np.zeros((self.K, self.K))
        sum_x = np.zeros((self.K, self.D))
        Nk = np.zeros(self.K)
        XtX = [np.zeros((1 + self.D, 1 + self.D)) for _ in range(self.K)]
        XtY = [np.zeros((1 + self.D, self.D)) for _ in range(self.K)]

        for x, z in zip(X, labels):
            pi_counts[z[0]] += 1
            for t in range(len(z) - 1):
                A_counts[z[t], z[t + 1]] += 1
            for k in range(self.K):
                pts = x[z == k]
                if len(pts) > 0:
                    sum_x[k] += pts.sum(axis=0)
                    Nk[k] += len(pts)
            for t in range(1, len(x)):
                k = int(z[t])
                y = np.hstack([1.0, x[t - 1]])
                XtX[k] += np.outer(y, y)
                XtY[k] += np.outer(y, x[t])

        self.pi = np.maximum(pi_counts, 1e-12)
        self.pi /= np.maximum(self.pi.sum(), 1e-12)

        Acounts = A_counts + self.sticky * np.eye(self.K) + 1e-9
        if self.left_right:
            mask = _left_right_mask(self.K)
            Acounts *= mask
            Acounts += 1e-12 * mask
        self.A = Acounts / np.maximum(Acounts.sum(axis=1, keepdims=True), 1e-12)

        for k in range(self.K):
            if Nk[k] > 0:
                self.mu[k] = sum_x[k] / Nk[k]
            XtXk = XtX[k] + self.ridge * np.eye(1 + self.D)
            if np.count_nonzero(XtX[k]) == 0:
                self.W[k] = 0.0
                continue
            try:
                self.W[k] = np.linalg.solve(XtXk, XtY[k])
            except np.linalg.LinAlgError:
                self.W[k] = np.linalg.lstsq(XtXk, XtY[k], rcond=None)[0]

        res_ss = np.zeros((self.K, self.D))
        res_Nk = np.zeros(self.K)
        for x, z in zip(X, labels):
            for t in range(1, len(x)):
                k = int(z[t])
                y = np.hstack([1.0, x[t - 1]])
                pred = y @ self.W[k]
                diff = x[t] - pred
                res_ss[k] += diff ** 2
                res_Nk[k] += 1.0
        for k in range(self.K):
            if res_Nk[k] > 0:
                self.var[k] = np.clip(res_ss[k] / res_Nk[k], self.min_covar, None)
            else:
                self.var[k] = np.ones(self.D)

    def _log_emiss(self, x):
        T=x.shape[0]; K=self.K; D=self.D
        logB=np.zeros((T,K))
        var0=np.clip(self.var*4.0, self.min_covar, None)
        for k in range(K):
            diff0 = x[0]-self.mu[k]
            term0 = -0.5*np.sum((diff0**2)/var0[k])
            norm0 = -0.5*np.sum(np.log(2*np.pi*var0[k]))
            logB[0,k]=norm0+term0
        for t in range(1,T):
            xtm1=x[t-1]; y=np.hstack([1.0, xtm1])
            for k in range(K):
                mean = y @ self.W[k]
                diff = x[t]-mean
                vark=np.clip(self.var[k], self.min_covar, None)
                term=-0.5*np.sum((diff**2)/vark); norm=-0.5*np.sum(np.log(2*np.pi*vark))
                logB[t,k]=norm+term
        return logB

    def _fb(self, x):
        T=x.shape[0]; logB=self._log_emiss(x)
        logA=np.log(self.A+1e-12); logpi=np.log(self.pi+1e-12)
        alpha=np.zeros((T,self.K)); beta=np.zeros((T,self.K))
        alpha[0]=logpi+logB[0]
        for t in range(1,T): alpha[t]=logB[t]+logsumexp(alpha[t-1][:,None]+logA,axis=0)
        for t in range(T-2,-1,-1): beta[t]=logsumexp(logA+(logB[t+1]+beta[t+1])[None,:],axis=1)
        logZ=logsumexp(alpha[-1],axis=0)
        gamma=np.exp(alpha+beta-logZ)
        xi=np.zeros((T-1,self.K,self.K))
        for t in range(T-1):
            m=alpha[t][:,None]+logA+logB[t+1][None,:]+beta[t+1][None,:]
            xi[t]=np.exp(m-logsumexp(m,axis=(0,1)))
        return gamma, xi, logZ

    def fit(self, X, n_iter=30, verbose=True, true_taus=None, tau_init=None):
        K,D=self.K,self.D
        hist_loglik = []
        hist_metrics = {"MeanAbsCutpointError": [], "CutpointExactMatchRate": []}
        if tau_init is not None:
            self._warm_start_from_taus(X, tau_init)
        for it in range(n_iter):
            sum_pi=np.zeros(K); sum_A=np.zeros((K,K))
            Nk=np.zeros(K); sum_x=np.zeros((K,D)); sum_x2=np.zeros((K,D))
            XtX=[np.zeros((1+D,1+D)) for _ in range(K)]
            XtY=[np.zeros((1+D,D))   for _ in range(K)]
            res_ss=np.zeros((K,D)); res_Nk=np.zeros(K)
            total=0.0
            for x in X:
                gamma,xi,logZ=self._fb(x); total+=logZ
                sum_pi+=gamma[0]; sum_A+=np.sum(xi,axis=0)
                Nk+=np.sum(gamma,axis=0); sum_x+=gamma.T@x; sum_x2+=gamma.T@(x**2)
                Tseq=x.shape[0]
                for t in range(1,Tseq):
                    y=np.hstack([1.0, x[t-1]]); Y=x[t]; g=gamma[t]
                    yyT=np.outer(y,y); yY=np.outer(y,Y)
                    for k in range(K):
                        w=g[k];
                        if w<=1e-12: continue
                        XtX[k]+=w*yyT; XtY[k]+=w*yY
            self.pi=np.maximum(sum_pi,1e-12); self.pi/=np.maximum(self.pi.sum(),1e-12)
            Acounts=sum_A + self.sticky*np.eye(K) + 1e-9
            if self.left_right:
                mask = _left_right_mask(K)
                Acounts *= mask
                Acounts += 1e-12 * mask
            self.A=Acounts/ np.maximum(Acounts.sum(axis=1,keepdims=True),1e-12)

            for k in range(K):
                XtXk=XtX[k]+self.ridge*np.eye(1+D)
                try: self.W[k]=np.linalg.solve(XtXk, XtY[k])
                except np.linalg.LinAlgError: self.W[k]=np.linalg.lstsq(XtXk, XtY[k], rcond=None)[0]
            for x in X:
                gamma,_,_=self._fb(x); Tseq=x.shape[0]
                for t in range(1,Tseq):
                    y=np.hstack([1.0,x[t-1]])
                    pred=np.stack([y@self.W[k] for k in range(K)],axis=0)
                    diff=x[t][None,:]-pred; g=gamma[t][:,None]
                    res_ss+=g*(diff**2); res_Nk+=gamma[t]
            for k in range(K):
                denom=max(res_Nk[k],1e-9)
                self.var[k]=np.clip(res_ss[k]/denom, 1e-6, None)
            self.mu = sum_x/(Nk[:,None]+1e-12)
            avg_ll = total / len(X)
            hist_loglik.append(float(avg_ll))
            iter_metrics = {}
            if true_taus is not None:
                Z_cur = [self.viterbi(x) for x in X]
                cutpoints_cur = [np.where(np.diff(z.astype(int)) != 0)[0].astype(int) for z in Z_cur]
                if self.K == 2:
                    true_cutpoints = [None if t is None else np.asarray([int(t)], dtype=int) for t in true_taus]
                else:
                    true_cutpoints = [None if t is None else np.asarray(t, dtype=int).reshape(-1) for t in true_taus]
                iter_metrics = compute_cutpoint_metrics(cutpoints_cur, true_cutpoints, X)
                for name in hist_metrics:
                    if name in iter_metrics:
                        hist_metrics[name].append(iter_metrics.get(name, np.nan))
            should_log = ((it + 1) % 10 == 0) or (it == n_iter - 1)
            if verbose and should_log:
                print(format_training_log("ARHMM", it, losses={"loss": avg_ll}, metrics=iter_metrics))
        Z=[self.viterbi(x) for x in X]
        hist = {"loglik": hist_loglik}
        nonempty_metrics = {k: v for k, v in hist_metrics.items() if v}
        if nonempty_metrics:
            hist.update(nonempty_metrics)
        return Z, hist

    def viterbi(self, x):
        T=x.shape[0]; logB=self._log_emiss(x)
        logA=np.log(self.A+1e-12); logpi=np.log(self.pi+1e-12)
        dp=np.zeros((T,self.K)); ptr=np.zeros((T,self.K),int)
        dp[0]=logpi+logB[0]
        for t in range(1,T):
            scores=dp[t-1][:,None]+logA
            ptr[t]=np.argmax(scores,axis=0)
            dp[t]=logB[t]+np.max(scores,axis=0)
        z=np.zeros(T,int); z[-1]=np.argmax(dp[-1])
        for t in range(T-2,-1,-1): z[t]=ptr[t+1,z[t+1]]
        return z

def segment_with_hmm(
    X_full,
    env=None,
    true_taus=None,
    method="ar",
    n_states=2,
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
        mdl = ARHMM(n_states=K_try, n_dims=D, sticky=sticky, rng=_rng, left_right=left_right)
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
    if isinstance(n_states, (list, tuple, range)):
        K_cands = list(n_states)
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
        mdl, Z_states_list, seg_hist = _fit_for_K(int(n_states))
        chosen_model = mdl

    Z_list = [np.asarray(z, dtype=int) for z in Z_states_list]
    return Z_list, chosen_model, seg_hist
