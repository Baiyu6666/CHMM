# methods/changepoint_backend.py
import numpy as np

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

def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return s if keepdims else np.squeeze(s, axis=axis)

def make_edge_features(
    X,
    use_velocity=True,
    vel_weight=1.0,
    standardize=True,
    env=None,
    use_env_features=True,
    selected_raw_feature_ids=None,
    standardize_stats=None,
):
    X = np.asarray(X, float)
    pos = X[:-1]
    if use_velocity:
        vel = X[1:] - X[:-1]
        Y = np.hstack([pos, vel_weight*vel])
    else:
        Y = pos.copy()
    if use_env_features:
        if env is None:
            raise ValueError("env must be provided when use_env_features=True.")
        feature_cols = _resolve_selected_feature_columns(env, selected_raw_feature_ids)
        F = np.asarray(env.compute_all_features_matrix(X), float)[:-1, feature_cols]
        Y = np.hstack([Y, F])
    if standardize:
        if standardize_stats is None:
            mu = Y.mean(axis=0, keepdims=True)
            std= Y.std(axis=0, keepdims=True)+1e-8
        else:
            mu, std = standardize_stats
        Y = (Y-mu)/std
    return Y


def compute_dataset_edge_standardization(
    X_list,
    use_velocity=True,
    vel_weight=1.0,
    env=None,
    use_env_features=True,
    selected_raw_feature_ids=None,
):
    Ys = []
    for X in X_list:
        Y = make_edge_features(
            X,
            use_velocity=use_velocity,
            vel_weight=vel_weight,
            standardize=False,
            env=env,
            use_env_features=use_env_features,
            selected_raw_feature_ids=selected_raw_feature_ids,
        )
        Ys.append(Y)
    Y_all = np.concatenate(Ys, axis=0)
    mu = Y_all.mean(axis=0, keepdims=True)
    std = Y_all.std(axis=0, keepdims=True) + 1e-8
    return mu, std

def _prefix_sums_edges(Y):
    Y = np.asarray(Y, float)
    N,D = Y.shape
    S1 = np.zeros((N+1, D)); S2=np.zeros(N+1)
    S1[1:] = np.cumsum(Y, axis=0)
    S2[1:] = np.cumsum(np.sum(Y*Y, axis=1))
    return S1,S2

def _sse(S1,S2,l,r):
    n=r-l+1; sumy=S1[r]-S1[l-1]; sumyy=S2[r]-S2[l-1]
    mean=sumy/max(n,1)
    sse=float(sumyy - n*np.sum(mean*mean))
    return max(sse,0.0)

def _gauss_ml(S1,S2,l,r,D,eps=1e-8):
    n=r-l+1
    if n<=1: return 0.0
    sse=_sse(S1,S2,l,r)
    return 0.5*n*D*np.log((sse/max(n*D,1))+eps)

def segment_fixed_K_CP(
    X,
    K=2,
    use_velocity=True,
    vel_weight=1.0,
    standardize=True,
    cost_type="gaussian",
    min_len=1,
    env=None,
    use_env_features=True,
    selected_raw_feature_ids=None,
    standardize_stats=None,
):
    X=np.asarray(X,float); N_edges=X.shape[0]-1
    Y=make_edge_features(
        X,
        use_velocity,
        vel_weight,
        standardize,
        env=env,
        use_env_features=use_env_features,
        selected_raw_feature_ids=selected_raw_feature_ids,
        standardize_stats=standardize_stats,
    ); N,D=Y.shape
    S1,S2=_prefix_sums_edges(Y)
    C = (lambda l,r:_sse(S1,S2,l,r)) if cost_type=="sse" else (lambda l,r:_gauss_ml(S1,S2,l,r,D))
    dp=np.full((K+1, N+1), np.inf); prv=np.full((K+1,N+1), -1, int); dp[0,0]=0.0
    for k in range(1,K+1):
        R={k-1}
        for t in range(k, N+1):
            best=np.inf; arg=-1
            for s in list(R):
                if t-s < min_len: continue
                val=dp[k-1,s] + C(s+1,t)
                if val<best: best=val; arg=s
            dp[k,t]=best; prv[k,t]=arg
            R={s for s in R if dp[k-1,s]+C(s+1,t)<=dp[k,t]}
            R.add(t)
    edge_endpoints=[]; k,t=K,N
    while k>0:
        s=prv[k,t]; edge_endpoints.append(t); t=s; k-=1
    edge_endpoints.reverse()
    z=np.zeros(N,int); start=1
    for seg_id, t_end in enumerate(edge_endpoints):
        z[start-1:t_end]=seg_id
        start=t_end+1
    z_time=np.empty(N+1,int); z_time[:-1]=z; z_time[-1]=z[-1]
    taus = [int(t_end - 1) for t_end in edge_endpoints[:-1]]
    return taus, z_time

def segment_changepoint(
    X_full,
    env=None,
    K=2,
    seg_dims=None,
    **kwargs
):
    kwargs = dict(kwargs)
    standardize = bool(kwargs.get("standardize", True))
    standardize_stats = None
    if standardize:
        standardize_stats = compute_dataset_edge_standardization(
            [(x if seg_dims is None else x[:, seg_dims]) for x in X_full],
            use_velocity=kwargs.get("use_velocity", True),
            vel_weight=kwargs.get("vel_weight", 1.0),
            env=env,
            use_env_features=kwargs.get("use_env_features", True),
            selected_raw_feature_ids=kwargs.get("selected_raw_feature_ids"),
        )
    kwargs["standardize_stats"] = standardize_stats

    X_seg = [ (x if seg_dims is None else x[:, seg_dims]) for x in X_full ]
    Z=[]
    for x in X_seg:
        _, z = segment_fixed_K_CP(x, K=K, env=env, **kwargs)
        Z.append(z)
    return Z
