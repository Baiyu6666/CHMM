# methods/changepoint_backend.py
import numpy as np

def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return s if keepdims else np.squeeze(s, axis=axis)

def make_edge_features(X, use_velocity=True, vel_weight=1.0, standardize=True):
    X = np.asarray(X, float)
    pos = X[:-1]
    if use_velocity:
        vel = X[1:] - X[:-1]
        Y = np.hstack([pos, vel_weight*vel])
    else:
        Y = pos.copy()
    if standardize:
        mu = Y.mean(axis=0, keepdims=True)
        std= Y.std(axis=0, keepdims=True)+1e-8
        Y = (Y-mu)/std
    return Y

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

def segment_fixed_K_CP(X, K=2, use_velocity=True, vel_weight=1.0, standardize=True, cost_type="gaussian", min_len=1):
    X=np.asarray(X,float); N_edges=X.shape[0]-1
    Y=make_edge_features(X,use_velocity,vel_weight,standardize); N,D=Y.shape
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
    cps=[]; k,t=K,N
    while k>0:
        s=prv[k,t]; cps.append(t); t=s; k-=1
    cps.reverse()
    z=np.zeros(N,int); start=1
    for seg_id, t_end in enumerate(cps):
        z[start-1:t_end]=seg_id
        start=t_end+1
    z_time=np.empty(N+1,int); z_time[:-1]=z; z_time[-1]=z[-1]
    return cps, z_time

# helper
from utils.subgoals import compute_per_demo_lastpoint_subgoals, average_subgoals_from_per_demo, take_first2_for_plot, take_first2_array

def segment_changepoint(
    X_full,
    K=2, average_subgoal=False,
    seg_dims=None, cl_dims=None,
    **kwargs
):
    X_seg = [ (x if seg_dims is None else x[:, seg_dims]) for x in X_full ]
    Z=[]
    for x in X_seg:
        _, z = segment_fixed_K_CP(x, K=K, **kwargs)
        Z.append(z)
    per_demo_vec, K_global = compute_per_demo_lastpoint_subgoals(X_full, Z, cl_dims=cl_dims)
    avg_vec = average_subgoals_from_per_demo(per_demo_vec, K_target=K_global) if average_subgoal else None
    per_demo_xy = take_first2_for_plot(per_demo_vec)
    avg_xy = take_first2_array(avg_vec)
    return Z, per_demo_vec, avg_vec, per_demo_xy, avg_xy
