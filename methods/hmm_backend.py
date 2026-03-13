# methods/hmm_backend.py
from __future__ import annotations

import numpy as np

# ---------- module-level switch for left-to-right constraint ----------
LEFT_RIGHT_DEFAULT = True  # 默认开启左→右（Bakis）结构

def _left_right_mask(K: int):
    """Bakis：只允许 i->i 或 i->i+1。最后一行只允许自环。"""
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

def _make_point_features_list(X_list, use_velocity=False, vel_weight=1.0, standardize=False):
    """
    把每条序列 x(t) 变成按时刻的特征：
      - 若 use_velocity=False:  y[t] = x[t]
      - 若 use_velocity=True:   y[t] = [ x[t], vel_weight * (x[t]-x[t-1]) ]，其中 v[0]=0
    如果 standardize=True，就在“所有序列拼接后的整体”上做标准化（再分发回每条序列）。
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
    if standardize:
        allY = np.vstack(Ys)
        mu = allY.mean(axis=0, keepdims=True)
        std = allY.std(axis=0, keepdims=True) + 1e-8
        Ys = [ (y - mu)/std for y in Ys ]
    return Ys

# ---------- BIC helpers ----------
def _count_free_trans_params(K: int, left_right: bool) -> int:
    """转移矩阵自由参数数（不含初始分布）。"""
    if not left_right:
        # 每行有 K-1 个自由度（行和=1）
        return K * (K - 1)
    # 左→右：第 i 行只允许 i->i 和 i->i+1（最后一行只有自环）
    # 每行自由度 = (允许的出边数 - 1)
    free = 0
    for i in range(K):
        allowed = 1 + (1 if i + 1 < K else 0)
        free += max(allowed - 1, 0)
    return free

def _num_params_standard(K: int, D: int, left_right: bool) -> int:
    """
    StandardHMM(对角方差) 参数量：
      - 初始分布 pi: (K-1) 自由度
      - 转移 A: _count_free_trans_params
      - 发射：每状态均值 D + 方差 D  => K * (2D)
    """
    p_pi = K - 1
    p_A  = _count_free_trans_params(K, left_right)
    p_em = K * (2 * D)
    return p_pi + p_A + p_em

def _num_params_ar(K: int, D: int, left_right: bool) -> int:
    """
    ARHMM(对角噪声) 参数量：
      - 初始分布 pi: (K-1)
      - 转移 A: _count_free_trans_params
      - 线性动力学：每状态 W 维度 (1+D) x D  => K * (D*(1+D))
      - 噪声方差：每状态 D                  => K * D
    """
    p_pi = K - 1
    p_A  = _count_free_trans_params(K, left_right)
    p_W  = K * (D * (1 + D))
    p_var= K * D
    return p_pi + p_A + p_W + p_var

def _total_loglik_standard(model, X_list):
    """用当前参数重新计算总对数似然（StandardHMM）。"""
    total = 0.0
    for x in X_list:
        _, _, logZ = model._fb(x)
        total += float(logZ)
    return total

def _total_loglik_ar(model, X_list):
    """用当前参数重新计算总对数似然（ARHMM）。"""
    total = 0.0
    for x in X_list:
        _, _, logZ = model._fb(x)
        total += float(logZ)
    return total


# ---------- Standard Gaussian HMM ----------
class StandardHMM:
    def __init__(self, n_states, n_dims, sticky=10.0, rng=None, min_covar=1e-3, left_right: bool = LEFT_RIGHT_DEFAULT):
        self.K = n_states; self.D = n_dims
        self.rng = rng or np.random.RandomState(0)
        self.min_covar = min_covar
        self.pi = np.ones(self.K)/self.K
        self.sticky = sticky
        self.left_right = bool(left_right)

        # 转移矩阵初始化
        if self.left_right:
            A = np.zeros((self.K, self.K))
            for i in range(self.K):
                A[i, i] = 0.9
                if i + 1 < self.K:
                    A[i, i + 1] = 0.1
            self.A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        else:
            A = 0.1*self.rng.rand(self.K, self.K) + 0.9*np.eye(self.K)
            self.A = A/ A.sum(axis=1, keepdims=True)

        self.mu  = self.rng.randn(self.K, self.D)*0.1
        self.var = np.ones((self.K, self.D))*0.5

    def _log_gauss_diag(self, x):
        T = x.shape[0]; logB = np.zeros((T, self.K))
        for k in range(self.K):
            diff = x - self.mu[k]
            v = np.clip(self.var[k], self.min_covar, None)
            term = -0.5*np.sum((diff**2)/v, axis=1)
            norm = -0.5*np.sum(np.log(2*np.pi*v))
            logB[:,k] = norm + term
        return logB

    def _fb(self, x):
        T = x.shape[0]
        logB = self._log_gauss_diag(x)
        logA = np.log(self.A + 1e-12)
        logpi= np.log(self.pi + 1e-12)
        alpha=np.zeros((T,self.K)); beta=np.zeros((T,self.K))
        alpha[0]=logpi+logB[0]
        for t in range(1,T):
            alpha[t] = logB[t] + logsumexp(alpha[t-1][:,None] + logA, axis=0)
        for t in range(T-2,-1,-1):
            beta[t] = logsumexp(logA + (logB[t+1]+beta[t+1])[None,:], axis=1)
        logZ = logsumexp(alpha[-1], axis=0)
        gamma = np.exp(alpha + beta - logZ)
        xi = np.zeros((T-1,self.K,self.K))
        for t in range(T-1):
            m = alpha[t][:,None]+logA+logB[t+1][None,:]+beta[t+1][None,:]
            xi[t] = np.exp(m - logsumexp(m, axis=(0,1)))
        return gamma, xi, logZ

    def fit(self, X, n_iter=30, verbose=True):
        hist_loglik = []
        for it in range(n_iter):
            sum_pi = np.zeros(self.K);
            sum_A = np.zeros((self.K, self.K))
            Nk = np.zeros(self.K);
            sum_x = np.zeros((self.K, self.D));
            sum_x2 = np.zeros((self.K, self.D))
            total_logZ = 0.0
            for x in X:
                gamma, xi, logZ = self._fb(x)
                total_logZ += logZ
                sum_pi += gamma[0];
                sum_A += np.sum(xi, axis=0)
                Nk += np.sum(gamma, axis=0)
                sum_x += gamma.T @ x
                sum_x2 += gamma.T @ (x ** 2)

            # M-step
            self.pi = np.maximum(sum_pi, 1e-12);
            self.pi /= np.maximum(self.pi.sum(), 1e-12)

            Acounts = sum_A + self.sticky * np.eye(self.K) + 1e-9
            if self.left_right:
                mask = _left_right_mask(self.K)
                Acounts *= mask
                Acounts += 1e-12 * mask  # 防止某行全 0
            self.A = Acounts / np.maximum(Acounts.sum(axis=1, keepdims=True), 1e-12)

            self.mu = sum_x / (Nk[:, None] + 1e-12)
            var = (sum_x2 / (Nk[:, None] + 1e-12)) - (self.mu ** 2)
            self.var = np.clip(var, self.min_covar, None)

            avg_ll = total_logZ / len(X)
            hist_loglik.append(float(avg_ll))
            if verbose and ((it + 1) % 5 == 0 or it == 0):
                print(f"[HMM] iter {it + 1:2d} | avg log-lik={avg_ll:.3f}")

        Z = [self.viterbi(x) for x in X]
        return Z, {"loglik": hist_loglik}

    def viterbi(self, x):
        T=x.shape[0]; logB=self._log_gauss_diag(x)
        logA=np.log(self.A+1e-12); logpi=np.log(self.pi+1e-12)
        dp=np.zeros((T,self.K)); ptr=np.zeros((T,self.K),int)
        dp[0]=logpi+logB[0]
        for t in range(1,T):
            scores = dp[t-1][:,None]+logA
            ptr[t]=np.argmax(scores, axis=0)
            dp[t]=logB[t]+np.max(scores, axis=0)
        z=np.zeros(T,int); z[-1]=np.argmax(dp[-1])
        for t in range(T-2,-1,-1):
            z[t]=ptr[t+1,z[t+1]]
        return z

# ---------- Sticky HDP-HMM（轻量实现，保守 birth） ----------
class StickyHDPHMM:
    def __init__(self, alpha=0.2, gamma=0.1, kappa=30.0, sigma=0.4, seed=42, left_right: bool = LEFT_RIGHT_DEFAULT):
        self.alpha=alpha; self.gamma=gamma; self.kappa=kappa; self.sigma=sigma
        self.rng=np.random.RandomState(seed)
        self.K=0; self.D=None
        self.pi=None; self.A=None; self.mu=None; self.var=None
        self.means=[]; self.counts={}
        self.left_right = bool(left_right)

    def _lik(self, x, m): return np.exp(-0.5*np.sum((x-m)**2)/max(self.sigma**2,1e-9))
    def _p0(self, x):
        s0=3.0; D=x.shape[0]; var=(s0**2+self.sigma**2)
        return np.exp(-0.5*np.dot(x,x)/var)/((2*np.pi*var)**(D/2))

    def _sample_state(self, prev_k, xt, max_new=2, new_tau=3.0, noise=0.05):
        """
        采样下一个状态。
        若启用左→右：只允许 prev->prev 或 prev->prev+1；且仅在 prev+1==K 时允许“新状态”（新生在末尾）。
        """
        K=self.K
        probs=[]; best_old=0.0

        # 候选集合
        if self.left_right:
            candidates = [prev_k]
            if prev_k + 1 < K:
                candidates.append(prev_k + 1)
        else:
            candidates = list(range(K))

        denom = (sum(self.counts.get((prev_k,j),0) for j in range(K)) + self.alpha + self.kappa + 1e-9)

        for k in candidates:
            trans = (self.counts.get((prev_k,k),0) + (self.kappa if prev_k==k else 0.0) + self.alpha/max(K,1)) / denom
            emit = self._lik(xt, self.means[k])
            p = trans*emit; probs.append((k,p)); best_old=max(best_old,p)

        # 新状态
        allow_new = False
        if self.left_right:
            # 只允许在“末尾”出生，且相当于 prev -> prev+1 == K
            allow_new = (prev_k + 1 == K)
        else:
            allow_new = True

        p_new = 0.0
        if allow_new:
            p_new = (self.alpha/max(K,1)) / denom * self._p0(xt)

        # 阈值控制
        if p_new <= new_tau * (best_old + 1e-12):
            allow_new = False
            p_new = 0.0

        # 归一化并采样
        vals = [p for _,p in probs]
        if allow_new: vals.append(p_new)
        s = sum(vals)
        if not np.isfinite(s) or s <= 0:
            # 均匀退化
            if allow_new:
                probs_norm = [1.0/(len(probs)+1)]*(len(probs)+1)
            else:
                probs_norm = [1.0/len(probs)]*len(probs)
        else:
            probs_norm = [p/s for p in vals]

        choice_idx = self.rng.choice(len(probs_norm), p=probs_norm)
        if allow_new and choice_idx == len(probs):
            # 出生在末尾：prev->K（新状态）
            self.means.append(xt + noise*self.rng.randn(*xt.shape))
            self.K += 1;
            return self.K - 1
        else:
            return probs[choice_idx][0]

    # --- 在 StickyHDPHMM.fit 中加入历史记录 ---
    def fit(self, X, n_iter=30, verbose=True):
        self.D = X[0].shape[1];
        self.K = 1
        self.means = [np.mean(np.vstack(X), axis=0)];
        self.counts = {}
        Z = [np.zeros(len(x), int) for x in X]
        hist_pseudo_ll = []
        hist_K = []

        for it in range(n_iter):
            self.born_in_iter = 0;
            self.counts = {}
            for n, x in enumerate(X):
                Tn = x.shape[0]
                for t in range(Tn):
                    prev = Z[n][t - 1] if t > 0 else 0
                    z_new = self._sample_state(prev, x[t])
                    Z[n][t] = z_new
                    if t > 0:
                        self.counts[(prev, z_new)] = self.counts.get((prev, z_new), 0) + 1
            # ML means
            new_means = []
            for k in range(self.K):
                pts = [x[Z[n] == k] for n, x in enumerate(X) if (Z[n] == k).any()]
                new_means.append(self.means[k] if len(pts) == 0 else np.mean(np.vstack(pts), axis=0))
            self.means = new_means

            # 伪 log-lik（用当前 means、sigma 计算各点到最近均值的密度）
            tot = 0.0;
            cnt = 0
            for x in X:
                for xt in x:
                    best = max(self._lik(xt, m) for m in self.means) if self.means else 1.0
                    tot += np.log(best + 1e-12);
                    cnt += 1
            hist_pseudo_ll.append(float(tot / max(cnt, 1)))
            hist_K.append(int(self.K))

            if verbose and ((it + 1) % 5 == 0 or it == 0):
                print(f"[HDP-HMM] iter {it + 1:2d} | K={self.K}")

        self._finalize(X, Z)
        return Z, {"pseudo_loglik": hist_pseudo_ll, "K": hist_K}

    def _finalize(self, X, Z):
        used=set()
        for z in Z:
            for k in np.unique(z): used.add(int(k))
        remap={k:i for i,k in enumerate(sorted(list(used)))}
        Kp=len(remap); self.K=Kp
        pi_counts=np.zeros(Kp); A_counts=np.zeros((Kp,Kp))
        for z in Z:
            if len(z)>0: pi_counts[ remap[z[0]] ] += 1
            for t in range(len(z)-1):
                A_counts[ remap[z[t]], remap[z[t+1]] ] += 1
        for i in range(Kp): A_counts[i,i]+=self.kappa

        # 左→右掩码
        if self.left_right:
            mask = _left_right_mask(Kp)
            A_counts *= mask
            A_counts += 1e-12 * mask

        self.pi = pi_counts/ max(pi_counts.sum(),1e-9);
        if self.pi.sum()<=0 or not np.isfinite(self.pi).all(): self.pi=np.ones(Kp)/Kp
        self.A = A_counts/ np.maximum(A_counts.sum(axis=1,keepdims=True),1e-9)
        D=self.D; self.mu=np.zeros((Kp,D)); self.var=np.ones((Kp,D))
        for k_old,k_new in remap.items():
            pts=[]
            for x,z in zip(X,Z):
                pt=x[z==k_old]
                if len(pt): pts.append(pt)
            if len(pts):
                pts=np.vstack(pts); self.mu[k_new]=pts.mean(axis=0); self.var[k_new]=np.var(pts,axis=0)+1e-3
            else:
                self.mu[k_new]=self.means[k_old]; self.var[k_new]=np.ones(D)*0.5

# ---------- ARHMM（对角方差，EM） ----------
class ARHMM:
    def __init__(self, n_states, n_dims, sticky=0.0, rng=None, min_covar=1e-4, ridge=1e-6, left_right: bool = LEFT_RIGHT_DEFAULT):
        self.K=n_states; self.D=n_dims; self.rng=rng or np.random.RandomState(0)
        self.sticky=sticky; self.min_covar=min_covar; self.ridge=ridge
        self.left_right = bool(left_right)

        self.pi = np.ones(self.K)/self.K
        # 转移矩阵初始化
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

    def fit(self, X, n_iter=30, verbose=True):
        K,D=self.K,self.D
        hist_loglik = []
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
            if verbose and ((it+1)%5==0 or it==0):
                print(f"[ARHMM] iter {it+1:2d} | avg log-lik={total/len(X):.3f}")
        Z=[self.viterbi(x) for x in X]
        return Z, {"loglik": hist_loglik}

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

# ---------- helper（统一接口 + subgoal 计算） ----------
from utils.subgoals import (
    compute_per_demo_lastpoint_subgoals,
    average_subgoals_from_per_demo,
    take_first2_for_plot, take_first2_array
)

def _relabel_to_runs(z):
    """把状态ID序列 z 重标为按时间顺序的段编号（0,1,2,...）"""
    z = np.asarray(z, dtype=int)
    if z.size == 0:
        return z
    out = np.zeros_like(z)
    k = 0
    t = 0
    T = len(z)
    while t < T:
        s = t
        val = z[t]
        while t + 1 < T and z[t + 1] == val:
            t += 1
        e = t
        out[s:e + 1] = k   # 这一段赋段号 k
        k += 1
        t += 1
    return out

def segment_with_hmm(
    X_full,
    method="standard",     # "standard" | "hdp" | "ar"
    n_states=2,
    sticky=10.0,
    n_iter=30,
    verbose=True,
    average_subgoal=False,
    seg_dims=None,
    cl_dims=None,
    seed: int = 0,
    use_velocity: bool = False,
    vel_weight: float = 1.0,
    standardize: bool = False,
    left_right: bool | None = None,
):
    if left_right is None:
        left_right = LEFT_RIGHT_DEFAULT
    else:
        left_right = bool(left_right)

    # 1) 选分割维度
    X_seg_raw = [ (x if seg_dims is None else x[:, seg_dims]) for x in X_full ]
    # 2) 点特征：是否拼速度
    X_seg = _make_point_features_list(
        X_seg_raw, use_velocity=use_velocity,
        vel_weight=vel_weight, standardize=standardize
    )
    D = X_seg[0].shape[1]

    import numpy as _np
    _rng = _np.random.RandomState(int(seed))

    m = method.lower()

    def _fit_for_K(K_try: int):
        if m == "standard":
            mdl = StandardHMM(n_states=K_try, n_dims=D, sticky=sticky, rng=_rng, left_right=left_right)
            Z_list_, hist = mdl.fit(X_seg, n_iter=n_iter, verbose=verbose)
            return mdl, Z_list_, hist
        elif m == "ar":
            mdl = ARHMM(n_states=K_try, n_dims=D, sticky=sticky, rng=_rng, left_right=left_right)
            Z_list_, hist = mdl.fit(X_seg, n_iter=n_iter, verbose=verbose)
            return mdl, Z_list_, hist
        elif m == "hdp":
            mdl = StickyHDPHMM(kappa=sticky, seed=int(seed), left_right=left_right)
            Z_list_, hist = mdl.fit(X_seg, n_iter=n_iter, verbose=verbose)
            return mdl, Z_list_, hist
        else:
            raise ValueError("method must be standard|hdp|ar")

    # --- BIC model selection when K is a sequence and method in {standard, ar} ---
    chosen_model = None
    Z_states_list = None
    seg_hist = {}
    if m in ("standard", "ar") and (isinstance(n_states, (list, tuple, range))):
        K_cands = list(n_states)
        if len(K_cands) == 0:
            raise ValueError("Empty K candidate list for BIC.")
        best = {"bic": np.inf, "K": None, "mdl": None, "Z": None, "hist": None}
        N_data = sum(len(x) for x in X_seg)  # BIC 中的样本数
        bic_table = []
        for K_try in K_cands:
            mdl, Z_tmp, hist = _fit_for_K(int(K_try))
            # 计算总对数似然 & 参数量
            if m == "standard":
                logL = _total_loglik_standard(mdl, X_seg)
                p = _num_params_standard(K_try, D, left_right)
            else:  # m == "ar"
                logL = _total_loglik_ar(mdl, X_seg)
                p = _num_params_ar(K_try, D, left_right)
            bic = -2.0 * logL + p * np.log(max(N_data, 1))
            if verbose:
                print(f"[BIC] method={m} K={K_try} | logL={logL:.3f} p={p} N={N_data} BIC={bic:.3f}")
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
        if m == "hdp":
            chosen_model, Z_states_list, seg_hist = None, None, None
            mdl, Z_states_list, seg_hist = _fit_for_K(n_states)
            chosen_model = mdl
        else:
            mdl, Z_states_list, seg_hist = _fit_for_K(int(n_states))
            chosen_model = mdl

    Z_list = [_relabel_to_runs(z) for z in Z_states_list]
    # Z_list = [z for z in Z_states_list]

    per_demo_vec, K_global = compute_per_demo_lastpoint_subgoals(X_full, Z_list, cl_dims=cl_dims)
    avg_vec = average_subgoals_from_per_demo(per_demo_vec, K_target=K_global) if average_subgoal else None
    per_demo_xy = take_first2_for_plot(per_demo_vec)
    avg_xy = take_first2_array(avg_vec)
    return Z_list, chosen_model, per_demo_vec, avg_vec, per_demo_xy, avg_xy, seg_hist
