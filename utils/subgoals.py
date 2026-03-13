# utils/subgoals.py
import numpy as np
from typing import List, Tuple, Optional, Sequence

def compute_per_demo_lastpoint_subgoals(
    X_list: List[np.ndarray],
    Z_list: List[np.ndarray],
    cl_dims: Optional[Sequence[int]] = None,
) -> Tuple[List[List[np.ndarray]], int]:
    out=[]; K_global=0
    for X,z in zip(X_list, Z_list):
        X=np.asarray(X,float); z=np.asarray(z,int)
        Xv = X if cl_dims is None else X[:, cl_dims]
        K_i = int(z.max())+1 if z.size>0 else 0
        K_global=max(K_global,K_i)
        goals=[]
        if K_i>0:
            T=len(z); t=0
            while t<T:
                k=int(z[t]); s=t
                while t+1<T and z[t+1]==k: t+=1
                e=t; goals.append(Xv[e].copy()); t+=1
        out.append(goals)
    return out, K_global

def average_subgoals_from_per_demo(subgoals_per_demo: List[List[np.ndarray]], K_target=None):
    if K_target is None: K_target=max((len(sgi) for sgi in subgoals_per_demo), default=0)
    if K_target==0: return np.zeros((0,0))
    first=None
    for sgi in subgoals_per_demo:
        if len(sgi): first=sgi[0]; break
    pos_dim=0 if first is None else first.shape[0]
    out=[]
    for k in range(K_target):
        pts=[sgi[k] for sgi in subgoals_per_demo if k < len(sgi)]
        out.append(np.zeros((pos_dim,),float) if len(pts)==0 else np.mean(np.vstack(pts),axis=0))
    return np.vstack(out)

def take_first2_for_plot(subgoals_vecs: List[List[np.ndarray]]):
    out=[]
    for sgi in subgoals_vecs:
        out.append([g[:2].copy() for g in sgi])
    return out

def take_first2_array(arr):
    if arr is None or arr.size==0: return None
    return arr[:, :2].copy()
