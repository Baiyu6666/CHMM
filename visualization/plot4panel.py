# visualization/plot4panel.py
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa
except ModuleNotFoundError:
    Axes3D = None

from utils.vmf import _unit, vmf_logC_d
from .io import learner_plot_dir, save_figure

PAPER_FIGSIZE = (7.2, 5.2)
PAPER_TITLE_SIZE = 9
PAPER_LABEL_SIZE = 8
PAPER_TICK_SIZE = 7
PAPER_LEGEND_SIZE = 6.5


def _draw_cylinder_wire(ax, center_xy, radius, z0=0.0, height=0.5,
                        color="gray", alpha=0.6, n_theta=80, n_z=10):
    """3D 线框圆柱（只用于 panel1 的障碍物可视化）"""
    cx, cy = center_xy
    theta = np.linspace(0, 2*np.pi, n_theta)
    z = np.linspace(z0, z0 + height, n_z)
    th, zz = np.meshgrid(theta, z)
    xx = cx + radius * np.cos(th)
    yy = cy + radius * np.sin(th)
    ax.plot_wireframe(xx, yy, zz, color=color, alpha=alpha, linewidth=0.6)

def _set_axes_equal_3d_from_xyz(ax, xyz):
    """
    xyz: (N,3) array of points you want to fit.
    Forces equal data scale AND equal box aspect.
    """
    xyz = np.asarray(xyz)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)

    spans = maxs - mins
    max_span = spans.max()
    centers = (maxs + mins) / 2.0

    # equal data limits
    ax.set_xlim(centers[0] - max_span/2, centers[0] + max_span/2)
    ax.set_ylim(centers[1] - max_span/2, centers[1] + max_span/2)
    ax.set_zlim(centers[2] - max_span/2, centers[2] + max_span/2)

    # equal box aspect (if supported)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def _env_has_xy_obstacle(env) -> bool:
    return hasattr(env, "obs_center") and hasattr(env, "obs_radius")


def _env_has_3d_obstacle(env) -> bool:
    return hasattr(env, "obs_center_xy") and hasattr(env, "obs_radius")


def _is_pickplace(env) -> bool:
    return getattr(env, "eval_tag", "") == "PickPlace"


def _xy_point(point):
    arr = np.asarray(point, dtype=float).reshape(-1)
    return arr[:2]



def plot_results_4panel(learner, taus, it, gammas, alphas, betas, xis_list, aux_list):
    if plt is None:
        return
    """
    Panel2-4 完全沿用旧版。
    Panel1: 仅在 3D 情况下使用 3D 画法，其余符号/颜色/legend 名称不变。

    learner 需要提供的属性/方法（与你旧代码一致）：
      - demos, true_taus, env
      - g1, g2, g1_hist, g2_hist
      - loss_loglik, loss_feat, loss_prog, loss_trans
      - get_bounds_for_plot(k_sigma=2)
      - _features_for_demo(X)
      - _transition_logprob(X, return_aux=False/True)
      - prog_kappa1, prog_kappa2
      - trans_eps, d0_trans (如果你现在叫 trans_delta，也保留别名 d0_trans)
    """

    # ==========================================================
    # (1) Trajectories + obstacle + goals + g history + cutpoints
    # ==========================================================
    fig = plt.figure(figsize=PAPER_FIGSIZE)
    X_dim = learner.demos[0].shape[1]
    if X_dim == 3:
        ax = fig.add_subplot(2, 2, 1, projection='3d')
    else:
        ax = fig.add_subplot(2, 2, 1)

    ax.set_title(f"Iter {it}: demos & goals", fontsize=PAPER_TITLE_SIZE, pad=4)

    X_dim = learner.demos[0].shape[1]

    # ================== demos + cutpoints ==================
    if X_dim == 3:
        # 3D: 用 ax（3D axis）
        for i, (X, tau_hat, gamma) in enumerate(zip(learner.demos, taus, gammas)):
            X = np.asarray(X)
            T = len(X)
            th = int(tau_hat)

            # split
            X_pre = X[:th + 1]
            X_post = X[th + 1:] if th + 1 < T else None

            # pre-cutpoint (orange)
            ax.scatter(
                X_pre[:, 0], X_pre[:, 1], X_pre[:, 2],
                c='orange', s=3, alpha=0.35, depthshade=False
            )

            # post-cutpoint (red)
            if X_post is not None and len(X_post) > 0:
                ax.scatter(
                    X_post[:, 0], X_post[:, 1], X_post[:, 2],
                    c='red', s=3, alpha=0.35, depthshade=False
                )

            # learned cutpoint (blue x)
            ax.scatter(
                X[th, 0], X[th, 1], X[th, 2],
                c='blue', marker='x', s=34, linewidths=1.1,
                label='learned cutpoint' if i == 0 else "",
                depthshade=False, zorder=12
            )

            # true cutpoint (green x)
            if learner.true_taus[i] is not None:
                tt = int(learner.true_taus[i])
                ax.scatter(
                    X[tt, 0], X[tt, 1], X[tt, 2],
                    c='green', marker='x', s=34, linewidths=1.1,
                    label='true cutpoint' if i == 0 else "",
                    depthshade=False, zorder=12
                )

    else:
        # 2D: 只用 x,y
        for i, (X, tau_hat, gamma) in enumerate(zip(learner.demos, taus, gammas)):
            X = np.asarray(X)
            T = len(X)
            th = int(tau_hat)

            X_pre = X[:th + 1]
            X_post = X[th + 1:] if th + 1 < T else None

            ax.scatter(
                X_pre[:, 0], X_pre[:, 1],
                c='orange', s=4, alpha=0.35
            )

            if X_post is not None and len(X_post) > 0:
                ax.scatter(
                    X_post[:, 0], X_post[:, 1],
                    c='red', s=4, alpha=0.35
                )

            ax.scatter(
                X[th, 0], X[th, 1],
                c='blue', marker='x', s=24, linewidths=1.1,
                label='learned cutpoint' if i == 0 else "", zorder=10
            )

            if learner.true_taus[i] is not None:
                tt = int(learner.true_taus[i])
                ax.scatter(
                    X[tt, 0], X[tt, 1],
                    c='green', marker='x', s=24, linewidths=1.1,
                    label='true cutpoint' if i == 0 else "", zorder=10
                )

    # ================== obstacle ==================
    if X_dim == 3 and _env_has_3d_obstacle(learner.env):
        # 3D env: obs_center_xy + obs_radius
        cx, cy = learner.env.obs_center_xy
        r = learner.env.obs_radius
        # _draw_cylinder_wire(
        #     ax,
        #     center_xy=(cx, cy),
        #     radius=r,
        #     z0=0.0,
        #     height=0.5,
        #     color='gray'
        # )
    elif X_dim != 3 and _env_has_xy_obstacle(learner.env):
        # 2D env: obs_center + obs_radius
        cx, cy = learner.env.obs_center
        r = learner.env.obs_radius
        circle = plt.Circle(
            (cx, cy),
            r,
            color='gray',
            fill=False,
            linestyle='-',
            label='obstacle'
        )
        ax.add_patch(circle)

    # ================== true goals ==================
    if X_dim == 3:
        # 3D env3d 里有 subgoal, goal (3D 向量)
        sg = learner.env.subgoal
        gg = learner.env.goal
        ax.scatter(
            sg[0], sg[1], sg[2],
            c='green', marker='*', s=40, label='true subgoal'
        )
        ax.scatter(
            gg[0], gg[1], gg[2],
            c='green', marker='P', s=40, label='true goal'
        )
    else:
        # 2D env2d: subgoal, goal 是 2D
        if not _is_pickplace(learner.env):
            sg = _xy_point(learner.env.subgoal)
            gg = _xy_point(learner.env.goal)
            ax.scatter(
                sg[0], sg[1],
                c='green', marker='*', s=40, label='true subgoal'
            )
            ax.scatter(
                gg[0], gg[1],
                c='green', marker='P', s=40, label='true goal'
            )

    # ================== g history ==================
    if len(learner.g1_hist) > 1:
        G1 = np.stack(learner.g1_hist, axis=0)
        if X_dim == 3:
            ax.plot(
                G1[:, 0], G1[:, 1], G1[:, 2],
                '-', lw=1.0, alpha=0.35, color='blue', label='g1 history'
            )
            ax.scatter(
                G1[:, 0], G1[:, 1], G1[:, 2],
                s=4, alpha=0.25, color='blue'
            )
        else:
            ax.plot(
                G1[:, 0], G1[:, 1],
                '-', lw=1.0, alpha=0.35, color='blue', label='g1 history'
            )
            ax.scatter(
                G1[:, 0], G1[:, 1],
                s=4, alpha=0.25, color='blue'
            )

    if len(learner.g2_hist) > 1:
        G2 = np.stack(learner.g2_hist, axis=0)
        if X_dim == 3:
            ax.plot(
                G2[:, 0], G2[:, 1], G2[:, 2],
                '-', lw=1.0, alpha=0.35, color='navy', label='g2 history'
            )
            ax.scatter(
                G2[:, 0], G2[:, 1], G2[:, 2],
                s=4, alpha=0.25, color='navy'
            )
        else:
            ax.plot(
                G2[:, 0], G2[:, 1],
                '-', lw=1.0, alpha=0.35, color='navy', label='g2 history'
            )
            ax.scatter(
                G2[:, 0], G2[:, 1],
                s=4, alpha=0.25, color='navy'
            )

    # ================== estimated goals ==================
    if X_dim == 3:
        ax.scatter(
            learner.g1[0], learner.g1[1], learner.g1[2],
            c='blue', marker='D', s=34, label='est. subgoal g1'
        )
        ax.scatter(
            learner.g2[0], learner.g2[1], learner.g2[2],
            c='navy', marker='P', s=34, label='est. goal g2'
        )
    else:
        g1_xy = _xy_point(learner.g1)
        g2_xy = _xy_point(learner.g2)
        ax.scatter(g1_xy[0], g1_xy[1], c='blue', marker='D', s=34, label='est. subgoal g1')
        ax.scatter(g2_xy[0], g2_xy[1], c='navy', marker='P', s=34, label='est. goal g2')

    # ================== axes labels + scaling ==================
    if X_dim == 3:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # ---- collect all points for 3D scaling ----
        xyz_all = []

        for X in learner.demos:
            X = np.asarray(X)
            if X.shape[1] == 2:
                X = np.pad(X, ((0, 0), (0, 1)), mode='constant')
            xyz_all.append(X)

        # true goals
        xyz_all.append(learner.env.subgoal[None, :])
        xyz_all.append(learner.env.goal[None, :])

        # learned goals + history
        xyz_all.append(np.asarray(learner.g1_hist))
        xyz_all.append(np.asarray(learner.g2_hist))
        xyz_all.append(learner.g1[None, :])
        xyz_all.append(learner.g2[None, :])

        # obstacle cylinder 的包围盒
        if _env_has_3d_obstacle(learner.env):
            cx, cy = learner.env.obs_center_xy
            r = learner.env.obs_radius
            z0, z1 = 0.0, 0.5
            cyl_bbox = np.array([
                [cx - r, cy - r, z0],
                [cx + r, cy + r, z1]
            ])
            xyz_all.append(cyl_bbox)

        xyz_all = np.concatenate(xyz_all, axis=0)
        _set_axes_equal_3d_from_xyz(ax, xyz_all)

    else:
        ax.set_xlabel("x")
        ax.set_ylabel("z" if _is_pickplace(learner.env) else "y")
        ax.set_aspect("equal", adjustable="box")

        # 简单用 demos + goals + obstacle 做一下范围
        pts = []
        for X in learner.demos:
            pts.append(np.asarray(X)[:, :2])
        if _is_pickplace(learner.env):
            if hasattr(learner.env, "pick_point"):
                pick_xy = _xy_point(learner.env.pick_point)
                ax.scatter(pick_xy[0], pick_xy[1], c="green", marker="*", s=40, label="pick")
                pts.append(pick_xy[None, :])
            if hasattr(learner.env, "place_point"):
                place_xy = _xy_point(learner.env.place_point)
                ax.scatter(place_xy[0], place_xy[1], c="green", marker="s", s=34, label="place")
                pts.append(place_xy[None, :])
            if hasattr(learner.env, "retreat_point"):
                retreat_xy = _xy_point(learner.env.retreat_point)
                ax.scatter(retreat_xy[0], retreat_xy[1], c="green", marker="P", s=34, label="retreat")
                pts.append(retreat_xy[None, :])
        else:
            pts.append(_xy_point(learner.env.subgoal)[None, :])
            pts.append(_xy_point(learner.env.goal)[None, :])

        if _env_has_xy_obstacle(learner.env):
            cx, cy = learner.env.obs_center
            r = learner.env.obs_radius
            pts.append(np.array([[cx - r, cy - r], [cx + r, cy + r]]))

        pts = np.concatenate(pts, axis=0)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        pad = 0.05 * max(dx, dy)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    # ================== legend 去重 ==================
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l is None:
            continue
        l = str(l).strip()
        if l == "" or l.startswith("_"):
            continue
        if l not in by_label:
            by_label[l] = h

    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=PAPER_LEGEND_SIZE, frameon=False, handlelength=1.2, borderpad=0.2)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)

    # ==========================================================
    # (2) log-likelihood curve + metrics (双 y 轴)
    # ==========================================================
    plt.subplot(2, 2, 2)
    iters = np.arange(len(learner.loss_loglik))

    # 左轴：log-likelihood
    ax1 = plt.gca()
    ax1.plot(iters, learner.loss_loglik, '-o', color='black', label='total log p(X)', markersize=2.5, linewidth=1.0)
    ax1.set_xlabel("Iteration", fontsize=PAPER_LABEL_SIZE)
    ax1.set_ylabel("Log-likelihood", fontsize=PAPER_LABEL_SIZE)
    ax1.set_title("Learning curves", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax1.tick_params(labelsize=PAPER_TICK_SIZE)

    # 右轴：metrics（来自 metrics_hist dict）
    ax2 = ax1.twinx()
    ax2.set_ylabel("Metrics", fontsize=PAPER_LABEL_SIZE)
    ax2.tick_params(labelsize=PAPER_TICK_SIZE)
    ax2.set_ylim(0.0, 0.8)

    metrics_hist = getattr(learner, "metrics_hist", None)
    if isinstance(metrics_hist, dict) and len(metrics_hist) > 0:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        from itertools import cycle
        color_iter = cycle(color_cycle)

        for name, seq in metrics_hist.items():
            if not seq:
                continue
            Tm = min(len(seq), len(iters))
            ax2.plot(
                iters[:Tm],
                seq[:Tm],
                linestyle='--',
                linewidth=1.0,
                label=name,
                color=next(color_iter),
            )

        # 合并 legend（左右轴）
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=PAPER_LEGEND_SIZE, frameon=False, handlelength=1.2, borderpad=0.2)
    else:
        # 只有 log-likelihood 时，就只要左侧 legend
        ax1.legend(loc='best', fontsize=PAPER_LEGEND_SIZE, frameon=False, handlelength=1.2, borderpad=0.2)

    # ==========================================================
    # (3) Feature evolution (demo 0) + ranges + r + P(z=1) + P(1->2)
    # ==========================================================
    plt.subplot(2, 2, 3)

    X0 = learner.demos[0]
    gamma0 = gammas[0]
    T0 = len(X0)
    t_axis = np.arange(T0)

    # ---------- 取 z-space feature 并还原到 raw ----------
    Fz = learner._features_for_demo_matrix(X0)  # (T, M)
    M = learner.num_features

    feat_raw = Fz * learner.feat_std[learner.feature_ids] + learner.feat_mean[learner.feature_ids]
    # feat_raw_list = []
    # for m in range(M):
    #     raw_m = Fz[:, m] * learner.feat_std[m] + learner.feat_mean[m]
    #     feat_raw_list.append(raw_m)

    ax_main = plt.gca()

    # 颜色循环
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if len(color_cycle) < M:
        from itertools import cycle, islice
        color_cycle = list(islice(cycle(color_cycle), M))

    # ---------- 画所有 feature 的时间序列 ----------
    for m in range(M):
        ax_main.plot(
            t_axis,
            feat_raw[:, m],
            "-",
            color=color_cycle[m],
            label=f"f{learner.feature_ids[m]}(t)",
        )

    # ---------- cutpoints ----------
    tau_hat0 = taus[0]
    true_tau0 = learner.true_taus[0]

    ax_main.axvline(
        tau_hat0,
        color="blue",
        linestyle="--",
        label="learned cutpoint",
    )
    if true_tau0 is not None:
        ax_main.axvline(
            true_tau0,
            color="green",
            linestyle="--",
            label="true cutpoint",
        )

    ax_main.set_xlabel("t", fontsize=PAPER_LABEL_SIZE)
    ax_main.set_ylabel("feature values", fontsize=PAPER_LABEL_SIZE)
    ax_main.tick_params(labelsize=PAPER_TICK_SIZE)

    # t 轴分段
    t1 = t_axis[: tau_hat0 + 1]
    t2 = t_axis[tau_hat0:] if tau_hat0 < T0 else np.array([tau_hat0])

    for m in range(M):
        color_m = color_cycle[m]

        # ---- stage1 bands ----
        if learner.r[0, m] == 1 and len(t1) > 0:
            model1 = learner.feature_models[0][m]
            info1 = model1.get_summary()
            t1_type = info1.get("type", "base")

            if t1_type in ("gauss", "margin_exp_lower", "gauss_zero"):
                z_low1 = float(model1.L)
                z_up1 = float(model1.U)

                fid = learner.feature_ids[m]
                L1_raw = z_low1 * learner.feat_std[fid] + learner.feat_mean[fid]
                U1_raw = z_up1 * learner.feat_std[fid] + learner.feat_mean[fid]

                ax_main.fill_between(
                    t1, L1_raw, U1_raw,
                    color=color_m,
                    alpha=0.12,
                    linewidth=0,
                )
                ax_main.axhline(0.4, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        # ---- stage2 bands ----
        if learner.r[1, m] == 1 and len(t2) > 0:
            model2 = learner.feature_models[1][m]
            info2 = model2.get_summary()
            t2_type = info2.get("type", "base")

            if t2_type in ("gauss", "margin_exp_lower", "gauss_zero"):
                z_low2 = float(model2.L)
                z_up2 = float(model2.U)

                fid = learner.feature_ids[m]
                L2_raw = z_low2 * learner.feat_std[fid] + learner.feat_mean[fid]
                U2_raw = z_up2 * learner.feat_std[fid] + learner.feat_mean[fid]

                ax_main.fill_between(
                    t2, L2_raw, U2_raw,
                    color=color_m,
                    alpha=0.06,
                    linewidth=0,
                )
                ax_main.axhline(0.0554, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # ---------- P(z_t=1 | X) & P(1→2 | x_t) ----------
    ax_prob = ax_main.twinx()
    ax_prob.plot(
        t_axis,
        gamma0[:, 0],
        "--",
        color="tab:green",
        label="P(z_t=1 | X)",
    )

    logA0 = learner._transition_logprob(X0, return_aux=False)
    p12 = np.zeros(T0)
    if logA0.shape[0] > 0:
        p12[:-1] = np.exp(logA0[:, 0, 1])
        p12[-1] = p12[-2]
    ax_prob.plot(
        t_axis,
        p12,
        ":",
        color="tab:orange",
        label="P(1→2 | x_t)",
    )

    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_ylabel("prob.", fontsize=PAPER_LABEL_SIZE)
    ax_prob.tick_params(labelsize=PAPER_TICK_SIZE)

    # ---------- 合并 legend ----------
    lines, labels = [], []
    for ax_ in (ax_main, ax_prob):
        ls, lb = ax_.get_legend_handles_labels()
        lines += ls
        labels += lb
    ax_main.legend(lines, labels, loc="best", fontsize=PAPER_LEGEND_SIZE, frameon=False, handlelength=1.2, borderpad=0.2)

    plt.title("Feature evolution", fontsize=PAPER_TITLE_SIZE, pad=4)

    # ==========================================================
    # (4) Dominance split with alpha & beta decomposition (new multi-feature version)
    # ==========================================================
    plt.subplot(2, 2, 4)

    alpha0 = alphas[0]
    beta0 = betas[0]
    gamma0 = gammas[0]
    X0 = learner.demos[0]
    T0 = len(X0)
    t_axis = np.arange(T0)
    eps = 1e-8

    # ---------- Feature part: Δ_feat = log p_feat(z=2) - log p_feat(z=1) ----------
    # 这里严格复现 SegCons._emission_loglik 里的 feature 部分：
    #   ll_feat_state[t, k] = Σ_m [ r[k,m] * log N_rel(k,m) + (1-r[k,m]) * log N_irrel(m) ]
    # 再乘上 feat_weight。
    F = learner._features_for_demo_matrix(X0)  # (T0, M)
    T0, M = F.shape
    K = learner.num_states

    rel_logpdf = np.zeros((T0, K, M))
    irrel_logpdf = np.zeros((T0, M))

    # 背景（irrelevant）logpdf：只依赖 feature 维度，不依赖 state
    for m in range(M):
        irrel_logpdf[:, m] = learner._log_irrelevant(F[:, m])

    # relevant logpdf：依赖 state k, feature m
    for k in range(K):
        for m in range(M):
            rel_logpdf[:, k, m] = learner.feature_models[k][m].logpdf(F[:, m])

    # 组合：得到每个 state 的 feature log-likelihood 序列
    ll_feat_state = np.zeros((T0, K))
    for k in range(K):
        for m in range(M):
            if learner.r[k, m] == 1:
                ll_feat_state[:, k] += rel_logpdf[:, k, m]
            else:
                ll_feat_state[:, k] += irrel_logpdf[:, m]

    ll_feat_state *= learner.feat_weight

    # 对应 old code 的 ll_feat1 / ll_feat2 / d_feat
    ll_feat1 = ll_feat_state[:, 0]
    ll_feat2 = ll_feat_state[:, 1]
    d_feat = ll_feat2 - ll_feat1

    # ---------- Progress part: Δ_prog ----------
    ll_prog1 = np.zeros(T0)
    ll_prog2 = np.zeros(T0)
    if learner.prog_weight > 0 and T0 > 1:
        from utils.vmf import _unit, vmf_logC_d

        D = X0.shape[1]
        logC1 = vmf_logC_d(learner.prog_kappa1, D)
        logC2 = vmf_logC_d(learner.prog_kappa2, D)

        Vs = _unit(X0[1:] - X0[:-1])
        U1 = _unit(learner.g1[None, :] - X0[:-1])
        U2 = _unit(learner.g2[None, :] - X0[:-1])

        cos1 = np.sum(Vs * U1, axis=1)
        cos2 = np.sum(Vs * U2, axis=1)

        ll_prog1[:-1] = learner.prog_weight * (logC1 + learner.prog_kappa1 * cos1)
        ll_prog2[:-1] = learner.prog_weight * (logC2 + learner.prog_kappa2 * cos2)

    d_prog = ll_prog2 - ll_prog1

    # ---------- Emission total difference ----------
    d_emit = d_feat + d_prog

    # ---------- Transition part: Δ_trans = log (p12 / (1-p12)) ----------
    logA0 = learner._transition_logprob(X0, return_aux=False)
    p12 = np.zeros(T0)
    if logA0.shape[0] > 0:
        # logA0.shape[0] = T0-1
        p12[:-1] = np.exp(logA0[:, 0, 1])
        p12[-1] = p12[-2]  # 补最后一个，保证长度对齐
    d_trans = np.log((p12 + eps) / (1.0 - p12 + eps))

    # ---------- alpha / beta / posterior log-odds ----------
    alpha_odds = alpha0[:, 1] - alpha0[:, 0]
    beta_odds = beta0[:, 1] - beta0[:, 0]
    post_odds = np.log((gamma0[:, 1] + eps) / (gamma0[:, 0] + eps))

    # ---------- Jump posterior: xi(0->1) ----------
    xi0 = xis_list[0]  # (T0-1, K, K)
    xi01 = xi0[:, 0, 1]  # length T0-1
    xi01_pad = np.r_[xi01, 0.0]

    # ---------- Plot ----------
    plt.plot(t_axis, d_feat, '-', lw=1.1, color='tab:red', label='Δ_feat')
    plt.plot(t_axis, d_prog, '-', lw=1.1, color='tab:blue', label='Δ_prog')
    plt.plot(t_axis, d_trans, '--', lw=1.2, color='tab:orange', label='Δ_trans')
    # plt.plot(t_axis, d_emit, ':', lw=1.5, color='gray', label='Δ_emit = Δ_feat + Δ_prog')

    # plt.plot(t_axis, alpha_odds, '-.', lw=2, color='tab:purple',
    #          label='alpha log-odds (past evidence)')
    # plt.plot(t_axis, beta_odds, '-', lw=2, color='tab:green',
    #          label='beta log-odds (future evidence)')
    plt.plot(t_axis, post_odds, '-', lw=1.6, color='black', label='posterior log-odds')

    # cutpoints
    plt.axvline(tau_hat0, color='blue', linestyle='--', label='learned cutpoint')
    if learner.true_taus[0] is not None:
        plt.axvline(learner.true_taus[0], color='green', linestyle='--', label='true cutpoint')

    plt.axhline(0, color='gray', lw=1, alpha=0.4)
    plt.title("Posterior decomposition", fontsize=PAPER_TITLE_SIZE, pad=4)
    plt.xlabel("t", fontsize=PAPER_LABEL_SIZE)
    plt.ylabel("log-odds diff", fontsize=PAPER_LABEL_SIZE)
    plt.legend(loc='best', fontsize=PAPER_LEGEND_SIZE, frameon=False, handlelength=1.2, borderpad=0.2)
    plt.ylim([-30, 20])

    ax = plt.gca()
    ax2 = ax.twinx()
    # ax2.plot(t_axis, xi01_pad, '-', lw=2, color='tab:cyan', label='xi(0→1) (jump posterior)')
    ax2.set_ylabel('prob.', fontsize=PAPER_LABEL_SIZE)
    ax2.plot(t_axis, gammas[0][:, 0], '-', lw=1.1, color='purple', label='gamma(0)')
    ax2.set_ylim([0.0, 1.0])
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    ax2.tick_params(labelsize=PAPER_TICK_SIZE)

    # 合并 legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=PAPER_LEGEND_SIZE, frameon=False, handlelength=1.2, borderpad=0.2)

    fig.tight_layout(pad=0.5, w_pad=0.6, h_pad=0.8)
    save_figure(fig, learner_plot_dir(learner) / f"plot4panel_iter_{int(it):04d}.png", dpi=220)


    # # ==========================================================
    # # Diagnostic A_t curve (旧版原样)
    # # ==========================================================
    # plt.figure(figsize=(10, 4))
    # xi0 = xis_list[0]
    # aux0 = aux_list[0]
    #
    # p12 = aux0["p12"][:-1]
    # p12 = np.clip(p12, learner.trans_eps, 1.0 - learner.trans_eps)
    #
    # xi01 = xi0[:, 0, 1]
    # xi00 = xi0[:, 0, 0]
    #
    # A_pos = xi01 / (p12 + 1e-12)
    # A_neg = xi00 / (1.0 - p12 + 1e-12)
    # A = A_pos - A_neg
    #
    # t_axis2 = np.arange(len(A))
    # plt.plot(t_axis2, A_pos, label="A_pos = xi01 / p_t")
    # plt.plot(t_axis2, A_neg, label="A_neg = xi00 / (1-p_t)")
    # plt.plot(t_axis2, A, lw=3, label="A = A_pos - A_neg", color="black")
    # plt.axhline(0, color="gray", lw=1)
    # plt.xlabel("t")
    # plt.ylabel("value")
    # plt.title("Transition gradient difference term")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # # effective force
    # sigma = getattr(learner, "d0_trans", getattr(learner, "trans_delta", 0.2))
    # d = aux0["dists"][:-1]
    # e = np.exp(-0.5 * (d ** 2) / (sigma ** 2 + 1e-12))
    # F = A * e * d / (sigma ** 2 + 1e-12)
    #
    # plt.figure(figsize=(10, 4))
    # plt.plot(F, lw=2, label="effective force magnitude (signed)")
    # plt.axhline(0, color="gray")
    # plt.legend()
    # plt.title("Signed effective transition force on g1")
    # plt.show()


def plot_feature_model_debug(learner, gammas, stages=(0, 1), max_bins=40):
    if plt is None:
        return
    """
    可视化：对每个 stage k、每个 feature m，画出
      - 该 stage 下该 feature 的 z-space 数据分布（带权 histogram）
      - 背景分布（irrelevant Gaussian, learner._log_irrelevant）
      - 当前 emission 模型（Gaussian / MarginExpLowerEmission）的曲线
    并在图中标出：
      - weighted mean log p_rel
      - weighted mean log p_bg
      - diff = rel - bg

    参数：
      learner : SegCons 实例
      gammas  : EM 之后每条 demo 的 posterior 列表（fit() 返回的 posts）
      stages  : 要画哪些 state（缺省 (0,1)）
    """
    stages = tuple(stages)
    K = learner.num_states
    M = learner.num_features

    assert max(stages) < K, f"stages={stages} 超出 state 数量 K={K}"

    # --------- 预取所有 demo 的 z-space feature + gamma ----------
    Fz_all = []      # 对应每条 demo 的 Fz (T_i, M)
    gamma_all = []   # 对应每条 demo 的 gamma (T_i, K)
    for X, gamma in zip(learner.demos, gammas):
        Fz = learner._features_for_demo_matrix(X)   # (T_i, M)
        Fz_all.append(Fz)
        gamma_all.append(gamma)

    # --------- 对每个 stage 单独画一个 figure ----------
    for k in stages:
        fig, axes = plt.subplots(
            nrows=int(np.ceil(M / 2)),
            ncols=2,
            figsize=(7.2, 2.4 * int(np.ceil(M / 2))),
        )
        axes = np.asarray(axes).reshape(-1)  # 展平成 1D list
        fig.suptitle(f"Stage {k}: feature distributions", fontsize=PAPER_TITLE_SIZE)

        for m in range(M):
            ax = axes[m]
            # ------- 收集该 stage、该 feature 的 z & gamma 权重 -------
            z_list = []
            w_list = []
            for Fz, gamma in zip(Fz_all, gamma_all):
                z_list.append(Fz[:, m])
                w_list.append(gamma[:, k])
            z_all = np.concatenate(z_list, axis=0).astype(float)
            w_all = np.concatenate(w_list, axis=0).astype(float)

            # 有效权重过滤
            w_all = np.maximum(w_all, 0.0)
            if np.sum(w_all) <= 1e-8:
                ax.set_title(f"f{m} (no effective data)")
                ax.axis("off")
                continue

            # ------- 计算 weighted histogram 的范围 -------
            # 用分位数裁剪一下，避免极端值拉太远
            z_min = np.percentile(z_all, 1.0)
            z_max = np.percentile(z_all, 99.0)
            if z_max <= z_min:
                z_min = z_all.min()
                z_max = z_all.max()
            pad = 0.1 * (z_max - z_min + 1e-6)
            z_min -= pad
            z_max += pad

            # ------- 画 weighted histogram (z-space) -------
            bins = min(max_bins, int(len(z_all) / 10) + 5)
            ax.hist(
                z_all,
                bins=bins,
                range=(z_min, z_max),
                weights=w_all,
                density=False,
                alpha=0.4,
                edgecolor="none",
                label="data (weighted hist)",
            )

            # histogram 的最大高度用于曲线缩放
            y_max = 1e-8
            for p in ax.patches:
                y_max = max(y_max, p.get_height())
            if y_max <= 0:
                y_max = 1.0

            # ------- 构造 z-grid，用于画 bg / rel 曲线 -------
            z_grid = np.linspace(z_min, z_max, 300)

            # 背景 logpdf & pdf（z-space）
            logp_bg_grid = learner._log_irrelevant(z_grid)
            # 当前 emission 模型
            model = learner.feature_models[k][m]
            logp_rel_grid = model.logpdf(z_grid)

            # 为了和 hist 尺度匹配，把 pdf 做一个相对缩放：
            #   curve(x) = exp(logp(x) - max_logp) * y_max
            max_bg = np.max(logp_bg_grid)
            max_rel = np.max(logp_rel_grid)

            curve_bg = np.exp(logp_bg_grid - max_bg) * y_max
            curve_rel = np.exp(logp_rel_grid - max_rel) * y_max

            ax.plot(
                z_grid,
                curve_bg,
                "--",
                linewidth=1.8,
                label="background (shape only)",
            )
            ax.plot(
                z_grid,
                curve_rel,
                "-",
                linewidth=1.8,
                label="feature model (shape only)",
            )

            # ------- 计算 weighted mean log-likelihood -------
            w_norm = w_all / (np.sum(w_all) + 1e-12)
            logp_bg_all = learner._log_irrelevant(z_all)
            logp_rel_all = model.logpdf(z_all)

            mean_ll_bg = float(np.sum(w_norm * logp_bg_all))
            mean_ll_rel = float(np.sum(w_norm * logp_rel_all))
            diff = mean_ll_rel - mean_ll_bg

            # feature 类型标签
            feat_type = getattr(learner, "feature_types", None)
            if feat_type is not None:
                type_str = feat_type[m]
            else:
                type_str = type(model).__name__

            ax.set_title(
                f"Stage {k}, f{m} ({type_str})\n"
                f"⟨log p_rel⟩={mean_ll_rel:.2f}, "
                f"⟨log p_bg⟩={mean_ll_bg:.2f}, Δ={diff:.2f}",
                fontsize=7,
            )
            ax.set_xlabel("z", fontsize=PAPER_LABEL_SIZE)
            ax.set_ylabel("weighted counts", fontsize=PAPER_LABEL_SIZE)
            ax.tick_params(labelsize=PAPER_TICK_SIZE)

            ax.legend(fontsize=PAPER_LEGEND_SIZE, loc="best", frameon=False, handlelength=1.2, borderpad=0.2)

        # 把多余的子图关掉（如果 M 是奇数）
        for j in range(M, len(axes)):
            axes[j].axis("off")

        plt.tight_layout(rect=[0, 0.02, 1, 0.95], pad=0.5, w_pad=0.5, h_pad=0.8)
        save_figure(fig, learner_plot_dir(learner) / f"feature_debug_stage_{int(k)}.png", dpi=220)
