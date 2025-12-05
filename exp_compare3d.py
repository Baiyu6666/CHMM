# exp_init_sensitivity.py
# ------------------------------------------------------------
# 实验：比较 GoalHMM vs GMM-HMM 在不同“共享初始化 τ”方案下的敏感性
#
# - 使用同一批 demos
# - 两种共享初始化方案：
#       "random"     : 每条轨迹随机选 τ （clip 在 [1, T-2]）
#       "heuristic"  : candidate centers + 平均距离 softmax 采样
# - 对于每个 scheme：
#       * 采样一组 taus_init（长度 = n_demos）
#       * 用这组 taus_init 跑一次 GoalHMM（from_tau）
#       * 用同一组 taus_init 初始化 GMMHMM
# - 记录：
#       loglik
#       tau MAE / NMAE
#       ||g1 - g1_true||, ||g2 - g2_true||
#       RelErr(d_safe), RelErr(v2_max)
# - 最后画 boxplot，对比：
#       random-goal, random-gmm, heuristic-goal, heuristic-gmm
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from env.env2d import ObsAvoidEnv
from env.env3d import ObsAvoidEnv3D
from learner.goal_hmm_backup import GoalHMM3D
from learner.gmm_hmm import GMMHMM

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------
USE_3D = True          # True: 用 ObsAvoidEnv3D, False: 用 ObsAvoidEnv (2D)
N_DEMOS = 12            # demo 数量
DEMO_SEED = 123         # 用于生成 demos 的 seed（固定数据，只改初始化）
N_RUNS_PER_SCHEME = 10   # 每种 init scheme 重复次数
MAX_ITER = 40           # EM 迭代次数

INIT_SCHEMES = [
    "random",       # shared_random
    "heuristic",    # shared_heuristic (softmax 距离)
]

MODELS = [
    "goal",         # GoalHMM
    "gmm",          # GMMHMM
]


# ------------------------------------------------------------
# 生成一批固定 demos
# ------------------------------------------------------------
def generate_demos(use_3d=True, n_demos=10, seed=0):
    np.random.seed(seed)

    if use_3d:
        env = ObsAvoidEnv3D(
            start_xy=(-1.5, 0.0),
            subgoal_xy=(0.5, 0.0),
            goal_xy=(0.2, 0.5),
            obs_center_xy=(-0.5, 0.0),
            obs_radius=0.3,
            start_z_range=(0.2, 0.7),
            subgoal_z=0.4,
            goal_z=0.05,
        )
    else:
        env = ObsAvoidEnv(
            start=(-1.5, 0.0),
            subgoal=(0.5, 0.0),
            goal=(0.0, 0.3),
            obs_center=(-0.5, 0.0),
            obs_radius=0.3,
        )

    demos, true_taus = [], []
    for _ in range(n_demos):
        if use_3d:
            X, tau = env.generate_demo_3d(n1=20, direction=None)
        else:
            X, tau = env.generate_demo(n1=20, direction=None)
        demos.append(X)
        true_taus.append(int(tau))

    return env, demos, true_taus


# ------------------------------------------------------------
# τ 初始化：shared random
# ------------------------------------------------------------
def sample_taus_random_shared(demos, rng):
    """
    每条轨迹独立随机一个 tau ∈ [1, T-2]
    """
    taus = []
    lam = rng.rand()
    for X in demos:
        T = len(X)
        t = int(round(lam * (T - 1)))
        t = np.clip(t, 1, T - 1)
        taus.append(int(t))
    return np.array(taus, dtype=int)

# ------------------------------------------------------------
# τ 初始化：shared heuristic + softmax over distance
# ------------------------------------------------------------
def sample_taus_heuristic_softmax(demos, rng, n_cand=200, temperature=0.3):
    """
    按照你在 GoalHMM 里说的那套逻辑实现一个简单版：

    1. 把所有点堆起来 all_pts
    2. 随机挑 n_cand 个 candidate center
    3. 对每个 candidate c：
         - 对每条轨迹，找离 c 最近的点的距离 d_i
         - score(c) = 这些距离的平均值 mean(d_i)
    4. 用 score 经过 softmax 转成概率：
         p(c) ∝ exp( - score(c) / temperature )
       越靠近“大家”的点，score 越小，概率越大
    5. 按这个分布 sample 一个 candidate
    6. 对每条轨迹，用“这个 candidate”找最近点的 index 作为 τ_i
    """
    # 全部点堆一起
    all_pts = np.concatenate(demos, axis=0)
    n_all = len(all_pts)

    n_cand = min(n_cand, n_all)
    cand_idx = rng.choice(n_all, size=n_cand, replace=False)
    cand_pts = all_pts[cand_idx]  # (n_cand, D)

    scores = []
    taus_for_each_cand = []

    for c in cand_pts:
        dists_this_c = []   # 每条轨迹最近点的距离
        taus_this_c = []    # 每条轨迹对应的 tau

        for X in demos:
            dists_i = np.linalg.norm(X - c[None, :], axis=1)  # (T,)
            t_i = int(np.argmin(dists_i))
            t_i = np.clip(t_i, 1, len(X) - 2)
            d_i = float(dists_i[t_i])
            dists_this_c.append(d_i)
            taus_this_c.append(t_i)

        mean_d = float(np.mean(dists_this_c))
        scores.append(mean_d)
        taus_for_each_cand.append(np.array(taus_this_c, dtype=int))

    scores = np.array(scores, dtype=float)
    # softmax over -score/temperature
    logits = -scores / max(temperature, 1e-6)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum() + 1e-12

    # sample 1 candidate according to probs
    idx = rng.choice(len(cand_pts), p=probs)
    taus_init = taus_for_each_cand[idx]

    return taus_init


# ------------------------------------------------------------
# 单次：GoalHMM 训练 + 提取 metrics
# ------------------------------------------------------------
def run_single_goal(env, demos, true_taus, taus_init, max_iter, seed):
    np.random.seed(seed)

    learner = GoalHMM3D(
        demos=demos,
        env=env,
        true_taus=true_taus,
        # 关键：用同一组 taus_init，启用 from_tau 初始化
        tau_init=taus_init,
        g1_init="from_tau",
        g2_init="from_tau",
        feat_weight=1.0,
        prog_weight=1.0,
        trans_weight=1.0,
        delta_init=0.15,
        learn_delta=True,
        vmf_lr=8e-4,
        g_step=0.2,
        vmf_steps=3,
        plot_every=None,   # 不在这里画 4-panel
    )

    learner.fit(max_iter=max_iter, verbose=False)

    last_idx = -1

    def _safe(arr_name):
        arr = getattr(learner, arr_name, None)
        if arr is None or len(arr) == 0:
            return np.nan
        return float(arr[last_idx])

    return {
        "loglik": _safe("loss_loglik"),
        "tau_mae": _safe("metric_tau_mae"),
        "tau_nmae": _safe("metric_tau_nmae"),
        "g1_err": _safe("metric_g1_err"),
        "g2_err": _safe("metric_g2_err"),
        "d_relerr": _safe("metric_d_relerr"),
        "v_relerr": _safe("metric_v_relerr"),
    }


# ------------------------------------------------------------
# 单次：GMM-HMM 训练 + 提取 metrics
# ------------------------------------------------------------
def run_single_gmm(env, demos, true_taus, taus_init, max_iter, seed):
    np.random.seed(seed)

    gmm = GMMHMM(
        demos=demos,
        env=env,
        true_taus=true_taus,
        use_xy_vel=True,
        plot_every=None,
    )

    # fit
    # （假设和 GoalHMM 一样有 loss_loglik，可自己在 GMMHMM 中补一条记录逻辑）
    if "verbose" in gmm.fit.__code__.co_varnames:
        gmm.fit(max_iter=max_iter, verbose=False)
    else:
        gmm.fit(max_iter=max_iter)

    last_idx = -1

    def _safe(obj, arr_name):
        arr = getattr(obj, arr_name, None)
        if arr is None or len(arr) == 0:
            return np.nan
        return float(arr[last_idx])

    # 如果你也给 GMMHMM 实现了对应 metric_xxx，这里会自动用；否则是 NaN
    return {
        "loglik": _safe(gmm, "loss_loglik"),
        "tau_mae": _safe(gmm, "metric_tau_mae"),
        "tau_nmae": _safe(gmm, "metric_tau_nmae"),
        "g1_err": _safe(gmm, "metric_g1_err"),
        "g2_err": _safe(gmm, "metric_g2_err"),
        "d_relerr": _safe(gmm, "metric_d_relerr"),
        "v_relerr": _safe(gmm, "metric_v_relerr"),
    }


# ------------------------------------------------------------
# 主实验：两种 shared init × 两个模型，多次重复
# ------------------------------------------------------------
def run_init_sensitivity_experiment():
    # 1) 生成固定 demos
    print(f"[Exp] Generating demos (USE_3D={USE_3D}) ...")
    env, demos, true_taus = generate_demos(
        use_3d=USE_3D,
        n_demos=N_DEMOS,
        seed=DEMO_SEED,
    )
    print(f"[Exp] Demos generated: {len(demos)}, example shape = {demos[0].shape}")

    # 2) metrics[metric_name][scheme][model] = list
    metric_names = [
        "loglik",
        "tau_mae",
        "tau_nmae",
        "g1_err",
        "g2_err",
        "d_relerr",
        "v_relerr",
    ]

    metrics = {
        m: {scheme: {model: [] for model in MODELS} for scheme in INIT_SCHEMES}
        for m in metric_names
    }

    rng_master = np.random.RandomState(999)

    # 3) scheme × runs
    for scheme in INIT_SCHEMES:
        print(f"\n[Exp] ===== Init scheme: {scheme} =====")

        for run_id in range(N_RUNS_PER_SCHEME):
            # 每个 run 使用独立 rng，但保证 random/heuristic 比较时可控
            rng = np.random.RandomState(rng_master.randint(0, 10**9))

            # 3.1 先根据 scheme 采一组共享的 taus_init
            if scheme == "random":
                taus_init = sample_taus_random_shared(demos, rng)
            elif scheme == "heuristic":
                taus_init = sample_taus_heuristic_softmax(
                    demos,
                    rng=rng,
                    n_cand=200,
                    temperature=0.02,
                )
            else:
                raise ValueError(f"Unknown init scheme: {scheme}")

            # 3.2 在这组 taus_init 下，分别跑 GoalHMM / GMMHMM
            seed_goal = rng_master.randint(0, 10**9)
            seed_gmm = rng_master.randint(0, 10**9)

            res_goal = run_single_goal(env, demos, true_taus, taus_init, MAX_ITER, seed_goal)
            res_gmm = run_single_gmm(env, demos, true_taus, taus_init, MAX_ITER, seed_gmm)

            # 3.3 记录
            for k in metric_names:
                metrics[k][scheme]["goal"].append(res_goal[k])
                metrics[k][scheme]["gmm"].append(res_gmm[k])

            print(
                f"  [Run {run_id:02d}] "
                f"scheme={scheme}, "
                f"Goal: log={res_goal['loglik']:.2f}, NMAE_tau={res_goal['tau_nmae']:.3f}, g1_err={res_goal['g1_err']:.3f}; "
                f"GMM:  log={res_gmm['loglik']:.2f}, NMAE_tau={res_gmm['tau_nmae']:.3f}, g1_err={res_gmm['g1_err']:.3f}"
            )

    # 4) Summary
    print("\n[Exp] ===== Summary (mean ± std) =====")
    for m in metric_names:
        print(f"\nMetric: {m}")
        for scheme in INIT_SCHEMES:
            for model in MODELS:
                vals = np.array(metrics[m][scheme][model], dtype=float)
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                print(f"  {scheme:9s}-{model:4s}: {mean:.4f} ± {std:.4f}")

    # 5) 画 boxplot
    plot_boxplots(metrics)


# ------------------------------------------------------------
# 画 boxplot（flatten：random-goal, random-gmm, heuristic-goal, heuristic-gmm）
# ------------------------------------------------------------
def plot_boxplots(metrics):
    metric_order = [
        "loglik",
        "tau_nmae",
        "g1_err",
        "g2_err",
        "d_relerr",
        "v_relerr",
    ]

    n_metrics = len(metric_order)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    plt.figure(figsize=(4.2 * n_cols, 3.2 * n_rows))

    for idx, mname in enumerate(metric_order, start=1):
        if mname not in metrics:
            continue
        plt.subplot(n_rows, n_cols, idx)

        data = []
        labels = []
        for scheme in INIT_SCHEMES:
            for model in MODELS:
                data.append(metrics[mname][scheme][model])
                labels.append(f"{scheme}-{model}")

        # plt.boxplot(
        #     data,
        #     labels=labels,
        #     showmeans=True,
        #     meanline=True,
        # )
        # plt.title(mname)
        # plt.xticks(rotation=20)
        # plt.grid(alpha=0.3, axis='y')


        # enhanced violin + box + points, colored, with medians and means
        positions = np.arange(1, len(data) + 1)
        cleaned = [np.array(d, dtype=float) for d in data]

        # create violins
        vp = plt.violinplot(cleaned, positions=positions, showmeans=False, showextrema=False, widths=0.8, points=200)

        # color palette
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i / max(1, len(cleaned) - 1)) for i in range(len(cleaned))]

        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[i])
            body.set_edgecolor("black")
            body.set_alpha(0.75)

        # compute medians and means and plot them
        medians = [np.nanmedian(d) if len(d) else np.nan for d in cleaned]
        means = [np.nanmean(d) if len(d) else np.nan for d in cleaned]

        for pos, med in zip(positions, medians):
            if not np.isnan(med):
                plt.plot([pos - 0.28, pos + 0.28], [med, med], color="white", linewidth=2, zorder=3)

        for pos, mean in zip(positions, means):
            if not np.isnan(mean):
                plt.scatter(pos, mean, marker="D", color="black", s=24, zorder=4)

        # overlay a slim boxplot to show quartiles
        bp = plt.boxplot(cleaned, positions=positions, widths=0.12, patch_artist=True, showfliers=False,
                         manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("black")
            patch.set_alpha(0.9)
        for whisker in bp["whiskers"]:
            whisker.set_color("black")
        for cap in bp["caps"]:
            cap.set_color("black")
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        # jittered raw points for distribution visualization
        for i, d in enumerate(cleaned):
            y = d[~np.isnan(d)]
            if len(y):
                x = np.random.normal(positions[i], 0.06, size=len(y))
                plt.scatter(x, y, s=6, color="black", alpha=0.18, zorder=2, rasterized=True)

        # final styling
        plt.title(mname, fontsize=12, fontweight="semibold")
        plt.xticks(positions, labels, rotation=20)
        plt.ylabel(mname)
        plt.grid(axis="y", alpha=0.28)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
if __name__ == "__main__":
    run_init_sensitivity_experiment()
