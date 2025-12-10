# exp_init_sensitivity.py
# ------------------------------------------------------------
# 实验：在 SineCorridorEnv3D 上比较 GoalHMM 在不同初始化 τ 方案下的敏感性
#
# - Env:   SineCorridorEnv3D（正弦走廊 + 两阶段速度约束）
# - Demo:  env.generate_demos(...)
# - Model: GoalHMM3D（配置参考 main_sine_auto.py）
#
# - 两种共享初始化方案：
#       "random"     : shared random λ ∈ (0,1)，每条轨迹 tau = round(λ (T-1))，clip 到 [1, T-1]
#       "heuristic"  : candidate centers + 平均距离 softmax 采样（全局几何启发式）
#
# - 对于每个 scheme：
#       * 采样一组 taus_init（长度 = n_demos）
#       * 用这组 taus_init 跑一次 GoalHMM（tau_init=taus_init, g1_init/g2_init="from_tau"）
# - 记录：
#       loglik
#       tau MAE / NMAE
#       ||g1 - g1_true||, ||g2 - g2_true||
#       RelErr(d_safe), RelErr(v2_max)
# - 最后画 boxplot，对比：
#       random, heuristic
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import random

from env.sine_corridor_3d import SineCorridorEnv3D
from learner.goal_hmm import GoalHMM3D
# from utils.learned_feature import SineCorridorResidual  # 如需开启可恢复

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------
N_DEMOS = 12             # demo 数量
DEMO_SEED = 123          # 用于生成 demos 的 seed（固定数据，只改初始化）
N_RUNS_PER_SCHEME = 10   # 每种 init scheme 重复次数
MAX_ITER = 30            # EM 迭代次数（参考 main_sine_auto.py）

INIT_SCHEMES = [
    "random",       # shared_random
    "heuristic",    # shared_heuristic (softmax 距离)
]

# 只测 GoalHMM 一个模型
MODEL_NAME = "goal"


# ------------------------------------------------------------
# 工具：设定全局随机种子
# ------------------------------------------------------------
def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)


# ------------------------------------------------------------
# 构建 SineCorridorEnv3D（参考 main_sine_auto.build_env）
# ------------------------------------------------------------
def build_env():
    env = SineCorridorEnv3D(
        A=0.1,
        omega=10.0,
        bias=0.2,
        phase=0.0,
        x_start_range=(-2.0, -1.0),
        z_start_range=(0, 0.5),
        x_sub=0.0,
        z_sub=0.6,
        goal=(0.6, 0, 0),
        dt=1,
    )
    return env


# ------------------------------------------------------------
# 生成一批固定 demos（参考 main_sine_auto.generate_demos）
# ------------------------------------------------------------
def generate_demos(env, n_demos=12, seed=0):
    np.random.seed(seed)
    demos, true_taus = env.generate_demos(
        n_demos=n_demos,
        T_stage1=120,
        noise_y_std=0.0,
        noise_z_std=0.0,
        v2_max=None,
    )
    env.estimate_oracle_constraints(demos, true_taus)
    return demos, true_taus


# ------------------------------------------------------------
# τ 初始化：shared random
# ------------------------------------------------------------
def sample_taus_random_shared(demos, rng):
    """
    共享 random λ：
      lam ~ U(0,1)，对所有轨迹共用；
      对第 i 条轨迹：tau_i = round(lam * (T_i - 1))，clip 到 [1, T_i - 1]。
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
# τ 初始化：shared heuristic + softmax over distance（几何启发式）
# ------------------------------------------------------------
def sample_taus_heuristic_softmax(demos, rng, n_cand=200, temperature=0.3):
    """
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
        taus_this_c = np.array(taus_this_c, dtype=int)
        taus_for_each_cand.append(taus_this_c)

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
# 单次：GoalHMM 在一组 taus_init 下训练 + 提取 metrics
# ------------------------------------------------------------
def run_single_goal(env, demos, true_taus, taus_init, max_iter, seed):
    set_seed(seed)

    # ============= main_sine_auto 里的配置 =============
    # optional learnable feature，如果你要打开，可以取消注释并在 GoalHMM3D 里加 learned_features 参数
    # learned_features = [
    #     SineCorridorResidual(
    #         A_init=env.A,
    #         w_init=env.omega,
    #         phi_init=env.phase,
    #         state_index=0,  # 主约束在 stage1
    #     )
    # ]

    # 假设：
    #   - raw 维度 0: stage1 的主约束
    #   - raw 维度 1: stage2 的主约束（速度）
    main_feat_stage1_raw = 0
    main_feat_stage2_raw = 1

    learner = GoalHMM3D(
        demos=demos,
        env=env,
        true_taus=true_taus,

        # ★ 关键：用外部给定的 taus_init，并从 tau 初始化 g1/g2
        tau_init=taus_init,
        g1_init="from_tau",
        g2_init="from_tau",

        main_feat_stage1_raw=main_feat_stage1_raw,
        main_feat_stage2_raw=main_feat_stage2_raw,
        feature_types=None,

        auto_feature_select=False,
        r_sparse_lambda=0.3,

        # learned_features=learned_features,
        f_lr=1e-2,
        f_mstep_steps=5,

        # ===== EM 权重 / 超参（保持与 main_sine_auto 一致）=====
        feat_weight=1.0,
        prog_weight=1.0,
        trans_weight=1.0,

        prog_kappa1=8.0,
        prog_kappa2=6.0,

        fixed_sigma_irrelevant=1.0,

        trans_eps=1e-6,
        delta_init=0.15,
        learn_delta=True,
        lr_delta=5e-4,

        vmf_steps=3,
        vmf_lr=5e-4,
        g_step=0.1,
        g_grad_clip=None,
        g1_vmf_weight=1.0,
        g1_trans_weight=1.0,

        plot_every=None,   # 实验里不画 4-panel
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
# 主实验：两种 shared init × GoalHMM，多次重复
# ------------------------------------------------------------
def run_init_sensitivity_experiment():
    # 1) 生成固定 demos
    print(f"[Exp] Building SineCorridorEnv3D …")
    env = build_env()
    print(f"[Exp] Generating demos (N_DEMOS={N_DEMOS}) …")
    demos, true_taus = generate_demos(env, n_demos=N_DEMOS, seed=DEMO_SEED)
    print(f"[Exp] Demos generated: {len(demos)}, example shape = {demos[0].shape}")

    # 2) metrics[metric_name][scheme] = list
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
        m: {scheme: [] for scheme in INIT_SCHEMES}
        for m in metric_names
    }

    rng_master = np.random.RandomState(999)

    # 3) scheme × runs
    for scheme in INIT_SCHEMES:
        print(f"\n[Exp] ===== Init scheme: {scheme} =====")

        for run_id in range(N_RUNS_PER_SCHEME):
            # 每个 run 使用独立 rng
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

            # 3.2 在这组 taus_init 下，跑 GoalHMM
            seed_goal = rng_master.randint(0, 10**9)
            res_goal = run_single_goal(env, demos, true_taus, taus_init, MAX_ITER, seed_goal)

            # 3.3 记录
            for k in metric_names:
                metrics[k][scheme].append(res_goal[k])

            print(
                f"  [Run {run_id:02d}] "
                f"scheme={scheme}, "
                f"log={res_goal['loglik']:.2f}, "
                f"NMAE_tau={res_goal['tau_nmae']:.3f}, "
                f"g1_err={res_goal['g1_err']:.3f}"
            )

    # 4) Summary
    print("\n[Exp] ===== Summary (mean ± std) =====")
    for m in metric_names:
        print(f"\nMetric: {m}")
        for scheme in INIT_SCHEMES:
            vals = np.array(metrics[m][scheme], dtype=float)
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            print(f"  {scheme:9s}: {mean:.4f} ± {std:.4f}")

    # 5) 画 boxplot
    plot_boxplots(metrics)


# ------------------------------------------------------------
# 画 boxplot（random, heuristic）
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
            data.append(metrics[mname][scheme])
            labels.append(scheme)

        positions = np.arange(1, len(data) + 1)
        cleaned = [np.array(d, dtype=float) for d in data]

        # violin
        vp = plt.violinplot(
            cleaned,
            positions=positions,
            showmeans=False,
            showextrema=False,
            widths=0.8,
            points=200,
        )

        cmap = plt.get_cmap("Set2")
        colors = [cmap(i / max(1, len(cleaned) - 1)) for i in range(len(cleaned))]

        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[i])
            body.set_edgecolor("black")
            body.set_alpha(0.75)

        medians = [np.nanmedian(d) if len(d) else np.nan for d in cleaned]
        means = [np.nanmean(d) if len(d) else np.nan for d in cleaned]

        for pos, med in zip(positions, medians):
            if not np.isnan(med):
                plt.plot(
                    [pos - 0.28, pos + 0.28],
                    [med, med],
                    color="white",
                    linewidth=2,
                    zorder=3,
                )

        for pos, mean in zip(positions, means):
            if not np.isnan(mean):
                plt.scatter(pos, mean, marker="D", color="black", s=24, zorder=4)

        # box
        bp = plt.boxplot(
            cleaned,
            positions=positions,
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
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

        # raw points
        for i, d in enumerate(cleaned):
            y = d[~np.isnan(d)]
            if len(y):
                x = np.random.normal(positions[i], 0.06, size=len(y))
                plt.scatter(x, y, s=6, color="black", alpha=0.18, zorder=2, rasterized=True)

        plt.title(mname, fontsize=12, fontweight="semibold")
        plt.xticks(positions, labels, rotation=0)
        plt.ylabel(mname)
        plt.grid(axis="y", alpha=0.28)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
if __name__ == "__main__":
    run_init_sensitivity_experiment()
