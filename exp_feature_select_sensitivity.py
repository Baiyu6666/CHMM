# exp_feature_select_sensitivity.py
# ------------------------------------------------------------
# 实验：Feature 选择敏感性分析
#
# 固定一个初始化方式（例如 heuristic），
# 在不同的 lambda 下、多次随机种子，观察：
#   - loglik / tau_nmae / g1_err / g2_err / d_relerr / v_relerr
#       的均值、方差（boxplot、violin）
#   - r 矩阵在不同 λ 下被选中的频率 + 方差（线图 + errorbar）
# ------------------------------------------------------------

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from env.env2d import ObsAvoidEnv
from env.env3d import ObsAvoidEnv3D
from learner.goal_hmm import GoalHMM3D


# ==========================================================
# 配置
# ==========================================================
USE_3D = True
N_DEMOS = 12
DEMO_SEED = 123

INIT_MODE = "heuristic"      # 固定初始化方式
N_RUNS = 30                  # 每个 λ 重复跑多少次

LAMBDA_LIST = [0., 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1]

MAX_ITER = 40
SAVE_DIR = "exp_results_feature_select"


# ==========================================================
# 生成 demos
# ==========================================================
def generate_demos(use_3d=True, n_demos=10, seed=0):
    np.random.seed(int(seed))

    if use_3d:
        env = ObsAvoidEnv3D(
            start_xy=(-1.5, 0.0),
            subgoal_xy=(0.5, 0.0),
            goal_xy=(-0.2, 0.5),
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


# ==========================================================
# 单次 run：返回 metrics + r 矩阵
# ==========================================================
def run_single(env, demos, true_taus, seed, r_lambda):
    np.random.seed(int(seed))

    learner = GoalHMM3D(
        demos=demos,
        env=env,
        true_taus=true_taus,

        g1_init=INIT_MODE,
        g2_init=None,
        tau_init=None,

        feat_weight=1.0,
        prog_weight=1.0,
        trans_weight=1.0,

        prog_kappa1=8.0,
        prog_kappa2=6.0,

        trans_eps=1e-6,
        delta_init=0.15,
        learn_delta=True,
        lr_delta=5e-4,

        vmf_steps=3,
        vmf_lr=8e-4,
        g_step=0.2,
        g_grad_clip=None,
        g1_vmf_weight=1.0,
        g1_trans_weight=1.0,

        q_low=0.1,
        q_high=0.9,
        width_reg=0.0,

        plot_every=None,

        # 自动 feature 选择
        auto_feature_select=True,
        r_sparse_lambda=r_lambda,
    )

    learner.fit(max_iter=MAX_ITER, verbose=False)

    last = -1

    def safe(arr):
        return float(arr[last]) if (arr is not None and len(arr) > 0) else np.nan

    res = {
        "loglik": safe(learner.loss_loglik),
        "tau_mae": safe(learner.metric_tau_mae),
        "tau_nmae": safe(learner.metric_tau_nmae),
        "g1_err": safe(learner.metric_g1_err),
        "g2_err": safe(learner.metric_g2_err),
        "d_relerr": safe(learner.metric_d_relerr),
        "v_relerr": safe(learner.metric_v_relerr),
    }

    r_mat = getattr(learner, "r", None)
    if r_mat is not None:
        res["r"] = np.array(r_mat, dtype=int)
    else:
        res["r"] = None

    return res


# ==========================================================
# Boxplot（每个 λ 一组）
# ==========================================================
def plot_boxplots(metrics, lambdas, save_path=None):
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

    plt.figure(figsize=(4.8 * n_cols, 3.1 * n_rows))

    for idx, m in enumerate(metric_order, 1):
        plt.subplot(n_rows, n_cols, idx)
        data = [metrics[m][lam] for lam in lambdas]
        plt.boxplot(
            data,
            labels=[str(l) for l in lambdas],
            showmeans=True,
            meanline=True,
        )
        plt.title(m)
        plt.xticks(rotation=25)
        plt.grid(alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220)
    plt.show()


# ==========================================================
# Violin（看方差形状）
# ==========================================================
def plot_violins(metrics, lambdas, save_path=None):
    metric_order = [
        "loglik",
        "tau_nmae",
        "g1_err",
        "g2_err",
        "d_relerr",
        "v_relerr",
    ]

    n_metrics = len(metric_order)
    fig, axes = plt.subplots(
        n_metrics, 1,
        figsize=(9, 2.4 * n_metrics),
        constrained_layout=True
    )

    if n_metrics == 1:
        axes = [axes]

    cmap = plt.cm.tab10

    for ax, m in zip(axes, metric_order):
        vals = [metrics[m][lam] for lam in lambdas]
        pos = np.arange(1, len(lambdas) + 1)

        parts = ax.violinplot(
            vals,
            positions=pos,
            widths=0.7,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for i, body in enumerate(parts["bodies"]):
            color = cmap(i % 10)
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.7)

        ax.boxplot(
            vals,
            positions=pos,
            widths=0.16,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
            medianprops={"color": "firebrick", "linewidth": 1.2},
        )

        # jitter raw points
        for i, v in enumerate(vals):
            v = np.asarray(v, float)
            if len(v) == 0:
                continue
            x_jit = np.random.normal(pos[i], 0.06, size=len(v))
            ax.scatter(x_jit, v, s=8, color="k", alpha=0.35)

        ax.set_xticks(pos)
        ax.set_xticklabels([str(l) for l in lambdas], rotation=25)
        ax.set_ylabel(m)
        ax.set_title(f"Violin: {m}")
        ax.grid(axis="y", alpha=0.25)

    if save_path:
        fig.savefig(save_path, dpi=220)
    plt.show()


# ==========================================================
# r 选择概率 vs lambda（带方差）
# ==========================================================
def plot_r_vs_lambda(r_records, lambdas, save_path=None):
    """
    r_records[lam] = [ (K x M) array, ... ]

    对每个 (state k, feature m)：
      计算： 对每个 lambda 的
        p_km(λ) = mean_r
        std_km(λ) = std_r （二项分布就等价于 sqrt(p(1-p))，这里直接 sample std）
    然后画成小图：
      x 轴: lambda
      y 轴: p_km(λ)
      误差条: std_km(λ)
    """
    lam_list = list(lambdas)

    # 推断 K, M
    example_r = None
    for lam in lam_list:
        if len(r_records[lam]) > 0:
            example_r = r_records[lam][0]
            break
    if example_r is None:
        print("[Warn] no r records found, skip r-vs-lambda plot.")
        return

    K, M = example_r.shape

    # 为每个 (k,m) 创建一个子图
    n_cells = K * M
    n_cols = min(n_cells, 4)
    n_rows = int(np.ceil(n_cells / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        constrained_layout=True
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])

    # 对每个 (k,m) 画一条 p(λ) 曲线
    for k in range(K):
        for m in range(M):
            idx = k * M + m
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            means = []
            stds = []
            for lam in lam_list:
                Rs = r_records[lam]
                if len(Rs) == 0:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    vals = np.array([R[k, m] for R in Rs], dtype=float)
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))

            means = np.array(means, float)
            stds = np.array(stds, float)
            x = np.arange(len(lam_list))

            ax.errorbar(
                x,
                means,
                yerr=stds,
                fmt="-o",
                capsize=4,
                lw=1.4,
            )
            ax.set_xticks(x)
            ax.set_xticklabels([str(l) for l in lam_list], rotation=25)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("lambda")
            ax.set_ylabel("P(r=1)")
            ax.set_title(f"state {k}, feature {m}")

            ax.grid(alpha=0.3)

    # 把多余的空 subplot 去掉（如果有）
    for idx in range(n_cells, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    if save_path:
        fig.savefig(save_path, dpi=220)
    plt.show()


# ==========================================================
# 主实验
# ==========================================================
def run_experiment():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M")
    json_path = os.path.join(SAVE_DIR, f"feature_select_sensitivity_{tag}.json")

    # --- demos ---
    print("[Exp] Generating demos...")
    env, demos, true_taus = generate_demos(
        use_3d=USE_3D,
        n_demos=N_DEMOS,
        seed=DEMO_SEED,
    )

    # --- metrics & r_records 数据结构 ---
    metrics = {
        "loglik": {lam: [] for lam in LAMBDA_LIST},
        "tau_mae": {lam: [] for lam in LAMBDA_LIST},
        "tau_nmae": {lam: [] for lam in LAMBDA_LIST},
        "g1_err": {lam: [] for lam in LAMBDA_LIST},
        "g2_err": {lam: [] for lam in LAMBDA_LIST},
        "d_relerr": {lam: [] for lam in LAMBDA_LIST},
        "v_relerr": {lam: [] for lam in LAMBDA_LIST},
    }
    r_records = {lam: [] for lam in LAMBDA_LIST}

    # --- sweep λ & seeds ---
    for lam_idx, lam in enumerate(LAMBDA_LIST):
        print(f"\n===== λ = {lam} =====")
        for run in range(N_RUNS):
            seed = int(lam_idx * 10000 + run)
            res = run_single(env, demos, true_taus, seed, lam)

            for k in metrics.keys():
                metrics[k][lam].append(res[k])

            if res["r"] is not None:
                r_records[lam].append(res["r"])

            print(
                f"  run {run:02d}: loglik={res['loglik']:.2f}, "
                f"tau_nmae={res['tau_nmae']:.3f}, r={res['r']}"
            )

    # --- 存结果（方便之后离线分析） ---
    metrics_json = {
        "metrics": {
            m: {str(lam): vals for lam, vals in metrics[m].items()}
            for m in metrics.keys()
        },
        "r_records": {
            str(lam): [r.tolist() for r in r_records[lam]]
            for lam in LAMBDA_LIST
        },
        "lambdas": LAMBDA_LIST,
        "init_mode": INIT_MODE,
        "n_runs": N_RUNS,
    }
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print("[Saved] JSON:", json_path)

    # --- boxplot & violin ---
    box_path = os.path.join(SAVE_DIR, f"{tag}_metrics_boxplot.png")
    violin_path = os.path.join(SAVE_DIR, f"{tag}_metrics_violin.png")
    plot_boxplots(metrics, LAMBDA_LIST, save_path=box_path)
    plot_violins(metrics, LAMBDA_LIST, save_path=violin_path)

    # --- r vs lambda ---
    r_path = os.path.join(SAVE_DIR, f"{tag}_r_vs_lambda.png")
    plot_r_vs_lambda(r_records, LAMBDA_LIST, save_path=r_path)


# ==========================================================
if __name__ == "__main__":
    run_experiment()
