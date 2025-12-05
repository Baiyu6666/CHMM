# exp_init_sensitivity.py
# ------------------------------------------------------------
# 实验：GoalHMM3D 初始化敏感性分析
#
# 新增：
#   - 保存实验数据（json + 可选 csv）
#   - boxplot 和 violin plot
# ------------------------------------------------------------

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from env.env2d import ObsAvoidEnv
from env.env3d import ObsAvoidEnv3D
from learner.goal_hmm_backup import GoalHMM3D
import scipy.stats as stats


# ==========================================================
# 全局配置
# ==========================================================
USE_3D = True
N_DEMOS = 12
DEMO_SEED = 123
N_RUNS_PER_MODE = 25
MAX_ITER = 40

SAVE_DIR = "exp_results"

INIT_MODES = [
    "true_tau",
    "heuristic",
    "random",
]


# ==========================================================
# Demo 生成
# ==========================================================
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

# ==========================================================
# Correlation analysis
# ==========================================================
import scipy.stats as stats


def compute_correlations(metrics, init_modes):
    """
    将所有 mode 的 run 合并后，计算 loglik 与其他指标的 Pearson/Spearman.
    返回字典：
        corr["pearson"]["tau_nmae"] = 值
        corr["spearman"]["tau_nmae"] = 值
    """
    loglik_all = []
    metric_all = {m: [] for m in metrics.keys() if m != "loglik"}

    # flatten 所有 run
    for m in init_modes:
        loglik_all += metrics["loglik"][m]
        for k in metric_all.keys():
            metric_all[k] += metrics[k][m]

    loglik_all = np.array(loglik_all, float)

    corr_res = {"pearson": {}, "spearman": {}}

    for k, vals in metric_all.items():
        vals = np.array(vals, float)

        # Pearson
        p_r, p_val = stats.pearsonr(loglik_all, vals)
        # Spearman
        s_r, s_val = stats.spearmanr(loglik_all, vals)

        corr_res["pearson"][k] = p_r
        corr_res["spearman"][k] = s_r

    return corr_res


# ==========================================================
# 单次试验
# ==========================================================
def run_single_training(env, demos, true_taus, init_mode, max_iter, seed):
    np.random.seed(seed)

    learner = GoalHMM3D(
        demos=demos,
        env=env,
        true_taus=true_taus,
        g1_init=init_mode,
        g2_init=None,
        feat_weight=1.0,
        prog_weight=1.0,
        trans_weight=1.0,
        delta_init=0.15,
        learn_delta=True,
        vmf_lr=8e-4,
        g_step=0.2,
        vmf_steps=3,
        plot_every=None,
    )

    learner.fit(max_iter=max_iter, verbose=False)

    last = -1

    def safe(arr):
        return float(arr[last]) if (arr is not None and len(arr) > 0) else np.nan

    return {
        "loglik": safe(learner.loss_loglik),
        "tau_mae": safe(learner.metric_tau_mae) if hasattr(learner, "metric_tau_mae") else np.nan,
        "tau_nmae": safe(learner.metric_tau_nmae) if hasattr(learner, "metric_tau_nmae") else np.nan,
        "g1_err": safe(learner.metric_g1_err) if hasattr(learner, "metric_g1_err") else np.nan,
        "g2_err": safe(learner.metric_g2_err) if hasattr(learner, "metric_g2_err") else np.nan,
        "d_relerr": safe(learner.metric_d_relerr) if hasattr(learner, "metric_d_relerr") else np.nan,
        "v_relerr": safe(learner.metric_v_relerr) if hasattr(learner, "metric_v_relerr") else np.nan,
    }


# ==========================================================
# Boxplot
# ==========================================================
def plot_boxplots(metrics, init_modes, save_path=None):
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

    plt.figure(figsize=(4 * n_cols, 3.2 * n_rows))

    for idx, m in enumerate(metric_order, 1):
        plt.subplot(n_rows, n_cols, idx)
        vals = [metrics[m][mode] for mode in init_modes]
        plt.boxplot(vals, labels=init_modes, showmeans=True, meanline=True)
        plt.title(m)
        plt.xticks(rotation=25)
        plt.grid(alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


# ==========================================================
# Violin Plot
# ==========================================================
# python
def plot_violin(metrics, init_modes, save_path=None):
    """
    Improved violin plot:
      - wider violins, colored bodies with alpha
      - overlay small boxplots (no fliers)
      - draw quartile lines and median
      - jittered raw data points
    """
    metric_order = [
        "loglik",
        "tau_nmae",
        "g1_err",
        "g2_err",
        "d_relerr",
        "v_relerr",
    ]

    n_metrics = len(metric_order)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 2.2 * n_metrics), constrained_layout=True)

    if n_metrics == 1:
        axes = [axes]

    palette = plt.cm.tab10  # color map for different modes

    for ax, m in zip(axes, metric_order):
        vals = [metrics[m][mode] for mode in init_modes]
        # positions and basic violinplot
        positions = np.arange(1, len(init_modes) + 1)
        parts = ax.violinplot(vals,
                              positions=positions,
                              widths=0.6,
                              showmeans=False,
                              showmedians=False,
                              showextrema=False)

        # style violin bodies
        for i, body in enumerate(parts['bodies']):
            color = palette(i % 10)
            body.set_facecolor(color)
            body.set_edgecolor('black')
            body.set_alpha(0.7)

        # small boxplot overlay for quartiles and median
        bp = ax.boxplot(vals,
                        positions=positions,
                        widths=0.12,
                        patch_artist=True,
                        showfliers=False,
                        boxprops={'facecolor':'white', 'edgecolor':'black', 'linewidth':0.8},
                        medianprops={'color':'firebrick', 'linewidth':1.2},
                        whiskerprops={'linewidth':0.8, 'color':'black'},
                        capprops={'linewidth':0.8, 'color':'black'})

        # jittered raw points for visibility
        for i, v in enumerate(vals):
            if len(v) == 0:
                continue
            x = np.random.normal(positions[i], 0.06, size=len(v))
            ax.scatter(x, v, color='k', s=8, alpha=0.35, rasterized=True)

        # draw explicit quartile lines (thin)
        for i, v in enumerate(vals):
            if len(v) == 0:
                continue
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            pos = positions[i]
            ax.plot([pos - 0.18, pos + 0.18], [q2, q2], color='firebrick', lw=1.2)  # median
            ax.plot([pos - 0.08, pos + 0.08], [q1, q1], color='black', lw=0.9)
            ax.plot([pos - 0.08, pos + 0.08], [q3, q3], color='black', lw=0.9)

        ax.set_xticks(positions)
        ax.set_xticklabels(init_modes, rotation=25)
        ax.set_title(f"Violin: {m}")
        ax.grid(axis='y', alpha=0.25)
        ax.set_ylabel(m)

    if save_path:
        fig.savefig(save_path, dpi=220)
    plt.show()


def plot_log_scatter(metrics, init_modes, save_path=None):
    loglik_all = []
    metric_all = {m: [] for m in metrics.keys() if m != "loglik"}

    for m in init_modes:
        loglik_all += metrics["loglik"][m]
        for k in metric_all.keys():
            metric_all[k] += metrics[k][m]

    loglik_all = np.array(loglik_all, float)

    keys = list(metric_all.keys())
    n = len(keys)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for idx, k in enumerate(keys, 1):
        plt.subplot(n_rows, n_cols, idx)
        y = np.array(metric_all[k], float)

        plt.scatter(loglik_all, y, alpha=0.5, s=18, label='data')

        # regression line
        if len(loglik_all) > 1:
            a, b = np.polyfit(loglik_all, y, deg=1)
            xs = np.linspace(min(loglik_all), max(loglik_all), 100)
            ys = a * xs + b
            plt.plot(xs, ys, 'r--', lw=2, label='linear fit')

        plt.xlabel("log-likelihood")
        plt.ylabel(k)
        plt.title(f"loglik vs {k}")
        plt.grid(alpha=0.3)
        plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

def plot_corr_heatmap(metrics, init_modes, save_path=None):
    keys = list(metrics.keys())
    keys.remove("loglik")
    keys = ["loglik"] + keys  # 把 loglik 放最前

    # assemble matrix
    data = {k: [] for k in keys}

    for mode in init_modes:
        for k in keys:
            data[k] += metrics[k][mode]

    # 转成 numpy
    mat = np.array([data[k] for k in keys], float)

    corr_mat = np.corrcoef(mat)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr_mat, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, shrink=0.8)

    plt.xticks(range(len(keys)), keys, rotation=45)
    plt.yticks(range(len(keys)), keys)
    plt.title("Correlation Heatmap")

    for i in range(len(keys)):
        for j in range(len(keys)):
            plt.text(j, i, f"{corr_mat[i,j]:.2f}",
                     ha='center', va='center', fontsize=8,
                     color='white' if abs(corr_mat[i,j]) > 0.6 else 'black')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220)
    plt.show()

# ==========================================================
# 主实验
# ==========================================================
def run_experiment():
    # --- 创建存储路径 ---
    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M")
    json_path = os.path.join(SAVE_DIR, f"init_sensitivity_{tag}.json")
    csv_path = os.path.join(SAVE_DIR, f"init_sensitivity_{tag}.csv")

    # --- 生成 demos ---
    print("[Exp] Generating demos...")
    env, demos, true_taus = generate_demos(
        use_3d=USE_3D,
        n_demos=N_DEMOS,
        seed=DEMO_SEED
    )

    # --- 初始化数据结构 ---
    metrics = {
        "loglik": {m: [] for m in INIT_MODES},
        "tau_mae": {m: [] for m in INIT_MODES},
        "tau_nmae": {m: [] for m in INIT_MODES},
        "g1_err": {m: [] for m in INIT_MODES},
        "g2_err": {m: [] for m in INIT_MODES},
        "d_relerr": {m: [] for m in INIT_MODES},
        "v_relerr": {m: [] for m in INIT_MODES},
    }

    # --- 多次测试 ---
    for mode in INIT_MODES:
        print(f"\n[Init Mode] {mode}")
        for r in range(N_RUNS_PER_MODE):
            seed = 1000 * INIT_MODES.index(mode) + r
            res = run_single_training(env, demos, true_taus, mode, MAX_ITER, seed)

            for k in metrics.keys():
                metrics[k][mode].append(res[k])

            print(f"  Run {r:02d}: L={res['loglik']:.2f}, tau={res['tau_nmae']:.3f}")

            if mode == "true_tau" and r>=2:
                break

    # ========================================================
    # 保存结果
    # ========================================================
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("[Saved] JSON:", json_path)

    # 可选 CSV
    with open(csv_path, "w") as f:
        header = "mode,metric,value\n"
        f.write(header)
        for m in metrics:
            for mode in INIT_MODES:
                for v in metrics[m][mode]:
                    f.write(f"{mode},{m},{v}\n")
    print("[Saved] CSV:", csv_path)

    # ========================================================
    # 画图
    # ========================================================
    box_path = os.path.join(SAVE_DIR, f"init_sensitivity_box_{tag}.png")
    violin_path = os.path.join(SAVE_DIR, f"init_sensitivity_violin_{tag}.png")

    print("[Plot] Boxplot...")
    plot_boxplots(metrics, INIT_MODES, save_path=box_path)
    print("[Plot] Violin plot...")
    plot_violin(metrics, INIT_MODES, save_path=violin_path)
    # ========================================================
    # Correlation analysis
    # ========================================================
    print("\n[Analysis] Computing correlations...")
    corr = compute_correlations(metrics, INIT_MODES)
    print(json.dumps(corr, indent=2))

    scatter_path = os.path.join(SAVE_DIR, f"init_sensitivity_scatter_{tag}.png")

    print("[Plot] Scatter regressions...")
    plot_log_scatter(metrics, INIT_MODES, save_path=scatter_path)

# ==========================================================
if __name__ == "__main__":
    run_experiment()
