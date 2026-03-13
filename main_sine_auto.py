# main_sine_auto.py
# ------------------------------------------------------------
# 测试：SineCorridorEnv3D + GoalHMM3D + learnable sine feature
#
# - Env:   SineCorridorEnv3D（正弦走廊 + 两阶段速度约束）
# - Demo:  env.generate_demos(...)
# - Model: GoalHMM3D
#          * env 提供物理特征
#          * 额外 learnable feature: SineCorridorResidual (y - A sin(wx+phi))
#
# 用法：
#   python main_sine_auto.py
# ------------------------------------------------------------

import os
import numpy as np
import random

from env.sine_corridor_3d import SineCorridorEnv3D   # 按你文件里的类名来改
from learner.goal_hmm import GoalHMM3D
from utils.learned_feature import SineCorridorResidual

def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)


def build_env():
    """
    根据你在 sine_corridor_3d.py 里的定义来改参数。
    这里给一个比较保守的默认配置。
    """
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


def generate_demos(env, n_demos=12):
    """
    封装一下 env.generate_demos，方便后面替换配置。
    你在 env 里定义 generate_demos(...) 的签名如果不一样，
    在这里改一下即可。
    """
    demos, true_taus = env.generate_demos(
        n_demos=n_demos,
        T_stage1=120,
        noise_y_std=0.0,
        noise_z_std=0.0,
        v2_max=None,
    )
    env.estimate_oracle_constraints(demos, true_taus)
    return demos, true_taus


def run_experiment(
    seed=0,
    n_demos=12,
    max_iter=30,
):
    set_seed(seed)

    print("Building SineCorridorEnv3D …")
    env = build_env()

    print(f"Generating {n_demos} demos …")
    demos, true_taus = generate_demos(env, n_demos=n_demos)

    # ============= Learnable feature: sine residual (绑定在阶段1) =============
    learned_features = [
        SineCorridorResidual(
            A_init=env.A,
            w_init=env.omega,
            phi_init=env.phase,
            state_index=0,  # 主约束在 stage1，所以绑定在第0维（x）。更新的时候只用 stage1 的数据，但是是否应该使用所有stage的数据存疑
        )
    ]

    print("Building GoalHMM3D …")
    learner = GoalHMM3D(
        demos=demos,
        env=env,
        true_taus=true_taus,

        g1_init="random",
        g2_init="heuristic",

        feature_ids=[4], # 默认用所有 raw feature（含 learnable）
        feature_types=None,  # 按 raw index 设置每维 emission type

        auto_feature_select=not True, #feature select优先级高于fixed feature
        # fixed_feature_mask = [[1],[1]],  # 如果 auto_feature_select=False，则用这个 mask 指定哪些 feature 用于哪一阶段
        r_sparse_lambda=0.3,  # 你可以在这里扫一下 lambda 看 Sensitivity

        learned_features=learned_features,
        f_lr=1e-2,
        f_mstep_steps=20,  # 每次 EM 对 g 做 5 个小步

        # ===== EM 权重 / 超参 =====
        feat_weight=1.0,
        prog_weight=1,
        trans_weight=1,
        posterior_temp = 1.0,

        prog_kappa1=8.0,
        prog_kappa2=6.0,

        fixed_sigma_irrelevant=1.0,

        trans_eps=1e-6,
        delta_init=0.05,
        trans_b_init=-2.0,
        learn_transition=False,  # None -> follow learn_delta
        lr_delta=5e-3,
        lr_b=5e-3,

        g_steps=10,
        g_lr=10e-4,
        g_grad_clip=None,
        g1_vmf_weight=1.0,
        g1_trans_weight=1,

        plot_every=12222,                   # 每 10 轮画一次 4panel（最后一轮一定会画）
    )
    print("Training GoalHMM3D on sine corridor demos …")
    posts = learner.fit(max_iter=max_iter, verbose=True)

    print("Training finished.")

    return learner, posts


def main():
    learner, posts = run_experiment(
        seed=426,   #421 靠左  #425远远靠左 ，426 靠右
        n_demos=12,
        max_iter=60,
    )

if __name__ == "__main__":
    main()
