# eval/constraint_eval.py
import numpy as np

# ==========================================================
# 1) Obs-avoid 任务的 metrics
# ==========================================================
def obs_avoid_metrics(learner, taus_hat):
    """
    Obs-avoid 类环境的 task-level evaluator：
      - 输入:
          learner : GoalHMM3D 或其它模型实例（需要有 env, true_taus 等）
          taus_hat: list[int]，每条 demo 的估计 cutpoint
      - 输出:
          metrics : dict[str, float]
    """
    metrics = {}

    # ---------- 1) segmentation error ----------
    mae_list, nmae_list = [], []
    if learner.true_taus is not None:
        for tau_hat, tau_true, X in zip(taus_hat, learner.true_taus, learner.demos):
            if tau_true is None:
                continue
            tau_true = int(tau_true)
            T = len(X)
            err = abs(tau_hat - tau_true)
            mae_list.append(err)
            nmae_list.append(err / max(T, 1))

    metrics["MAE_tau"] = float(np.mean(mae_list)) if mae_list else np.nan
    metrics["NMAE_tau"] = float(np.mean(nmae_list)) if nmae_list else np.nan

    # ---------- 2) goal position error ----------
    if hasattr(learner.env, "subgoal") and hasattr(learner.env, "goal"):
        g1_true = np.asarray(learner.env.subgoal, float)
        g2_true = np.asarray(learner.env.goal, float)
        e_g1 = float(np.linalg.norm(learner.g1 - g1_true))
        e_g2 = float(np.linalg.norm(learner.g2 - g2_true))
    else:
        e_g1 = np.nan
        e_g2 = np.nan

    metrics["e_g1"] = e_g1
    metrics["e_g2"] = e_g2

    # ---------- 3) constraint threshold relative error ----------
    if hasattr(learner.env, "true_constraints"):
        oracle = learner.env.true_constraints

        d_safe_true = float(oracle["d_safe"])
        v1_max_true = float(oracle["v1_max"])
        v2_max_true = float(oracle["v2_max"])

        model1 = learner.feature_models[0][0]  # stage1 main feature (distance-like)
        model2 = learner.feature_models[1][1]  # stage2 main feature (speed-like)

        info1 = model1.get_summary()
        info2 = model2.get_summary()

        t1 = info1.get("type", "base")
        t2 = info2.get("type", "base")

        # stage1: 下界型约束，用 z-space 的 L
        if t1 in ("gauss", "margin_exp_lower"):
            z_d_safe_est = float(model1.L)
            d_safe_est_raw = z_d_safe_est * learner.feat_std[0] + learner.feat_mean[0]
            e_d = abs(d_safe_est_raw - d_safe_true)
            metrics["RelErr_d"] = e_d #/ (abs(d_safe_true) + 1e-8)
        else:
            metrics["RelErr_d"] = np.nan

        # stage2: 上界型约束，用 z-space 的 U
        if t2 in ("gauss", "margin_exp_lower"):
            z_v2_max_est = float(model2.U)
            v2_max_est_raw = z_v2_max_est * learner.feat_std[1] + learner.feat_mean[1]
            e_v = abs(v2_max_est_raw - v2_max_true)
            metrics["RelErr_v"] = e_v #/ (abs(v2_max_true) + 1e-8)
        else:
            metrics["RelErr_v"] = np.nan
    else:
        metrics["RelErr_d"] = np.nan
        metrics["RelErr_v"] = np.nan

    return metrics

# ==========================================================
# 2) Sine corridor 环境的 metrics
# ==========================================================
def sine_corridor_metrics(learner, taus_hat):
    """
    SineCorridorEnv3D 的 evaluator：

      1) segmentation / goal 误差：和 obs_avoid 一样
      2) 等式约束误差 EqAbsErr：
           - env.true_constraints["stage1_X_list"] 提供所有阶段1轨迹段
           - learner.learned_features 里找 name=="sine_residual" 的 feature f
           - EqAbsErr = mean_{all stage1 samples} |f(X)|
      3) 速度约束误差 RelErr_v（用 stage2 最大速度 v2_max）：
           - env.true_constraints["v2_max"] 作为 raw-space GT
           - learner.feature_models[1][idx_speed] 给出 z-space 上界 U
             其中 idx_speed = learner.feature_ids.index(1)  (#1 是 speed feature)
           - 速度估计 = U * std + mean
           - RelErr_v = |v2_est - v2_true|   （如果你想可以改回相对误差）
      4) 距离阈值 RelErr_d：Sine 环境没有障碍距离不等式，这里设为 NaN。
    """
    metrics = {}

    # ---------- 1) segmentation error ----------
    mae_list, nmae_list = [], []
    if learner.true_taus is not None:
        for tau_hat, tau_true, X in zip(taus_hat, learner.true_taus, learner.demos):
            if tau_true is None:
                continue
            tau_true = int(tau_true)
            T = len(X)
            err = abs(tau_hat - tau_true)
            mae_list.append(err)
            nmae_list.append(err / max(T, 1))

    metrics["MAE_tau"] = float(np.mean(mae_list)) if mae_list else np.nan
    metrics["NMAE_tau"] = float(np.mean(nmae_list)) if nmae_list else np.nan

    # ---------- 2) goal position error ----------
    if hasattr(learner.env, "subgoal") and hasattr(learner.env, "goal"):
        g1_true = np.asarray(learner.env.subgoal, float)
        g2_true = np.asarray(learner.env.goal, float)
        e_g1 = float(np.linalg.norm(learner.g1 - g1_true))
        e_g2 = float(np.linalg.norm(learner.g2 - g2_true))
    else:
        e_g1 = np.nan
        e_g2 = np.nan

    metrics["e_g1"] = e_g1
    metrics["e_g2"] = e_g2

    # ---------- 3) 等式约束误差：EqAbsErr ----------
    eq_err = np.nan
    env = learner.env

    stage1_X_list = None
    if hasattr(env, "true_constraints") and isinstance(env.true_constraints, dict):
        stage1_X_list = env.true_constraints.get("stage1_X_list", None)

    # 找到 sine residual 这个 learned feature
    lf = None
    if hasattr(learner, "learned_features"):
        for f in learner.learned_features:
            if getattr(f, "name", "") == "sine_residual":
                lf = f
                break

    if stage1_X_list is not None and lf is not None:
        total_abs = 0.0
        total_count = 0
        for X_stage1 in stage1_X_list:
            X_stage1 = np.asarray(X_stage1, float)
            if X_stage1.ndim != 2 or X_stage1.shape[0] == 0:
                continue
            g_vals = lf.eval_raw_numpy(X_stage1)  # (T1,)
            total_abs += float(np.sum(np.abs(g_vals)))
            total_count += g_vals.size

        if total_count > 0:
            eq_err = total_abs / float(total_count)

    metrics["RelErr_d"] = float(eq_err) if np.isfinite(eq_err) else np.nan

    # ---------- 4) 速度阈值误差：RelErr_v（用 stage2 的 v2_max） ----------
    rel_v = np.nan
    if hasattr(env, "true_constraints") and isinstance(env.true_constraints, dict):
        v2_max_true = env.true_constraints.get("v2_max", None)

        if v2_max_true is not None and np.isfinite(v2_max_true):
            try:
                # raw feature 中 speed 的维度 = 1
                raw_speed_fid = 1
                idx_speed = learner.feature_ids.index(raw_speed_fid)

                # stage2（state=1）对应的速度模型
                model2 = learner.feature_models[1][idx_speed]
                info2 = model2.get_summary()
                t2 = info2.get("type", "base")

                if t2 in ("gauss", "margin_exp_lower") and hasattr(model2, "U"):
                    # z-space 上界（U）
                    z_v2_max_est = float(model2.U)

                    # 反变换回 raw-space
                    v2_max_est_raw = (
                        z_v2_max_est * learner.feat_std[idx_speed]
                        + learner.feat_mean[idx_speed]
                    )

                    rel_v = abs(v2_max_est_raw - float(v2_max_true))
            except Exception:
                rel_v = np.nan

    metrics["RelErr_v"] = float(rel_v) if np.isfinite(rel_v) else np.nan

    return metrics


# ==========================================================
# 3) env-name -> metrics 函数的注册表
# ==========================================================
ENV_METRICS_REGISTRY = {
    # Obs-avoid 系列
    "ObsAvoidEnv":       obs_avoid_metrics,
    "ObsAvoidEnv2D":     obs_avoid_metrics,
    "ObsAvoidEnv3D":     obs_avoid_metrics,

    # Sine corridor 系列
    "SineCorridorEnv3D": sine_corridor_metrics,
    "SineCorridor3D":    sine_corridor_metrics,
}


def _get_env_key(env):
    """
    给一个 env，返回用来查表的 key。
    优先使用 env.eval_tag，其次用类名。
    """
    # 你可以在各个 env 里加一个 eval_tag 属性，显式指定
    tag = getattr(env, "eval_tag", None)
    if isinstance(tag, str) and tag:
        return tag

    # 否则就退回到 class name
    return env.__class__.__name__


# ==========================================================
# 4) GoalHMM 的通用 evaluator：自动根据 env 选择 metrics 函数
# ==========================================================
def eval_goalhmm_auto(learner, gammas, xis_list):
    """
    通用版 GoalHMM evaluator:
      - 从 gammas 里得到每条 demo 的 tau_hat
      - 根据 learner.env 的“环境名”选择合适的 metrics 函数
      - 返回一个 metrics dict
    """
    # -------- 从 gammas 里算 MAP cutpoints --------
    taus_hat = []
    for gamma in gammas:
        idx = np.where(gamma[:, 1] > 0.5)[0]
        tau_hat = int(idx[0]) if len(idx) > 0 else int(np.argmax(gamma[:, 1]))
        taus_hat.append(tau_hat)

    # -------- 选 metrics 函数 --------
    env_key = _get_env_key(learner.env)
    metrics_fn = ENV_METRICS_REGISTRY.get(env_key, None)

    if metrics_fn is None:
        # 没找到专用 evaluator，就退到一个超简单的默认版本：只算 segmentation
        metrics = {}

        mae_list, nmae_list = [], []
        if learner.true_taus is not None:
            for tau_hat, tau_true, X in zip(taus_hat, learner.true_taus, learner.demos):
                if tau_true is None:
                    continue
                tau_true = int(tau_true)
                T = len(X)
                err = abs(tau_hat - tau_true)
                mae_list.append(err)
                nmae_list.append(err / max(T, 1))

        metrics["MAE_tau"] = float(np.mean(mae_list)) if mae_list else np.nan
        metrics["NMAE_tau"] = float(np.mean(nmae_list)) if nmae_list else np.nan
        return metrics

    # -------- 调用对应任务的 metrics 函数 --------
    metrics = metrics_fn(learner, taus_hat)
    return metrics