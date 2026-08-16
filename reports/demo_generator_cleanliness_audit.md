# Demonstration generator cleanliness audit

本文档审计当前三个任务的 demonstration 生成路径：

- `S3ObsAvoid`
- `S4SlideInsert`
- `S5SphereInspect`

重点不是算法表现，而是：在一个最干净的几何路径、轨迹优化或仿真执行基础上，代码额外加了哪些 small modifications；这些 modification 哪些可以解释为合理 demonstration noise，哪些更像为了让 demo 符合当前 stage-wise constraint modeling 而强行塑形。

## 总体判断

这三个任务都不是独立的、自然专家策略产生的数据。它们都是 procedural demonstration generator，并且都会把 stage structure、true cutpoints、true constraints 显式带进 `TaskBundle`。这本身作为 synthetic benchmark 是可以接受的，但不应该描述成干净的人类/机器人专家 demonstration source。

干净程度排序：

1. `S3ObsAvoid`: 相对最干净。轨迹主要由几何路径、速度/加速度约束优化和投影产生，feature 大多可由轨迹重算。
2. `S5SphereInspect`: 中等偏脏。默认配置是 analytic reference + PyBullet IK execution，但 tool-axis trace 是额外生成/缓存的 observation state，不完全由位置轨迹决定。
3. `S4SlideInsert`: 最不干净。`normal_force` 是 label-conditioned synthetic trace，通过 cache 侧通道进入 feature；速度、姿态、normal load 都明显被塑造成适合 constraint-learning 的统计形状。

## S3ObsAvoid

### 默认生成路径

入口是 `load_S3ObsAvoid -> rollout_demo -> generate_demo`，位于 `envs/S3ObsAvoid.py`。

最干净的底层生成器可以理解为：

1. stage 1 生成绕障几何路径。
2. stage 2 从 stage 1 末端走到 terminal arc entry。
3. stage 3 沿 terminal circular arc 到 goal。
4. 每段通过 `resample_polyline` 和 `optimize_trajectory` 满足速度/加速度限制；stage 1 和 stage 3 分别用 projector 保证 obstacle clearance 或 terminal arc 几何。

返回的 demo 是二维 trajectory；true cutpoints 是 stage 拼接点。feature 后处理从 trajectory 计算：

- `obs_dist`
- `speed`
- `arc_dist`
- `heading`
- `line_dist`
- `noise`

### 在干净轨迹基础上叠加的 mods

#### 可以解释为合理 demonstration variation/noise

- 随机 start 位置和绕障方向。  
  这是合理的 demo diversity。不同专家从不同起点绕同一障碍是正常的。

- `stage1_end` 和 arc entry 的 jitter。  
  可解释为 subgoal variation。只要 jitter 不改变任务语义，这属于合理 demonstration variability。

- stage-specific speed scale jitter。  
  每条 demo 的速度稍有不同是合理的；它也让 demo 不至于完全 deterministic。

- smooth process noise 和 small Gaussian noise。  
  轨迹级平滑扰动可以作为执行噪声或演示者风格差异。

- 对 noisy trajectory 进行 `repair_trajectory_constraints`。  
  如果把 noise 当作扰动，repair 是为了保证物理/任务可行性，这本身合理。

- stage 1 对 clearance boundary 的 projection。  
  合理，因为这是 obstacle avoidance task 的核心可行性约束。

- stage 3 对 terminal arc 的 projection。  
  合理，因为 stage 3 的语义就是 terminal arc following。

#### 更像 modelling preference 的 mods

- stage 2 和 stage 3 的速度被强烈 stage-specific shaping。  
  代码先按 `stage2_speed_max`、`stage3_speed_max` resample，再 repair，使 speed feature 在对应 stage 有清楚的 active pattern。这有任务解释，但也明显服务于让 speed constraint 可学习。

- stage 3 radial jitter 只围绕 terminal arc 加，而不是一般执行噪声。  
  这会专门控制 `arc_dist` 的分布，让 stage 3 的 equality-like feature 更接近可学的窄分布。

- stage 2 noisy path 被 fixed-count uniform reparameterization 混合。  
  这不是一般 physical noise，而是在调 speed profile 的统计形状，降低局部速度波动。

- `line_dist` 是明确的 decoy feature。  
  它不是任务真实约束，只是用来测试学习器是否会误选 irrelevant/progress-correlated feature。

- `noise` 是 deterministic sinusoid auxiliary feature。  
  这不是传感器噪声，而是人为添加的 irrelevant feature。

#### Questionable points

- `heading` 被配置为 `trunc_t_auto_z` 候选 feature，但 true constraints 里没有 heading constraint。  
  这会测试算法抗干扰能力，但如果作为 natural demo generator 解释，比较牵强。

- `obs_dist` 使用 composite obstacle 的 effective distance，而不是单个物理障碍的真实 signed distance。  
  这对 benchmark 合理，但它是 feature engineering，不是原始传感器观测。

- true constraints 直接由 environment constants 给出，demo generator 和 oracle constraint 共用同一套参数。  
  作为 synthetic task 正常；作为 clean expert source 则不独立。

### S3 小结

S3 是一个比较干净的 synthetic generator。脏点主要在 feature set 和 stage-wise speed/arc shaping，而不是通过隐藏侧通道把 label 信息注入 feature。它适合描述为：

> A synthetic three-stage trajectory-optimization generator with controlled decoy features and stage-shaped speed/arc-distance statistics.

不要描述成：

> Pure expert demonstrations from an unconstrained planner.

## S4SlideInsert

### 默认生成路径

默认配置在 `configs/envs/S4SlideInsert.json` 中指定：

- `rollout_backend = analytic`
- `observation_backend = analytic`
- `seg_lengths = [35, 14, 67, 21]`
- `normal_load_scale = 10.0`

入口是 `load_S4SlideInsert -> S4SlideInsertEnv.rollout_demo -> S4SlideInsertEnv.generate_demo`。

当前默认实际使用的是 `S4SlideInsertEnv` 的 robot-friendly 4D 版本，不是旧的 `_S4SlideInsertBase`。trajectory state 是：

```text
[x, y, z, theta]
```

最干净的底层生成器可以理解为：

1. 采样四段长度。
2. 采样 start、stage1 end、stage2 end、stage3 end、stage4 end。
3. 用四条 smooth planar curve 连接这些 waypoint。
4. 给每段 resample，拼成四阶段 slide/insert 轨迹。
5. 根据 theta profile 生成 orientation trajectory。

feature 后处理包括：

- `surf_dist`
- `center_dist`
- `orient_err`
- `speed`
- `angular_speed`
- `normal_force`
- `noise`
- `start_dist`
- `insert_err`

### 在干净轨迹基础上叠加的 mods

#### 可以解释为合理 demonstration variation/noise

- start、endpoints、segment lengths 的随机 jitter。  
  可以解释为不同 demo 的起点、接触点、插入深度略有不同。

- path wobble 和 smooth noise。  
  小幅曲线扰动可以解释为人工演示或控制误差。

- stage boundary blending。  
  用 Hermite blending 消除拼接 kink，可解释为让轨迹更物理、更平滑。

- `theta` 加 smooth noise，并在边界附近 blend。  
  如果只是小幅噪声和连续性修正，这是合理的执行噪声。

- optional PyBullet backend。  
  如果使用 PyBullet execution，执行误差、接触误差和控制器行为可以算合理 demo noise。不过当前默认 benchmark 配置走 analytic，不走 PyBullet。

#### 更像 modelling preference 的 mods

- stage endpoints 是根据 target speed 反推的。  
  代码不是先决定 task geometry 再自然得到速度，而是用 `v2_demo/v3_demo/v4_demo` 和 segment length 反推 align/insert/seat 的距离。这会让 speed feature 更接近预设 stage-wise constraints。

- `_speed_profile_weights` 为不同 stage 手写 valley、micro-slowdown、noise。  
  这些速度形状不是物理仿真的自然结果，而是专门控制 speed feature 的统计轮廓。

- stage 3/4 的 orientation error 被 half-wave profile 和 clipping 控制在 `orient_err_max_stage3/4` 附近。  
  这是典型 near-boundary shaping：让 inequality constraint 不仅成立，而且经常贴近边界，方便算法识别 active upper bound。

- normal force 是 label-conditioned synthetic signal。  
  `_compute_force_signal` 使用 `labels` 和 stage lower bounds，先给 stage 1/2/3 后的 constrained stages 赋 lower bound，再叠加 margin profile，最后强制 `force >= stage_lower_bound`。这不是自然从 trajectory 或接触物理中产生的 normal force。

- `normal_force` 通过 cache 侧通道进入 feature。  
  `compute_observation` 先把 latent rollout 里的 `normal_load` 注册到 environment cache，然后 `compute_all_features_matrix` 通过 trajectory key 取回。也就是说，同一个 `[x,y,z,theta]` trajectory 本身并不唯一决定 `normal_force`。

- `normal_load_scale = 10.0`。  
  这会放大 normal force 的数值尺度，明显影响学习器的 feature distribution。

- y dimension 是后加的 robot-friendly lateral clearance trace。  
  `_lift_planar_demo_to_4d` 按 stage 给 y 加极小波形和 clipping。这可以用于渲染/机器人友好化，但不是原始 planar task 自然产生。

- `surf_dist`、`center_dist`、`insert_err` 都被构造成很强的 progress/stage signal。  
  它们有任务意义，但也很容易制造可分段统计。

- deterministic `noise` feature。  
  和 S3 一样，这是人工 irrelevant feature。

#### Questionable points

- `normal_force` 是最 questionable 的点。  
  它同时满足三个不干净条件：依赖 true labels、依赖 oracle lower bounds、通过 cache 注入 feature。它更像 benchmark annotation/control signal，而不是 clean observed force.

- analytic backend 下的 `normal_force` 不应被称为 measured force。  
  代码的 observation spec 写的是 measured normal contact force，但默认 analytic 生成时它是 planned/synthetic normal load。

- speed feature 是几何有限差分算出来的，但 trajectory endpoint 和 resampling 已经被 target speed 反向塑形。  
  所以 speed feature 表面上由 trajectory 决定，实际上 trajectory 是为了 speed constraint 被设计出来的。

- `start_dist` 和 `insert_err` 可能产生强 progress-correlated one-sided distributions。  
  它们不一定是任务 active constraints，却可能被 inequality model 解释为 lower/upper bound。

- 保留的 `_S4SlideInsertBase` 也有 force/speed cache。  
  当前默认不走这个旧类，但代码历史上确实存在通过 cache 注入 speed/force trace 的设计倾向。

### S4 小结

S4 不是 clean demonstration generator。它更适合描述为：

> A procedural slide-insert data generator with stage-conditioned trajectory shaping and synthetic normal-load observations.

不要描述成：

> Demonstrations produced by a physical contact simulator with naturally measured force.

如果你要清理 S4，优先级最高的是：

1. 把 `normal_force` 从 feature 中移除，或明确改名为 `planned_normal_load`.
2. 如果要保留 force，必须让它来自 PyBullet contact measurement，且不要用 labels/oracle bounds 合成。
3. 去掉或弱化 near-boundary orientation/speed clipping，让 demonstrations 只满足 constraints，而不是专门贴边。

## S5SphereInspect

### 默认生成路径

默认配置在 `configs/envs/S5SphereInspect.json` 中指定：

- `rollout_backend = pybullet`
- `observation_backend = pybullet`
- `cache_demos = true`
- `seed = 127`

loader 还会通过 `_apply_default_s5_loader_config` 设置：

- `split_stage3_transition = True`
- stage2/stage4 speed valley 参数
- stage-specific noise scale
- PyBullet IK filter threshold

实际入口是：

```text
load_S5SphereInspect
-> _build_sphere_inspect_bundle
-> env.rollout_demo
-> _rollout_demo_pybullet
-> _rollout_demo_analytic
-> generate_demo
-> simulate_s5_demo_from_reference
```

如果 cache hit，则直接从 `envs/demo_cache/S5SphereInspect/*.npz` 加载 demos、cutpoints、tool_axis traces。

最干净的底层生成器可以理解为：

1. 在球面上采样 contact normal。
2. stage 1 approach sphere。
3. stage 2 沿 sphere surface trace。
4. stage 3 从 surface 过渡到 offset shell。
5. stage 4 沿 shell inspect/reposition。
6. stage 5 depart。
7. 用 PyBullet UR5 IK 执行 reference，得到 executed trajectory 和 realized tool axis。

feature 后处理包括：

- `surf_dist`
- `normal_err`
- `speed`
- `ang_speed`
- `noise`
- `start_dist`
- `goal_dist`

其中 `normal_err` 和 `ang_speed` 都依赖 tool-axis trace。

### 在干净轨迹基础上叠加的 mods

#### 可以解释为合理 demonstration variation/noise

- contact point、trace direction、top shell target 的随机采样。  
  合理，表示不同 inspect demonstration 覆盖球面不同区域。

- segment length jitter。  
  合理，表示每条 demo 时间长短不同。

- stage 1 approach 和 stage 5 depart 的 geometric control path。  
  合理，属于任务规划的一部分。

- smooth trajectory noise。  
  如果幅度小，并且之后投影回 contact/shell surface，可以解释为执行噪声加可行性修复。

- PyBullet IK execution。  
  默认配置确实会执行 reference，而不是只返回 analytic path。这比纯 analytic generator 干净。

- IK precheck 和 rollout validity filter。  
  从“确保机器人可执行”的角度合理；但它也会引入选择偏差，见下面 questionable 部分。

#### 更像 modelling preference 的 mods

- 默认把 4 个 base segment 拆成 5 个 stage。  
  `split_stage3_transition=True` 是 loader 默认设置，不是类初始化默认值。它让任务更贴合 `S5` 的 stage count。

- stage 2 surface path 使用固定 latitude/lateral trace template。  
  这让 surface trace 更规则、更可学习，而不是自由 inspect trajectory。

- stage 2 和 stage 4 speed valleys 是手写的。  
  `_make_cruise_valley_weights`、`_stage4_speed_profile_weights` 通过固定 center/width/depth 控制局部速度降低。它们不是物理必然结果。

- stage 1 speed taper、stage 3 speed jitter、stage 4 tail stabilization。  
  这些都在塑造 speed feature 的 stage-wise distribution。

- stage 3 transition 被 regularized 到线性 radial profile。  
  `_regularize_stage3_transition_path` 强化从 sphere surface 到 shell offset 的干净 radial transition。这服务于 `surf_dist` 在 stage 3/4 的可解释模式。

- stage 4 被反复 project to shell，并 repair departure tail。  
  这保证 shell-stage 的 `surf_dist` 非常接近 target shell offset，使 equality-like feature 更干净。

- tool-axis trace 是另行生成的。  
  `_generate_tool_axis_trace` 使 stage 2 near-normal aligned，其他 stages irregular transition。它不是仅由 position trajectory 决定，而是 generator 的额外 latent channel。

- stage 2 tool normal error 被控制在 `tool_align_max_stage2` 附近。  
  `_make_aligned_axis_trace` 使用 margin profile 让 normal error 分布贴近上界但不超过上界。这和 S4 orientation 类似，是 near-boundary inequality shaping。

- deterministic `noise` feature。  
  和 S3/S4 一样，是人为 irrelevant feature。

- `goal_dist` 是 nominal goal 相关 progress feature。  
  它很可能产生 monotonic 或 one-sided distribution，容易被 inequality model 吸收。

#### Questionable points

- `normal_err` 和 `ang_speed` 依赖 tool-axis side channel。  
  如果没有 tool_axis，fallback 是用 sphere normal 估计 tool_axis，这会让 `normal_err` 近似 0。说明 feature matrix 不完全由 trajectory 自洽决定。

- cache 保存了 `tool_axis`。  
  cache hit 时会重新 register tool_axis trace；如果只看 demo trajectory 文件，会缺少重建 feature 所需的隐藏状态。

- cache key 只包含 env/run kwargs 和 `cache_version=17`。  
  如果 generator 代码改了但 cache version 没 bump，可能继续使用旧 demo。这对实验可复现有帮助，但对“当前 generator 到底产生什么”会造成混淆。

- PyBullet IK-validity rejection sampling 会改变 demo distribution。  
  默认最多 80 次尝试；失败的 references 被丢弃。这意味着最终 demo 是“容易被 UR5 IK 跟踪”的子集，不是原始 task distribution。

- `observation_backend=pybullet` 和 `analytic_raw` 在 feature 计算上差异很小。  
  当前 `compute_observation` 对两者都调用 `compute_all_features_matrix`，主要差别来自 latent rollout 是否带 executed trajectory/tool_axis，而不是一个独立传感器模型。

### S5 小结

S5 比 S4 更像真实机器人 demonstration，因为默认有 PyBullet execution。但 reference 和 tool-axis 仍然高度 scripted。它适合描述为：

> An analytic spherical-inspection reference generator followed by IK-filtered PyBullet execution, with cached tool-axis observations.

不要描述成：

> A fully physical robot demonstration generator whose constraints emerge naturally from interaction.

如果要清理 S5，优先级最高的是：

1. 让 feature computation 显式要求 `tool_axis`，缺失时 fail fast，而不是 fallback 到 sphere normal。
2. 在 dataset artifact 中把 trajectory 和 tool_axis 视为共同 observation，不要只保存 trajectory。
3. 把 speed valley、near-boundary normal_err shaping 标成 benchmark shaping，而不是 demonstration noise。
4. 对 cache version 做严格管理，任何 generator 逻辑改动都 bump cache version。

## 横向分类

### 合理 demonstration noise / diversity

这些 modification 可以较自然地解释为 demonstration variation：

- start/end/subgoal jitter
- segment length jitter
- smooth low-amplitude trajectory noise
- boundary smoothing
- velocity/acceleration repair after noise
- PyBullet execution error
- IK feasibility filtering，如果明确目标是 robot-executable dataset
- randomized contact location / trace direction / style phase

### Modeling preference / benchmark shaping

这些 modification 更像为了让 SWCL 或 stage-wise constraint learner 更容易看到目标统计：

- stage-specific speed target ratios
- speed valleys at hand-picked stage locations
- near-boundary inequality traces
- clipping features to oracle upper/lower bounds
- feature-specific deterministic noise channel
- decoy line distance
- progress-correlated `start_dist`, `insert_err`, `goal_dist`
- synthetic normal-load / force trace
- tool-axis trace independently generated to satisfy `normal_err` behavior
- default `split_stage3_transition=True` to force S5 stage count

### Most questionable

最需要在论文或实验说明里主动交代的点：

1. S4 `normal_force` is label-conditioned synthetic side-channel.  
   这是最脏的点。它不是从 analytic trajectory 自然计算，也不是默认 PyBullet contact measurement。

2. S5 `tool_axis` is an extra observation channel.  
   `normal_err` 和 `ang_speed` 不是 position-only features。trajectory 缺少 tool_axis 时不能完整复现实验 feature。

3. S5 cache can hide generator changes.  
   cache 对重现实验有用，但 audit 当前 generator 时必须确认是否 cache hit。

4. Near-boundary shaping appears in S4 orientation, S4 normal force, S5 normal_err, and S5 speed.  
   这让 inequality constraints 显得 active，但它不是自然 expert behavior 的必然结果。

5. Progress features can be misread as constraints.  
   `start_dist`, `insert_err`, `goal_dist` 有很强时间/阶段相关性，容易让 inequality model 找到统计边界，但它们不一定是 task constraint。

## 建议的写法

如果后续论文或报告需要描述 dataset，建议用以下口径。

### 推荐表述

```text
We use procedural staged demonstration generators. Each generator creates
geometrically feasible trajectories with controlled stage-wise feature
statistics, then injects smooth execution-style variation. Some auxiliary
and progress-correlated features are included to test whether the learner
can distinguish demonstrated active constraints from irrelevant signals.
```

对于 S4 需要额外说明：

```text
For S4, the analytic backend uses a synthetic normal-load trace generated
from stage labels and task-specific lower-bound profiles. This signal should
be interpreted as a planned/load-control observation rather than a naturally
measured contact force.
```

对于 S5 需要额外说明：

```text
For S5, position trajectories are accompanied by a tool-axis trace. Features
such as normal alignment error and angular speed depend on this trace; the
dataset is therefore not position-only.
```

### 不推荐表述

避免说：

```text
The demonstrations are generated by clean expert policies.
```

避免说：

```text
All features are directly measured from the trajectory.
```

避免说：

```text
S4 normal force is a physical contact measurement in the default setting.
```

这些说法和当前代码不一致。
