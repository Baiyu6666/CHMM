# S5 demonstration generator 专项审计

## 结论

S5 当前并不是一个单纯的“任务几何 + 控制器执行”生成器，而是以下几层机制叠加：

1. 采样一套五阶段球面检查任务几何；
2. 分阶段手写速度目标、局部 speed valley 和 taper；
3. 对 transition 与 shell path 做多轮 regularize / repair / projection；
4. 在 position path 完成后，按阶段额外合成 tool-axis；
5. 用 PyBullet IK 跟踪 reference，再拒绝不满足阈值的样本；
6. 保存 executed observation、reference、tool-axis、goal 与 materialized features。

因此，S5 可以作为 **controlled synthetic benchmark**，但 Phase A 前的默认实现里“任务定义”“demonstrator style”“benchmark shaping”“数值修复”没有分层。代码审阅者最容易质疑的不是存在 shaping，而是很多 feature-specific 常数和修复函数曾散落在同一个 3500 行模块里，看起来像为了当前模型逐项调出来的。

建议保留仿真 shaping，但把它变成一个显式、可切换、可报告的策略：

- `natural_controller`：可选的干净生成链，不读取待学习 constraint bound，不手写 feature waveform；
- `shaped_benchmark`：允许由 constraint bound 条件化的 near-bound、周期倾斜、速度事件等受控 shaping，并在 metadata 中记录；
- `real_replay`：将来从真实机器人轨迹或估计出的 demonstrator-style 参数采样。

这样论文可以明确说 synthetic experiment 使用了 controlled demonstrations，而不是暗示它们完全自然产生。

## Phase A 实施状态

Phase A 已完成，且没有改变 v20 cache 或 analytic generator 的数值输出：

- `envs/S5SphereInspect.py` 已缩减为兼容 import façade；
- task、generator、execution、planner、features、dataset、rendering 已迁入 `envs/s5/`；
- 正式配置由分组 dataclass 与唯一 `S5_SYNTHETIC_V22` preset 提供，`S5_SYNTHETIC_V20/V21` 仅用于冻结与中间版本回归；
- 四阶段分支、自由 surface 分支、无效 length-scale ranges 和已确认未调用的 helper 已从 active implementation 删除；
- 历史固定参数只保留在 v20 cache-key compatibility 层；
- frozen cache fingerprint 与 analytic generator fingerprint 由 `experiments/check_s5_v20_regression.py` 验证。

确定性训练回归也已通过：`map` 与 `map_pooled` 的 cutpoint predictions 和 semantic F1 与现有结果一致；`nominal_shared` 的 MAE 保持 `0.525`，per-demo `demo_goal` 的 MAE 保持 `0.625`，两种模式的 semantic F1 都是 `1.0`。

以下问题清单中的结构性问题已经由 Phase A 解决；time-parameterization 与 stage-2 orientation profile 已完成兼容优先的第一步，repair 行为仍留给后续阶段。

## Time-parameterization 初步实施状态

兼容优先的 timing 重构已经完成，并保持 v20 analytic trajectory fingerprint 逐位不变：

- `envs/s5/time_parameterization.py` 现在唯一负责几何路径的 fixed-step 时间参数化；
- parameterizer 显式输出 positions、timestamps、speed-intent weights 与 reference speeds；
- stage 1 taper、stage 3 correlated variation、stage 2/4 Gaussian valleys 现在都是显式 speed intent；
- stage 2/4 valley 被明确标记为 deliberate slowdown，位置、深度和宽度暂时保留当前设置；
- dataset 保存 rollout 传递的 timestamps，不再在保存端重新假定采样时间；
- timing 重构本身没有增加随机 slowdown、改变 valley 位置或改变 controller 行为，因此仍可通过 v20 regression；orientation profile 的两次行为变化分别使用 v21 和 v22 cache 隔离。

这一步只规范生成流程。后续若随机化 slowdown、替换 motion-limit 算法或改变输出轨迹，应升新 dataset version 并重新生成 demo。

## 当前正式调用链

正式配置使用 PyBullet backend，cache miss 时的调用链为：

```text
load_S5SphereInspect
-> _build_sphere_inspect_bundle
-> env.rollout_demo
-> _rollout_demo_pybullet
-> _rollout_demo_analytic
-> generate_demo
-> simulate_s5_demo_from_reference
-> compute_observation
```

cache hit 时加载与正式 preset 隔离的 v22 dataset；冻结的 v20/v21 dataset 只由显式 regression 路径读取。各版本均显式保存：

- executed position 与 tool-axis；
- per-demo `goal_position`；
- timestamps、stage labels 与 cutpoints；
- reference position 与 reference tool-axis；
- materialized features 及 extractor version；
- 可独立于本项目读取的 manifest。

所以数据保存形式本身已经不是当前主要问题；主要问题在 reference / orientation / timing 的生成方式。

## 缓存轨迹给出的直接证据

对当前正式 v20 cache 的 10 条 PyBullet demonstration 做了阶段归一化统计：

| 现象 | 当前结果 | 含义 |
|---|---:|---|
| demo 长度 | 77--83 samples | 总时长变化很小 |
| stage 1 `surf_dist` profile 平均两两相关 | 0.99997 | approach 几何几乎是同一模板 |
| stage 2 `normal_err` profile 平均两两相关 | 0.99244 | tool-axis 误差波形高度固定 |
| stage 3 `surf_dist` profile 平均两两相关 | 0.99985 | radial transition 基本被规则化成同一形状 |
| stage 5 `surf_dist` profile 平均两两相关 | 0.99997 | departure 几何模板化明显 |
| stage 2 `normal_err` 中位数 | 0.03957 rad | 是 true bound 0.04 的 98.9% |
| stage 2 `speed` 中位数 | 0.00833 m/s | 是 true bound 0.00846 的 98.4% |
| stage 4 `surf_dist` 范围 | 0.032398--0.032402 m | 几乎精确等于 target 0.0324 |
| stage 4 `speed` 中位数 | 0.00801 m/s | 是同一速度 bound 的 94.7% |

stage 2 的平均速度最低区域落在归一化进度约 `0.59--0.64`，也能看到手写 valley center `0.58` 的影响。PyBullet execution 增加了一些跟踪误差，但没有消除 reference 的固定 feature pattern。

## 必须优先整改

### 1. 用统一 time-parameterization 承载 speed intent

旧实现由 generator 直接执行不均匀空间重采样。现在相同数值过程已提炼到统一 parameterizer：generator 提供几何与 speed intent，parameterizer 在固定 `dt` 上生成 reference samples 和 timestamps。

按当前实验选择，stage 2/4 的固定 Gaussian valleys 暂时保留，并明确解释为 controlled demonstrator slowdown，而不是自然由曲率产生的速度 feature。它们仍然属于 benchmark shaping，后续可以通过现有 depth/center/width 参数调整。

当前边界：

- 已完成 geometry 与 timing 代码职责分离；
- 已保留当前 target speed、taper、jitter 和 valley 行为；
- 尚未随机化 valley，也尚未引入 real-data-calibrated human suboptimality process；
- motion-limit refinement 仍以兼容当前输出为优先，不能描述成 time-optimal controller。

固定 valley 的合理性来自“故意 slowdown”的实验定义，而不是它恰好由机器人动力学产生。论文或配置中应继续明确这一点。

### 2. 删除多轮 transition/tail 修复链

当前 split path 的顺序包括：

```text
stage3 regularize
-> stage4 tail spacing
-> departure-tail repair
-> stage4 resample
-> shell projection
-> add noise
-> stage3 regularize again
-> tail spacing again
-> shell projection
-> departure-tail repair again
-> fixed-count resample
-> shell projection again
```

这类链条是最明显的“小补丁感”。每一步局部都合理，但组合后说明上游 path representation 没有直接表达阶段边界条件。

建议：

- stage 3 用一个带起止位置、切向和径向边界条件的 C1/C2 transition；
- stage 4 直接在 shell manifold 上规划，并在末端约束 departure tangent；
- 几何只生成一次，noise 进入 controller/command 后不再通过 feature-specific repair 修回模板；
- 只保留通用数值 projection 作为 manifold integrator 的内部步骤，而不是多次后处理。

### 3. stage-2 tool-axis 已改为随机控制点轨迹

position path 生成完成后，`_generate_tool_axis_trace` 再合成各阶段 tool-axis。v20 的 stage 2 将倾角裁剪在 `0.48--0.99 * bound`，并对每条 demo 使用同一个 4.6 周期模板，因此不同 demo 的 `normal_err` profile 几乎完全相关。

在当前 controlled synthetic benchmark 定义下，读取 oracle bound 并不构成错误：生成器已知约束且只生成 feasible demonstration，使用 bound 控制 margin 是合理的 shaping。真正需要消除的是所有 demo 共享同一逐点模板。

v21 的随机正弦版本虽然打散了 demo 间的相位，但单条轨迹仍保留约 4--5 次等幅振荡。v22 因此改用 `random_control_points_quantile_matched` policy：

- 保留 v20 angle samples 作为经验分位数模板，因此均值、方差和 near-bound 比例基本不变；
- 每条 demo 采样 6--10 个随机间距、随机高度的控制点，经插值和平滑后形成 correlated carrier；
- 再按 carrier 的秩重新排列 v20 分位数，并采样轻微 depth scale 和 bias；
- 使用独立局部 RNG，不改变 position generator 或后续 orientation 阶段的随机流；
- 固定 stage-2 首尾角度，避免变化通过连续性传播到 stage 3--5；
- 将采样参数写入 `orientation_policy` metadata，并用 v22 cache 与 v20/v21 隔离。

seed 127 的 10 条 analytic demo 中，stage-2 `normal_err / bound` 的均值从 v20 的 `0.8372` 变为 v22 的 `0.8346`，标准差从 `0.1865` 变为 `0.1873`，大于 `0.95` 的比例从 `0.5358` 变为 `0.5473`。position 与 stage 3--5 tool-axis 逐位不变；固定五周期结构被每条 demo 不同宽度、不同数量的慢变化 valley 取代。

同配置 MAP 验证中，analytic `MeanAbsCutpointError` 从 v20 的 `0.35` 变为 v22 的 `0.30`。正式 PyBullet 数据上，v20/v21/v22 分别为 `0.625/0.575/0.600`，semantic precision/recall/F1 始终保持 `1.0`，没有观察到实质性能变化。

### 4. 把近百个构造参数收敛成配置对象

Phase A 前，`S5SphereInspectEnv.__init__` 同时包含任务几何、五阶段运动、tool-axis、PyBullet、IK、过滤、feature 和 dataset 行为。现在正式 loader 通过唯一 dataclass preset 解析配置，低层构造参数暂时保留用于兼容直接实例化。

建议拆为 dataclass：

- `S5TaskDistributionConfig`：球体、contact、coverage、shell task；
- `S5PathPolicyConfig`：阶段几何与边界条件；
- `TimeParameterizationConfig`：速度、加速度和 style；
- `OrientationPolicyConfig`：姿态跟踪与扰动；
- `ExecutionConfig`：PyBullet、IK、controller 和 sensor noise；
- `DatasetConfig`：cache、schema、version 和 feature extractor。

正式默认值只能在一个命名 preset 中定义，例如 `S5SyntheticV21Preset`；JSON 只覆盖它。不要再让 loader 暗中补一套与类默认值不同的实验设定。

### 5. 五阶段应有一致的 duration 表示

当前 `seg_lengths` 和 `seg_length_jitter` 只有四项，但 split mode 输出五个阶段。stage 4 的点数由 path/speed 自然算出，第四个长度又用于 stage 5。这会让阶段长度语义很难从配置判断。

二选一：

- 显式保存五个阶段的 nominal duration prior；或
- 不再配置离散点数，由统一 time-parameterizer 根据五段几何生成 timestamps 和 samples。

推荐第二种。阶段时长可以变化，但不应靠复用一个四元素 tuple 表达五段。

## 建议继续整改

### 6. 把固定几何模板改成任务分布

当前 stage 2 默认使用以 `theta=0` 为中心的窄 latitude trace，trace angle、top cap 和 shell detour 也在窄范围内。它可以代表一个特定 inspection protocol，但不应散落成很多 magic numbers。

建议将其定义为具名 task distribution，例如：

- `lateral_band_inspection`；
- `geodesic_patch_inspection`；
- `top_cap_reposition`。

先明确真实机器人实验会采用哪一种工作流，再匹配范围。几何范围的拓宽会显著改变 dataset，应单独升版本，不要与纯代码重构混在一起。

### 7. 把 noise 放到统一执行模型

当前 smooth Cartesian noise 是加在 reference position 上的，stage 2/4 另有 noise scale，之后 constrained stages 又被投影或 repair。正式 PyBullet 调用没有传入非零 joint execution noise，所以它更像“reference 扰动 + 几何清洗”，不是完整的执行噪声。

通用平滑噪声、overshoot 和 correlated error 都可以保留。建议集中到：

- pose command noise；
- joint command / actuator noise；
- controller lag；
- observation noise。

它们可由统一 `DemonstratorStyle` 控制，不要按 feature 或阶段分别补偿。

### 8. 显式记录 IK rejection 带来的选择偏差

每次 PyBullet attempt 会重新采样整条 reference，最多 80 次，直到 IK/rollout filter 接受。因此最终数据实际来自“原始 task distribution 中容易被当前 UR5 与阈值执行的子集”。

可执行性过滤本身合理，但建议：

- 记录全部 attempt 数、失败原因和 acceptance rate；
- task sampling 与 execution seed 分离；
- 尽可能对同一 task/reference 尝试 controller/IK，而不是每次失败就改变任务；
- 报告 accepted distribution 相对 proposed distribution 的偏移。

### 9. 区分 reference stage 与 realized event label

目前 cutpoints 来源于 reference 拼接索引，PyBullet 执行可能存在滞后。如果 stage 表示高层 policy phase，这种标注没有问题；如果论文把它解释为物理接触、进入 shell 或离开表面时刻，就应由 executed observation 的事件重新计算。

建议 dataset 同时保存：

- `command_stage_labels`；
- `event_stage_labels`（若定义了可靠物理事件）。

### 10. 拆分 generator、planner、dataset 和 renderer

当前同一文件还包含 learned-constraint planner、PyBullet execution、feature extraction、rendering 与 cache migration。即使不改变任何 demo，代码也会显得历史包袱很重。

建议目标结构：

```text
envs/s5/task.py
envs/s5/path_policy.py
envs/s5/orientation_policy.py
envs/s5/time_parameterization.py
envs/s5/execution.py
envs/s5/dataset.py
envs/s5/features.py
planning/s5_constraint_planner.py
```

## 可以保留的机制

以下机制本身合理，不建议为了追求“自然”而删除：

- approach / surface trace / transition / shell inspect / depart 五个任务阶段；
- stage 2 在 sphere surface、stage 4 在 offset shell 上的任务定义；
- task-level contact、coverage、方向和目标变化；
- 通用低频相关噪声、偶发 overshoot 和不同 demonstrator speed style；
- 周期 wrist/tool-axis 动作，但应作为显式且随机化的 demonstrator style；
- PyBullet execution、IK feasibility 检查与真实机器人可达性筛选；
- synthetic benchmark 中的 controlled shaping，但必须可切换、可报告，并与默认 natural preset 分离；
- materialized features，同时保存 raw observations、schema 和 extractor version。

“球面阶段精确位于球面/壳层”不是问题，因为这是任务定义。问题是通过多轮后处理把加入的噪声和边界行为重新修成高度固定的 feature profile。

## 死代码与遗留分支

Phase A 扫描出的以下无调用 helper 已删除：

- `_smoothstep`；
- `_blend_segment_boundary`；
- `_make_target_stage_trace`；
- `_make_irregular_positive_stage_trace`；
- `_split_polyline_by_fraction`；
- `_normal_with_geodesic_angle`。

以下原先由正式 preset 固定的开关和范围已从 active generator 删除：

- `split_stage3_transition=True` 与旧四阶段分支；
- `stage2_robot_lateral_trace=True` 与另一套 surface path；
- `stage2_surface_detour_angle=0`；
- stage 2/4 的 `(1, 1)` length-scale ranges。

v20 cache compatibility 值集中在 `envs/s5/config.py`，不再进入 active generator API。cache migration 仍保留在独立 dataset 模块中。

## 推荐实施顺序

### Phase A：只清代码，不改变数据

1. 固定当前 v20 cache 和轨迹 fingerprint 作为 regression fixture；
2. 引入配置 dataclass 与唯一正式 preset；
3. 按模块拆分 generator / execution / dataset / planner；
4. 删除确认无调用的 helper、恒定开关和失效范围；
5. 保证 v20 cache load 与现有训练结果不变。

这一阶段不需要重新生成 demo。

### Phase B：修改 demonstrator 行为

1. 新建统一五阶段 time-parameterizer，删除 valleys/taper/tail speed patches；
2. 将 stage-2 tool-axis 固定模板改为显式、可记录的 bound-conditioned stochastic style；
3. 用带边界条件的 path 一次生成 transition 和 shell departure；
4. 将噪声集中到 controller/execution/observation；
5. 增加 proposed/accepted task statistics 和可选 event labels；
6. 生成 v22 dataset，并与 v20/v21、真实机器人轨迹分别比较。

这一阶段必须重新生成 demo。当前 v22 只随机化 stage-2 `normal_err` 的时间排列与轻微 demo-level scale/bias：analytic position 和 stage 3--5 orientation 保持逐位不变，PyBullet execution 只产生小幅跟踪差异。后续若再修改 transition repair 或噪声模型，应另行升版本并重新评估。

## 最小验收标准

当前 controlled benchmark 的最低验收标准是：

- bound-conditioned shaping 必须由具名 policy 表达并写入 metadata；
- 不同 demo 不共享同一个逐点 `normal_err` waveform；
- 每条轨迹只经过一次几何生成和一次统一 time-parameterization；
- tool-axis 与 position 作为同一个 pose policy 的输出；
- noise 后没有 stage-specific feature repair stack；
- task rejection 的统计可见；
- dataset 明确记录 demonstrator preset、style sample、generator version 和 feature extractor version；
- v20/v21 preset 能复现历史与中间实验，v22 默认路径使用独立 cache。
