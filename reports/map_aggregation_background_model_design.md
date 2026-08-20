# MAP 约束学习：数据聚合与 Background Model 改进方案

更新时间：2026-08-20

## 1. 文档目的

本文档整理真实机器人 Bar Inspection 数据上的 tuning 结果，以及由这些结果引出的算法改进方案。当前最需要研究的两个问题是：

1. 多条 demonstration 以及单条 demonstration 内的样本应当如何聚合；
2. inactive/background 应当使用什么概率模型。

本文档不把环境配置中的 `true_constraints` 当作完整 ground truth。真实任务可能存在配置中没有写出的合理约束，例如接近阶段的 surface distance 下界。对真实数据的判断主要依据：

- 任务语义是否合理；
- 是否具有跨 demonstration 的一致证据；
- 对采样率、执行速度、静止停留和先验是否稳定；
- 学到的约束能否在新轨迹或下游规划中产生合理作用。

## 2. 当前工作基线

当前较合理的配置是：

- 聚合方式：`demo_balanced_pooled`；
- activation prior：每个 feature 为 0.3；
- active mode prior：不使用 feature-specific 先验；
- 候选模式：`inactive / eq / lb / ub`；
- fixed cutpoints：当前 Bar Inspection 实验使用人工阶段边界；
- inactive：局部 Gaussian background；
- inequality：soft half-Student-t 型单边模型。

当前输出的约束语义总体合理：

| 阶段 | 当前学到的约束 | 当前判断 |
|---|---|---|
| S1 | obstacle `lb`、surface `lb`、pitch `eq` | 语义合理，但两个下界证据较弱 |
| S2 | surface `eq`、pitch `eq`、plane error `eq` | 合理且证据较强 |
| S3 | surface `eq`、plane error `eq` | 合理且证据较强 |
| S4 | 全部 inactive | 合理，撤离趋势没有继续被误判为 inequality |

需要注意，当前“结果合理”不等于算法问题已经解决：

- S1 obstacle 的 `lb` 与 inactive 总成本约为 7.3 对 7.6；
- S1 surface 的 `lb` 与 inactive 总成本约为 7.9 对 8.3；
- S3 obstacle 的 inactive 与 `ub` 总成本约为 7.7 对 7.8。

这些模式仍会随先验、数据组成或 background model 的轻微变化而翻转。

## 3. 前期 tuning 得到的主要结论

### 3.1 末尾静止数据确实会污染分布

人工采集轨迹曾经包含末尾长时间静止。重复采样会在固定状态附近产生大量质量，影响 equality 和 inequality 的相对得分。当前已经移除了明确的末尾纯静止段。

但是，不能把所有低平移速度样本都删除，因为机器人可能：

- 原地调整 orientation；
- 低速执行有意义的接近或扫描；
- 沿某个等值面运动，使某个 feature 近似不变，但整体 pose 仍在变化。

因此，后续算法应当降低重复静止采样的权重，而不是继续依靠激进裁剪。

### 3.2 Hard voting 丢失证据强度

`shared_vote` 先让每条 demonstration 产生一个硬模式，再做多数投票。它强调跨 demonstration 一致性，但存在两个问题：

- 三个非常微弱的投票可以压过两个非常强的证据；
- 每条 demonstration 的 best mode 与最终 shared mode 之间不容易解释。

### 3.3 Raw pooled 对样本数量过于敏感

普通 pooled 使用所有样本 NLL 的总和：

`Cₘ = Σ_d Σ_t NLLₘ(x_dt) + C_prior(m)`

而模式先验只加一次。随着样本数量增加，每个样本非常小的拟合优势也会累积成很大的总优势。长 demonstration、低速运动和局部停留因此可能主导模式选择。

此外，`lb/ub` 可以把边界优化到经验分布的一侧，因此偏斜分布、单调趋势和阶段端点都容易被解释为 inequality。

### 3.4 Demo-balanced pooled 明显更合理

`demo_balanced_pooled` 对每条 demonstration 使用 mean NLL，再跨 demonstration 求和。它基本消除了不同阶段长度和不同 demonstration 样本数的直接影响。

但它还没有完全解决：

- demonstration 内部的静止重复采样；
- 少数 demonstration 的强证据推动共享模式；
- 单调趋势冒充 inequality；
- 不同 demonstration 合理边界存在小幅偏差。

### 3.5 Activation prior 可以抑制弱约束，但不是根本修复

将 activation prior 从 0.5 调为 0.3 后，当前输出更加合理。该先验是统一稀疏先验，不是 feature-specific mode 限制，因此可以作为工作配置。

但是先验只是在模式证据接近时决定取舍，不能修复 likelihood family 本身的错误归因。最终算法不应依赖某一个先验值才能避免明显错误。

### 3.6 当前 constant background 无法表达阶段趋势

当前 inactive 模型近似为：

`xₜ ∼ N(a, σ²)`

它只能解释一团没有时间结构的数据。因此：

- S3 obstacle clearance 的持续下降曾被解释为 `ub`；
- S4 surface distance 的持续上升曾被解释为 `ub`；
- 阶段起点或终点形成的硬边缘容易被当作约束边界。

这些现象表明 background 至少需要表达简单的阶段趋势。

## 4. 研究问题一：如何聚合数据

聚合问题应当分成三个层次：

1. demonstration 内部的样本如何加权；
2. 不同 demonstration 的模式证据如何聚合；
3. 不同 demonstration 的约束参数如何形成 shared parameter。

### 4.1 设计目标

理想的聚合方法应满足：

- demonstration 等权：长轨迹不能因为样本更多自动获得更大权重；
- time-warp 稳定：同一路径执行得快或慢不应改变约束模式；
- 静止复制稳定：复制一段纯静止样本不应显著改变结果；
- orientation-aware：原地调整姿态不能被当作无信息；
- 保留证据强度：不能退化成只看硬投票数量；
- 对异常 demonstration 稳健；
- shared constraint parameter 仍然是任务级参数，而不是每条 demonstration 各学一套约束。

### 4.2 Demonstration 内部：SE(3) 运动加权

不建议只按机器人平移路径长度重采样。纯平移路径会忽略原地 orientation 调整。

建议对相邻 pose 定义联合运动量：

`Δsₜ = √[(‖Δpₜ‖／lₚ)² + (Δθₜ／lᵣ)²]`

其中：

- `Δpₜ` 是 TCP 平移变化；
- `Δθₜ` 是两个 orientation 之间的 geodesic rotation angle；
- `lₚ` 是平移特征尺度；
- `lᵣ` 是旋转特征尺度。

构造样本权重：

`wₜ ∝ ε + clip(Δsₜ, 0, s_max)`

并在每条 demonstration、每个阶段内归一化：

`Σₜ wₜ = 1`

其中：

- `ε` 保留少量时间占用信息，避免完全忽略真正稳定维持的状态；
- clip 防止单次追踪跳变或异常 pose 获得过大权重；
- orientation-only motion 仍然通过 `Δθₜ` 获得权重；
- 纯静止重复点只获得很小权重。

优先实现加权 likelihood，而不是物理删除或重采样。这样可以保留原始数据与时间戳，调试也更容易。

### 4.3 Demonstration 之间：从 hard vote 改为 soft evidence pooling

对 demonstration `d` 和候选模式 `m`，先计算归一化模式损失：

`ℓ_d(m, η) = Σ_t w_dt · [−log p_m(x_dt | η)]`

定义相对 inactive 的证据：

`e_d(m, η) = ℓ_d(inactive) − ℓ_d(m, η)`

解释如下：

- `e_d > 0`：该 demonstration 更支持模式 `m`；
- `e_d < 0`：该 demonstration 更支持 inactive；
- `|e_d|`：保留证据强度。

跨 demonstration 不再做硬多数投票，而是使用 robust soft aggregation：

`E(m, η) = RobustMean_d[e_d(m, η)]`

第一版可以比较：

- arithmetic mean；
- trimmed mean；
- Huber mean；
- median。

建议从 Huber mean 或小比例 trimmed mean 开始。median 非常稳健，但在 demonstration 数量只有 5 时可能过于离散。

最终模式选择可以写成：

`Score(m) = −max_η E(m, η) + ComplexityPenalty(m)`

选择 Score 最小的模式。

这套方案相对于 hard vote 的优势是：

- 每条 demonstration 等权；
- 保留每条 demonstration 的证据强度；
- 单条异常 demonstration 不容易主导；
- shared boundary 仍然通过所有 demonstration 联合优化；
- 可以直接画出每条 demonstration 的 `e_d`，解释 shared mode 为什么成立。

### 4.4 Shared parameter 的第一版方案

第一版继续使用严格共享参数：

- equality 使用一个 shared target `η`；
- inequality 使用一个 shared boundary `η`；
- 每条 demonstration 只允许有自己的 background location、scale 和 trend nuisance parameters。

这样模型仍然明确区分：

- 任务级约束参数；
- demonstration 级执行变化。

### 4.5 可选升级：Hierarchical shared boundary

如果实验发现不同 demonstration 对同一个合理约束存在持续的小幅边界偏差，可以考虑：

`η_d = η_shared + u_d`

`u_d ∼ N(0, τ_η²)`

其中：

- `η_shared` 是最终报告的任务约束；
- `u_d` 描述 demonstration 的保守程度或校准误差；
- `τ_η` 描述跨 demonstration 的边界离散程度。

这一方案比“每条 demonstration 独立参数后取 median”更系统，但会增加优化和 identifiability 复杂度。因此只作为第二阶段研究，不应优先于 demo 内加权和 soft evidence pooling。

## 5. 研究问题二：Background Model 应该使用什么

### 5.1 Background 的职责

inactive/background 应当解释：

- 普通的自由变化；
- demonstration 间不同的局部中心和尺度；
- 简单的接近、撤离和姿态调整趋势；
- 少量追踪异常或执行异常。

但它不应：

- 使用任意窄尺度模仿 equality；
- 使用过强的平滑器解释所有结构；
- 吸收真实、跨 demonstration 一致的支持边界；
- 成为一个无法解释的 universal model。

### 5.2 推荐模型：带收缩线性趋势的 Student-t

建议第一版 background 使用：

`x_dt = a_d + b_d · s_dt + ε_dt`

`ε_dt ∼ StudentT(0, σ_d, ν_bg)`

`b_d ∼ N(0, τ_b²)`

其中：

- `s_dt ∈ [0, 1]` 是当前阶段的 normalized progress；
- `a_d` 是 demonstration 级局部中心；
- `b_d` 是 demonstration 级趋势；
- `σ_d` 是 demonstration 级 background scale；
- Student-t 噪声降低少量异常值的影响；
- `b_d` 的零均值收缩先验抑制不必要的趋势。

这个设计仍然只有一个 inactive/background 模式：

- 当没有明显趋势时，`b_d ≈ 0`，模型退化为原来的 constant background；
- 当存在稳定的阶段趋势时，`b_d ≠ 0`；
- 输出中仍然只显示 inactive，不额外增加 constant/trend 两个模式。

### 5.3 防止 background 挤占 equality

保留 background scale floor：

`σ_d ≥ c_bg · σ_eq`

同时 equality 继续使用共享窄目标模型：

`x_dt ∼ N(η_shared, σ_eq²)`

或者使用固定尺度的 Student-t equality。

因此，即使 background 可以拟合趋势，也不能通过收缩到任意小的 `σ_d` 来模仿 equality。真正集中在共同目标附近的特征仍应由 equality 获胜。

### 5.4 Inactive、lb、ub 应共享同一个基础趋势模型

如果只有 inactive 能拟合趋势，而 inequality 仍然只能使用无趋势的单边分布，那么 background 可能反过来吞掉“趋势 + 真边界”同时存在的数据。

更公平的设计是让 inactive、lb、ub 共享基础分布：

`μ_d(s) = a_d + b_d · s`

inactive：

`x_dt ∼ StudentT(μ_d(s_dt), σ_d, ν_bg)`

lower bound：

`p_lb(x_dt) ∝ StudentT(x_dt | μ_d(s_dt), σ_d, ν_bg) · B((x_dt − η)／κ)`

upper bound：

`p_ub(x_dt) ∝ StudentT(x_dt | μ_d(s_dt), σ_d, ν_bg) · B((η − x_dt)／κ)`

其中：

- `B(·)` 是 soft barrier；
- `η` 是 shared task boundary；
- `κ` 控制边界 softness；
- 三种模式都能解释同样的简单趋势；
- `lb/ub` 只有在额外的共同支持边界确实改善预测时才会获胜。

这样比较的核心变成“是否需要边界”，而不是“单边模型和常数背景谁更擅长拟合趋势”。

### 5.5 暂时不使用 spline 或 Gaussian Process

自由 spline、Gaussian Process 或高阶多项式可能过于强大：

- 容易吸收 equality；
- 容易吸收真实边界附近的结构；
- 小数据下复杂度难以可靠估计；
- 结果会依赖平滑超参数；
- 与阶段切分联合优化时容易产生新的 local optimum。

当前 S3/S4 的主要错误由近似单调趋势产生。带收缩的一阶趋势已经足够作为第一版。

只有在去除线性趋势后，background residual 仍表现出稳定且跨 demonstration 一致的非线性结构时，才考虑二次项或低自由度 spline。

## 6. 模式先验与复杂度惩罚

当前 activation prior 为 0.3，可以作为 baseline，但不应成为唯一的防误判机制。

建议遵循以下原则：

- 保持统一 activation prior，不为具体 feature 指定允许模式；
- 不使用 `obstacle 只能 lb`、`plane error 只能 eq` 之类的硬先验；
- 将先验用于表达“约束应该稀疏”，而不是表达任务答案；
- 用 held-out evidence 或 inactive null simulation 校准 active mode 的最小证据要求。

可以从 inactive background 拟合后生成无约束模拟数据，重复计算：

`Δ_null = Score(inactive) − min[Score(lb), Score(ub)]`

使用 `Δ_null` 的高分位数估计 inequality 假阳性阈值。这样阈值会自动考虑：

- demonstration 数量；
- 每条 demonstration 的有效样本数；
- background scale；
- 边界优化带来的额外自由度；
- soft barrier 的归一化优势。

这比手工不断调整 `map_c_ineq` 更有统计意义。`map_c_ineq` 本质上控制 inequality scale floor，并不是明确的模式复杂度惩罚。

## 7. 建议的分阶段实现路线

### Phase 0：冻结当前 baseline

保留：

- `map_pooled`；
- `map_balanced_pooled`；
- activation prior = 0.3；
- 当前 fixed-cutpoint Bar Inspection 结果。

不要覆盖已有输出，使后续消融实验能够直接比较。

### Phase 1：只修改聚合，不修改概率模型

实现：

1. SE(3) motion weight；
2. demonstration 内 weighted mean NLL；
3. 跨 demonstration soft evidence；
4. mean、Huber、trimmed mean 三种聚合器；
5. 每个 stage-feature 输出 per-demo evidence table。

这一阶段仍使用当前 constant Gaussian background 和现有 inequality，目的是单独验证聚合方式的作用。

### Phase 2：只修改 background

在 demo-balanced soft pooling 下比较：

1. constant Gaussian；
2. constant Student-t；
3. shrinkage linear Gaussian；
4. shrinkage linear Student-t。

重点检查 S3 obstacle 和 S4 surface distance 的 false inequality 是否消失，以及 S2/S3 equality 是否仍然保留。

### Phase 3：统一 inactive/lb/ub 的趋势基础

让 inactive、lb、ub 使用相同的 `μ_d(s)` 和 residual family，只通过 soft boundary 区分。

这一阶段重点研究：

- inequality normalization 是否仍产生结构性优势；
- shared boundary 是否具有 held-out predictive gain；
- true inequality 与单纯阶段趋势是否能够区分。

### Phase 4：可选 hierarchical boundary

只有在严格 shared boundary 明显限制合理的跨 demonstration 变化时再实现。

## 8. 消融实验设计

聚合轴：

- hard vote；
- raw pooled；
- demo-balanced pooled；
- demo-balanced + SE(3) weighting；
- robust soft evidence pooling。

Background 轴：

- constant Gaussian；
- constant Student-t；
- shrinkage linear Gaussian；
- shrinkage linear Student-t；
- unified trend base for inactive/lb/ub。

优先使用 fixed cutpoints 分离“约束识别问题”和“分段问题”。约束模型稳定后，再恢复 joint segmentation。

## 9. 必须通过的稳定性测试

### 9.1 采样率不变性

将同一数据转换为 5 Hz、10 Hz、20 Hz。shared mode 和参数不应因为重复采样密度改变。

### 9.2 静止复制不变性

人为复制 1–3 秒纯静止数据。结果不应产生新的 equality 或 inequality。

### 9.3 Time-warp 不变性

对同一几何轨迹做局部时间拉伸和压缩。约束模式应保持稳定。

### 9.4 Orientation-only 测试

构造位置不变但 orientation 持续变化的片段。SE(3) weighting 应保留该片段的信息，纯 translation weighting 则会失败。

### 9.5 Leave-one-demo-out

每次留下 4 条 demonstration 训练，检查：

- shared mode 是否稳定；
- shared target/boundary 的方差；
- 被留下 demonstration 的预测成本；
- 是否由某一条 demonstration 单独推动模式。

### 9.6 Prior sensitivity

至少测试 activation prior：

`0.20 / 0.25 / 0.30 / 0.35 / 0.40 / 0.50`

真正稳定的约束不应只在某一个 prior 值出现。结果应报告模式稳定区间，而不是只展示最符合预期的单次结果。

### 9.7 无约束趋势模拟

生成只有上升或下降趋势、没有支持边界的数据。测试 inequality false-positive rate。

### 9.8 合成约束恢复

在模拟数据中明确生成：

- equality；
- lower bound；
- upper bound；
- trend only；
- trend + lower bound；
- trend + upper bound。

只有合成实验中的生成约束才作为严格 ground truth。真实机器人实验主要用于语义合理性、稳定性和迁移验证。

## 10. 评估指标

建议至少报告：

- mode stability across prior；
- mode stability across sample rate；
- leave-one-demo-out agreement；
- false inequality rate on trend-only data；
- shared parameter dispersion；
- held-out weighted NLL；
- per-demo evidence distribution；
- segmentation stability；
- transfer planning success。

真实机器人数据不应只报告与手工 `true_constraints` 的 classification accuracy，因为手工约束集合可能不完整。

## 11. 当前建议的算法主线

最终建议可以概括为：

`SE(3) motion-aware sample weighting`

→ `per-demo normalized likelihood evidence`

→ `robust soft aggregation across demonstrations`

→ `shared task-level target or boundary`

配合：

`shrinkage linear Student-t background`

并让：

`inactive / lb / ub`

共享同一个趋势基础，只通过是否存在 shared support boundary 来区分。

当前 `demo_balanced_pooled + activation_prior 0.3` 是一个有用 baseline，但还不是最终算法。最优先实现的是：

1. demo 内 SE(3) 加权；
2. 跨 demo soft evidence pooling；
3. 收缩线性 Student-t background；
4. inactive/lb/ub 共享趋势基础。

hierarchical boundary、非线性 spline 和更复杂的 Bayesian marginalization 应当放在这些基础消融完成之后。

## 12. 实验命令约定

所有正式实验统一通过 benchmark runner 启动，不使用 `run_one.py`：

```bash
/home/baiyu/miniforge3/envs/segment/bin/python runners/run_benchmark.py \
  --methods <method_name> \
  --datasets BarInsepect \
  --method-seeds 0
```

每个新聚合器或 background variant 使用独立 method name 和独立输出目录，避免覆盖 baseline。
