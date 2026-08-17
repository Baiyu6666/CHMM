# Real-World 插座插入实验构想

## 实验目标

使用真实但完全断电的日常插头和插座，验证算法能否从真实机器人 demonstrations 中识别插入任务的阶段和几何约束，并将学到的约束用于后续插入。

插座必须与市电物理隔离并固定在实验台上。实验只研究机械插入，不涉及通电。

## 任务阶段

1. **Approach**：机器人从任意初始位置移动到插座上方。
2. **Pre-Insertion Alignment**：在入口上方保持固定距离，调整横向位置和插头姿态。
3. **Straight Insertion**：保持对准，沿插座轴向低速插入，达到目标深度后停止。

## Feature List

- `axial_gap`：插头尖端到插座入口平面的轴向距离。
- `center_err`：插头中心轴与插座中心轴的横向距离。
- `axis_angle_err`：插头轴与插座轴之间的夹角。
- `key_yaw_err`：插头防呆方向与插座方向之间的旋转误差。
- `speed`：末端平移速度，可作为 inactive 或辅助 feature。
- `start_dist`：到轨迹起点的距离，作为 progress/irrelevant feature。

这些 feature 由机器人 FK、已标定的插座位姿和插头 TCP 直接计算，不需要复杂视觉系统。

## 预期约束

- Alignment：`axial_gap = h`，并保持较小的角度误差。
- Insertion：`center_err ≤ c`、`axis_angle_err ≤ α`、`key_yaw_err ≤ ψ`。
- `axial_gap` 和 insertion depth 在插入阶段主要表示 progress，不作为持续约束。

几何 ground truth 可由插头/插座尺寸、机械间隙和独立标定获得。为辨识 upper-bound，demonstrations 应覆盖多个安全的横向和角度偏差，而不是每次都完美居中。

## Force 的处理

Force 不加入学习 feature list。插入力随深度、摩擦、速度和对准误差变化，不适合直接建模成单个 stage 内的固定 equality 或 inequality。

如果机器人能够估计外力或配有 F/T sensor，只将 force 用作独立安全信号：

- 横向力过高时停止；
- 轴向力较高且插入深度不再增加时判定为卡住；
- 达到最大深度、最大时间或 force safety envelope 时停止。

## 调试与正式实验

### 调试阶段

使用 CAD/oracle 几何阈值检查 pose，确认正常插入流程、标定和 force guard 均可工作。

### 正式实验

1. 使用 learned constraints 判断当前 pose 是否允许开始插入。
2. 保留一个更宽松的 hard pose safety envelope，防止明显危险的插入姿态。
3. 使用简单 position/velocity controller 沿插座轴向向下运动。
4. force guard 在插入过程中独立监控，异常时停止并撤回。

Safety layer 只负责阻止设备损坏，不自动修正 learned pose。被 hard safety 或 force guard 拒绝的执行均计为算法失败。

## 评价指标

- Cutpoint error。
- Semantic constraint F1。
- 几何 parameter error。
- 插入成功率。
- Learned-pose rejection、hard-safety rejection 和 force-abort 次数。
- 最大横向力、最大轴向力和 protective-stop 次数。

