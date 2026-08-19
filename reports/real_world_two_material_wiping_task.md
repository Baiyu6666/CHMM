# Task 1：插座避让与双区域压缩量擦拭任务



## 0. practical
- 需要搞一个ee，把海绵固定上去
- 需要一个合适的钢条或者别的什么bar，呈现出不同的结构
- 需要研究如何检测钢条的位置，以及障碍物的位置。


## 1. 任务描述

机器人使用固定在末端的海绵完成以下连续任务：

- 将湿海绵移动到擦拭起点，但不能从插座上方经过；
- 带标签或保护膜的区域，只能轻压擦拭，避免损伤或翘起；
- 带顽固污渍的裸露区域，需要更深地压缩海绵才能一次擦除。

该任务只用于采集 demonstration 和学习约束，不要求机器人根据 learned constraints 复现。

## 2. 实验场景

擦拭区域位于同一平面。在海绵的初始位置与擦拭起点之间放置一个插座，使二者之间的直线路径穿过插座禁湿区：

```text
海绵初始位置          插座禁湿区          擦拭起点
     ●  -------------  (插座)  -------------  ●
       \_____________________________________/
                    合法绕行路径

擦拭起点 | 保护膜/标签区域 | 顽固污渍区域 | 终点
              轻压               深压
```

Stage 1 的起点、终点和插座位置应固定，并满足：不考虑插座约束时的最短直线路径一定穿过禁湿区。因此，`socket_distance` 下界会在 demonstration 中实际改变轨迹，而不是一个始终不会触发的冗余约束。

建议使用：

- 一块平整的木板、塑料板或金属板；
- 一个断电的插座、插线板或插座模型；
- 可替换的透明保护膜、木纹贴膜或 vinyl sticker；
- 干燥白板笔、水洗颜料或粉笔浆制作标准化污渍。

实验中不要在带电插座附近使用湿海绵。插座只用于表达真实的“电气部件不能沾水”语义，应保持断电，并在其周围标出清晰的圆形禁湿区。

两个擦拭区域不能有明显高度差，demonstrator 在交界处不改变 yaw，也不进行额外横向转弯。

## 3. End Effector

使用刚性背板固定海绵：

```text
机器人法兰
    |
刚性背板
    |
海绵
```

不要松散地绑一块布，否则工具尺寸和海绵自由厚度难以稳定测量。

在海绵侧面可以增加毫米刻度或颜色标记，使轻压和深压在视频中直接可见。

## 4. Feature

在整个擦拭区域建立统一坐标系：

```text
u：沿整条擦拭路径的位置
v：相对共同中心线的横向位置
backing_height：刚性背板到材料表面的法向距离
compression：海绵压缩量
socket_distance：海绵中心在平面上的投影到插座中心的距离
```

Compression 由以下关系计算：

```text
compression = sponge_free_thickness - backing_height
```

代码可以直接计算 `backing_height`，论文和图表中使用更直观的 `compression`。

不使用 force、yaw、normal angle 或 `surface_distance=0` 作为 ground-truth constraints。

## 5. 四个 Stage

### Stage 1：绕开插座到达擦拭起点

Human 将湿海绵移动到保护膜区域起点。由于插座位于直线路径上，海绵必须从禁湿区外侧绕行。

```text
Goal：到达保护膜区域起点上方
Progress：到擦拭起点的距离
Constraints：
    socket_distance >= D_socket_safe
```

其中：

```text
D_socket_safe = 插座禁湿区半径 + 海绵半径
```

建议使用圆形海绵，使安全距离不依赖 yaw。这个约束不是机械碰撞或机器人 workspace 限制：海绵在几何上可以从插座上方经过，但任务语义禁止湿海绵进入该区域。
由于距离按海绵中心在平面上的投影计算，即使将海绵抬高后飞越插座，也仍然属于违反约束。

### Stage 2：轻压擦拭保护膜区域

```text
Goal：到达两个区域的交界处
Progress：u
Constraints：
    v = center
    compression <= C_delicate_max
```

`C_delicate_max` 是保护膜或标签允许的最大海绵压缩量。

### Stage 3：在交界处增加压缩量

工具在区域交界处保持横向位置不变，并向下移动刚性背板，使海绵达到清除顽固污渍所需的压缩状态。

```text
Goal：compression 达到 C_stain_min
Progress：compression
Constraints：
    u = junction
    v = center
```

这个阶段避免要求 compression 在 Stage 2 和 Stage 4 之间瞬间跳变。

### Stage 4：深压擦除顽固污渍

```text
Goal：到达污渍区域终点
Progress：u
Constraints：
    v = center
    compression >= C_stain_min
```

`C_stain_min` 是一次擦除达到预设效果所需的最小海绵压缩量。

## 6. 约束矩阵

| Feature | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---:|---:|---:|---:|
| `socket_distance` | lower bound | inactive | inactive | inactive |
| `u` | inactive | progress | equality | progress |
| `v` | inactive | equality | equality | equality |
| `compression` | inactive | upper bound | progress/goal | lower bound |

Stage 1 学习任务语义上的禁湿区域；Stage 2 和 Stage 4 的几何路径相同，主要区别来自 compression constraint；Stage 3 是有明确任务语义的压缩模式切换阶段。

## 7. Oracle 标定

### 7.1 插座安全距离

测量插座周围标记的圆形禁湿区半径和圆形海绵半径，得到：

```text
socket_distance >= D_socket_safe
```

该值是实验预先规定且可直接测量的 task-specified oracle。学习算法可以使用 `socket_distance` 这个 feature，但不能预先获得 `D_socket_safe`。

### 7.2 保护膜最大压缩量

独立测试不同压缩量，观察保护膜是否出现翘起、压痕或表面损伤。将最大安全压缩量定义为：

```text
compression <= C_delicate_max
```

如果材料没有稳定的物理损伤阈值，可以在实验 protocol 中明确规定一个最大允许压缩量。此时应称为 task-specified oracle，而不是材料的固有物理阈值。

### 7.3 污渍最低有效压缩量

使用相同海绵和相同污渍，测试不同压缩量下的单次擦除效果。将能够稳定达到预设擦除比例的最小值定义为：

```text
compression >= C_stain_min
```

`C_delicate_max` 和 `C_stain_min` 之间应留出明显间隔，避免机器人定位噪声和海绵变形导致两个阶段重叠。

## 8. Demonstration 采集

- 使用 human kinesthetic teaching；
- Stage 1 从固定初始位置开始，并从禁湿区两侧展示不同的合法绕行轨迹；
- Stage 1 的部分 demonstrations 应靠近禁湿区边界，否则很难准确识别下界；
- Human 主要控制沿路径方向的运动和背板高度；
- 在海绵侧面显示压缩刻度，或在界面中实时显示 compression；
- Stage 2 在合法侧展示不同程度的轻压；
- Stage 4 在合法侧展示不同程度的深压；
- 在区域交界处不要额外改变 yaw、横向位置或速度；
- 使用相同型号、相同干湿状态的海绵；
- 每轮实验前重新测量海绵自由厚度；
- 海绵出现永久压缩后及时更换。

如果 demonstrations 只围绕两个窄目标值对称波动，数据更支持 equality，而不是 inequality。正式采集前应先通过 pilot demonstrations 检查 compression 分布是否具有预期的单侧边界。

## 9. Ground-Truth Cutpoints

```text
Stage 1 -> Stage 2：绕过插座并到达保护膜区域的擦拭起点
Stage 2 -> Stage 3：工具中心到达区域交界并停止沿 u 方向运动
Stage 3 -> Stage 4：compression 达到顽固污渍擦拭要求并重新开始沿 u 方向运动
```

Cutpoints 可以通过 calibrated geometry、compression 和沿路径速度共同标注。

## 10. 评价指标

- 四阶段 segmentation accuracy；
- feature activation accuracy；
- equality、upper-bound 和 lower-bound mode accuracy；
- `u`、`v` equality 参数误差；
- `D_socket_safe` 参数误差；
- `C_delicate_max` 和 `C_stain_min` 参数误差；
- demonstration 中的 constraint violation；
- 保护膜损伤情况和污渍擦除比例，作为辅助任务质量指标。

可以增加两项 ablation：移除 `socket_distance` 后，Stage 1 的最短路径会直接经过插座；移除 compression 后，模型应更难区分 Stage 2、Stage 3 和 Stage 4。

该实验只说明算法能够从受控真实 demonstrations 中恢复给定 feature library 内的 stage-wise constraints，不应宣称 learned constraints 已足以完成自主清洁。
