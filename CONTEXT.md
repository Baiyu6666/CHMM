# Project Context

更新时间：2026-08-20

## 当前目标

这个项目正在把 stage-wise constraint learning 与实验室 iiwa14 工作站接到同一套可复现的执行流程中。当前工作的重点是
`robot/stage_cons_iiwa14`：

- 实验室电脑负责采集 demonstration、连接 OptiTrack，并在确认安全后控制真实 iiwa14；
- 笔记本使用相同 Git revision、Dockerfile 和 ROS workspace 运行 PyBullet 仿真；
- 两台电脑通过普通 Git 工作流同步代码，通过 Git LFS 同步 rosbag、模型和较大的仿真输出；
- 仿真和真机使用同一个网页 Task GUI、同一个 Cartesian planner 接口，以及同一个 Cartesian-to-joint trajectory 编译器；
- 标准 task 流程是：移动到 Start（不记录）→ 执行 Start 到 Goal 的 planner trajectory（可记录）→ 到达 Goal 后停止并保存。

机器人 workspace 的完整运行说明见
[`robot/stage_cons_iiwa14/README.md`](robot/stage_cons_iiwa14/README.md)。

## 当前系统架构

### Planner 边界

当前 `stage_placeholder_planner` 是一个接口占位符：

- 输入：
  - `/stage_cons/planner/start`
  - `/stage_cons/planner/goal`
- 服务：`/straight_line_planner/plan`
- 输出：`/stage_cons/plan`，类型为 `nav_msgs/Path`
- 路径内容：Cartesian position 与 quaternion orientation 的直线/球面线性插值采样
- planner 本身不做 IK、不切换 controller，也不直接命令仿真或真机。

将来 learned planner 可以替换这个 package，只要继续发布相同的 `/stage_cons/plan` 接口。

### 共享 Cartesian trajectory 编译器

`stage_cartesian_trajectory` 是仿真和真机共同使用的 Python/catkin package。它把 planner 的 Cartesian samples 转换成两个 timed joint segments：

1. `approach`：从当前关节状态移动到 planner 的 Start；
2. `task`：从 Start 执行到 Goal。

编译器统一负责：

- 位置 + Tool-Z orientation 约束；
- 从当前实际/模拟关节角开始的连续数值 IK；
- Tool-X 连续传递和冗余姿态分支选择；
- joint limit 与最大相邻关节步长检查；
- self-collision 检查；
- Jacobian minimum singular value 检查；
- 速度和加速度受限的时间参数化；
- approach/task duration 与 IK 误差指标。

因此 planner 输出的正式任务路径不再由 PyBullet 每个 step 临时求 IK。仿真和真机首先获得相同格式的 timed joint trajectory，然后交给不同的执行后端。

### PyBullet 执行后端

`stage_iiwa_sim`：

- 在隔离的 bridge-network Compose 模式中运行；
- 不启动 FRI driver、OptiTrack、virtual fixture 或真机 recorder；
- 场景包含 iiwa14、桌子、bar 和球形 obstacle；
- 桌面高度为 `0.14584 m`；
- bar 高度暂定为 `0.06 m`；
- 使用共享编译器生成 `approach` 和 `task` joint trajectories；
- 每个 physics step 根据共享时间戳插值关节目标，再使用 PyBullet `POSITION_CONTROL`；
- 只在到达 Start 后记录 task segment；
- 可保存 `metadata.json`、`trajectory.csv`、`goal_reaching.mp4` 和 ffmpeg 日志；
- CSV 包含实际 `q/dq`、共享轨迹的 `q_target`、末端/目标 pose 和碰撞计数。

仿真刚启动时，为了把机器人放到默认展示姿态，仍会单独调用一次 PyBullet IK；planner 输出的正式路径全部使用共享编译器。

### 真机执行后端

`stage_real_executor` 已实现，但尚未在物理机器人上验证：

- robot namespace：`/iiwa14`；
- bare flange tip：`iiwa14_link_7`，不使用 force sensor、grabber 或其他末端器；
- orientation：位置 + Tool-Z；
- IK seed：最新的 `/iiwa14/joint_states`；
- controller：`PositionTrajectoryController`；
- action：`/iiwa14/PositionTrajectoryController/follow_joint_trajectory`；
- 接收 `/stage_cons/plan` 不会自动运动；必须显式 `prepare` 后再 `execute`；
- `prepare` 要求 FRI `POSITION` + `COMMANDING_ACTIVE`，并运行全部轨迹检查；
- `execute` 前重新检查 current joint state 与 prepare 时的起点漂移；
- recording 只覆盖 task segment，输出 rosbag 和 metadata；
- abort、controller failure、heartbeat loss 或 external torque threshold 会取消目标并关闭/回退 position-command gate；
- protective stop 会锁存，需要检查现场并重启 real station。

主要服务：

- `/iiwa14/real_executor/validate`：只做离线 kinematic validation；
- `/iiwa14/real_executor/prepare`：检查硬件状态并准备轨迹；
- `/iiwa14/real_executor/execute`：显式开始已准备轨迹；
- `/iiwa14/real_executor/abort`：取消执行；
- `/iiwa14/real_executor/set_recording`：设置是否记录 task。

`iiwa_driver` 的 position command gate 默认关闭。只有 real executor 先同步 controller 当前关节角、启动 20 Hz heartbeat，并通过 0.5° 对齐检查后，driver 才允许 position command。heartbeat 超过 0.2 秒未更新时，driver 自动回退到 measured joint position。

## 网页 GUI

宿主机 supervisor：<http://127.0.0.1:8080>

Task GUI 当前支持：

- 设置 Start/Goal：`x y z qx qy qz qw`；
- 选择 `PyBullet simulator` 或 `Real iiwa14`；
- 设置是否 recording；
- 自动执行 planner、approach、task 和 finalize；
- 显示 `starting/planning/moving_to_start/executing/complete/failed/aborted`；
- 显示保存目录；
- 仿真完成后播放 MP4；
- 提供 abort 控件；
- 真机执行前要求浏览器中的显式安全确认。

真机模式不会复用已经运行的 SafeTorque demo driver。如果检测到已有 iiwa driver、仿真容器仍在运行，或机器人网络未配置，它会拒绝启动 real station。

## Demonstration 采集

Kinesthetic demonstration 仍使用 `SafeTorqueController` 和 `inspection_virtual_fixture`：

- Demo 模式本身不锁住机器人；
- orientation assistance 与 vertical damping 可独立启用；
- 旧的 demo position-reference hold 已移除；
- 新增的 gated position command 只属于独立 real task executor，不能从 Demo 模式启用；
- recorder 保存 joint state、TF、OptiTrack、assistance 状态、markers 和 metadata。

不要同时运行 Demo station 和 real task station。

## Git、Docker 与两台电脑

推荐工作流：

```bash
git pull --rebase
cd robot/stage_cons_iiwa14
./scripts/start.sh
```

仿真使用同一个 Dockerfile/image，只更换 Compose override：

```bash
cd robot/stage_cons_iiwa14
./scripts/start_sim.sh
./scripts/logs_sim.sh
# 查看完成状态后按 Ctrl+C 退出日志
./scripts/stop_sim.sh
```

`.env` 保持每台电脑本地独立，不提交到 Git。实验室电脑保存真实 interface/IP；笔记本不需要复制实验室网络设置。

Git LFS 当前配置覆盖：

- `*.bag`
- `*.pt`、`*.pth`、`*.ckpt`、`*.onnx`
- 仿真 `trajectory.csv`
- 仿真 `goal_reaching.mp4`

安装了 Git LFS 后，日常仍然使用普通 `git add`、`git commit`、`git push` 和 `git pull`。不需要把 `git lfs ls-files` 放进日常流程。

## 已完成验证

最新共享轨迹版本已经完成：

- Python syntax check；
- `git diff --check`；
- Docker image build；
- catkin 构建全部 12 个 ROS packages；
- PyBullet 完整离线 task：17 个 Cartesian/task joint samples、15 cm、约 6.5 秒，最终状态 `complete`；
- 仿真最大 IK position error 约 `0.086 mm`；
- 仿真最大 Tool-Z error 约 `0.00042°`；
- 仿真最大 joint step 约 `0.0358 rad`；
- 仿真 minimum Jacobian singular value 约 `0.2066`；
- real executor 使用同一个编译器完成独立离线 validate；
- real validate 最大 IK position error 约 `0.062 mm`，task duration 同样约 `6.5 s`。

没有连接、启动或运动真实机器人。真机链路仍需要现场低速、低刚度、空工作区和可用 E-stop 条件下的分阶段验证。

## 当前工作区状态

本 session 的 ROS/Docker/GUI/driver 修改尚未 commit。重要新增目录包括：

- `robot/stage_cons_iiwa14/ros_ws/src/stage_cartesian_trajectory/`
- `robot/stage_cons_iiwa14/ros_ws/src/stage_iiwa_sim/`
- `robot/stage_cons_iiwa14/ros_ws/src/stage_real_executor/`
- `robot/stage_cons_iiwa14/compose.sim.yaml`
- `robot/stage_cons_iiwa14/scripts/start_sim.sh`
- `robot/stage_cons_iiwa14/scripts/logs_sim.sh`
- `robot/stage_cons_iiwa14/scripts/stop_sim.sh`

`robot/stage_cons_iiwa14/data/sim_runs/` 中有之前保留的仿真输出。提交前需要人工决定是否把示例 run 一起提交。

本机的新 Docker image 已构建，但当前长期运行的 `stage_cons_iiwa14` 容器没有被强制重建，以免中断已有环境。使用 GUI 的构建/启动操作，或重新 Compose，才会让该容器加载最新镜像。

## 推荐下一步

1. 查看完整 `git diff`，决定是否提交现有 `data/sim_runs` 示例。
2. 在浏览器中人工检查 Task GUI 的布局、状态显示和确认交互。
3. commit 并 push 当前统一轨迹版本，在另一台电脑重新 build/up 做仿真复现。
4. 真机无故障且现场安全时，先只验证 joint state、FRI mode、controller synchronization 和 `validate/prepare`，再进行很小范围、低速 execute。
5. 物理验证完成前，不把当前 real executor 当作已验证的机器人控制系统。
6. 之后用 learned planner 替换 placeholder，同时保持 `/stage_cons/plan` contract。

## 新 session 的继续方式

在另一台电脑或新的 Codex session 中：

```bash
git pull --rebase
git status
```

然后告诉 Codex：

> 先阅读 AGENTS.md、CONTEXT.md 和 robot/stage_cons_iiwa14/README.md，再检查 git status 和最近 commit，继续 CONTEXT.md 中的下一步。
