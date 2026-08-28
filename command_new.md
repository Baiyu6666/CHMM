# 常用命令（当前版本）

所有 Python 命令默认从项目根目录运行：

```bash
cd /home/baiyu/LearnStageConstraints
conda activate segment
```

## 1. 处理 demonstration 数据

### 新 rosbag：自动处理并打开 Matplotlib reviewer

每个新数据集使用一个新的 `dataset-dir`：

```bash
python experiments/bar_clean_data_pipeline.py run \
  robot/stage_cons_iiwa14/data/demos/BarClean/新录包目录/demo.bag \
  --dataset-dir robot/stage_cons_iiwa14/data/processed/新数据集名称
```

打开后直接在 GUI 中完成以下操作：

- 拖动 `Start、CP1–CP4、End`；松开鼠标自动保存草稿。
- 使用 `Previous/Next` 检查所有 demo。
- 检查完成后点击 `Confirm all`。
- 需要设为默认训练数据时勾选 `Activate`，然后点击 `Export`。

### 继续上次保存的审核

当前 `demo_fixed_scene3`：

```bash
python experiments/bar_clean_data_pipeline.py review \
  robot/stage_cons_iiwa14/data/processed/demo_fixed_scene3
```

`review` 会读取该数据集的 `review.json`。不要重新运行 `run/prepare`，否则不会继续当前审核状态。

### 确实需要清空并重新自动处理

警告：`--overwrite` 会删除该 `dataset-dir` 中已有的人工审核和导出结果。

```bash
python experiments/bar_clean_data_pipeline.py run \
  robot/stage_cons_iiwa14/data/demos/BarClean/20260828T071134_960601Z_demo_fixed_scene3/demo.bag \
  --dataset-dir robot/stage_cons_iiwa14/data/processed/demo_fixed_scene3 \
  --overwrite
```

## 2. 训练 MAP balanced pooled

使用 `Activate` 选中的 BarClean 数据，只运行一个 method seed：

```bash
python runners/run_benchmark.py \
  --methods map_balanced_pooled \
  --datasets BarClean \
  --method-seeds 0 \
  --outdir outputs/benchmark/barclean_fixed_scene3_map_balanced_pooled
```

主要输出：

```text
outputs/map_balanced_pooled/BarClean/method_seed_000/learned_constraints.json
outputs/benchmark/barclean_fixed_scene3_map_balanced_pooled/benchmark_results.json
outputs/benchmark/barclean_fixed_scene3_map_balanced_pooled/benchmark_results.csv
```

继续相同 benchmark，并跳过数据 fingerprint 一致的已完成结果：

```bash
python runners/run_benchmark.py \
  --methods map_balanced_pooled \
  --datasets BarClean \
  --method-seeds 0 \
  --outdir outputs/benchmark/barclean_fixed_scene3_map_balanced_pooled \
  --resume
```

## 3. 快速检查当前默认数据

```bash
jq '{n_demos, source_demo_ids, processed_demo_path}' configs/envs/BarClean.json
```

## 4. 机器人工作站和录包

```bash
cd /home/baiyu/LearnStageConstraints/robot/stage_cons_iiwa14
```

构建并启动工作站：

```bash
./scripts/start.sh
```

机器人 FRI 正处于激活装填或运动状态时，不要执行构建或重启。

配置机器人网卡：

```bash
./scripts/connect_robot_network.sh
```

启动现场 demonstration 录制 GUI：

```bash
./scripts/start_demo_gui.sh
```

停止工作站：

```bash
./scripts/stop.sh
```

## 5. 仿真

以下命令从 `robot/stage_cons_iiwa14` 目录运行：

```bash
./scripts/start_sim.sh
./scripts/logs_sim.sh
./scripts/stop_sim.sh
```
