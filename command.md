# LearnStageConstraint Commands

## Benchmarks: Original Synthetic Datasets

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods swcl \
  --datasets S3ObsAvoid,S4SlideInsert,S5SphereInspect  \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1 \
  --refresh-demo-cache

```
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods swcl \
  --datasets S4SlideInsert  \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1 

python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods swcl \
  --datasets S3ObsAvoid  \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1 

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods fchmm,arhsmm,cluster \
  --datasets S3ObsAvoid,S4SlideInsert,S5SphereInspect  \
  --method-seeds 0,1,2,3,4,5,6,7,8,9 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1
```

## Benchmarks: S5SphereInspect

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py   --methods fchmm --datasets S5SphereInspect   --method-seeds 0   --dataset-seed 0   --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1   --n_demos 1   --plot_every 10
```

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods fchmm,arhsmm,cluster \
  --datasets S5SphereInspect \
  --method-seeds 0,1,2,3,4,5,6,7,8,9 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1
```

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods swcl \
  --datasets S5SphereInspect \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1 \
  --plot-every 10 \
  --refresh-demo-cache
```

## Benchmarks: S4SlideInsert

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods fchmm,arhsmm,cluster \
  --datasets S4SlideInsert \
  --method-seeds 0,1,2,3,4,5,6,7,8,9 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1
```

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods swcl \
  --datasets S4SlideInsert \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1 \
  --plot-every 5 \
  --refresh-demo-cache
```

## Paper / Benchmark Plotting

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/experiments/plot_benchmark_comparison.py \
  --input /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1/benchmark_results.json
```

```bash
python experiments/collect_swcl_paper_figures.py  --demo-map S3ObsAvoid:7,S4SlideInsert:2,S5SphereInspect:5
```

## S5 Render

```bash
python experiments/render_s5_demonstrations.py   --n-demos 3   --seed 127   --gui 1   --fps 10 --width 1360 --height 900 --outdir outputs/swcl/videos/s5_demonstrations --playback-speed 1.5   --render-frame-stride 1 

```

```bash
python experiments/render_s5_planned_trajectory.py \
  --constraints-json outputs/swcl/S5SphereInspect/method_seed_000/constraints.json \
  --constraint-source learned \
  --gui 1 \
  --fps 10 \
  --width 1360 \
  --height 900 \
  --outdir outputs/swcl/videos/s5_planned_render \
  --render-frame-stride 1 \
  --feature-overlay 1 \
  --n-plans 2 \
  --save-frame-indices 10,50,90,115,130 \
    --render-frame-stride 20

```

## S4 Demonstration Render

```bash
python experiments/render_s4_demonstrations.py   --n-demos 2   --seed 7   --outdir outputs/swcl/videos/s4_demonstrations   --gui 1   --fps 20   --camera-target 0.72,0.14,0.54   --feature-overlay 1 

```

```bash
python experiments/render_s4_planned_trajectory.py   --constraints-json outputs/swcl/S4SlideInsert/method_seed_000/constraints.json    --planner optimizer   --gui 1   --fps 20   --outdir outputs/swcl/video/s4_optimizer_gt_gui2   --constraint-source learned
```

## S4 Transfer Render

```bash
python experiments/render_s4_planned_trajectory.py \
  --constraints-json outputs/swcl/S4SlideInsert/method_seed_000/constraints.json \
  --constraint-source learned \
  --rail-shape sine \
  --rail-bend-amp 0.03 \
  --gui 0 \
  --fps 20 \
  --width 1360 \
  --height 900 \
  --camera-target 0.72,0.14,0.54 \
  --outdir outputs/swcl/videos/s4_transfer_curve_guide \
  --feature-overlay 1 \
  --execution-control torque_preload \
  --n-plans 1 \
  --render-frame-stride 1
#  --save-frame-indices 0,50,100,130 

```

```bash
python experiments/render_s4_planned_trajectory.py \
  --constraints-json outputs/swcl/S4SlideInsert/method_seed_000/constraints.json \
  --constraint-source learned \
  --rail-shape straight \
  --surface-tilt-x 0.2 \
  --gui 1 \
  --fps 20 \
  --outdir outputs/swcl/videos/s4_transfer_tilted_table \
  --feature-overlay 1 \
  --n-plans 1 \
  --render-frame-stride 1

#  --save-frame-indices 0,50,100,130
```
