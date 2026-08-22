# LearnStageConstraint Commands


ssh -N -L 18080:127.0.0.1:8080   baiyu@128.178.145.250

http://127.0.0.1:18080

## Benchmarks: Original Synthetic Datasets

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods map \
  --datasets S3ObsAvoid,S4SlideInsert,S5SphereInspect  \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1 \
  --n-demos 10 \
  --refresh-demo-cache

```
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods map \
  --datasets S4SlideInsert  \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1 \
  --method-param save_paper_figures=true \
  --n-demos 10

python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods map \
  --datasets S3ObsAvoid  \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1 \
  --method-param save_paper_figures=true \
  --n-demos 10

python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods map \
  --datasets S5SphereInspect  \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1 \
  --method-param save_paper_figures=true \
  --n-demos 10


python runners/run_benchmark.py \
  --methods map \
  --datasets S3ObsAvoidReal \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir outputs/benchmark/s3_real_restore \
  --n-demos 4 \
  --method-param save_paper_figures=true

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods changeforest,arhsmm,cluster,gmmhmm \
  --datasets S3ObsAvoid,S4SlideInsert,S5SphereInspect  \
  --method-seeds 0,1,2,3,4,5,6,7,8,9 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1
```

## Benchmarks: S5SphereInspect

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py   --methods gmmhmm --datasets S5SphereInspect   --method-seeds 0   --dataset-seed 0   --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1   --n_demos 1   --plot_every 10
```

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods changeforest,arhsmm,cluster,gmmhmm \
  --datasets S5SphereInspect \
  --method-seeds 0,1,2,3,4,5,6,7,8,9 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1
```

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods map \
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
  --methods changeforest,arhsmm,cluster,gmmhmm \
  --datasets S4SlideInsert \
  --method-seeds 0,1,2,3,4,5,6,7,8,9 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/paper1
```

```bash
python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods map \
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

## S5 Render

```bash
python experiments/render_s5_demonstrations.py   --n-demos 5   --seed 127   --gui 1   --fps 10 --width 1360 --height 900 --outdir outputs/map/videos/s5_demonstrations --playback-speed 1.5   --render-frame-stride 1 

```

```bash
python experiments/render_s5_planned_trajectory.py \
  --constraints-json outputs/map/S5SphereInspect/method_seed_000/constraints.json \
  --constraint-source learned \
  --gui 1 \
  --fps 10 \
  --width 1360 \
  --height 900 \
  --outdir outputs/map/videos/s5_planned_render \
  --render-frame-stride 1 \
  --feature-overlay 1 \
  --n-plans 2 \
  --save-frame-indices 10,50,90,115,130 \
    --render-frame-stride 1

```

## S4 Demonstration Render

```bash
python experiments/render_s4_demonstrations.py   --n-demos 2   --seed 7   --outdir outputs/map/videos/s4_demonstrations   --gui 1   --fps 20   --camera-target 0.72,0.14,0.54   --feature-overlay 1 

```

```bash
python experiments/render_s4_planned_trajectory.py   --constraints-json outputs/map/S4SlideInsert/method_seed_000/constraints.json    --planner optimizer   --gui 1   --fps 20   --outdir outputs/map/video/s4_optimizer_gt_gui2   --constraint-source learned
```

## S4 Transfer Render

```bash
python experiments/render_s4_planned_trajectory.py \
  --constraints-json outputs/map/S4SlideInsert/method_seed_000/constraints.json \
  --constraint-source learned \
  --rail-shape sine \
  --rail-bend-amp 0.03 \
  --gui 1 \
  --fps 20 \
  --width 1360 \
  --height 900 \
  --camera-target 0.72,0.14,0.54 \
  --outdir outputs/map/videos/s4_transfer_curve_guide \
  --output-prefix s4_curve_guide \
  --video-filename s4_curve_guide.mp4 \
  --feature-overlay 1 \
  --execution-control torque_preload \
  --n-plans 1 \
  --render-frame-stride 1
#  --save-frame-indices 0,50,100,130 

```

```bash
python experiments/render_s4_planned_trajectory.py \
  --constraints-json outputs/map/S4SlideInsert/method_seed_000/constraints.json \
  --constraint-source learned \
  --rail-shape straight \
  --surface-tilt-x 0.2 \
  --gui 1 \
  --fps 20 \
  --outdir outputs/map/videos/s4_transfer_tilted_table \
  --output-prefix s4_tilted_table \
  --video-filename tilted_table.mp4 \
  --feature-overlay 1 \
  --n-plans 1 \
  --render-frame-stride 1

#  --save-frame-indices 0,50,100,130
```
