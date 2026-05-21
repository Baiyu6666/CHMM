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
python experiments/render_s5_demonstrations.py   --n-demos 3   --seed 127   --gui 1   --fps 10 --outdir outputs/swcl/videos/s5_demonstrations --playback-speed 1.5
```

```bash
python experiments/render_s5_planned_trajectory.py \
  --constraints-json outputs/swcl/S5SphereInspect/method_seed_000/constraints.json \
  --constraint-source target \
  --gui 1 \
  --fps 10 \
  --outdir outputs/swcl/videos/s5_planned_render \
  --render-frame-stride 1 \
  --feature-overlay 1 \
  --save-frame-indices 10,50,90,115,130 
```

## S4 Demonstration Render

```bash
python experiments/render_s4_demonstrations.py   --n-demos 1   --seed 7   --outdir outputs/swcl/videos/s4_demonstrations   --gui 0   --fps 20    --feature-overlay 0  --save-frame-indices 0,50,100,130 

```

```bash
python experiments/render_s4_planned_trajectory.py   --constraints-json outputs/swcl/S4SlideInsert/method_seed_000/constraints.json    --planner optimizer   --gui 1   --fps 20   --outdir outputs/swcl/video/s4_optimizer_gt_gui2   --constraint-source learned
```

## S4 Transfer Render

```bash
python experiments/render_s4_planned_trajectory.py \
  --constraints-json outputs/swcl/S4SlideInsert/method_seed_000/constraints.json \
  --constraint-source target \
  --rail-shape sine \
  --rail-bend-amp 0.03 \
  --gui 0 \
  --fps 20 \
  --outdir outputs/swcl/videos/s4_transfer_curve_guide \
  --feature-overlay 0 \
    --n-plans 3 \
  --save-frame-indices 0,50,100,130 

```

```bash
python experiments/render_s4_planned_trajectory.py \
  --constraints-json outputs/swcl/S4SlideInsert/method_seed_000/constraints.json \
  --constraint-source target \
  --rail-shape straight \
  --surface-tilt-x 0.2 \
  --gui 0 \
  --fps 20 \
  --outdir outputs/swcl/videos/s4_transfer_tilted_table \
  --feature-overlay 0 \
    --n-plans 3 \
  --save-frame-indices 0,50,100,130
```

## Paper Wording: Active Constraints

在 paper 里应该把 claim 收窄成 active / behavior-shaping constraints。这不是坏事，反而更严谨。

可以这样写：

> We aim to infer stage-wise constraints that are active in, and therefore shape, the demonstrated behavior. In this sense, feature relevance is evidence-based: a feature is considered relevant for a stage when the demonstrations exhibit statistical signatures consistent with an equality or inequality constraint on that feature.

然后马上说明不可辨识性：

> Constraints that are valid for the task but remain inactive in the demonstrations are generally not identifiable from observation alone. For example, a loose upper bound on speed may be physically valid, but if the demonstrated motion never approaches the bound, the data cannot distinguish this constraint from an unconstrained low-speed behavior induced by the stage geometry.

关于 relevant feature，可以写成：

> Therefore, our learned relevance masks should be interpreted as identifying features whose constraints are active in the demonstrations, rather than enumerating all possible task constraints.

如果你要放在 limitations：

> A limitation of our formulation is that inactive or weakly active inequality constraints may be classified as irrelevant, even when they are valid task constraints. This is an inherent ambiguity in learning constraints from demonstrations: without interventions, failed attempts, or demonstrations near the boundary, loose constraints leave little statistical evidence in the observed trajectories.

更简洁的版本可以是：

> Our method recovers demonstrated active constraints, not the complete feasible set.

这句话我建议一定放进去。它能避免 reviewer 说“你没有 recover 所有 constraints”。你的回答就是：我们本来学的是 demonstrations 中 shaping behavior 的 stage-wise constraints。
