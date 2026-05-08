python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods fchmm,arhsmm,cluster \
  --datasets S3ObsAvoid,S4SlideInsert,S5SphereInspect \
  --method-seeds 0,1,2,3,4,5,6,7,8,9 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/new_s5

python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py   --methods fchmm --datasets S5SphereInspectRaw   --method-seeds 0   --dataset-seed 0   --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/new_s5   --n_demos 1   --plot_every 10

python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods fchmm,arhsmm,cluster \
  --datasets S5SphereInspectRaw \
  --method-seeds 0,1,2,3,4,5,6,7,8,9 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/new_s5

  python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods swcl \
  --datasets S5SphereInspectRaw \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/new_s5


  python /home/baiyu/PycharmProjects/LearnStageConstraint/experiments/plot_benchmark_comparison.py \
  --input /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/new_s5/benchmark_results.json

  python experiments/collect_swcl_paper_figures.py  --demo-map S3ObsAvoid:7,S4SlideInsert:2,S5SphereInspect:5


python experiments/render_s5_demonstrations.py   --n-demos 1   --seed 7   --demo-indices 0   --gui 2   --fps 10 --outdir outputs/swcl/s5_planned_render

python experiments/render_s5_planned_trajectory.py   --constraints-json outputs/swcl/S5SphereInspectRaw/method_seed_000/constraints.json   --gui 1   --fps 15   --outdir outputs/swcl/s5_planned_render