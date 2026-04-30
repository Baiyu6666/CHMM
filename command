python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods fchmm,arhsmm,cluster \
  --datasets S3ObsAvoid,S4SlideInsert,S5SphereInspect \
  --method-seeds 0,1,2,3,4,5,6,7,8,9 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/more_features

python /home/baiyu/PycharmProjects/LearnStageConstraint/runners/run_benchmark.py \
  --methods swcl \
  --datasets S3ObsAvoid,S4SlideInsert,S5SphereInspect \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/more_features

  python /home/baiyu/PycharmProjects/LearnStageConstraint/experiments/plot_benchmark_comparison.py \
  --input /home/baiyu/PycharmProjects/LearnStageConstraint/outputs/benchmark/more_features/benchmark_results.json

  python experiments/collect_swcl_paper_figures.py  --demo-map S3ObsAvoid:7,S4SlideInsert:2,S5SphereInspect:5