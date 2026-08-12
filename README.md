# LearnStageConstraint

Learning stage-wise active constraints from Expert demonstrations.

Author:
- Baiyu Peng, LASA, EPFL

Project summary:

Robotic demonstrations often contain multiple stages, and each stage may be shaped by a different subset of task constraints. Some constraints are equality-like, such as keeping a feature close to a target value, while others are inequality-like, such as staying above a safety margin or below a speed bound.

This repository studies how to jointly infer stage segmentation and stage-wise active constraints from demonstrations. The main method is Stage-Wise Constraint Learning (`SWCL`), which searches for stage boundaries while fitting feature-wise constraint models and selecting which constraints are active in each stage.

The learned constraints should be interpreted as demonstrated active constraints: features whose constraints visibly shape the observed behavior. Loose or inactive task constraints may be valid for the environment but are generally not identifiable from successful demonstrations alone.

This repository contains:
- the `SWCL` joint segmentation-and-constraint-learning method
- baseline segmentation pipelines with post-hoc constraint estimation
- synthetic and manipulation-inspired stage-wise environments
- benchmark runners, parameter-search utilities, and artifact saving
- visualization utilities for segmentation, activation masks, constraint margins, and paper figures
- demonstration and planned-trajectory rendering scripts for supported tasks

## Overview

Typical workflow:

1. choose a method and dataset
2. train with `runners/run_one.py` or `runners/run_benchmark.py`
3. inspect segmentation and constraint metrics
4. inspect saved plots and `constraints.json`
5. for supported environments, render demonstrations or planned trajectories
6. aggregate paper figures from saved SWCL outputs

## Repository Layout

- `methods/`: learning methods and wrappers
  - `methods/cores/swcl.py`: main SWCL implementation
  - `methods/cores/fchmm_core.py`: factorized constrained HMM baseline
  - `methods/cores/posthoc_constraint_model.py`: post-hoc constraint learner
  - `methods/wrappers/`: joint and sequential method wrappers
- `envs/`: benchmark environments and feature APIs
  - `S3ObsAvoid`: 3-stage obstacle-avoidance task
  - `S4SlideInsert`: 4-stage slide-and-insert task
  - `S5SphereInspect`: 5-stage sphere-inspection task
- `configs/`: method and environment config files
- `experiments/`: experiment orchestration, rendering, and figure utilities
- `evaluation/`: segmentation and constraint metrics
- `pipelines/`: joint and sequential experiment pipelines
- `planner/`: trajectory repair/refinement utilities
- `runners/`: CLI entrypoints for single runs, benchmarks, and parameter search
- `utils/`: feature-emission and constraint models
- `visualization/`: plotting and rendering helpers
- `outputs/`: generated results, plots, videos, and benchmark summaries

## Environment

The current development environment is the conda environment named `segment`.

```bash
conda activate segment
pip install -r requirements.txt
```

Core Python dependencies include:
- `numpy`
- `scipy`
- `matplotlib`
- `pybullet` for PyBullet-backed rendering and S5 rollouts

Notes:
- Some environments can run with analytic backends; PyBullet is still needed for the rendering scripts and PyBullet-backed S5 demonstrations.
- `outputs/` and `envs/demo_cache/` are generated artifacts and may be large.
- `requirements.txt` is generated from source imports with `pipreqs` and then adjusted to the versions used in the current `segment` environment.

## Configuration

Configs are layered in this order:

1. `configs/methods/<method>.json`
2. `configs/envs/<dataset>.json`
3. dataset-specific `method_overrides`
4. CLI overrides such as `--method-param key=value`

Supported method names are:
- `swcl`
- `fchmm`
- `hmm`
- `arhsmm`
- `changepoint`
- `cluster`

Supported dataset names are:
- `S3ObsAvoid`
- `S4SlideInsert`
- `S5SphereInspect`

Important SWCL config fields:
- `n_stages`: number of task stages
- `selected_raw_feature_ids`: feature subset used for learning
- `feature_model_types`: per-feature equality/inequality model family
- `feature_activation_mode`: activation-selection mode, usually `score`
- `truncated_z_score_mode`: explicit/auto lower-upper inequality score mode. `fast_fit` uses the half-t slack profile score; `fast_fit_minus_baseline` uses active NLL minus a Student-t baseline NLL; `soft_fit_minus_baseline` uses the soft-boundary half-t score from `s4_start_dist_inequality_cutpoint_test`.
- `truncated_z_soft_boundary_scale`: soft-boundary width as a multiple of the fitted half-t slack scale, used only by `soft_fit_minus_baseline`.
- `truncated_inequality_z_threshold`: inequality activation threshold. For `fast_fit_minus_baseline`, the score is active NLL minus baseline NLL, so negative thresholds require a stronger one-sided fit.
- `lambda_eq_constraint`: equality constraint weight
- `lambda_ineq_constraint`: inequality constraint weight
- `lambda_param_consensus`: cross-demo parameter-consensus weight
- `lambda_activation_consensus`: cross-demo activation-consensus weight
- `duration_min` / `duration_max`: optional stage duration bounds; use `null` when duration should not constrain learning
- `force_inactive_feature_ids`: keep selected features in reports while forcing their activation mask to zero
- `plot_every`: plot interval; `null` disables periodic plots

## Training and Evaluation

Run one experiment from config files:

```bash
python runners/run_one.py \
  --env-config configs/envs/S4SlideInsert.json \
  --method-config configs/methods/swcl.json \
  --method-seed 0 \
  --max-iter 10 \
  --output-root outputs
```

Run a benchmark:

```bash
python runners/run_benchmark.py \
  --methods swcl,fchmm,hmm,arhsmm,changepoint,cluster \
  --datasets S3ObsAvoid,S4SlideInsert,S5SphereInspect \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir outputs/benchmark/paper
```

Run a faster SWCL-only benchmark without plots:

```bash
python runners/run_benchmark.py \
  --methods swcl \
  --datasets S4SlideInsert \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir outputs/benchmark/s4_debug \
  --method-param disable_plots=true \
  --method-param plot_every=null
```

The benchmark runner prints the key metrics after each completed run:

- `MeanAbsCutpointError`: segmentation error
- `MeanConstraintError`: normalized constraint error
- `MeanConstraintErrorRaw`: raw constraint error

Per-run artifacts are saved under:

```text
outputs/<method>/<dataset>/method_seed_<seed>/
```

Important artifacts include:
- `config_snapshot.json`: resolved dataset and method config
- `metadata.json`: run metadata
- `metrics.json`: scalar and matrix metrics
- `segmentation.json`: true and predicted cutpoints
- `constraints.json`: true constraints, learned values, activation masks, and errors
- `objectives.json`: objective and loss history when available
- `*.png`: diagnostic and paper-style plots when enabled

## Parameter Search

Parameter grids are JSON files mapping dotted parameter paths to candidate lists. Example:

```json
{
  "lambda_ineq_constraint": [0.25, 0.5, 1.0],
  "truncated_inequality_z_threshold": [-0.5, -0.4, -0.3, -0.2, -0.1]
}
```

Run a parameter search:

```bash
python runners/run_param_search.py \
  --dataset S4SlideInsert \
  --method swcl \
  --grid-config configs/param_search_swcl_S4SlideInsert.json \
  --method-seeds 0 \
  --output-root outputs/param_search
```

By default, parameter search disables plots for speed unless full per-run artifacts are explicitly saved.

## Visualization and Paper Figures

SWCL can produce several diagnostic plots:
- activation masks and activation-rate matrices
- constraint margin heatmaps
- true constraint active masks
- key feature traces
- training summaries
- selected paper-style figures

Periodic plotting is controlled by `plot_every`. For expensive tuning runs, use:

```bash
--method-param disable_plots=true
--method-param plot_every=null
```

Collect the latest SWCL paper figures after running benchmarks:

```bash
python experiments/collect_swcl_paper_figures.py \
  --demo-map S3ObsAvoid:7,S4SlideInsert:2,S5SphereInspect:5
```

## Rendering and Planning

Rendering scripts are available for the S4 and S5 environments. Not every dataset or method has downstream planning support.

### S4SlideInsert

Render demonstrations:

```bash
python experiments/render_s4_demonstrations.py \
  --n-demos 1 \
  --seed 1342 \
  --outdir outputs/swcl/videos/s4_demonstrations \
  --gui 1
```

Render a planned trajectory from learned or target constraints:

```bash
python experiments/render_s4_planned_trajectory.py \
  --constraints-json outputs/swcl/S4SlideInsert/method_seed_000/constraints.json \
  --constraint-source learned \
  --planner optimizer \
  --gui 1 \
  --outdir outputs/swcl/videos/s4_planned_render
```

### S5SphereInspect

Render demonstrations:

```bash
python experiments/render_s5_demonstrations.py \
  --n-demos 1 \
  --seed 127 \
  --outdir outputs/swcl/videos/s5_demonstrations \
  --gui 1
```

Render a planned trajectory:

```bash
python experiments/render_s5_planned_trajectory.py \
  --constraints-json outputs/swcl/S5SphereInspect/method_seed_000/constraints.json \
  --constraint-source learned \
  --planner optimizer \
  --gui 1 \
  --outdir outputs/swcl/videos/s5_planned_render
```

### GUI Modes

For rendering scripts that expose `--gui`:

- `--gui 0`: no interactive GUI
- `--gui 1`: off-screen rendering with saved video outputs
- `--gui 2`: interactive PyBullet GUI for manual inspection

## Naming Conventions

- `S3ObsAvoid`: 3-stage obstacle-avoidance environment
- `S4SlideInsert`: 4-stage slide-insert environment
- `S5SphereInspect`: 5-stage sphere-inspection environment
- `swcl`: joint stage-wise constraint learning
- `fchmm`, `hmm`, `arhsmm`, `changepoint`, `cluster`: baseline segmentation pipelines with constraint evaluation
- `method_seed_<seed>`: method initialization or search seed
- `dataset_seed`: demonstration-generation seed from the environment config or CLI

## Interpretation Notes

This project learns constraints that are active in the demonstrations, not the complete feasible set of the task.

An inactive or loose inequality may be physically valid but statistically invisible if the demonstrations never approach its boundary. Conversely, short stages or progress-correlated features can create apparent one-sided feature distributions. SWCL therefore treats activation masks as evidence-based summaries of demonstrated behavior rather than complete task specifications.

Environment configs may include pragmatic experiment controls such as `force_inactive_feature_ids` when a feature should remain visible in reports but be excluded from activation.

## Reference Commands

Common benchmark command:

```bash
python runners/run_benchmark.py \
  --methods swcl,fchmm,hmm,arhsmm,changepoint,cluster \
  --datasets S3ObsAvoid,S4SlideInsert,S5SphereInspect \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir outputs/benchmark/paper
```

S4 SWCL quick run:

```bash
python runners/run_benchmark.py \
  --methods swcl \
  --datasets S4SlideInsert \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir outputs/benchmark/s4_quick \
  --method-param disable_plots=true \
  --method-param plot_every=null
```

S5 SWCL run with plots every 10 iterations:

```bash
python runners/run_benchmark.py \
  --methods swcl \
  --datasets S5SphereInspect \
  --method-seeds 0 \
  --dataset-seed 0 \
  --outdir outputs/benchmark/s5_debug \
  --method-param plot_every=10
```

Render S4 planned trajectory from target constraints:

```bash
python experiments/render_s4_planned_trajectory.py \
  --constraints-json outputs/swcl/S4SlideInsert/method_seed_000/constraints.json \
  --constraint-source target \
  --planner optimizer \
  --gui 1 \
  --outdir outputs/swcl/videos/s4_target_plan
```
