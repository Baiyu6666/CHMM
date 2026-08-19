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
  - `BarInsepect`: 4-stage real-task prototype for obstacle-aware bar inspection
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
- `map`
- `map_pooled`
- `map_balanced_pooled`
- `map_balanced_vote`
- `fchmm`
- `gmmhmm`
- `hmm`
- `arhsmm`
- `changepoint`
- `changeforest`
- `cluster`

Supported dataset names are:
- `BarInsepect`
- `S3ObsAvoid`
- `S4SlideInsert`
- `S5SphereInspect`

`BarInsepect` is the four-stage prototype for the physical steel-bar task. Its
default learner uses obstacle clearance, EE-to-calibrated-table distance, tool pitch,
and tool-axis plane error. Motion/bar-axis alignment, lateral centerline offset,
speed, and angular speed are diagnostic columns only. The pose state
uses `[x, y, z, qx, qy, qz, qw]`; the feature extractor can additionally bind
time-aligned `baiyu_bar` and `baiyu_obs_ball` OptiTrack pose traces. The former
provides the bar-local `+X` direction, while the latter supplies the center of a
0.10 m-diameter infinite vertical obstacle cylinder at every sample. The
checked-in synthetic demonstrations exercise the API
until recorded demonstrations are converted into the same pose/OptiTrack
representation.

Convert a recorded BarInsepect bag into synchronized EE, bar, obstacle, and
feature arrays with:

```bash
python experiments/extract_bar_inspection_rosbag.py INPUT.bag OUTPUT.npz
```

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

Important MAP config fields:
- `map_mode_aggregation`: MAP M-step mode selection. `pooled` fits and scores candidates by summed point-wise NLL; `shared_vote` retains point-pooled candidate fits but gives each demo one mean-NLL vote; `demo_balanced_pooled` fits and scores candidates by the sum of per-demo mean NLLs; `demo_balanced_vote` combines demo-balanced candidate fits with strict-majority voting.
- `map_pooled` is the controlled MAP ablation with the same dataset settings and deterministic initialization but `map_mode_aggregation=pooled`
- `map_balanced_pooled` and `map_balanced_vote` inherit each dataset's `map` override automatically and change only the M-step aggregation.
- `map_activation_prior`: feature-wise vector of `P(active)` values, shared by all stages; its mode cost is split evenly across demo vote scores under voting aggregation and applied once under direct pooled/balanced aggregation
- `map_active_mode_prior`: `eq`, `lb`, and `ub` feature-wise vectors for `P(mode | active)`; the three probabilities for each feature must sum to one
- `map_mstep_boundary_trim`: samples removed on both sides of each internal recovered stage boundary only when fitting/voting shared MAP modes and parameters; DP segmentation still scores the full intervals
- `map_demo_num_workers`: number of MAP demo-segmentation worker processes; `null` uses up to one worker per demo, while `1` disables multiprocessing
- `map_progress_kappa`: `null` fits one nonnegative goal-progress concentration per stage; a nonnegative scalar fixes the same concentration for every stage and demo; `0` disables MAP progress
- `map_progress_kappa_max`: numerical upper bound used only while fitting stage-wise progress concentrations
- MAP uses unweighted likelihoods; SWCL's equality/inequality weights do not apply to MAP

Important post-hoc constraint config fields for `gmmhmm`, `hmm`, `arhsmm`, `changepoint`, `changeforest`, and `cluster`:
- `posthoc_training_mode`: `swcl` keeps the original fixed-mask/fixed-mode per-demo fit followed by a coordinate-wise parameter median; `pooled` runs MAP candidate selection with summed NLL; `voting` runs the same MAP shared-candidate demo voting used by joint MAP. ARHSMM and Cluster default to `voting`.
- `fixed_feature_mask` and `feature_model_types`: used by `swcl`; `pooled` and `voting` infer inactive/equality/lower/upper modes instead
- Baseline post-hoc learners automatically inherit MAP likelihood, prior, feature-selection, and trimming parameters from the dataset's `map` override; explicit CLI method overrides still take precedence
- `fchmm` trains its factorized constraint emissions internally and does not use the post-hoc learner
- `gmmhmm` is a mode-agnostic left-to-right HMM with a three-component diagonal GMM emission per stage; its configured voting post-hoc is a controlled MAP-decoder segmentation ablation
- `changeforest` is a fixed-K adaptation of changeforest: it repeatedly applies the upstream classifier gain to the best current interval until exactly `n_stages` ordered segments are obtained. It uses the same velocity-only representation and shallow random-forest settings across all datasets; significance testing is disabled because K is supplied to every baseline.

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
  --methods swcl,map,map_pooled,gmmhmm,fchmm,hmm,arhsmm,changepoint,changeforest,cluster \
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
- `SemanticConstraintF1`: exact `(stage, feature, mode)` recovery F1, excluding inactive true negatives
- `MeanParameterError`: normalized parameter error over constraints whose active mode is correctly identified
- `MeanParameterErrorRaw`: raw parameter error over constraints whose active mode is correctly identified

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

S5 cache files with `cache_version >= 20` are self-describing standalone NPZ datasets. They can be read without importing this project:

```python
import json
import numpy as np

with np.load("seed_127_....npz", allow_pickle=False) as data:
    manifest = json.loads(str(data["dataset_manifest_json"].item()))
    demo_index = 0
    position = data[f"position_{demo_index:03d}"]
    tool_axis = data[f"tool_axis_{demo_index:03d}"]
    goal_position = data[f"goal_position_{demo_index:03d}"]
    timestamps = data[f"timestamps_{demo_index:03d}"]
    features = data[f"features_{demo_index:03d}"]
    stage_labels = data[f"stage_labels_{demo_index:03d}"]
    cutpoints = data[f"cutpoints_{demo_index:03d}"]
```

The manifest records units, coordinate frame, stage semantics, feature schema and extractor version, constraints, required fields, and optional robot-state fields. Training uses the materialized `features` arrays by default; recomputation is reserved for explicit dataset migration or verification. Per-demo sampling and PyBullet acceptance metadata are stored in `demo_metadata_json`.

The S5 implementation is split by responsibility under `envs/s5/`: `task.py`, `generator.py`, `time_parameterization.py`, `execution.py`, `planner.py`, `features.py`, `dataset.py`, and `rendering.py`. Geometry is supplied to one fixed-step time-parameterizer; stage taper, correlated variation, and the existing Gaussian valleys are explicit speed-intent inputs. The valleys are retained as deliberate slowdown events rather than being embedded implicitly in geometry generation. `envs/S5SphereInspect.py` remains a compatibility import façade. Formal loader defaults come from the single `S5_SYNTHETIC_V23` dataclass preset in `envs/s5/config.py`. Its stage-2 tool-normal error uses randomly spaced control points plus smoothing, then quantile-matches the v20 near-bound marginal distribution. Subsequent orientation stages sample after their shared boundary pose, matching the position concatenation semantics and avoiding artificial zero angular-speed samples. `S5_SYNTHETIC_V20`, `S5_SYNTHETIC_V21`, and `S5_SYNTHETIC_V22` remain available for frozen and intermediate regression.

Verify the time-parameterization semantics and confirm that the frozen v20 dataset plus analytic generator retain their expected fingerprints:

```bash
python experiments/check_s5_v20_regression.py
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
- `map`: joint MAP stage-wise constraint learning with automatic feature modes
- `map_pooled`: MAP ablation using pooled point-wise M-step aggregation
- `map_balanced_pooled`: MAP ablation using summed per-demo mean NLLs for fitting and mode selection
- `map_balanced_vote`: MAP ablation using demo-balanced fitting followed by strict-majority mode voting
- `swcl`: joint stage-wise constraint learning
- `gmmhmm`, `fchmm`, `hmm`, `arhsmm`, `changepoint`, `changeforest`, `cluster`: baseline segmentation pipelines with constraint evaluation
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
  --methods swcl,map,map_pooled,gmmhmm,fchmm,hmm,arhsmm,changepoint,changeforest,cluster \
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
