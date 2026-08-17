from __future__ import annotations

from pathlib import Path

import numpy as np

from .cache import (
    load_s4_demo_cache,
    make_s4_bundle,
    make_s4_cache_manifest,
    s4_demo_cache_path,
    save_s4_demo_cache,
)


def _observation_record(observation: dict, env) -> dict:
    trajectory = np.asarray(observation["trajectory"], dtype=float)
    measured_force = np.asarray(
        observation.get(
            "measured_normal_force_trace",
            observation.get("normal_force_trace", observation.get("normal_load_trace")),
        ),
        dtype=float,
    )
    planned_force = np.asarray(
        observation.get(
            "planned_normal_force_trace",
            observation.get("planned_normal_load_trace", measured_force),
        ),
        dtype=float,
    )
    commanded_force = np.asarray(observation.get("preload_command_trace", measured_force), dtype=float)
    record = {
        "trajectory": trajectory,
        "features": np.asarray(observation["features"], dtype=float),
        "stage_labels": np.asarray(observation["true_labels"], dtype=np.int64),
        "cutpoints": np.asarray(observation["true_cutpoints"], dtype=np.int64),
        "timestamps": np.arange(len(trajectory), dtype=float) * float(env.dt),
        "reference_trajectory": np.asarray(
            observation.get("planned_trajectory", observation.get("reference_trajectory", trajectory)),
            dtype=float,
        ),
        "planned_normal_force": planned_force,
        "commanded_normal_force": commanded_force,
        "measured_normal_force": measured_force,
    }
    optional_fields = {
        "contact_slider_trajectory": "contact_slider_trajectory",
        "joint_positions": "joint_positions",
        "joint_velocities": "joint_velocities",
        "joint_position_commands": "joint_position_commands",
        "joint_position_commands_nominal": "joint_position_commands_nominal",
        "joint_torque_commands": "joint_torque_commands",
        "execution_joint_noise": "execution_joint_noise",
        "preload_indent_trace": "preload_indent",
    }
    for observation_key, record_key in optional_fields.items():
        if observation.get(observation_key) is not None:
            record[record_key] = np.asarray(observation[observation_key])
    return record


def load_S4SlideInsert(
    n_demos: int = 10,
    seed: int = 123,
    env_kwargs=None,
    demo_kwargs=None,
    **extra_env_kwargs,
):
    from ..S4SlideInsert import S4SlideInsertEnv

    task_name = "S4SlideInsert"
    env_config = dict(env_kwargs or {})
    env_config.update(extra_env_kwargs)
    cache_demos = bool(env_config.pop("cache_demos", False))
    cache_dir = env_config.pop("demo_cache_dir", None)
    explicit_cache_path = env_config.pop("demo_cache_path", None)
    rollout_config = dict(demo_kwargs or {})
    env = S4SlideInsertEnv(**env_config)

    cache_path = None
    if cache_demos:
        cache_path = (
            Path(explicit_cache_path)
            if explicit_cache_path is not None
            else s4_demo_cache_path(
                task_name=task_name,
                seed=int(seed),
                env_config=env_config,
                rollout_config=rollout_config,
                cache_dir=cache_dir,
            )
        )
        cached_bundle = load_s4_demo_cache(
            cache_path=cache_path,
            task_name=task_name,
            n_demos=int(n_demos),
            seed=int(seed),
            env=env,
            rollout_config=rollout_config,
        )
        if cached_bundle is not None:
            return cached_bundle

    records = []
    scene_specs = []
    for demo_index in range(int(n_demos)):
        scene = env.sample_scene()
        scene["demo_index"] = int(demo_index)
        latent = env.rollout_demo(scene, seed=int(seed) + int(demo_index), **rollout_config)
        observation = env.compute_observation(latent, scene)
        records.append(_observation_record(observation, env))
        scene_specs.append(dict(scene))

    manifest = make_s4_cache_manifest(
        task_name=task_name,
        env=env,
        count=len(records),
        rollout_config=rollout_config,
    )
    bundle = make_s4_bundle(
        task_name=task_name,
        seed=int(seed),
        env=env,
        records=records,
        scene_specs=scene_specs,
        manifest=manifest,
        cache_path=cache_path,
        cache_hit=False,
    )
    if cache_path is not None:
        save_s4_demo_cache(
            cache_path=cache_path,
            bundle=bundle,
            env_config=env_config,
            rollout_config=rollout_config,
        )
        print(f"\033[31m[S4 demo cache] saved {len(records)} demos to {cache_path}\033[0m", flush=True)
    return bundle
