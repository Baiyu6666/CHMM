#!/usr/bin/env python3
"""Select accepted runs, render each profile clip, then concatenate in order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_RUN_ROOT = PROJECT_ROOT / "robot" / "final_video_runs"
REQUIRED_RUN_FILES = ("execution.mp4", "visualization.json", "metadata.json", "result.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render and concatenate selected accepted single-camera runs"
    )
    parser.add_argument(
        "--final-run-root", type=Path, default=DEFAULT_FINAL_RUN_ROOT
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list renderable final runs")
    list_parser.add_argument("--task", default=None)

    render = subparsers.add_parser(
        "render", help="render every selected run, then concatenate them"
    )
    render.add_argument("--task", required=True)
    render.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run directory IDs or unique prefixes, in final-video order",
    )
    render.add_argument(
        "--output",
        type=Path,
        default=None,
        help="final montage path; defaults to FINAL_RUN_ROOT/selected_runs.mp4",
    )
    render.add_argument("--start-pause-seconds", type=float, default=1.0)
    render.add_argument("--end-pause-seconds", type=float, default=1.0)
    render.add_argument("--fps", type=float, default=30.0)
    render.add_argument("--crf", type=int, default=15)
    render.add_argument("--panel-width-ratio", type=float, default=0.28)
    render.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="render only the start of each run; intended for layout debugging",
    )
    return parser


def discover_runs(final_run_root: Path, task: str | None = None) -> list[dict]:
    root = final_run_root.expanduser().resolve()
    task_directories = [root / task] if task else sorted(root.iterdir()) if root.is_dir() else []
    runs = []
    for task_directory in task_directories:
        if not task_directory.is_dir():
            continue
        for run_directory in sorted(task_directory.iterdir()):
            if not run_directory.is_dir():
                continue
            missing = [
                name for name in REQUIRED_RUN_FILES if not (run_directory / name).is_file()
            ]
            if missing:
                continue
            try:
                metadata = json.loads(
                    (run_directory / "metadata.json").read_text(encoding="utf-8")
                )
                video_metadata_path = run_directory / "execution_video_metadata.json"
                video_metadata = (
                    json.loads(video_metadata_path.read_text(encoding="utf-8"))
                    if video_metadata_path.is_file()
                    else {}
                )
            except (OSError, ValueError, TypeError):
                continue
            runs.append(
                {
                    "task": task_directory.name,
                    "id": run_directory.name,
                    "directory": str(run_directory),
                    "started_at_utc": metadata.get("started_at_utc"),
                    "constraint_source": metadata.get("constraint_source", "true"),
                    "width": video_metadata.get("width"),
                    "height": video_metadata.get("height"),
                    "fps": video_metadata.get("fps"),
                }
            )
    return runs


def resolve_selected_runs(
    available: Sequence[dict], selected_ids: Sequence[str]
) -> list[dict]:
    resolved = []
    used_ids = set()
    for requested in selected_ids:
        requested = str(requested).strip()
        exact = [item for item in available if item["id"] == requested]
        matches = exact or [item for item in available if item["id"].startswith(requested)]
        if not matches:
            raise RuntimeError(f"unknown final run: {requested}")
        if len(matches) > 1:
            raise RuntimeError(
                f"ambiguous final-run prefix {requested}: "
                + ", ".join(item["id"] for item in matches)
            )
        item = matches[0]
        if item["id"] in used_ids:
            raise RuntimeError(f"final run selected more than once: {item['id']}")
        used_ids.add(item["id"])
        resolved.append(item)
    return resolved


def _run_command(command: Sequence[str]) -> dict:
    print("$ " + " ".join(str(value) for value in command), flush=True)
    completed = subprocess.run(
        list(command),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    try:
        return json.loads(completed.stdout)
    except (ValueError, TypeError) as error:
        raise RuntimeError("video-processing command did not return JSON") from error


def render_selected(options: argparse.Namespace) -> dict:
    task = str(options.task).strip()
    if not task or Path(task).name != task or task in {".", ".."}:
        raise RuntimeError("unsafe task name")
    available = discover_runs(options.final_run_root, task)
    selected = resolve_selected_runs(available, options.runs)
    final_run_root = options.final_run_root.expanduser().resolve()
    renderer = Path(__file__).with_name("render_experiment_profiles.py")
    concat = Path(__file__).with_name("concat_trajectory_videos.py")
    clips = []
    render_results = []
    for item in selected:
        clip = Path(item["directory"]) / "execution_profiles.mp4"
        command = [
            sys.executable,
            str(renderer),
            "--run-directory",
            item["directory"],
            "--output",
            str(clip),
            "--fps",
            str(float(options.fps)),
            "--crf",
            str(int(options.crf)),
            "--panel-width-ratio",
            str(float(options.panel_width_ratio)),
        ]
        if options.max_seconds is not None:
            command.extend(["--max-seconds", str(float(options.max_seconds))])
        render_results.append(_run_command(command))
        clips.append(clip)

    final_output = (
        options.output.expanduser().resolve()
        if options.output is not None
        else final_run_root / "selected_runs.mp4"
    )
    concat_command = [
        sys.executable,
        str(concat),
        "--videos",
        *[str(path) for path in clips],
        "--output",
        str(final_output),
        "--start-pause-seconds",
        str(float(options.start_pause_seconds)),
        "--end-pause-seconds",
        str(float(options.end_pause_seconds)),
        "--fps",
        str(float(options.fps)),
        "--crf",
        str(int(options.crf)),
    ]
    concat_result = _run_command(concat_command)
    manifest = {
        "schema_version": 1,
        "task": task,
        "selected_runs": [item["id"] for item in selected],
        "clips": [str(path) for path in clips],
        "output": str(final_output),
        "start_pause_seconds": float(options.start_pause_seconds),
        "end_pause_seconds": float(options.end_pause_seconds),
        "fps": float(options.fps),
        "crf": int(options.crf),
        "renders": render_results,
        "concatenation": concat_result,
    }
    manifest_path = final_run_root / "render_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    manifest["manifest"] = str(manifest_path)
    return manifest


def main() -> None:
    options = build_parser().parse_args()
    if options.command == "list":
        result = {
            "final_run_root": str(options.final_run_root.expanduser().resolve()),
            "runs": discover_runs(options.final_run_root, options.task),
        }
    else:
        result = render_selected(options)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
