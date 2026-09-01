#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
video="${1:-${repo_root}/robot/final_video_runs/BarClean/Asss10/execution.mp4}"
output_dir="${2:-${repo_root}/figs}"

crop_width=1142
crop_height=747
crop_x=282
crop_y=183
output_width=1200
output_height=785

timestamps=(03 06.5 10.5 13.5)
output_names=(
    barclear_asss10_stage1.png
    barclear_asss10_stage2.png
    barclear_asss10_stage3.png
    barclear_asss10_stage4.png
)

if [[ ! -f "$video" ]]; then
    printf 'Video not found: %s\n' "$video" >&2
    exit 1
fi

mkdir -p "$output_dir"
filter="crop=${crop_width}:${crop_height}:${crop_x}:${crop_y},scale=${output_width}:${output_height}:flags=lanczos"

for index in "${!timestamps[@]}"; do
    ffmpeg -hide_banner -loglevel error -y \
        -ss "${timestamps[$index]}" \
        -i "$video" \
        -frames:v 1 \
        -vf "$filter" \
        -compression_level 3 \
        "${output_dir}/${output_names[$index]}"
done

printf 'Saved four snapshots to %s\n' "$output_dir"
