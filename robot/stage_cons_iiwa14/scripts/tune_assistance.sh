#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

container=stage_cons_iiwa14
namespace=/iiwa14/demo_virtual_fixture

if ! docker ps --format '{{.Names}}' | grep -qx "${container}"; then
    echo "${container} is not running." >&2
    exit 1
fi

usage() {
    echo "Usage:" >&2
    echo "  $0 orientation STIFFNESS DAMPING MAX_MOMENT MAX_SPEED_DEG_S" >&2
    echo "  $0 vertical DAMPING MAX_FORCE" >&2
    echo "Both assistance channels must be off. Values are checked by the node." >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
case "$1" in
    orientation)
        [[ $# -eq 5 ]] || usage
        docker exec "${container}" rosparam set "${namespace}/orientation_stiffness" "$2"
        docker exec "${container}" rosparam set "${namespace}/orientation_damping" "$3"
        docker exec "${container}" rosparam set "${namespace}/max_orientation_moment" "$4"
        docker exec "${container}" rosparam set "${namespace}/max_orientation_recovery_speed_deg_s" "$5"
        ;;
    vertical)
        [[ $# -eq 3 ]] || usage
        docker exec "${container}" rosparam set "${namespace}/vertical_damping" "$2"
        docker exec "${container}" rosparam set "${namespace}/max_vertical_force" "$3"
        ;;
    *)
        usage
        ;;
esac

docker exec "${container}" rosservice call "${namespace}/reload_tuning"
