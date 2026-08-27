#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "Missing .env. Copy .env.example to .env and verify OptiTrack settings." >&2
    exit 1
fi
source .env

SCENE_POSE_SOURCE="${SCENE_POSE_SOURCE:-fixed}"
[[ "${SCENE_POSE_SOURCE}" == "fixed" || "${SCENE_POSE_SOURCE}" == "optitrack" ]] || {
    echo "SCENE_POSE_SOURCE must be fixed or optitrack." >&2
    exit 1
}
OPTITRACK_SERVER="${OPTITRACK_SERVER:-128.178.145.104}"
OPTITRACK_BASE="${OPTITRACK_BASE:-iiwa14}"
OPTITRACK_OBJECT="${OPTITRACK_OBJECT:-baiyu_bar}"
OPTITRACK_OBSTACLE="${OPTITRACK_OBSTACLE:-baiyu_obs_bar}"
USE_FIXED_SCENE=false
[[ "${SCENE_POSE_SOURCE}" == "fixed" ]] && USE_FIXED_SCENE=true

if ! docker ps --format '{{.Names}}' | grep -qx 'stage_cons_iiwa14'; then
    echo "stage_cons_iiwa14 is not running. Run ./scripts/start.sh first." >&2
    exit 1
fi

exec docker exec -it stage_cons_iiwa14 /entrypoint.sh \
    roslaunch stage_demo_recorder demo_collection.launch \
    "use_fixed_scene:=${USE_FIXED_SCENE}" \
    "optitrack_server:=${OPTITRACK_SERVER}" \
    "optitrack_base:=${OPTITRACK_BASE}" \
    "optitrack_object:=${OPTITRACK_OBJECT}" \
    "optitrack_obstacle:=${OPTITRACK_OBSTACLE}"
