#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "Missing .env. Copy .env.example to .env and verify OptiTrack settings." >&2
    exit 1
fi
source .env

: "${OPTITRACK_SERVER:?OPTITRACK_SERVER is required}"
: "${OPTITRACK_BASE:?OPTITRACK_BASE is required}"
: "${OPTITRACK_OBJECT:?OPTITRACK_OBJECT is required}"
: "${OPTITRACK_OBSTACLE:?OPTITRACK_OBSTACLE is required}"

if ! docker ps --format '{{.Names}}' | grep -qx 'stage_cons_iiwa14'; then
    echo "stage_cons_iiwa14 is not running. Run ./scripts/start.sh first." >&2
    exit 1
fi

exec docker exec -it stage_cons_iiwa14 /entrypoint.sh \
    roslaunch stage_demo_recorder demo_collection.launch \
    "optitrack_server:=${OPTITRACK_SERVER}" \
    "optitrack_base:=${OPTITRACK_BASE}" \
    "optitrack_object:=${OPTITRACK_OBJECT}" \
    "optitrack_obstacle:=${OPTITRACK_OBSTACLE}"
