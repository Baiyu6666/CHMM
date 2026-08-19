#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "Missing .env. Copy .env.example to .env and verify OptiTrack settings." >&2
    exit 1
fi
source .env

GUI_PORT="${STAGE_DEMO_GUI_PORT:-8081}"

: "${OPTITRACK_SERVER:?OPTITRACK_SERVER is required}"
: "${OPTITRACK_BASE:?OPTITRACK_BASE is required}"
: "${OPTITRACK_OBJECT:?OPTITRACK_OBJECT is required}"

if ! docker ps --format '{{.Names}}' | grep -qx 'stage_cons_iiwa14'; then
    echo "stage_cons_iiwa14 is not running. Run ./scripts/start.sh first." >&2
    exit 1
fi

echo "Demo GUI: http://127.0.0.1:${GUI_PORT}"
echo "The page can start without iiwa_driver; hardware readiness stays false until valid joint states arrive."
echo "Experimental position-reference hold is disabled; Demo mode gates acquisition and assistance only."
exec docker exec -it stage_cons_iiwa14 /entrypoint.sh \
    roslaunch stage_demo_gui demo_station.launch \
    "optitrack_server:=${OPTITRACK_SERVER}" \
    "optitrack_base:=${OPTITRACK_BASE}" \
    "optitrack_object:=${OPTITRACK_OBJECT}" \
    "gui_port:=${GUI_PORT}"
