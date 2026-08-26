#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

if docker ps --format '{{.Names}}' | grep -qx 'kuka14'; then
    echo "Refusing to start: the old dual-arm kuka14 container is still running." >&2
    echo "Stop the old stack first; two FRI clients must not target the same robot." >&2
    exit 1
fi

export USER_UID="$(id -u)"
mkdir -p data/demos data/models data/real_runs
docker compose build
docker compose up -d
docker compose ps

echo "Container started on the host network so VRPN/Motive UDP can reach it."
echo "No robot driver, OptiTrack client, or recorder was started automatically."
echo "After the hardware-limit fault is professionally recovered, configure the robot NIC explicitly with:"
echo "  ./scripts/connect_robot_network.sh"
