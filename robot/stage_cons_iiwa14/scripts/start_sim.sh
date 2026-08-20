#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

export USER_UID="$(id -u)"
mkdir -p data/demos data/models data/sim_runs

docker compose \
    -p stage_cons_iiwa14_sim \
    -f compose.yaml \
    -f compose.sim.yaml \
    build
docker compose \
    -p stage_cons_iiwa14_sim \
    -f compose.yaml \
    -f compose.sim.yaml \
    up -d
docker compose \
    -p stage_cons_iiwa14_sim \
    -f compose.yaml \
    -f compose.sim.yaml \
    ps

echo "PyBullet simulation started without the FRI driver or OptiTrack."
echo "Trajectory data: $(pwd)/data/sim_runs"
echo "Follow logs with: ./scripts/logs_sim.sh"
