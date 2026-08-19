#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "Missing .env. Copy .env.example to .env and verify the interface and IPs." >&2
    exit 1
fi
source .env

: "${IIWA14_IFACE:?IIWA14_IFACE is required}"
: "${FRI_HOST_IP:?FRI_HOST_IP is required}"
: "${IIWA14_ROBOT_IP:?IIWA14_ROBOT_IP is required}"

container=stage_cons_iiwa14

if docker ps --format '{{.Names}}' | grep -qx 'kuka14'; then
    echo "Refusing to connect while the old dual-arm kuka14 container is running." >&2
    exit 1
fi
if ! docker ps --format '{{.Names}}' | grep -qx "${container}"; then
    echo "${container} is not running. Run ./scripts/start.sh first." >&2
    exit 1
fi
if ! ip link show "${IIWA14_IFACE}" >/dev/null 2>&1; then
    echo "Robot interface not found: ${IIWA14_IFACE}" >&2
    exit 1
fi

if ! ip -4 addr show dev "${IIWA14_IFACE}" | grep -Fq "${FRI_HOST_IP}/24"; then
    echo "Adding ${FRI_HOST_IP}/24 to host interface ${IIWA14_IFACE}." >&2
    if [[ ${EUID} -eq 0 ]]; then
        ip addr add "${FRI_HOST_IP}/24" dev "${IIWA14_IFACE}"
    else
        sudo ip addr add "${FRI_HOST_IP}/24" dev "${IIWA14_IFACE}"
    fi
fi
if [[ ${EUID} -eq 0 ]]; then
    ip link set "${IIWA14_IFACE}" up
else
    sudo ip link set "${IIWA14_IFACE}" up
fi
ping -I "${IIWA14_IFACE}" -c 2 -W 1 "${IIWA14_ROBOT_IP}"

echo "Robot NIC configured on the host. The host-network container can now reach it."
echo "This does not start iiwa_driver or any torque controller."
