#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
template="${project_root}/systemd/stage-cons-supervisor.service.in"
unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
unit_file="${unit_dir}/stage-cons-supervisor.service"
python_executable="${STAGE_HOST_PYTHON:-/usr/bin/python3}"

if [[ ! -x "${python_executable}" ]]; then
    echo "Python executable not found: ${python_executable}" >&2
    echo "Set STAGE_HOST_PYTHON to an absolute Python 3 executable and retry." >&2
    exit 1
fi
if [[ ! -f "${template}" ]]; then
    echo "Missing service template: ${template}" >&2
    exit 1
fi

mkdir -p "${unit_dir}"
escaped_root=${project_root//&/\\&}
escaped_root=${escaped_root//|/\\|}
escaped_python=${python_executable//&/\\&}
escaped_python=${escaped_python//|/\\|}
sed \
    -e "s|@PROJECT_ROOT@|${escaped_root}|g" \
    -e "s|@PYTHON_EXECUTABLE@|${escaped_python}|g" \
    "${template}" > "${unit_file}"

# Make graphical authorization and BuildKit SSH forwarding available to the
# long-running user manager when these variables exist in this login session.
environment_names=()
for name in DISPLAY XAUTHORITY DBUS_SESSION_BUS_ADDRESS SSH_AUTH_SOCK; do
    if [[ -n "${!name:-}" ]]; then
        environment_names+=("${name}")
    fi
done
if ((${#environment_names[@]})); then
    systemctl --user import-environment "${environment_names[@]}"
fi

systemctl --user daemon-reload
systemctl --user enable --now stage-cons-supervisor.service

echo "Host GUI service installed and started."
echo "Open http://127.0.0.1:8080"
echo "The service does not start Docker or iiwa_driver automatically."
