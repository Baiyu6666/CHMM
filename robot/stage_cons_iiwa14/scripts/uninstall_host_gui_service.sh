#!/bin/bash
set -euo pipefail

unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
unit_file="${unit_dir}/stage-cons-supervisor.service"

systemctl --user disable --now stage-cons-supervisor.service 2>/dev/null || true
if [[ -f "${unit_file}" ]]; then
    rm "${unit_file}"
fi
systemctl --user daemon-reload
echo "Host GUI service removed. Docker containers were not changed."
