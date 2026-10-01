#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
python_command=python
if [[ "$(uname -s)" == Darwin ]]; then
  python_command=mjpython
fi
exec uv run --no-sync "$python_command" -m zbot_mpc_mjbatch.double_leg --viewer "$@"
