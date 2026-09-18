#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
if [[ -z "${MUJOCO_GL:-}" ]]; then
  case "$(uname -s)" in
    Darwin) export MUJOCO_GL=cgl ;;
    Linux) export MUJOCO_GL=egl ;;
  esac
fi
python_command=python
if [[ "$(uname -s)" == Darwin ]]; then
  for arg in "$@"; do
    if [[ "$arg" == --viewer ]]; then
      python_command=mjpython
      break
    fi
  done
fi
exec uv run --no-sync "$python_command" -m zbot_mpc_mjbatch.visualize "$@"
