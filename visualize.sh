#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec uv run --no-sync python -m zbot_mpc_mjbatch.visualize "$@"
