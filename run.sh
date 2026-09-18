#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
exec uv run --no-sync python -m zbot_mpc_mjbatch.stepping "$@"
