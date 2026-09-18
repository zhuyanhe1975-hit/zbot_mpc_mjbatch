#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
exec uv run --no-sync zbot-mpc-smoke "$@"
