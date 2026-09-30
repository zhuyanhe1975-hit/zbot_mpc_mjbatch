#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"

# Replay the saved measured trajectory; --recompute runs MPC again.
# Pass --viewer for an interactive MuJoCo window, or override any option,
# for example: ./replay_stepping.sh --seconds 5 --speed 0.5
recompute=0
if [[ "${1:-}" == --recompute ]]; then
  recompute=1
  shift
fi
for argument in "$@"; do
  case "$argument" in
    --frequency*|--seconds*|--candidates*|--iterations*|--threads*|--seed*|--rl-prior*|--no-rl-prior*)
      recompute=1 ;;
  esac
done
if [[ "$recompute" == 0 && -f dist/zbot_stable_stepping.npz ]]; then
  exec ./visualize.sh --replay dist/zbot_stable_stepping.npz \
    --output dist/zbot_stable_stepping.mp4 "$@"
fi
exec ./visualize.sh \
  --frequency 0.5 \
  --seconds 20 \
  --candidates 256 \
  --iterations 24 \
  --threads 8 \
  --seed 4 \
  --no-rl-prior \
  --trajectory dist/zbot_stable_stepping.npz \
  --output dist/zbot_stable_stepping.mp4 \
  "$@"
