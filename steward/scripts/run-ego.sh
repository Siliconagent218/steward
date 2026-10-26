#!/usr/bin/env bash
# Run the robot's Ego Harness on one task: scripts/run-ego.sh "把杯子放到桌上"
# Loads steward/.env (DEEPSEEK_API_KEY) and starts dsh in the robot's own empty workspace, so the
# agent's working directory holds no simulator data. Requires `steward up` (or `steward ego`).
set -euo pipefail

STEWARD="$(cd "$(dirname "$0")/.." && pwd)"
DSH="$STEWARD/dsh/node_modules/.bin/dsh"
HOME_DIR="${STEWARD_DSH_VAR:-$STEWARD/var/dsh}/robot-1"

if [ -f "$STEWARD/.env" ]; then
  set -a
  . "$STEWARD/.env"
  set +a
fi
mkdir -p "$HOME_DIR/workspace"
cd "$HOME_DIR/workspace"
exec env DSH_HOME="$HOME_DIR" "$DSH" --profile ego "$@"
