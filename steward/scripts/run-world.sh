#!/usr/bin/env bash
# World Harness launcher. Requires `steward up` or `steward world`; loads steward/.env.
#   scripts/run-world.sh                 browser UI on port 3091 (profile "world")
#   scripts/run-world.sh "<task>"        one-shot task (profile "world-task", headless)
set -euo pipefail

STEWARD="$(cd "$(dirname "$0")/.." && pwd)"
DSH="$STEWARD/dsh/node_modules/.bin/dsh"
HOME_DIR="${STEWARD_DSH_VAR:-$STEWARD/var/dsh}/world"

if [ -f "$STEWARD/.env" ]; then
  set -a
  . "$STEWARD/.env"
  set +a
fi
cd "$STEWARD"
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  exec env DSH_HOME="$HOME_DIR" "$DSH" --profile world-task "$@"
fi
exec env DSH_HOME="$HOME_DIR" "$DSH" --profile world "${@:---port 3091}"
