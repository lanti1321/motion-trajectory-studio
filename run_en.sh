#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOTION_STUDIO_LANG=en
exec bash "${ROOT}/run.sh" "$@"
