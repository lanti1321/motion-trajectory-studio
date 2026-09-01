#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Compatibility launcher. The application is English-only; run.sh is identical.
exec bash "${ROOT}/run.sh" "$@"
