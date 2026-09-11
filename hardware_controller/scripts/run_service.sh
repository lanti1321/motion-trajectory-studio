#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build"
BIN="${BUILD}/motion_studio_hardware_service"

if [[ ! -x "${BIN}" || "${ROOT}/src/main.cpp" -nt "${BIN}" ]]; then
  cmake -S "${ROOT}" -B "${BUILD}"
  cmake --build "${BUILD}" --target motion_studio_hardware_service -j2
fi

LEFT_CAN="${OPENARM_LEFT_FOLLOWER_CAN:-can1}"
RIGHT_CAN="${OPENARM_RIGHT_FOLLOWER_CAN:-can0}"
BITRATE="${OPENARM_CAN_BITRATE:-1000000}"
DBITRATE="${OPENARM_CAN_DBITRATE:-5000000}"

if [[ "${OPENARM_AUTO_CONFIG_CAN:-1}" == "1" ]]; then
  CONFIGURE_CAN="${ROOT}/scripts/configure_can.sh"
  if sudo -n true >/dev/null 2>&1; then
    sudo bash "${CONFIGURE_CAN}" "${LEFT_CAN}" "${RIGHT_CAN}" "${BITRATE}" "${DBITRATE}"
  elif command -v pkexec >/dev/null 2>&1; then
    pkexec bash "${CONFIGURE_CAN}" "${LEFT_CAN}" "${RIGHT_CAN}" "${BITRATE}" "${DBITRATE}"
  else
    echo "CAN configuration requires root; run this script from an interactive terminal." >&2
    exit 1
  fi
fi

exec "${BIN}" \
  --left-can "${LEFT_CAN}" \
  --right-can "${RIGHT_CAN}" \
  --command-port "${OPENARM_REPLAY_COMMAND_PORT:-47970}" \
  --status-port "${OPENARM_REPLAY_STATUS_PORT:-47971}"
