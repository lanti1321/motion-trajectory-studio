#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 LEFT_CAN RIGHT_CAN BITRATE DBITRATE" >&2
  exit 2
fi

LEFT_CAN="$1"
RIGHT_CAN="$2"
BITRATE="$3"
DBITRATE="$4"

if [[ ! "${LEFT_CAN}" =~ ^[A-Za-z0-9_.:-]+$ ]] ||
   [[ ! "${RIGHT_CAN}" =~ ^[A-Za-z0-9_.:-]+$ ]] ||
   [[ ! "${BITRATE}" =~ ^[0-9]+$ ]] ||
   [[ ! "${DBITRATE}" =~ ^[0-9]+$ ]]; then
  echo "invalid CAN configuration arguments" >&2
  exit 2
fi

configure_can() {
  local iface="$1"
  ip link show "${iface}" >/dev/null
  ip link set "${iface}" down
  ip link set "${iface}" type can bitrate "${BITRATE}" dbitrate "${DBITRATE}" fd on
  ip link set "${iface}" up
}

configure_can "${RIGHT_CAN}"
[[ "${LEFT_CAN}" == "${RIGHT_CAN}" ]] || configure_can "${LEFT_CAN}"
