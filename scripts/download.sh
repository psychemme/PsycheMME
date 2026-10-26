#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OVERRIDES=()
for name in DATA_DIR EGO4D_ACCESS_KEY_ID EGO4D_SECRET_ACCESS_KEY EGO4D_SESSION_TOKEN EGO4D_PROFILE EGO4D_SOURCE_DIR; do
  if [[ "${!name+x}" == x ]]; then OVERRIDES+=("$name=${!name}"); fi
done
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
for value in "${OVERRIDES[@]-}"; do
  if [[ -n "$value" ]]; then export "$value"; fi
done
psychemme download --bundle benchmark --output "${DATA_DIR:-data/PsycheMME}" "$@"
