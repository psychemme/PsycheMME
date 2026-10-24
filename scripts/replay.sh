#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/sync_resources.py logs
psychemme replay --root data/PsycheMME-logs --output runs/paper_replay.json
