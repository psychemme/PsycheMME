#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
: "${MODEL_BASE_URL:?Set MODEL_BASE_URL in .env}"
: "${MODEL_ID:?Set MODEL_ID in .env}"
: "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL in .env}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL in .env}"
DATA_DIR="${DATA_DIR:-data/PsycheMME}"
RUN_DIR="${RUN_DIR:-runs/my-model}"
if [[ -e "$RUN_DIR" ]]; then
  echo 'RUN_DIR already exists. Choose a fresh run directory in .env.' >&2
  exit 2
fi
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg before evaluation.' >&2; exit 2; }
psychemme download --bundle benchmark --output "$DATA_DIR"
psychemme infer --backend api \
  --inputs "$DATA_DIR/benchmark/public/test.jsonl" --asr "$DATA_DIR/benchmark/asr.jsonl" \
  --sources "$DATA_DIR/metadata/media.jsonl" --media-root "$DATA_DIR/benchmark/media" \
  --frame-cache runs/frames --base-url "$MODEL_BASE_URL" --api-key-env MODEL_API_KEY \
  --model "$MODEL_ID" --output "$RUN_DIR/predictions.jsonl"
python - "$RUN_DIR/predictions.jsonl" <<'PY'
import json,sys
with open(sys.argv[1]) as stream:
    rows=[json.loads(line) for line in stream if line.strip()]
if not any(row.get('status') == 'OK' for row in rows):
    raise SystemExit('No valid model responses. Check the endpoint and prediction statuses before judging.')
PY
psychemme judge \
  --inputs "$DATA_DIR/benchmark/public/test.jsonl" --references "$DATA_DIR/benchmark/ground_truth.jsonl" \
  --predictions "$RUN_DIR/predictions.jsonl" --base-url "$JUDGE_BASE_URL" \
  --api-key-env JUDGE_API_KEY --judge-model "$JUDGE_MODEL" --output "$RUN_DIR/judgments.jsonl"
psychemme score \
  --inputs "$DATA_DIR/benchmark/public/test.jsonl" --references "$DATA_DIR/benchmark/ground_truth.jsonl" \
  --predictions "$RUN_DIR/predictions.jsonl" --judgments "$RUN_DIR/judgments.jsonl" \
  --output "$RUN_DIR/scores.jsonl"
echo "Results: $RUN_DIR/scores.summary.json"
