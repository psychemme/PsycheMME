#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || { echo "Usage: bash scripts/evaluate.sh" >&2; exit 2; }
cd "$(dirname "$0")/.."
OVERRIDES=()
for name in MODEL_BASE_URL MODEL_ID MODEL_API_KEY JUDGE_BASE_URL JUDGE_MODEL JUDGE_API_KEY DATA_DIR RUN_DIR RETRY_ERRORS; do
  if [[ "${!name+x}" == x ]]; then OVERRIDES+=("$name=${!name}"); fi
done
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
for value in "${OVERRIDES[@]-}"; do
  if [[ -n "$value" ]]; then export "$value"; fi
done
: "${MODEL_BASE_URL:?Set MODEL_BASE_URL in .env}"
: "${MODEL_ID:?Set MODEL_ID in .env}"
: "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL in .env}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL in .env}"
DATA_DIR="${DATA_DIR:-data/PsycheMME}"
RUN_DIR="${RUN_DIR:-runs/my-model}"
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg before evaluation.' >&2; exit 2; }
mkdir -p "$RUN_DIR"
psychemme download --bundle benchmark --output "$DATA_DIR"
INPUTS="$DATA_DIR/benchmark/public/test.jsonl"; REFERENCES="$DATA_DIR/benchmark/ground_truth.jsonl"
ASR="$DATA_DIR/benchmark/asr.jsonl"; SOURCES="$DATA_DIR/metadata/media.jsonl"
RETRY_ARGS=(--retries 2)
if [[ "${RETRY_ERRORS:-0}" == 1 ]]; then RETRY_ARGS+=(--retry-errors); fi
psychemme infer --backend api --resume "${RETRY_ARGS[@]}" \
  --inputs "$INPUTS" --asr "$ASR" --sources "$SOURCES" \
  --media-root "$DATA_DIR/benchmark/media" --frame-cache runs/frames \
  --base-url "$MODEL_BASE_URL" --api-key-env MODEL_API_KEY \
  --model "$MODEL_ID" --output "$RUN_DIR/predictions.jsonl"
JUDGE_RETRY_ARGS=(--retries 2)
if [[ "${RETRY_ERRORS:-0}" == 1 ]]; then JUDGE_RETRY_ARGS+=(--retry-errors); fi
psychemme judge --resume "${JUDGE_RETRY_ARGS[@]}" --inputs "$INPUTS" --references "$REFERENCES" \
  --predictions "$RUN_DIR/predictions.jsonl" --base-url "$JUDGE_BASE_URL" \
  --api-key-env JUDGE_API_KEY --judge-model "$JUDGE_MODEL" --output "$RUN_DIR/judgments.jsonl"
psychemme score --inputs "$INPUTS" --references "$REFERENCES" \
  --predictions "$RUN_DIR/predictions.jsonl" --judgments "$RUN_DIR/judgments.jsonl" \
  --output "$RUN_DIR/scores.jsonl"
echo "Results: $RUN_DIR/scores.summary.json"
