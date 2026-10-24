# Quick start

## Install

Use Python 3.11 and [FFmpeg](https://ffmpeg.org/download.html).

```bash
git clone https://github.com/psychemme/PsycheMME.git
cd PsycheMME
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[data]'
cp .env.example .env
```

## Configure

Fill in the model and text Judge URLs, served model IDs and API keys in `.env`. The model endpoint accepts chat-completions requests containing images and text. Keys can be empty for your own unauthenticated local endpoints.

## Evaluate

Start with ten real benchmark questions:

```bash
RUN_DIR=runs/sample bash scripts/evaluate.sh 10
```

The sample covers the eight capabilities and downloads only the required videos. Run the full benchmark with:

```bash
bash scripts/evaluate.sh
```

Results are saved in the selected `RUN_DIR`:

```text
predictions.jsonl       Model answers and rationales
judgments.jsonl         Answer and rationale assessments
scores.jsonl            Question-level scores
scores.summary.json     Raw / Verified capability and macro scores
```

## Continue a run

Repeat the same command after interruption. Saved terminal records are reused, and each stage verifies that the inputs and settings match. Use a new directory for a different model or question population.

Temporary network errors are retried twice per inference request. After fixing an endpoint problem, retry saved execution errors explicitly:

```bash
RETRY_ERRORS=1 RUN_DIR=runs/sample bash scripts/evaluate.sh 10
```

Inference prints progress and reports successful responses, format failures, endpoint errors and pending questions. A failed stage returns a nonzero exit code. See [Troubleshooting](TROUBLESHOOTING.md) for recovery steps.

Next: [model integration](MODEL_INTEGRATION.md), [local video inference](VIDEO.md), [data format](DATA_FORMAT.md), [scoring](EVALUATION.md).

## Recompute paper scores

```bash
bash scripts/replay.sh
```

Checks 19,000 question-level scores and 190 table values from the released scoring factors. Output: `runs/paper_replay.json`.
