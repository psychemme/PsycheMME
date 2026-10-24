# Quick start

Use Python 3.11 for paper-score replay. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[data]'
psychemme doctor
psychemme demo --output runs/demo
```

The demo runs eight synthetic items on CPU, without API keys or downloads. Expected scores: **75.00% Raw**, **71.18% Reasoning-Verified**. Outputs include questions, references, predictions, judgments, per-question scores and a capability summary in `runs/demo/`.

## Recompute the paper table

```bash
bash scripts/replay.sh
```

This downloads [PsycheMME-logs](https://huggingface.co/datasets/psychemme/PsycheMME-logs) and checks **19,000 scores** and **190 table values**. It needs no videos, GPU or API key. The report is saved to `runs/paper_replay.json`.

## Get the benchmark

```bash
python scripts/sync_resources.py benchmark
```

The download includes 1,000 questions, reference answers, 630 ASR transcripts, sources and 630 videos (about 27.6 GB). Interrupted downloads resume. To download annotations without videos:

```bash
python scripts/sync_resources.py benchmark --metadata-only
```

Questions are in `data/PsycheMME/benchmark/public/test.jsonl`. References are in `benchmark/ground_truth.jsonl`; supply them only to the Judge and scoring steps. Follow [Model Integration](MODEL_INTEGRATION.md) to evaluate a model through an API or the local Qwen backend.

Paper-score replay uses the recorded paper outputs; model inference and judging are separate runs.

## Run a local video

Install FFmpeg and `python -m pip install '.[local-model]'`, then follow the [video example](VIDEO.md). The example uses the same Qwen loader and frame sampling as evaluation. A compatible GPU is required.

Next: [data format](DATA_FORMAT.md), [scoring protocol](EVALUATION.md), [reproduction guide](REPRODUCIBILITY.md), or [troubleshooting](TROUBLESHOOTING.md).
