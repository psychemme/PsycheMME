# Reproduction guide

The benchmark, scoring records and model adapters are registered in `manifests/resources.json`.

## Current scope

| Component | Status |
| :--- | :--- |
| Inference, Judge and scoring code | Available |
| CPU scoring demo | Available; synthetic inputs |
| SFT and GRPO adapters | Available on Hugging Face |
| Local video inference | Available; requires a compatible GPU and FFmpeg |
| 1,000-question / 630-episode benchmark | Available on Hugging Face |
| Paper scoring inputs and evaluation logs | Available in PsycheMME-logs |
| SFT/GRPO training recipes and data construction | Planned |

## Recompute the paper table

The replay code reads these local files:

```text
data/PsycheMME-logs/
├── dataset/
│   └── metadata_only.jsonl
└── results/
    ├── scoring_inputs.jsonl
    ├── per_question_scores.jsonl
    └── paper_table_2.json
```

```bash
python scripts/sync_resources.py logs
psychemme replay --root data/PsycheMME-logs --output runs/paper_replay.json
```

For the complete paper logs, replay recomputes all **19,000** question/model scores from answer credit, rationale coverage, support and logic labels. It compares each result with its saved score and checks **190** displayed table cells for **19** model entries. Use **Python 3.11** for the paper table display. Replay uses the saved answer credit and rationale labels; inference and judging are separate runs.

## Run the released models

Follow [Video Inference](VIDEO.md) for a single video and [Model Integration](MODEL_INTEGRATION.md) for benchmark inputs. The loader uses `Qwen3_5ForConditionalGeneration`, `AutoProcessor` and `PeftModel`.

| Setting | Paper comparison |
| :--- | :--- |
| Base | `Qwen/Qwen3.5-4B` |
| Base revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Adapters | [SFT](https://huggingface.co/psychemme/PsycheMME-Qwen3.5-4B-SFT) or [GRPO](https://huggingface.co/psychemme/PsycheMME-Qwen3.5-4B-GRPO), revision `v1.0-paper` |
| Frames | Full-timeline 1 fps, uniform cap64; no additional resizing |
| Precision / attention | BF16 / SDPA |
| Generation | Greedy, seed42, max768 new tokens; thinking disabled |

Both adapters load directly on the original base. GRPO is a complete adapter relative to that base, not an adapter to stack on SFT. Training used 32 frames; the evaluation cap is 64.

The runner uses the paper prompt, parser and frame-sampling settings. API evaluation requires a compatible image-capable chat-completions endpoint; local inference uses the shared Hugging Face/PEFT loader.

## Verification and versioning

Tests cover data boundaries, choice parsing, fixed-population scoring, API transport, frame extraction and memory handling. References and evidence are supplied only to judging and scoring.

Questions, references and ASR retain the paper final-1000 content. See [Release Status](RELEASE_STATUS.md) for resource availability.
