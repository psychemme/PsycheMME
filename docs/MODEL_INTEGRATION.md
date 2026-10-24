# Evaluate your model

Install the package and try the [offline demo](QUICKSTART.md) or [local video example](VIDEO.md). Download the benchmark with `python scripts/sync_resources.py benchmark` using the [documented layout](DATA_FORMAT.md). Use matching question populations for inference, judging and scoring.

## Quick evaluation

Configure your model and Judge endpoints, then run the complete evaluation:

```bash
cp .env.example .env
# Set the endpoint URLs, model IDs and API keys in .env.
bash scripts/evaluate.sh
```

The script downloads the benchmark, runs Infer → Judge → Score, and saves predictions, judgments and scores under `runs/my-model/`. Use a fresh `RUN_DIR` for each run. Install FFmpeg before evaluation.

## Compatible chat-completions API

Install [FFmpeg](https://ffmpeg.org/download.html) and provide your matching videos in `data/PsycheMME/benchmark/media/video/`. The source index binds every episode to its video and recorded file size; Hugging Face verifies downloaded objects. A supplied media checksum is also checked.

```sh
# Set MODEL_API_KEY in your shell only if your endpoint requires it.
psychemme infer --backend api \
  --inputs data/PsycheMME/benchmark/public/test.jsonl \
  --asr data/PsycheMME/benchmark/asr.jsonl \
  --sources data/PsycheMME/metadata/media.jsonl \
  --media-root data/PsycheMME/benchmark/media --frame-cache runs/frames \
  --base-url https://YOUR_ENDPOINT/v1 --api-key-env MODEL_API_KEY \
  --model YOUR_SERVED_MODEL_ID --output runs/my-model/predictions.jsonl
```

The endpoint must support image messages and honor requested generation controls. Input uses full-timeline 1 fps / cap64 frames, public prompt and ASR. The runner requests greedy generation, seed42, max768 and disabled thinking. Inspect per-row statuses and `.run.json` for run settings and failures. References are used only after inference.

## Your own runner

Use only public inputs, ASR and authorized media. `psychemme_eval.inference.build_input` prepares frames from the mapped release videos and the exact public prompt; it has no gold argument. Save one line per response:

```json
{"question_id":"YOUR_QUESTION_ID","response":{"answer":"A","G":"A brief cue-supported observation.","W":null,"N":null,"T":null,"S":null,"A":null,"R":null,"M":null},"finish_reason":"stop"}
```

`response` can be a raw string or an object. Single-choice uses a letter, multiple-select a nonempty letter array, open questions natural language. All nine keys and at least one informative rationale slot are required.

```sh
psychemme import-predictions \
  --inputs data/PsycheMME/benchmark/public/test.jsonl \
  --responses my-responses.jsonl --model YOUR_MODEL_ID \
  --output runs/my-model/predictions.jsonl
```

Duplicate/foreign IDs are rejected. Missing responses become `MISSING`; malformed responses and explicitly recorded non-`stop` completions remain invalid. Record the actual `finish_reason`; omitted values are retained as `UNRECORDED`. Use a fresh output path for each run.

For local Qwen, install `python -m pip install '.[local-model]'` with the appropriate PyTorch CUDA wheel, then `psychemme doctor --mode local`. Follow the pinned Base/SFT/GRPO settings in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Both LoRA adapters load directly on the original base. Keep the documented frame cap and full clip timeline.

## Judge separately, then score

The Judge endpoint is explicitly selected by you and may charge for calls. Demo, doctor, download and replay never invoke it. Record its exact served model/version and configuration.

```sh
psychemme judge \
  --inputs data/PsycheMME/benchmark/public/test.jsonl \
  --references data/PsycheMME/benchmark/ground_truth.jsonl \
  --predictions runs/my-model/predictions.jsonl \
  --base-url https://YOUR_JUDGE_ENDPOINT/v1 --api-key-env JUDGE_API_KEY \
  --judge-model YOUR_EXACT_JUDGE_MODEL --output runs/my-model/judgments.jsonl

psychemme score \
  --inputs data/PsycheMME/benchmark/public/test.jsonl \
  --references data/PsycheMME/benchmark/ground_truth.jsonl \
  --predictions runs/my-model/predictions.jsonl \
  --judgments runs/my-model/judgments.jsonl --output runs/my-model/scores.jsonl
```

Read `scores.summary.json`: capability counts, Raw/Verified macro, missing capabilities. Missing predictions stay in the denominator. An incomplete-capability population has official macro `null`, with the exploratory present-capability mean labeled separately. Open credit needs a successful saved semantic judgment. Failed judges receive zero Verified credit. Full rules: [EVALUATION.md](EVALUATION.md).

If using `--limit`, select matching input/reference subsets before judging. Report the evaluated population, model revisions, backend, sampling, generation settings and Judge version with results.
