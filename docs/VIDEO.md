# Video inference

The SFT and GRPO adapters use **`Qwen3_5ForConditionalGeneration`**, `AutoProcessor` and PEFT. This is a multimodal model; the generic Hub `AutoModelForCausalLM` example is not the video-loading recipe.

## Install

From the code repository root:

```bash
python -m pip install '.[local-model]'
psychemme doctor --mode local
```

Install FFmpeg separately and choose a PyTorch CUDA build for your GPU. The package pins the paper's Torch, Transformers, PEFT and Qwen-VL utilities versions.

## Try a video

Use your own local video, or download the example linked in the [official Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B):

```bash
mkdir -p examples
curl --fail --location \
  https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/video/N1cdUjctpG8.mp4 \
  --output examples/video.mp4

psychemme video --video examples/video.mp4 \
  --question "Summarize the video content." \
  --adapter psychemme/PsycheMME-Qwen3.5-4B-SFT \
  --adapter-revision v1.0-paper \
  --output runs/video.json
```

For GRPO, use `psychemme/PsycheMME-Qwen3.5-4B-GRPO`. Omit both adapter arguments for the base model. Each adapter loads directly on `Qwen/Qwen3.5-4B`; do not load SFT underneath GRPO.

For a spoken interaction, pass `--asr transcript.txt` with your UTF-8 transcript. The command uses the full video timeline, FFmpeg 1 fps, a uniform cap of 64 frames, the original processor, BF16/SDPA, seed 42, greedy generation and at most 768 new tokens. Thinking is disabled.

The JSON output contains the response, frame count, model revisions and finish reason. A token-limit completion is reported as `INCOMPLETE` with a nonzero exit code. Existing outputs are not overwritten.

The command shares `LocalQwen` and `frames_1fps` with benchmark inference. It answers a free-form question; use [`infer`, `judge` and `score`](MODEL_INTEGRATION.md) for the benchmark's structured responses and metrics. This sample video is supplied by Qwen and is not part of PsycheMME.
