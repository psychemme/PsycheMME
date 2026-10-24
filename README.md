<p align="center">
  <a href="https://psychemme.github.io/">
    <img src="assets/logo.png" alt="PsycheMME" width="480">
  </a>
</p>

<h2 align="center">A Psychology-Informed Benchmark<br>for Social Understanding and Interaction in Videos</h2>

<p align="center">
  <a href="https://psychemme.github.io/"><img src="https://img.shields.io/badge/Webpage-Website-8A2BE2?style=plastic&amp;logo=googlechrome&amp;logoColor=white" alt="Webpage"></a>
  <a href="docs/QUICKSTART.md"><img src="https://img.shields.io/badge/Docs-Document-007EC6?style=plastic&amp;logo=readthedocs&amp;logoColor=white" alt="Document"></a>
  <a href="https://arxiv.org/abs/2610.28625"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=plastic&amp;logo=arxiv&amp;logoColor=white" alt="arXiv Paper"></a>
  <a href="https://github.com/psychemme/PsycheMME/issues"><img src="https://img.shields.io/badge/GitHub-Community-2EA44F?style=plastic&amp;logo=github&amp;logoColor=white" alt="Community"></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> ·
  <a href="#-update">Update</a> ·
  <a href="#-installation">Installation</a> ·
  <a href="#-usage">Usage</a> ·
  <a href="#-experiment">Experiment</a> ·
  <a href="#-citation">Citation</a>
</p>

# 🧠 Overview

**Can a video model understand what a person needs, when to respond, and why?**

**PsycheMME** evaluates social understanding and interaction in videos through **1,000 questions**, **630 episodes**, and **eight psychology-informed capabilities**. It connects observable social cues with psychological understanding, needs, response timing, verbal responses, actions, memory, and risk.

<p align="center">
  <img src="assets/overview.webp" alt="PsycheMME overview: social situations, eight capabilities, and model evaluation" width="100%">
</p>

- **Understand the person.** Ground visible cues and infer psychological states and needs.
- **Choose a response.** Decide when to respond, what to say, and how to act.
- **Use context responsibly.** Incorporate relevant history and respect social boundaries.
- **Explain with evidence.** Evaluate answer correctness together with evidence support and logical consistency using **Reasoning-Verified** scoring.

> Start with the [PsycheMME Document](docs/QUICKSTART.md) for a guided first run, or explore the [project page](https://psychemme.github.io/) for results and examples.

## Resources

| Resource | Location |
| :--- | :--- |
| Benchmark: questions, references, videos, ASR and sources | [PsycheMME](https://huggingface.co/datasets/psychemme/PsycheMME) |
| Paper scoring records and tables | [PsycheMME-logs](https://huggingface.co/datasets/psychemme/PsycheMME-logs) |
| SFT LoRA adapter | [PsycheMME-Qwen3.5-4B-SFT](https://huggingface.co/psychemme/PsycheMME-Qwen3.5-4B-SFT) |
| GRPO LoRA adapter | [PsycheMME-Qwen3.5-4B-GRPO](https://huggingface.co/psychemme/PsycheMME-Qwen3.5-4B-GRPO) |

Resource IDs, types and download locations are maintained in [resources.json](manifests/resources.json).

# 📣 Update

- Evaluation tools and the [SFT / GRPO adapters](https://huggingface.co/psychemme/models) are available.
- The [benchmark](https://huggingface.co/datasets/psychemme/PsycheMME) and [paper scoring records](https://huggingface.co/datasets/psychemme/PsycheMME-logs) are available on Hugging Face.
- Training recipes and data construction will follow separately.

# 🛠️ Installation

## Quickstart

See the [installation guide](docs/QUICKSTART.md) for detailed instructions. **Python 3.10+** is required. Use **Python 3.11** for paper-score replay.

```bash
git clone https://github.com/psychemme/PsycheMME.git
cd PsycheMME

python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[data]'

psychemme doctor
```

Install [FFmpeg](https://ffmpeg.org/download.html) for video inference with either an API or a local model. For local models, also install the optional dependencies:

```bash
python -m pip install '.[local-model]'
psychemme doctor --mode local
```

Choose a PyTorch CUDA build suitable for your GPU. API-based evaluation does not require local model weights. On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

# 🚀 Usage

## Document

Full instructions are available in the **[PsycheMME Document](docs/QUICKSTART.md)**. Start with the quick start, then follow the guide for your task:

| Task | Guide |
| :--- | :--- |
| Run your first example | [Quick Start](docs/QUICKSTART.md) · [中文指南](docs/QUICKSTART.zh-CN.md) |
| Understand questions, ASR, and references | [Data Format](docs/DATA_FORMAT.md) |
| Evaluate your own model | [Model Integration](docs/MODEL_INTEGRATION.md) |
| Understand Raw and Reasoning-Verified scores | [Evaluation Protocol](docs/EVALUATION.md) |
| Reproduce the paper results | [Reproduction Guide](docs/REPRODUCIBILITY.md) |
| Resolve setup or runtime issues | [Troubleshooting](docs/TROUBLESHOOTING.md) |

## Getting Data

Download the **[benchmark](https://huggingface.co/datasets/psychemme/PsycheMME)** and **[paper scoring records](https://huggingface.co/datasets/psychemme/PsycheMME-logs)**:

```bash
python scripts/sync_resources.py
```

The benchmark includes **1,000 questions, references, 630 videos, ASR and source intervals** (about 27.6 GB of video). Downloads resume automatically. Add `--metadata-only` to skip videos, or select a resource: `python scripts/sync_resources.py logs`.

Questions, references and ASR retain the paper final-1000 content. See [Data Format](docs/DATA_FORMAT.md) and the [Reproduction Guide](docs/REPRODUCIBILITY.md).

## Evaluate Your Model

Configure your model and Judge endpoints, then run the complete evaluation:

```bash
cp .env.example .env
# Set the endpoint URLs, model IDs and API keys in .env.
bash scripts/evaluate.sh
```

The script downloads the benchmark, runs Infer → Judge → Score, and saves predictions, judgments and scores under `runs/my-model/`. Use a fresh `RUN_DIR` for each run. Install FFmpeg before evaluation.

### 1. Try the offline example

```bash
psychemme demo --output runs/demo
```

This synthetic example runs on CPU without data downloads or API keys. Expected macro scores: **75.00% Raw** and **71.18% Reasoning-Verified**.

### 2. Run model inference

After downloading the benchmark, install FFmpeg and follow the commands below. See [Model Integration](docs/MODEL_INTEGRATION.md) for backend settings.

Use a compatible image-capable chat-completions API. Replace the endpoint and model ID below, then set `MODEL_API_KEY` in your shell.

```bash
export MODEL_API_KEY="your-api-key"

psychemme infer --backend api \
  --inputs data/PsycheMME/benchmark/public/test.jsonl \
  --asr data/PsycheMME/benchmark/asr.jsonl \
  --sources data/PsycheMME/metadata/media.jsonl \
  --media-root data/PsycheMME/benchmark/media --frame-cache runs/frames \
  --base-url https://YOUR_ENDPOINT/v1 --api-key-env MODEL_API_KEY \
  --model YOUR_MODEL_ID --output runs/my-model/predictions.jsonl
```

For local Qwen inference, model adapters, or responses from your own runner, follow [Model Integration](docs/MODEL_INTEGRATION.md). Model inputs contain questions, public context, ASR, and video; reference answers are supplied only during judging and scoring.

### 3. Judge and score

Configure a separate text Judge and set `JUDGE_API_KEY` for its endpoint.

```bash
export JUDGE_API_KEY="your-judge-api-key"

psychemme judge \
  --inputs data/PsycheMME/benchmark/public/test.jsonl \
  --references data/PsycheMME/benchmark/ground_truth.jsonl \
  --predictions runs/my-model/predictions.jsonl \
  --base-url https://YOUR_JUDGE_ENDPOINT/v1 --api-key-env JUDGE_API_KEY \
  --judge-model YOUR_JUDGE_MODEL --output runs/my-model/judgments.jsonl

psychemme score \
  --inputs data/PsycheMME/benchmark/public/test.jsonl \
  --references data/PsycheMME/benchmark/ground_truth.jsonl \
  --predictions runs/my-model/predictions.jsonl \
  --judgments runs/my-model/judgments.jsonl \
  --output runs/my-model/scores.jsonl
```

Results include per-question JSONL scores and a `scores.summary.json` capability breakdown. Use the same question population throughout inference, judging, and scoring; comparisons on a subset should report that subset.

## Try a Video

With FFmpeg and the local-model dependencies installed, ask a question about a video:

```bash
psychemme video --video /path/to/clip.mp4 \
  --question "What happens in this video?" \
  --adapter psychemme/PsycheMME-Qwen3.5-4B-SFT \
  --adapter-revision v1.0-paper.1 \
  --output runs/video.json
```

This uses the evaluator's multimodal Qwen loader and full-timeline 1 fps sampling, capped at 64 frames. Follow the [video example](docs/VIDEO.md) for installation, a downloadable sample and optional ASR input.

## Reproduce Paper Scores

Replay recomputes Raw and Reasoning-Verified scores from the released answer credit and rationale labels. Use Python 3.11:

```bash
bash scripts/replay.sh
```

For the complete paper inputs, the expected result is **19 model entries**, **19,000 factor rows**, and **190 matching table cells**. Replay recomputes scoring; it does not run model inference or a fresh Judge. See the [Reproduction Guide](docs/REPRODUCIBILITY.md).

## Training

The paper's model adapters are available on Hugging Face:

| Model | Checkpoint |
| :--- | :--- |
| Qwen3.5-4B + SFT | [PsycheMME-Qwen3.5-4B-SFT](https://huggingface.co/psychemme/PsycheMME-Qwen3.5-4B-SFT) |
| Qwen3.5-4B + SFT + GRPO | [PsycheMME-Qwen3.5-4B-GRPO](https://huggingface.co/psychemme/PsycheMME-Qwen3.5-4B-GRPO) |

Both adapters load directly on the original base model. Loading instructions are in the [reproduction guide](docs/REPRODUCIBILITY.md).

**Planned:** SFT and GRPO training data, configurations, and launch scripts.

## Data Construction

**Planned.** Agent construction and the data-generation pipeline will be released separately. The [paper](https://arxiv.org/abs/2610.28625) describes the construction framework.

<details>
<summary><b>Repository Layout</b></summary>

```text
PsycheMME/
├── assets/                  # README logo and figures
├── code/
│   ├── psychemme_eval/      # Data loading, inference, judging, scoring, and CLI
│   ├── configs/             # Model and evaluation settings
│   ├── reference_core/      # Paper implementation
│   ├── scripts/             # Data and result checks
│   └── tests/               # Evaluation tests
├── docs/                    # Usage and reproduction guides
├── manifests/               # Resource locations and versions
├── scripts/                 # Downloads and paper-score replay
├── schemas/                 # Input and output formats
├── .github/workflows/       # CPU checks
├── pyproject.toml
├── requirements.txt
└── README.md
```

</details>

# 📊 Experiment

PsycheMME supports research on:

1. **Social understanding** — grounding cues, interpreting psychological states, and inferring needs.
2. **Interaction decisions** — response timing, verbal responses, and appropriate actions.
3. **Context and memory** — understanding when relevant history changes a response.
4. **Social constraints** — recognizing risk and respecting interpersonal boundaries.
5. **Reasoning quality** — measuring the gap between a correct answer and a supported explanation.

<p align="center">
  <img src="assets/results.svg" alt="Raw and Reasoning-Verified results for proprietary and open-weight models, with a separate SFT and GRPO comparison" width="100%">
</p>

The paper compares **17 frozen models** and separately studies **SFT and GRPO** adaptation. Both headline metrics are macro-averaged across the eight capabilities. Explore the **[full results and analysis](https://psychemme.github.io/#results)**, and explore the [paper scoring records](https://huggingface.co/datasets/psychemme/PsycheMME-logs).

# 📝 Citation

If you find PsycheMME useful in your research, please cite our work.

```bibtex
@misc{hu2026psychemme,
  title={PsycheMME: A Psychology-Informed Benchmark for Social Understanding and Interaction in Videos},
  author={Heran Hu and Yuhang Dong and Zhucun Xue and Kaiwei Chen and Yong Liu and Jiangning Zhang},
  year={2026},
  eprint={2610.28625},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.28625}
}
```

# 🤝 Acknowledgement

We thank the creators of the source datasets and the open-source model and software communities that support this work. Source attribution and access terms are provided in the dataset release, alongside the source-specific attribution and access terms.

Questions or suggestions? Join the conversation through [GitHub Issues](https://github.com/psychemme/PsycheMME/issues).

# 📄 License

Code is released under [Apache-2.0](LICENSE). Dataset annotations and third-party media follow their respective licenses and source terms, provided with the dataset.
