# PsycheMME evaluation tools

Install from the repository root with `python -m pip install .`, or install this directory with `python -m pip install ./code`. Python 3.10+ is required. The core package uses the standard library. Add `.[data]` for Hub downloads or `.[local-model]` for local inference. Use Python 3.11 for paper-score replay.

`psychemme --help` lists the commands: `doctor`, `demo`, `video`, `download`, `import-predictions`, `validate`, `infer`, `judge`, `score` and `replay`.

- `psychemme_eval/`: data validation, prompt/parser, frames, Qwen/API inference, Judge and scoring.
- `configs/`: model metadata and protocol settings.
- `scripts/`: result checks for matching local data.
- `tests/`: evaluation and transport tests.
- `reference_core/`: supporting paper implementation.

Start with the [Quick Start](../docs/QUICKSTART.md) or [Video Inference](../docs/VIDEO.md). Benchmark data and paper scoring records are published separately on Hugging Face; SFT and GRPO adapters are available on [Hugging Face](https://huggingface.co/psychemme/models). [Training data and settings](https://huggingface.co/datasets/psychemme/PsycheMME-Training) are available. Training scripts and construction recipes will be released separately.
