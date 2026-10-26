# Resources

The [resource table](../README.md#resources) lists the benchmark, model adapters and evaluation records. Download locations are maintained in `manifests/resources.json`.

## Versions

| Version | Entry | Use |
| :--- | :--- | :--- |
| Paper version | `v1.0-paper` in GitHub and all five Hugging Face repositories | Reproduce the paper tables with the frozen resources |
| Current version | `main` in GitHub and all five Hugging Face repositories | Evaluate models with the current code and benchmark |

The current benchmark rewords 32 multiple-select questions. Answers and options are unchanged. Use `v1.0-paper` for comparisons with the paper tables; scores from new runs on the current version are separate results.

The evaluation-record repository provides the paper's 19,000 scores and 190 table values in both entries. These numerical records are unchanged.

Start with [model evaluation](QUICKSTART.md). [Training data and settings](https://huggingface.co/datasets/psychemme/PsycheMME-Training) include the SFT set, GRPO candidate pool, videos and training frames. Training scripts and construction code will be released separately.
