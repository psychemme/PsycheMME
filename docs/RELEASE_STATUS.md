# Release status

| Resource | Availability |
| :--- | :--- |
| Evaluation code | GitHub: inference, Reasoning-Verified Judge, scoring and replay |
| Benchmark | [PsycheMME](https://huggingface.co/datasets/psychemme/PsycheMME): 1,000 questions, references, 630 videos, ASR and source intervals |
| Paper scoring records | [PsycheMME-logs](https://huggingface.co/datasets/psychemme/PsycheMME-logs): 19,000 records and paper tables |
| SFT / GRPO adapters | Hugging Face: weights, evaluation settings and video examples |
| Training | Training data, configurations and launch recipes are planned |
| Data construction | Planned |

Resource IDs and revisions are maintained in `manifests/resources.json`; `scripts/sync_resources.py` downloads them with resumable transfers. Training is listed as planned and is not downloaded.

Questions, references and ASR retain the paper final-1000 content. Paper scoring records are provided separately in the logs repository. The [reproduction guide](REPRODUCIBILITY.md) distinguishes score replay from new inference.

Code and adapters use Apache-2.0. Benchmark annotations and media follow the source terms in the dataset's `licenses/` directory. Numerical results and index metadata use CC BY 4.0.
