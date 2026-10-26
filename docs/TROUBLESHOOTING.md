# Troubleshooting

| Symptom | Meaning and next step |
| --- | --- |
| Command not found | Activate the installed environment; try `python -m psychemme_eval --help`. |
| Python requirement error | Create a fresh Python 3.10+ environment. |
| Doctor reports missing Torch/Transformers | Optional in core mode. Demo/download/replay need no model stack. Local inference needs `.[local-model]` with the appropriate CUDA wheel. |
| Network/TLS download failure | Retry; verified files are reused. Keep TLS checks enabled. Public files need no token. You can also clone the public HF dataset on a working connection and transfer it locally. |
| Existing file differs / hash mismatch | Preserve the conflicting file if needed, then use a fresh directory; do not disable verification. |
| Demo/prediction/Judge output exists | Use a new path to keep experiments separate. |
| Download support missing | Install `python -m pip install ".[data]"`, then rerun the download. |
| Paper cell differs by 0.01 | Use Python 3.11 for replay; Python 3.12+ changed floating-point summation at display ties. Do not edit saved results. |
| Missing replay data | Provide the four local files listed in [Reproducibility](REPRODUCIBILITY.md), then set `--root` to their parent directory. |
| Missing/mismatched video | Rerun the benchmark download and use `metadata/media.local.jsonl` with `--media-root data/PsycheMME/benchmark/media`. |
| Imported responses are invalid | Check nine keys, an informative rationale, answer type and `finish_reason`. Inspect `parse_reason`. |
| Judge population differs | Public inputs, references and predictions need matching IDs. Import to materialize missing predictions. |
| Unexpected score | Inspect inference/Judge failures and invalid formats. They remain in the denominator; Verified additionally applies fixed normalized rationale weights. |
| Macro is `null` | The declared population lacks one or more capabilities. Do not relabel the exploratory mean as the official eight-capability macro. |
| GPU out of memory | Check available memory and use the documented runtime. Changing frame count, resizing or clip duration changes the input protocol. Demo and Replay can run on CPU. |

For tracebacks use `psychemme --debug COMMAND ...`. Before [opening an issue](https://github.com/psychemme/PsycheMME/issues/new/choose), remove keys, signed URLs, personal paths and restricted text. Include code commit, Python version, exact sanitized command, question IDs, data revision/hashes and doctor output.

For Ego4D access errors, renew the official credentials, update the Ego4D fields in `.env`, and rerun `bash scripts/download.sh`. Existing source files and completed clips are reused. See [Download Guide](DOWNLOAD.md).
