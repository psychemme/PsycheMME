# Contributing and support

[Open an issue](https://github.com/psychemme/PsycheMME/issues/new/choose) for usage, reproducibility or suspected annotation problems. Include versions, a minimal command, expected/actual behavior and question IDs. Do not post credentials, restricted media or unreleased text.

For code changes, install from the root, run `psychemme demo --output runs/your-change` and `python -m unittest discover -s code/tests -v`. ffmpeg enables CPU media tests. Preserve model/evaluator separation, strict IDs, fixed populations and failure records.

Frozen answers and judgments must not be silently edited. Report suspected errors with IDs/evidence; corrections need old/new hashes, reason, date, score impact and a separate data version. A quality flag is not a confirmed error. Public examples must not be selected by favorable model performance.

Respect source-specific data rights; the code license does not override them. Maintain anonymous publication metadata during review.
