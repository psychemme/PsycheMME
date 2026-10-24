# Evaluation and Reasoning-Verified scoring

`Raw_i = A_i`, where single-choice uses exact option correctness, multiple-select uses exact-set correctness, and open answers use persisted semantic credit in `{0, 0.5, 1}`. Invalid inference/format responses receive zero.

`Verified_i = A_i × C_i × S_i × L_i / 1.44`.

`C_i` is 1 for explicit informative rationale coverage and 0 otherwise. Support and logic are separate judgments with these weights:

| Support | Logic | Weight | Interpretation |
| --- | --- | --- | --- |
| strong | fully_coherent | 1.2 | Material claims supported by cues / active judgments jointly support the answer without contradiction. |
| normal | acceptable | 1.0 | Adequate support or consistency, with limited omissions or minor gaps. |
| weak | weak | 0.8 | A material supporting inference or reasoning link is incomplete or inadequately justified. |
| contradicted | contradictory | 0.5 | A material claim conflicts with cues / active judgments conflict with one another or the final answer. |
| insufficient | unevaluable | 0 | No evaluable support or logical explanation. |

An incorrect answer does not by itself establish an internal logical contradiction. Null rationale slots are not implicit justification. The judge assesses explicit concise judgments rather than reconstructing hidden reasoning. Reference memory is distinct from the history supplied to the candidate; mismatched history is an experimental distractor, not a fact about the participant.

Capability means include all declared question IDs and failure records. The headline metric is the equal-weight mean over all eight capability means, multiplied by 100. A subset missing a capability receives no official eight-capability macro score; the tool separately labels its observed-capability mean as exploratory. A missing prediction stays in the reference population with zero credit. Missing/failed reasoning judgments receive zero Verified credit. Choice Raw correctness remains directly computable; open Raw credit requires a valid answer judgment.

## Commands

Follow the [installation instructions](../README.md#-installation). Download the benchmark with `python scripts/sync_resources.py benchmark`; the commands below use the [released layout](DATA_FORMAT.md).

```sh
psychemme validate --inputs data/PsycheMME/benchmark/public/test.jsonl \
  --references data/PsycheMME/benchmark/ground_truth.jsonl \
  --asr data/PsycheMME/benchmark/asr.jsonl

psychemme download --bundle replay
psychemme replay --root data/PsycheMME-logs --output runs/paper_factor_replay.json
```

`replay` recomputes 19,000 scores from saved answer credit and rationale factors, then checks 190 main-table values. It runs on CPU without model or Judge calls.

For new model outputs, use evaluator-only references after inference:

```sh
export JUDGE_API_KEY="your-judge-api-key"

psychemme judge --inputs data/PsycheMME/benchmark/public/test.jsonl \
  --references data/PsycheMME/benchmark/ground_truth.jsonl \
  --predictions runs/predictions.jsonl --output runs/judgments.jsonl \
  --base-url https://YOUR_JUDGE_ENDPOINT/v1 --api-key-env JUDGE_API_KEY \
  --judge-model YOUR_JUDGE_MODEL

psychemme score --inputs data/PsycheMME/benchmark/public/test.jsonl \
  --references data/PsycheMME/benchmark/ground_truth.jsonl \
  --predictions runs/predictions.jsonl --judgments runs/judgments.jsonl \
  --output runs/scores.jsonl
```

Configure the Judge endpoint and `JUDGE_API_KEY` for your run. The paper uses `Qwen/Qwen3.8-27B`; prompts, answer rubric and memory handling are in `code/psychemme_eval/judge.py` and `memory_verification_context.py`.

Use the CLI for inference, judging and fixed-population scoring. Supporting implementations are in `code/reference_core/`.
