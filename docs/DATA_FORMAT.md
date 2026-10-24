# Data format

The paper benchmark contains **1,000 questions and 630 episodes**. The benchmark and paper scoring records are published in separate Hugging Face repositories.

All JSONL files contain one JSON object per line.

## Benchmark: `psychemme/PsycheMME`

| File | Contents |
| :--- | :--- |
| `benchmark/public/test.jsonl` | Question/episode IDs, relative media ID, category, task type, question, options, source group and supplied memory. |
| `benchmark/ground_truth.jsonl` | Matching public fields plus zero-based answer indices, answer texts, open reference, evidence and reference reasoning. |
| `benchmark/asr.jsonl` | Episode/media IDs, transcript segments, language and status. |
| `benchmark/media/video/` | 630 video files. |
| `metadata/media.jsonl` | Video mapping, file size, dimensions, frame rate, duration and source interval. |
| `metadata/episodes.jsonl` | Episode/source mapping and question count. |
| `metadata/sources.csv` | Original source links, identifiers and crop intervals. |

`media_id` is relative to `benchmark/media/`. The model reads questions, ASR and media; references are supplied only to judging/scoring. Questions, reference answers and ASR retain the paper final-1000 content.

## Paper records: `psychemme/PsycheMME-logs`

| File | Contents |
| :--- | :--- |
| `dataset/metadata_only.jsonl` | 1,000 replay IDs, capability codes, task types and episode IDs. |
| `metadata/result_ids.jsonl` | Mapping from compact result IDs to original benchmark question and episode IDs. |
| `results/scoring_inputs.jsonl` | 19,000 answer credits, rationale coverage, support/logic labels and status records. |
| `results/per_question_scores.jsonl` | Saved Raw/Verified scores. |
| `results/paper_table_2.json` | 190 displayed main-table values. |
| `results/leaderboard.csv`, `results/per_capability_scores.csv` | Aggregate result tables. |

Use `original_question_id` and `original_episode_id` to join logs with the benchmark. `replay --root` points to the downloaded logs directory. Full generated rationale text and Judge transcripts are not included.

## Validation

The validator checks public/reference ID closure, option indices, duplicates, capability codes and forbidden reference fields in public inputs. It rejects foreign or duplicate predictions. JSON Schemas are in `schemas/`; cross-record constraints are checked by Python.

Single-choice answers use a letter string. Multiple-select uses an exact set of letters with no partial-set credit. Open-answer credit is 0, 0.5 or 1 from a separately saved semantic judgment. Model output has nine keys: `answer, G, W, N, T, S, A, R, M`, with at least one informative rationale slot.

## Memory context

`memory_context` is the public history supplied to the model. The loader passes this text unchanged to the prompt and to the reference-memory Judge context. It is present in 40 Memory questions; the paper's explicit-history ablation uses those 40 items. Video and ASR remain part of the input when this field is null.

`memory_dependency` describes the annotated dependence on history and is not model input. Fourteen dependency-marked questions have null supplied history in the paper source as well as the published data. Their annotations are retained for review; no context is generated from reference answers or evidence.

## Evidence and time

Evidence may be an object list or a free-text annotation. Objects may carry times, offsets, source/display coordinates and history/current roles. Clip-relative and source timestamps are different coordinates and must not be substituted for one another. Structural validation does not establish semantic correctness.

Observable behavior, inferred psychological state and a recommended response are separate judgments. Cue-supported interpretations are not clinical diagnoses. Benchmark corrections should preserve the paper release and use a new version.
