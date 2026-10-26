# Download the benchmark

For video preparation, first apply for [Ego4D access](https://ego4d-data.org/docs/start-here/). Approval may take time; request access before starting the video download.

Install the evaluation package and FFmpeg, then copy `.env.example` to `.env`.

## Ego4D access

Apply through the [official Ego4D page](https://ego4d-data.org/docs/start-here/). Once approved, paste the supplied AWS **Access Key ID** and **Secret Access Key** into these fields in your local `.env`:

```dotenv
EGO4D_ACCESS_KEY_ID=
EGO4D_SECRET_ACCESS_KEY=
```

Configure the supplied key pair in your local `.env`. The script passes it only to the official Ego4D downloader through a temporary local AWS profile, which is deleted after use. PsycheMME does not log the keys or upload them to GitHub or Hugging Face.

Fill `EGO4D_SESSION_TOKEN` only if it was included in your official credentials. An existing named AWS profile can be used instead through `EGO4D_PROFILE`.

## Prepare all videos

```bash
bash scripts/download.sh
```

The script downloads annotations and the other seven video sources from Hugging Face. For Ego4D it uses the official downloader to fetch only the **76 source videos** needed for the benchmark, then prepares **142 clips** at the recorded source offsets using the benchmark's video and loudness-processing recipe. All **630 clips** are placed in `data/PsycheMME/benchmark/media/video/` with their existing media IDs.

Allow approximately **76 GB** for the **48.3 GB** Ego4D source cache and **27.6 GB** prepared benchmark media tree, plus temporary files and frame caches. Source videos are reused on subsequent runs. The official CLI handles its download cache; completed clips are checked and reused after interruption. Renew expired credentials through Ego4D and rerun the same command.

If you already downloaded the official Ego4D originals, set `EGO4D_SOURCE_DIR` to that download root or its `v2/full_scale/` directory. Existing originals are used directly.

Preparation writes `metadata/media.local.jsonl`, the local video index used by `scripts/evaluate.sh`. Questions, references, ASR and the published metadata remain unchanged.

## Evaluate

Configure the model and Judge fields in `.env`, then run:

```bash
bash scripts/evaluate.sh
```

This also prepares missing media, so the separate download command is optional.

For annotations only:

```bash
bash scripts/download.sh --metadata-only
```
