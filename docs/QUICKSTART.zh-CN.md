# 快速开始

安装 Python 3.11 和 FFmpeg，然后在仓库根目录运行：

```bash
git clone https://github.com/psychemme/PsycheMME.git
cd PsycheMME
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[data]'
cp .env.example .env
```

在 `.env` 中填写模型与文本 Judge 的服务地址、模型 ID 和 API Key。视频下载还需要先到 [Ego4D 官方](https://ego4d-data.org/docs/start-here/)申请，将收到的 AWS Access Key ID 和 Secret Access Key 分别填入 `EGO4D_ACCESS_KEY_ID` 与 `EGO4D_SECRET_ACCESS_KEY`。已有官方原视频可设置 `EGO4D_SOURCE_DIR`，详见[下载说明](DOWNLOAD.md)。

运行全部 1,000 道题、630 个视频的完整评测：

```bash
bash scripts/evaluate.sh
```

首次运行准备约 27.6 GB Benchmark 视频，并缓存所需的 76 个 Ego4D 原视频（约 48.3 GB）。其余七个来源的视频从 Hugging Face 获取。请为抽帧缓存预留额外空间。API 请求使用你配置的模型和 Judge 服务。

中断后重复同一命令即可继续。切换模型或题目集合时使用新的 `RUN_DIR`。临时网络错误会有限重试；修复服务后可在命令前加 `RETRY_ERRORS=1`，显式重试保存的执行错误。已完成的正常回答和格式失败不会自动重跑。

结果位于运行目录：`predictions.jsonl`、`judgments.jsonl`、`scores.jsonl`、`scores.summary.json`。详见 [模型接入](MODEL_INTEGRATION.md) 和 [数据格式](DATA_FORMAT.md)。

## 复算论文分数

```bash
bash scripts/replay.sh
```

从发布的评分因子复算 19,000 条分数并核对 190 个主表值。
