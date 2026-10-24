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

在 `.env` 中填写模型与文本 Judge 的服务地址、模型 ID 和 API Key。

运行全部 1,000 道题、630 个视频的完整评测：

```bash
bash scripts/evaluate.sh
```

首次运行下载约 27.6 GB 视频及标注，请为抽帧缓存预留额外空间。API 请求使用你配置的模型和 Judge 服务。

中断后重复同一命令即可继续。切换模型或题目集合时使用新的 `RUN_DIR`。临时网络错误会有限重试；修复服务后可在命令前加 `RETRY_ERRORS=1`，显式重试保存的执行错误。已完成的正常回答和格式失败不会自动重跑。

结果位于运行目录：`predictions.jsonl`、`judgments.jsonl`、`scores.jsonl`、`scores.summary.json`。详见 [模型接入](MODEL_INTEGRATION.md) 和 [数据格式](DATA_FORMAT.md)。

## 复算论文分数

```bash
bash scripts/replay.sh
```

从发布的评分因子复算 19,000 条分数并核对 190 个主表值。
