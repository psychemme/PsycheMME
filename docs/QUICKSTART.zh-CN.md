# 中文快速上手

在仓库根目录使用 Python 3.11：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[data]'
psychemme doctor
psychemme demo --output runs/demo
```

预期：8 个合成示例，Raw **75.00%**、Verified **71.18%**。输入、评测答案、预测、Judge 标签及结果分别写入 `runs/demo/`。示例包含错答、缺失回答及降低的推理权重，不调用模型或 Judge，不需要 GPU 或 API Key。再次运行请换新输出目录。

## 视频推理

安装 FFmpeg 和 `python -m pip install '.[local-model]'` 后，按[视频示例](VIDEO.md)运行。SFT 和 GRPO 都直接加载在原始 Qwen3.5-4B 上；两者不叠加。示例复用评测器的多模态加载及全时段 1 fps / 最多 64 帧设置，需要兼容的 GPU。

## Benchmark 与论文结果

[Benchmark](https://huggingface.co/datasets/psychemme/PsycheMME) 和[评分记录](https://huggingface.co/datasets/psychemme/PsycheMME-logs)分开发布。下载完整 Benchmark：

```bash
python scripts/sync_resources.py benchmark
```

包括 1,000 道题、参考答案、630 段视频、ASR 和来源区间。视频约 27.6 GB，下载支持续传；加 `--metadata-only` 可跳过视频。

复算论文结果（不需要视频、GPU 或 API Key）：

```bash
bash scripts/replay.sh
```

完整论文日志预期包含 19 个模型条目、19,000 条逐题评分因子及 190 个主表数值。Replay 根据已保存的标签重算分数，不重新调用模型或 Judge。

参考答案和 Evidence 仅用于评测，不能进入被测模型输入。完整运行条件见[复现指南](REPRODUCIBILITY.md)，问题排查见[排错指南](TROUBLESHOOTING.md)。
