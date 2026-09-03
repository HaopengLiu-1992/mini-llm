# Mini LLM

[English](README.md)

一个用于学习 GPT-2 从模型推理到监督微调（SFT）流程的最小 LLM 实现。

## 功能

- 手写 GPT-2 模型
- GPT-2 预训练权重加载
- ModelRunner
- Greedy Sampler
- Scheduler
- Continuous Batching
- Alpaca 数据集预处理
- 基于 PyTorch 的监督微调训练器

## 安装

在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 运行推理

项目内置 `tiny-gpt2` 模型，不需要额外下载：

```bash
.venv/bin/python -m examples.run_tiny_gpt2
```

模型位于：

```text
inference/model/tiny-gpt2
```

## 运行 SFT 训练

以下命令会加载 Alpaca 数据集，使用 `distilbert/distilgpt2` 进行监督微调：

```bash
PYTHONPATH=post-training .venv/bin/python post-training/sft/sft-trainer.py
```

训练检查点默认保存到 `checkpoints/`。

## 项目结构

```text
inference/
├── layers/             # Transformer 基础层
├── model/              # GPT-2 和权重加载
└── engine/             # Engine、Scheduler 和 Sampler
post-training/
├── datamodule.py       # 数据集加载、格式化与批处理
├── trainer.py          # 通用训练器
└── sft/                # SFT 数据处理与训练入口
```
