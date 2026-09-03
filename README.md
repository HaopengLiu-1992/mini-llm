# Mini LLM

一个用于学习 GPT-2 从模型推理到监督微调（SFT）流程的最小 LLM 实现。

A minimal LLM implementation for learning the GPT-2 workflow, from model inference to supervised fine-tuning (SFT).

## 功能 | Features

- 手写 GPT-2 模型 | A hand-written GPT-2 model
- GPT-2 预训练权重加载 | GPT-2 pretrained weight loading
- ModelRunner
- Greedy Sampler
- Scheduler
- Continuous Batching
- Alpaca 数据集预处理 | Alpaca dataset preprocessing
- 基于 PyTorch 的监督微调训练器 | A PyTorch-based supervised fine-tuning trainer

## 安装 | Installation

在项目根目录执行以下命令：

Run the following commands from the project root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 运行推理 | Run Inference

项目内置 `tiny-gpt2` 模型，不需要额外下载。

The project includes a `tiny-gpt2` model, so no additional model download is required.

```bash
.venv/bin/python -m examples.run_tiny_gpt2
```

模型位于：

The model is located at:

```text
inference/model/tiny-gpt2
```

## 运行 SFT 训练 | Run SFT Training

以下命令会加载 Alpaca 数据集，使用 `distilbert/distilgpt2` 进行监督微调。

The following command loads the Alpaca dataset and performs supervised fine-tuning with `distilbert/distilgpt2`.

```bash
PYTHONPATH=post-training .venv/bin/python post-training/sft/sft-trainer.py
```

训练检查点默认保存到 `checkpoints/`。

Training checkpoints are saved to `checkpoints/` by default.

## 项目结构 | Project Structure

```text
inference/
├── layers/             # Transformer 基础层 | Transformer building blocks
├── model/              # GPT-2 和权重加载 | GPT-2 and weight loading
└── engine/             # Engine、Scheduler、Sampler
post-training/
├── datamodule.py       # 数据集加载、格式化与批处理 | Dataset loading, formatting, and batching
├── trainer.py          # 通用训练器 | Generic trainer
└── sft/                # SFT 数据处理与训练入口 | SFT data processing and training entry point
```
