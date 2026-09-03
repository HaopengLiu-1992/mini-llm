# Mini LLM

[简体中文](README_zh-CN.md)

A minimal LLM implementation for learning the GPT-2 workflow, from model inference to supervised fine-tuning (SFT) and direct preference optimization (DPO).

## Features

- A hand-written GPT-2 model
- GPT-2 pretrained weight loading
- ModelRunner
- Greedy Sampler
- Scheduler
- Continuous Batching
- Alpaca dataset preprocessing
- A PyTorch-based supervised fine-tuning trainer
- DPO preference-data generation and training

## Installation

Run the following commands from the project root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run Inference

The project includes a `tiny-gpt2` model, so no additional model download is required.

```bash
.venv/bin/python -m examples.run_tiny_gpt2
```

The model is located at:

```text
inference/model/tiny-gpt2
```

## Run SFT Training

The following command loads the Alpaca dataset and performs supervised fine-tuning with `distilbert/distilgpt2`.

```bash
PYTHONPATH=post-training .venv/bin/python post-training/sft/sft_trainer.py
```

Training checkpoints are saved to `checkpoints/` by default.

## Run DPO Training

First, generate preference data. The recommended starting point is the SFT checkpoint:

```bash
.venv/bin/python post-training/dpo/generate_alpaca_dpo.py \
  --model checkpoints/distilgpt2-sft-step20 \
  --size 2000 \
  --output data/alpaca-dpo/train.jsonl
```

The generated JSONL file contains `prompt`, `chosen`, and `rejected` fields. Then run DPO training:

```bash
.venv/bin/python post-training/dpo/dpo_trainer.py
```

The DPO script trains for one epoch by default, using 1,800 training examples and 200 evaluation examples. The final checkpoint is saved to `checkpoints/distilgpt2-dpo-epoch1/`.

## Project Structure

```text
inference/
├── layers/             # Transformer building blocks
├── model/              # GPT-2 and weight loading
└── engine/             # Engine, Scheduler, and Sampler
post-training/
├── datamodule.py       # Dataset loading, formatting, and batching
├── trainer.py          # Generic trainer
├── sft/                # SFT data processing and training entry point
└── dpo/                # DPO data generation and training entry point
```
