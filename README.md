# Mini LLM

一个用于学习 GPT-2 和 LLM 推理流程的最小实现。

当前包含：

- 手写 GPT-2 模型
- GPT-2 预训练权重加载
- ModelRunner
- Greedy Sampler
- Scheduler
- Continuous Batching

## 安装

在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 运行

项目内置 `tiny-gpt2` 模型，不需要额外下载：

```bash
.venv/bin/python -m examples.run_tiny_gpt2
```

模型位于：

```text
inference/model/tiny-gpt2
```

## 项目结构

```text
inference/
├── layers/    # Transformer 基础层
├── model/     # GPT-2 和权重加载
└── engine/    # Engine、Scheduler、Sampler
```
