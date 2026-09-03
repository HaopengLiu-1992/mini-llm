import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_prompt(example):
    if example["input"].strip():
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n"
        )

    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert/distilgpt2")
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default="data/alpaca-dpo/train.jsonl")
    args = parser.parse_args()

    device = get_device()
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    dataset = load_dataset(args.dataset, split="train")
    size = min(args.size, len(dataset))
    dataset = dataset.select(range(size))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    written = 0
    with open(args.output, "w", encoding="utf-8") as output_file:
        for start in range(0, size, args.batch_size):
            examples = [dataset[i] for i in range(start, min(start + args.batch_size, size))]
            prompts = [format_prompt(example) for example in examples]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
            ).to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            rejected_answers = tokenizer.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
            )

            for example, prompt, rejected in zip(examples, prompts, rejected_answers):
                rejected = rejected.strip()
                if not rejected:
                    continue

                row = {
                    "prompt": prompt,
                    "chosen": example["output"].strip(),
                    "rejected": rejected,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            if (start // args.batch_size + 1) % 10 == 0 or start + args.batch_size >= size:
                print(f"processed={min(start + args.batch_size, size)}/{size} written={written}")

    print(f"saved={args.output} rows={written}")


if __name__ == "__main__":
    main()
