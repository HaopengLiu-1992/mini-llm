import ast
import re
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "distilbert/distilgpt2"
DATASET_NAME = "yufan/recsys-genrec-dataset"
DATASET_CONFIG = "Video_Games_seqrec"
TRAIN_SIZE = 1_000
EVAL_SIZE = 200
BATCH_SIZE = 4
MAX_LENGTH = 256
EPOCHS = 1
LEARNING_RATE = 5e-5
OUTPUT_DIR = "checkpoints/distilgpt2-recsys-sft"
SID_PATTERN = re.compile(r"<[abc]_\d+>")


def parse_history(value):
    """The dataset stores history lists as strings."""
    if isinstance(value, list):
        return value
    return ast.literal_eval(value)


def collect_sid_tokens(dataset):
    sid_tokens = set()
    for split in ("train", "validation"):
        for value in dataset[split]["history_item_sid"]:
            sid_tokens.update(SID_PATTERN.findall(value))
        sid_tokens.update(
            token
            for value in dataset[split]["item_sid"]
            for token in SID_PATTERN.findall(value)
        )
    return sorted(sid_tokens)


def build_prompt(example):
    history = parse_history(example["history_item_sid"])
    history_text = "\n".join(
        f"{index}. {sid}"
        for index, sid in enumerate(history, start=1)
    )

    return (
        "### Instruction:\n"
        "Recommend the next video game based on the user's semantic-ID history.\n\n"
        "### User History (Semantic IDs):\n"
        f"{history_text}\n\n"
        "### Response (Semantic ID):\n"
    )


def tokenize_example(example, tokenizer):
    prompt_ids = tokenizer(
        build_prompt(example),
        add_special_tokens=False,
    )["input_ids"]
    response_ids = tokenizer(
        example["item_sid"] + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]

    # Preserve the complete target whenever possible. If the sequence is too
    # long, truncate older history from the left and keep the response.
    response_ids = response_ids[:MAX_LENGTH]
    prompt_budget = max(0, MAX_LENGTH - len(response_ids))
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []

    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


def collate_fn(batch, tokenizer):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attention_masks = [
        torch.tensor(item["attention_mask"], dtype=torch.long)
        for item in batch
    ]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]

    return {
        "input_ids": pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ),
        "attention_mask": pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        ),
        "labels": pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        ),
    }


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            total_loss += model(**batch).loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device={device}")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sid_tokens = collect_sid_tokens(dataset)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": sid_tokens}
    )
    print(f"added_sid_tokens={len(sid_tokens)}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    train_size = min(TRAIN_SIZE, len(dataset["train"]))
    eval_size = min(EVAL_SIZE, len(dataset["validation"]))

    train_dataset = dataset["train"].select(range(train_size)).map(
        lambda example: tokenize_example(example, tokenizer),
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
    )
    eval_dataset = dataset["validation"].select(range(eval_size)).map(
        lambda example: tokenize_example(example, tokenizer),
        remove_columns=dataset["validation"].column_names,
        load_from_cache_file=False,
    )

    collator = lambda batch: collate_fn(batch, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0 or step == 1:
                print(
                    f"epoch={epoch + 1} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        eval_loss = evaluate(model, eval_loader, device)
        print(f"epoch={epoch + 1} eval_loss={eval_loss:.4f}")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    train()
