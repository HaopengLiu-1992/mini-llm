import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("yahma/alpaca-cleaned", split="train")
ds = dataset.shuffle(seed=42)
train_dataset = ds.select(range(2000))
eval_dataset = ds.select(range(2000, 2200))
print(type(ds))

def format_example(example):
    if example["input"].strip():
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n"
        )
    response = example["output"]
    return {
        "prompt": prompt,
        "response": response,
        "text": prompt + response,
    }

formatted_dataset = train_dataset.map(format_example)


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
print(tokenizer.pad_token)
print(tokenizer.eos_token)
tokenizer.pad_token = tokenizer.eos_token

example = formatted_dataset[0]

prompt = example["prompt"]
response = example["response"]

prompt_ids = tokenizer(
    prompt,
    add_special_tokens=False,
)["input_ids"]

response_ids = tokenizer(
    response + tokenizer.eos_token,
    add_special_tokens=False,
)["input_ids"]


MAX_LENGTH = 256

def tokenize_example(example):
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=False,
    )["input_ids"]

    response_ids = tokenizer(
        example["response"] + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = prompt_ids + response_ids

    labels = [-100] * len(prompt_ids)+ response_ids

    # truncate
    input_ids = input_ids[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized_dataset = formatted_dataset.map(
    tokenize_example,
    remove_columns=formatted_dataset.column_names,
)

def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for x in batch:
        n = len(x["input_ids"])
        pad_len = max_len - n

        input_ids.append(
            x["input_ids"] +
            [tokenizer.pad_token_id] * pad_len
        )

        attention_masks.append(
            x["attention_mask"] +
            [0] * pad_len
        )

        labels.append(
            x["labels"] +
            [-100] * pad_len
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

x = tokenized_dataset[0]

print(len(x["input_ids"]))
print(len(x["attention_mask"]))
print(len(x["labels"]))

assert len(x["input_ids"]) == len(x["labels"])
assert len(x["input_ids"]) == len(x["attention_mask"])

train_loader = DataLoader(
    tokenized_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
)
batch = next(iter(train_loader))

# print(batch["input_ids"].shape)
# print(batch["attention_mask"].shape)
# print(batch["labels"].shape)
# print(batch["input_ids"][0])
# print(batch["attention_mask"][0])
# print(batch["labels"][0])

def train(epoches):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    model.to(device)

    print(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
    )

    for epoch in range(epoches):
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={step} "
                    f"loss={loss.item():.4f}"
                )
    save_path = "./checkpoints/distilgpt2-sft-step20"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

train(1)