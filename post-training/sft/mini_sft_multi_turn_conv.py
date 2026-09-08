import torch
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer



model_id = "Qwen/Qwen2.5-0.5B"
device = "cuda"
max_len = 1024
batch_size = 4
lr = 1e-5

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

def show_generation():
    model.eval()
    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id, im_end_id],
        )

    gen = out[0][ids["input_ids"].shape[1]:]
    print(f"generated {len(gen)} tokens, last token: {tokenizer.convert_ids_to_tokens([gen[-1].item()])}")
    print(tokenizer.decode(gen, skip_special_tokens=True))

class SftDataset(TorchDataset):
    def __init__(self, ds, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.processed_ds = ds.map(self.process_example, remove_columns=ds.column_names).filter(lambda x: any(l != -100 for l in x["labels"]))

    def process_example(self, example):
        messages = example["messages"]
        input_ids, labels = [], []
        prev_ids = []
        for i, msg in enumerate(messages):
            curr_ids = self.tokenizer.apply_chat_template(messages[: i + 1], tokenize=True)["input_ids"]
            seg = curr_ids[len(prev_ids):]
            input_ids += seg
            labels += seg if msg["role"] == "assistant" else [-100] * len(seg)
            prev_ids = curr_ids
        return { "input_ids": input_ids[: self.max_len], "labels": labels[: self.max_len] }
    
    def __len__(self):
        return len(self.processed_ds)

    def __getitem__(self, idx):
        return self.processed_ds[idx]

def collate_fn(batch):
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
    attention_mask = [torch.ones(len(x), dtype=torch.long) for x in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

data = load_from_disk("data/ultrachat_10k")
training_ds = SftDataset(data["train"], tokenizer, max_len)
eval_ds = SftDataset(data["eval"], tokenizer, max_len)

training_dl = DataLoader(training_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=lr)

def training():
    model.train()
    total_loss = 0
    for step, batch in enumerate(training_dl, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if step % 100 == 0:
            print(f"step {step}/{len(training_dl)}  running loss = {total_loss / step:.4f}")
    return total_loss / len(training_dl)

def evaluate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in eval_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            total_loss += model(**batch).loss.item()
    return total_loss / len(eval_dl)

show_generation()
train_loss = training()
eval_loss = evaluate()
print(f"training loss = {train_loss:.4f}  eval loss = {eval_loss:.4f}")
show_generation()