import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM

import time

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

class SftDataset(Dataset):
    def __init__(self, ds, tokenizer, max_len = 512):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.eos_token_id = self.tokenizer.eos_token_id
        self.processed_ds = self.ds.map(self.process_example, remove_columns=self.ds.column_names)
    
    def process_example(self, example):
        instruction = example['instruction']
        response = example["output"]
        prompt = (
            f"### Instruction:\n"
            f"{instruction}\n\n"
            f"### Response:\n"
        )
        prompt_ids = self.tokenizer(prompt,add_special_tokens=False)["input_ids"]
        resp_ids = self.tokenizer(response,add_special_tokens=False)["input_ids"]
        resp_ids.append(self.eos_token_id)
        
        labels = [-100] * len(prompt_ids) + resp_ids
        input_ids = prompt_ids + resp_ids

        labels = labels[:self.max_len]
        input_ids = input_ids[:self.max_len]
        return { "input_ids": input_ids, "labels": labels }
    
    def __len__(self):
        return len(self.processed_ds)
    
    def __getitem__(self, idx):
        example = self.processed_ds[idx]
        input_ids_tensor = torch.tensor(example['input_ids'], dtype = torch.long)
        labels_tensor = torch.tensor(example['labels'], dtype = torch.long)
        return { "input_ids": input_ids_tensor, 'labels': labels_tensor }

ds = load_dataset("yahma/alpaca-cleaned", split="train")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
training_dataset = SftDataset(ds.select(range(1000)), tokenizer)
eval_dataset = SftDataset(ds.select(range(1000, 1200)), tokenizer)

# padding and attention mask
def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [torch.ones(len(x["input_ids"]), dtype=torch.long) for x in batch]
    labels = [x["labels"] for x in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return { "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels }

training_dl= DataLoader(training_dataset, batch_size=8, shuffle=True, collate_fn = collate_fn)
eval_dl= DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn = collate_fn)

model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to(device)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

def training():
    model.to(device)
    model.train()
    total_loss = 0
    batch_times = []

    for step, batch in enumerate(training_dl):
        if step >= 2:  # skip the first two warm-up batches
            synchronize(device)
            start_time = time.perf_counter()

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()

        if step >= 2:
            synchronize(device)
            batch_times.append(time.perf_counter() - start_time)

        total_loss += loss.item()

        if step == 3:
            break

    avg_loss = total_loss / len(training_dl)
    avg_batch_time = sum(batch_times) / len(batch_times)

    print(f"Device: {device}")
    print(f"Training loss: {avg_loss:.4f}")
    print(f"Average batch time: {avg_batch_time:.4f} seconds")

def eval():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in eval_dl:
            batch = { k: v.to(device) for k, v in batch.items() }
            output = model(**batch)
            loss = output.loss
            print(f"loss = {loss.item():.4f}")
            total_loss += loss.item()
    avg_loss = total_loss / len(eval_dl)
    print(f"eval loss = {avg_loss:.4f}")


training()
eval()