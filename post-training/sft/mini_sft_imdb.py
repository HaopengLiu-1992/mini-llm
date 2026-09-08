from sympy.polys import total_degree
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

model_name = "distilbert-base-uncased"
dataset_name = "stanfordnlp/imdb"
device = 'mps'

model = AutoModelForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased', num_labels=2)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


ds = load_dataset(dataset_name, split = 'train')
ds = ds.shuffle(seed=42)
print(ds[0])

raw_training = ds.select(range(1000))
raw_eval = ds.select(range(1000, 1100))

class SftDataset(Dataset):
    def __init__(self, ds, tokenizer, max_len = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.processed_ds = ds.map(self.process_example, remove_columns=ds.column_names)
    
    def process_example(self, example):
        input_ids = self.tokenizer(example["text"], truncation=True)['input_ids']
        input_ids = input_ids[:self.max_len]
        return { 'input_ids': input_ids, 'label': example["label"]} 
    
    def __len__(self):
        return len(self.processed_ds)
    
    def __getitem__(self, idx):
        return self.processed_ds[idx]

training = SftDataset(raw_training, tokenizer)
eval = SftDataset(raw_eval, tokenizer)

# attn mask and padding
def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids'], dtype = torch.long) for item in batch]
    attn_mask = [torch.tensor([1] * len(item['input_ids']), dtype = torch.long) for item in batch]
    labels = [torch.tensor(item['label'], dtype = torch.long) for item in batch]
    labels = torch.stack(labels)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
    return { "input_ids": input_ids, "attention_mask": attn_mask, "labels": labels }

training_dl = DataLoader(training, batch_size=4, shuffle=True, collate_fn=collate_fn)
eval_dl = DataLoader(eval, batch_size=4, shuffle=False, collate_fn=collate_fn)
lr = 1e-5
optimizer = AdamW(model.parameters(), lr=lr)

def training():
    model.train()
    for batch in training_dl:
        batch = { k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()

def eval():
    model.eval();
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in eval_dl:
            batch = { k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            logits = output.logits
            pred = logits.argmax(dim=-1)
            correct += (pred == batch["labels"]).sum().item()
            total += batch["labels"].numel()
        accuracy = correct / total
        print(f"accuracy = {accuracy:.4f}")

eval()
training()
eval()