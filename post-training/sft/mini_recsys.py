import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch import nn


raw_data = [
    {"history": [1, 3, 5],       "candidate": 7,  "label": 1},
    {"history": [2, 4],          "candidate": 8,  "label": 0},
    {"history": [1, 2, 9, 10],   "candidate": 3,  "label": 1},
    {"history": [6],             "candidate": 2,  "label": 0},
    {"history": [3, 7, 8],       "candidate": 5,  "label": 1},
    {"history": [4, 9],          "candidate": 1,  "label": 0},
    {"history": [2, 5, 7, 11],   "candidate": 10, "label": 1},
    {"history": [1, 8],          "candidate": 6,  "label": 0},
    {"history": [5, 6, 9],       "candidate": 11, "label": 1},
    {"history": [2, 3],          "candidate": 9,  "label": 0},
]

num_items = 20
embedding_dim = 16
padding_idx = 0

items_emb = nn.Embedding(num_items, embedding_dim)

class RecDataset(Dataset):
    def __init__(self, ds, max_len=5):
        super().__init__()
        self.ds = ds
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        history = torch.tensor(
            item["history"][:self.max_len],
            dtype=torch.long,
        )

        candidate = torch.tensor(
            item["candidate"],
            dtype=torch.long,
        )

        label = torch.tensor(
            item["label"],
            dtype=torch.float,
        )

        return {
            "history": history,
            "candidate": candidate,
            "label": label,
        }

def collate_fn(batch):
    histories = [item["history"] for item in batch]

    history = pad_sequence(histories, batch_first=True, padding_value=0)
    attention_mask = history.ne(0).long()

    candidate = torch.stack([
        item["candidate"]
        for item in batch
    ])

    labels = torch.stack([
        item["label"]
        for item in batch
    ])

    return {
        "history": history,
        "attention_mask": attention_mask,
        "candidate": candidate,
        "labels": labels,
    }

class RecModel(nn.Module):
    def __init__(
        self,
        num_items,
        embedding_dim,
        padding_idx=0,
    ):
        super().__init__()

        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(
        self,
        history,
        attention_mask,
        candidate,
    ):
        # history: [B, T]
        history_emb = self.item_embedding(history)
        # [B, T, D]

        mask = attention_mask.unsqueeze(-1).float()
        # [B, T, 1]

        masked_history_emb = history_emb * mask
        # [B, T, D]

        user_emb = (
            masked_history_emb.sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        )
        # [B, D]

        candidate_emb = self.item_embedding(candidate)
        # [B, D]

        logits = (
            user_emb * candidate_emb
        ).sum(dim=-1)
        # [B]

        return logits

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in data_loader:
        history = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        candidate = batch["candidate"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            history=history,
            attention_mask=attention_mask,
            candidate=candidate,
        )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
        )

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()
        targets = labels.long()

        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == targets).sum().item()
        total_examples += labels.size(0)

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


# --------------------------------------------------
# 6. Training
# --------------------------------------------------

def train():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"device={device}")

    train_data = raw_data[:8]
    eval_data = raw_data[8:]

    train_dataset = RecDataset(
        train_data,
        max_len=5,
    )

    eval_dataset = RecDataset(
        eval_data,
        max_len=5,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = RecModel(
        num_items=num_items,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-2,
    )

    # 先检查 batch shape
    first_batch = next(iter(train_loader))

    print({
        key: tuple(value.shape)
        for key, value in first_batch.items()
    })

    for epoch in range(1, 101):
        model.train()

        total_train_loss = 0.0
        total_train_examples = 0

        for batch in train_loader:
            history = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            candidate = batch["candidate"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                history=history,
                attention_mask=attention_mask,
                candidate=candidate,
            )

            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)
            total_train_examples += labels.size(0)

        if epoch % 10 == 0 or epoch == 1:
            train_loss = (
                total_train_loss
                / total_train_examples
            )

            eval_loss, eval_accuracy = evaluate(
                model,
                eval_loader,
                device,
            )

            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.4f} "
                f"eval_loss={eval_loss:.4f} "
                f"eval_accuracy={eval_accuracy:.4f}"
            )


if __name__ == "__main__":
    train()

