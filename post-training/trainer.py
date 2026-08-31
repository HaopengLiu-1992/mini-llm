import os
import torch
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, model, train_loader, eval_loader = None, lr=5e-5, grad_clip=1.0):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.grad_clip = grad_clip
        self.global_step = 0

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,)

    def move_batch_to_device(self, batch):
        return { k: v.to(self.device) for k, v in batch.items() }

    @abstractmethod
    def compute_loss(self, batch):
        pass

    def train_step(self, batch):
        self.model.train()
        batch = self.move_batch_to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.compute_loss(batch)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip,
            )

        self.optimizer.step()
        self.global_step += 1
        return loss.detach().item()
    
    @torch.no_grad()
    def evaluate(self):
        if self.eval_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.eval_loader:
            batch = self.move_batch_to_device(batch)
            loss = self.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches
    
    def train(self, epochs=1, max_steps=None, log_every=10, eval_every=None, save_every=None, save_dir="./checkpoints",):
        for epoch in range(epochs):
            for batch in self.train_loader:
                loss = self.train_step(batch)
                if self.global_step % log_every == 0:
                    print(f"epoch={epoch} step={self.global_step} loss={loss:.4f}")
                if eval_every is not None and self.eval_loader is not None and self.global_step % eval_every == 0:
                    eval_loss = self.evaluate()
                    print(f"step={self.global_step} eval_loss={eval_loss:.4f}")
                if save_every is not None and self.global_step % save_every == 0:
                    path = os.path.join(save_dir, f"step-{self.global_step}")
                    self.save_checkpoint(path)
                if max_steps is not None and self.global_step >= max_steps:
                    return

    def save_checkpoint(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        torch.save({"optimizer_state_dict": self.optimizer.state_dict(), "global_step": self.global_step}, os.path.join(path, "trainer_state.pt"))
    
