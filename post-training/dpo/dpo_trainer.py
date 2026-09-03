from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# Allow the script to import shared training modules when run directly:
# `python post-training/dpo/dpo_trainer.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer import BaseTrainer
from datamodule import DPODataModule

class DPOTrainer(BaseTrainer):
    def __init__(self, ref_model, beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ref_model = ref_model.to(self.device)
        self.ref_model.eval()
        self.beta = beta
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def sequence_logprob(self, model, input_ids, attention_mask, labels):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        logits = outputs.logits[:, :-1, :]
        target = labels[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)

        safe_target = target.clamp_min(0)

        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=safe_target.unsqueeze(-1),
        ).squeeze(-1)

        response_mask = target.ne(-100)

        return (
            token_log_probs * response_mask
        ).sum(dim=-1)


    def compute_loss(self, batch):
        policy_chosen = self.sequence_logprob(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )

        policy_rejected = self.sequence_logprob(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )

        with torch.no_grad():
            ref_chosen = self.sequence_logprob(
                self.ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )

            ref_rejected = self.sequence_logprob(
                self.ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )
        policy_gap = policy_chosen - policy_rejected
        reference_gap = ref_chosen - ref_rejected
        logits = self.beta * (policy_gap - reference_gap)
        return -F.logsigmoid(logits).mean()


def train(
    epochs=1,
    model_name="checkpoints/distilgpt2-sft-step20",
    dataset_path="data/alpaca-dpo/train.jsonl",
    train_size=1800,
    eval_size=200,
    batch_size=4,
    max_length=256,
    lr=5e-6,
    beta=0.1,
    output_dir="checkpoints/distilgpt2-dpo-epoch1",
):
    """Run DPO training for one epoch by default."""
    if not Path(model_name).exists():
        print(
            f"warning: {model_name} not found; "
            "using distilbert/distilgpt2 instead"
        )
        model_name = "distilbert/distilgpt2"

    policy_model = AutoModelForCausalLM.from_pretrained(model_name)
    reference_model = AutoModelForCausalLM.from_pretrained(model_name)

    data_module = DPODataModule(
        dataset_path=dataset_path,
        model_name=model_name,
        train_size=train_size,
        eval_size=eval_size,
        batch_size=batch_size,
        max_length=max_length,
    )
    data_module.prepare_data()

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=reference_model,
        train_loader=data_module.train_dataloader(),
        eval_loader=data_module.eval_dataloader(),
        lr=lr,
        beta=beta,
    )

    trainer.train(
        epochs=epochs,
        log_every=10,
        eval_every=100,
    )

    trainer.save_checkpoint(output_dir)
    data_module.tokenizer.save_pretrained(output_dir)
    print(f"saved DPO checkpoint to {output_dir}")


if __name__ == "__main__":
    train()
