from transformers import AutoModelForCausalLM

from trainer import BaseTrainer
from datamodule import SFTDataModule

class SFTTrainer(BaseTrainer):
    def compute_loss(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        return outputs.loss


model = AutoModelForCausalLM.from_pretrained(
    "distilbert/distilgpt2"
)

data_module = SFTDataModule(
    dataset_name="yahma/alpaca-cleaned",
    model_name="distilbert/distilgpt2",
    train_size=2000,
    eval_size=200,
    batch_size=8,
    max_length=256,
)

data_module.prepare_data()

train_loader = data_module.train_dataloader()
eval_loader = data_module.eval_dataloader()

trainer = SFTTrainer(
    model=model,
    train_loader=train_loader,
    eval_loader=eval_loader,
    lr=5e-5,
)

trainer.train(
    epochs=3,
    max_steps=1000,
    log_every=10,
    eval_every=100,
    save_every=500,
)
