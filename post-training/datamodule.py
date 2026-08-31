import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from datasets import load_dataset
from transformers import AutoTokenizer

class BaseDataModule(ABC):
    def __init__(self, batch_size=8, seed=42,):
        self.batch_size = batch_size
        self.seed = seed
        self.train_dataset = None
        self.eval_dataset = None

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Call prepare_data() first")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def eval_dataloader(self):
        if self.eval_dataset is None:
            raise RuntimeError("Call prepare_data() first")

        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

class SFTDataModule(BaseDataModule):
    def __init__(
        self,
        dataset_name,
        model_name,
        train_size=2000,
        eval_size=200,
        batch_size=8,
        max_length=256,
        seed=42,
    ):
        super().__init__(
            batch_size=batch_size,
            seed=seed,
        )

        self.dataset_name = dataset_name
        self.model_name = model_name

        self.train_size = train_size
        self.eval_size = eval_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_example(self, example):
        if example["input"].strip():
            prompt = (
                f"### Instruction:\n"
                f"{example['instruction']}\n\n"
                f"### Input:\n"
                f"{example['input']}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"### Instruction:\n"
                f"{example['instruction']}\n\n"
                f"### Response:\n"
            )

        response = example["output"]

        return {
            "prompt": prompt,
            "response": response,
        }

    def tokenize_example(self, example):
        prompt_ids = self.tokenizer(
            example["prompt"],
            add_special_tokens=False,
        )["input_ids"]

        response_ids = self.tokenizer(
            example["response"] + self.tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + response_ids

        labels = (
            [-100] * len(prompt_ids)
            + response_ids
        )

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def process_dataset(self, dataset):
        formatted = dataset.map(
            self.format_example
        )

        tokenized = formatted.map(
            self.tokenize_example,
            remove_columns=formatted.column_names,
        )

        return tokenized

    def prepare_data(self):
        dataset = load_dataset(
            self.dataset_name,
            split="train",
        )

        dataset = dataset.shuffle(
            seed=self.seed
        )

        raw_train = dataset.select(
            range(self.train_size)
        )

        raw_eval = dataset.select(
            range(
                self.train_size,
                self.train_size + self.eval_size,
            )
        )

        self.train_dataset = self.process_dataset(
            raw_train
        )

        self.eval_dataset = self.process_dataset(
            raw_eval
        )

    def collate_fn(self, batch):
        max_len = max(
            len(x["input_ids"])
            for x in batch
        )

        input_ids = []
        attention_masks = []
        labels = []

        for x in batch:
            pad_len = (
                max_len - len(x["input_ids"])
            )

            input_ids.append(
                x["input_ids"]
                + [self.tokenizer.pad_token_id] * pad_len
            )

            attention_masks.append(
                x["attention_mask"]
                + [0] * pad_len
            )

            labels.append(
                x["labels"]
                + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }
