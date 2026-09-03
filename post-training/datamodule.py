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


class DPODataModule(BaseDataModule):
    """Data module for preference pairs with prompt/chosen/rejected fields."""

    def __init__(
        self,
        dataset_path,
        model_name,
        train_size=2000,
        eval_size=200,
        batch_size=4,
        max_length=256,
        seed=42,
    ):
        super().__init__(batch_size=batch_size, seed=seed)

        self.dataset_path = dataset_path
        self.model_name = model_name
        self.train_size = train_size
        self.eval_size = eval_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_pair(self, prompt, chosen, rejected):
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        chosen_ids = self.tokenizer(
            chosen + self.tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]
        rejected_ids = self.tokenizer(
            rejected + self.tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]

        # Use the same prompt tokens for both preference candidates. Reserve
        # enough room for the longer response and truncate the prompt from
        # the left so its ending remains visible.
        response_budget = max(len(chosen_ids), len(rejected_ids))
        prompt_budget = max(0, self.max_length - response_budget)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []

        response_budget = self.max_length - len(prompt_ids)
        if response_budget <= 0:
            prompt_ids = []
            response_budget = self.max_length

        chosen_ids = chosen_ids[:response_budget]
        rejected_ids = rejected_ids[:response_budget]

        def build_sequence(response_ids):
            return {
                "input_ids": prompt_ids + response_ids,
                "attention_mask": [1] * (len(prompt_ids) + len(response_ids)),
                "labels": [-100] * len(prompt_ids) + response_ids,
            }

        return build_sequence(chosen_ids), build_sequence(rejected_ids)

    def tokenize_example(self, example):
        chosen, rejected = self.tokenize_pair(
            example["prompt"],
            example["chosen"],
            example["rejected"],
        )

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
        }

    def prepare_data(self):
        dataset = load_dataset(
            "json",
            data_files=self.dataset_path,
            split="train",
        ).shuffle(seed=self.seed)

        required_columns = {"prompt", "chosen", "rejected"}
        missing_columns = required_columns - set(dataset.column_names)
        if missing_columns:
            raise ValueError(
                f"DPO dataset is missing columns: {sorted(missing_columns)}"
            )

        total_size = len(dataset)
        if total_size < 2:
            raise ValueError("DPO dataset must contain at least two examples")

        eval_size = min(self.eval_size, total_size - 1)
        train_size = min(self.train_size, total_size - eval_size)

        raw_train = dataset.select(range(train_size))
        raw_eval = dataset.select(
            range(train_size, train_size + eval_size)
        )

        columns = dataset.column_names
        self.train_dataset = raw_train.map(
            self.tokenize_example,
            remove_columns=columns,
            load_from_cache_file=False,
        )
        self.eval_dataset = raw_eval.map(
            self.tokenize_example,
            remove_columns=columns,
            load_from_cache_file=False,
        )

    def collate_side(self, batch, prefix):
        input_key = f"{prefix}_input_ids"
        attention_key = f"{prefix}_attention_mask"
        labels_key = f"{prefix}_labels"

        max_len = max(len(example[input_key]) for example in batch)
        input_ids = []
        attention_masks = []
        labels = []

        for example in batch:
            pad_len = max_len - len(example[input_key])
            input_ids.append(
                example[input_key]
                + [self.tokenizer.pad_token_id] * pad_len
            )
            attention_masks.append(
                example[attention_key] + [0] * pad_len
            )
            labels.append(
                example[labels_key] + [-100] * pad_len
            )

        return {
            input_key: torch.tensor(input_ids, dtype=torch.long),
            attention_key: torch.tensor(attention_masks, dtype=torch.long),
            labels_key: torch.tensor(labels, dtype=torch.long),
        }

    def collate_fn(self, batch):
        result = self.collate_side(batch, "chosen")
        result.update(self.collate_side(batch, "rejected"))
        return result
