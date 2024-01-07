from os import PathLike
from typing import Any

import lightning as L
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from vispool import GLUE_LOADER_COLUMNS, GLUE_NUM_LABELS, GLUE_TASKS, GLUE_TEXT_FIELDS


def get_glue_task_dataset(task_name: str) -> DatasetDict:
    adjusted_task_name = "mnli" if task_name == "mnli-mm" else task_name
    dataset_dict = load_dataset("glue", adjusted_task_name, trust_remote_code=True)
    if type(dataset_dict) is not DatasetDict:
        raise ValueError(f"Dataset {adjusted_task_name} is not a DatasetDict")

    if task_name in {"mnli", "mnli-mm"}:
        dataset_dict = _get_mnli_dataset(dataset_dict, task_name)
    return dataset_dict


def _get_mnli_dataset(dataset_dict: DatasetDict, task_name: str) -> DatasetDict:
    version = "mismatched" if task_name == "mnli-mm" else "matched"
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict[f"validation_{version}"]
    test_dataset = dataset_dict[f"test_{version}"]
    return DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
    )


class GLUEDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str | PathLike[str],
        task_name: str,
        max_seq_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        if task_name not in GLUE_TASKS:
            raise ValueError(f"Task name {task_name} not in {GLUE_TASKS}")
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_fields = GLUE_TEXT_FIELDS[task_name]
        self.num_labels = GLUE_NUM_LABELS[task_name]

    def prepare_data(self) -> None:
        get_glue_task_dataset(self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.dataset_dict = get_glue_task_dataset(self.task_name)
        for split in self.dataset_dict.keys():
            self.dataset_dict[split] = self.dataset_dict[split].map(self.encode, batched=True)
            self.columns = [c for c in self.dataset_dict[split].column_names if c in GLUE_LOADER_COLUMNS]
            self.dataset_dict[split].set_format(type="torch", columns=self.columns)

        self.dataset_dict.set_format(
            type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
        )

    def encode(self, batch: Any) -> Any:
        return batch
        # return self.tokenizer(
        #     examples["sentence1"],
        #     examples["sentence2"],
        #     padding="max_length",
        #     truncation=True,
        # )
