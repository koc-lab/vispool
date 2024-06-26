from os import PathLike
from typing import Any, Mapping

import lightning as L
from datasets import DatasetDict, disable_caching, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing_extensions import override

from vispool import GLUE_LOADER_COLUMNS, GLUE_NUM_LABELS, GLUE_TASKS, GLUE_TEXT_FIELDS

disable_caching()


def get_glue_task_dataset(task_name: str) -> DatasetDict:
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Task name {task_name} not in {GLUE_TASKS}")
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

    @override
    def prepare_data(self) -> None:
        get_glue_task_dataset(self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    @override
    def setup(self, stage: str) -> None:
        target_splits: set[str] = set()
        if stage == "fit":
            target_splits = {"train", "validation"}
        elif stage == "validate":
            target_splits = {"validation"}
        else:
            target_splits = {"test"}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.dataset_dict = get_glue_task_dataset(self.task_name)
        for split in self.dataset_dict.keys():
            if split in target_splits:
                self.dataset_dict[split] = self.dataset_dict[split].map(self.encode, batched=True)
                self.columns = [c for c in self.dataset_dict[split].column_names if c in GLUE_LOADER_COLUMNS]
                self.dataset_dict[split].set_format(type="torch", columns=self.columns)

    @override
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_dict["train"],  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_dict["validation"],  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_dict["test"],  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def encode(self, batch: Mapping[str, list]) -> Any:
        if len(self.text_fields) > 1:
            texts = list(zip(*[batch[field] for field in self.text_fields]))
        else:
            texts = batch[self.text_fields[0]]

        tokenized = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        if "label" in batch.keys():
            tokenized["labels"] = batch["label"]
        return tokenized
