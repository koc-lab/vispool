from typing import Any

import datasets
import lightning as L
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vispool import glue_loader_columns, glue_tasks_num_labels, glue_tasks_text_fields


class GLUEDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "cola",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._dataset = None
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = glue_tasks_text_fields[task_name]
        self.num_labels = glue_tasks_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    @property
    def dataset(self) -> datasets.DatasetDict:
        if self._dataset is None:
            self._dataset = datasets.load_dataset("glue", self.task_name)
        if type(self._dataset) != datasets.dataset_dict.DatasetDict:
            raise ValueError(f"Dataset type {type(self._dataset)}")
        return self._dataset

    def setup(self, stage: Any = None) -> None:
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                # remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in glue_loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self) -> None:
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=4
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=4)
                for x in self.eval_splits
            ]
        else:
            raise ValueError("No validation splits found")

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=4)
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=4)
                for x in self.eval_splits
            ]
        else:
            raise ValueError("No validation splits found")

    def convert_to_features(self, example_batch: Any, indices: Any = None) -> Any:
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    example_batch[self.text_fields[0]],
                    example_batch[self.text_fields[1]],
                )
            )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features
