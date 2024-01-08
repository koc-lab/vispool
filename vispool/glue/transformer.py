from os import PathLike
from typing import Any, Mapping

import lightning as L
import torch
from evaluate import EvaluationModule, load
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel

from vispool import GLUE_NUM_LABELS, GLUE_TASKS


def get_glue_task_metric(task_name: str) -> EvaluationModule:
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Task name {task_name} not in {GLUE_TASKS}")
    adjusted_task_name = "mnli" if task_name == "mnli-mm" else task_name
    return load("glue", adjusted_task_name)


def is_token_type_ids_input(transformer_model: PreTrainedModel) -> bool:
    return "token_type_ids" in transformer_model.forward.__code__.co_varnames


class GLUETransformer(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str | PathLike[str],
        task_name: str,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-8,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon

        self.num_labels = GLUE_NUM_LABELS[task_name]
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = get_glue_task_metric(task_name)
        self.use_token_type_ids = is_token_type_ids_input(self.model)

    def forward(self, **inputs: dict) -> Any:
        return self.model(**inputs)

    def training_step(self, batch: Mapping, batch_idx: int) -> dict:
        # inputs = batch if self.use_token_type_ids else {k: v for k, v in batch.items() if k != "token_type_ids"}
        inputs = batch
        outputs = self(**inputs)
        return {"loss": outputs["loss"]}

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self(**batch)
        val_loss, logits = outputs["loss"], outputs["logits"]
        preds = logits.squeeze() if self.num_labels == 1 else torch.argmax(logits, dim=-1)
        labels = batch["labels"]
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        log_dict = {"val_loss": val_loss} if metric_dict is None else {"val_loss": val_loss, **metric_dict}
        self.log_dict(log_dict, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)
        return optimizer
