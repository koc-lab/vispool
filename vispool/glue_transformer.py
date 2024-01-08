from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import evaluate
import lightning as L
import torch
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


class GLUETransformer(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
            "glue",
            self.hparams["task_name"],
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        )
        self.outputs: dict = defaultdict(list)

    def forward(self, **inputs: Any) -> Any:
        target_key = "token_type_ids"
        if target_key in inputs.keys() and (inputs[target_key] is None or not inputs["token_type_ids"].any()):
            return self.model(**{k: v for k, v in inputs.items() if k != target_key})
        return self.model(**inputs)

    def training_step(self, batch: Any, batch_idx: Any) -> torch.Tensor:
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch: Any, batch_idx: Any, dataloader_idx: int = 0) -> None:
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams["num_labels"] > 1:
            preds = torch.argmax(logits, dim=1)
        elif self.hparams["num_labels"] == 1:
            preds = logits.squeeze()
        else:
            raise ValueError("Invalid number of labels.")

        labels = batch["labels"]

        self.outputs[dataloader_idx].append({"loss": val_loss, "preds": preds, "labels": labels})

    def on_validation_epoch_end(self) -> None:
        if self.hparams["task_name"] == "mnli":
            loss = torch.tensor(torch.inf)
            for i, outputs in self.outputs.items():
                # matched or mismatched
                split = self.hparams["eval_splits"][i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in outputs]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)

                computed_metrics = self.metric.compute(predictions=preds, references=labels)
                if computed_metrics is not None:
                    split_metrics = {f"{k}_{split}": v for k, v in computed_metrics.items()}
                    self.log_dict(split_metrics, prog_bar=True)

        flat_outputs = []
        for lst in self.outputs.values():
            flat_outputs.extend(lst)

        preds = torch.cat([x["preds"] for x in flat_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in flat_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in flat_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        computed_metrics = self.metric.compute(predictions=preds, references=labels)
        if computed_metrics is not None:
            self.log_dict(computed_metrics, prog_bar=True)
        self.outputs.clear()

    def configure_optimizers(self) -> tuple[list, list]:
        """Prepare optimizer and schedule (linear warmup and decay)."""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams["learning_rate"],
            eps=self.hparams["adam_epsilon"],
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams["warmup_steps"],
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
