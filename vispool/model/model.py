from os import PathLike
from typing import Any, Mapping

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from transformers import AutoModel
from vector_vis_graph import WeightMethod

from vispool import GLUE_NUM_LABELS
from vispool.glue.transformer import get_glue_task_metric, is_token_type_ids_input
from vispool.model.gcn import OverallGCN, PoolStrategy
from vispool.vvg import VVGType, get_vvgs


class VVGTransformer(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str | PathLike[str],
        task_name: str,
        *,
        encoder_lr: float = 1e-5,
        gcn_lr: float = 1e-4,
        dropout: float = 0.1,
        layer_norm: bool = True,
        gcn_hidden_dim: int = 128,
        pool: PoolStrategy = PoolStrategy.CLS,
        vvg_type: VVGType = VVGType.NATURAL,
        timeline: np.ndarray | None = None,
        weight_method: WeightMethod = WeightMethod.UNWEIGHTED,
        penetrable_limit: int = 0,
        directed: bool = False,
        degree_normalize: bool = False,
        parameter_search: bool = False,
        define_metric: str | None = None,
    ) -> None:
        super().__init__()

        # Hyperparameters
        if not parameter_search:
            self.save_hyperparameters()
        self.define_metric = define_metric

        # Network Hyperparameters
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.encoder_lr = encoder_lr
        self.gcn_lr = gcn_lr
        self.gcn_hidden_dim = gcn_hidden_dim

        # VVG Hyperparameters
        self.vvg_type = vvg_type
        self.timeline = timeline
        self.weight_method = weight_method
        self.penetrable_limit = penetrable_limit
        self.directed = directed
        self.degree_normalize = degree_normalize

        # Metric and Loss
        self.metric = get_glue_task_metric(task_name)
        self.num_labels = GLUE_NUM_LABELS[task_name]
        self.loss_fn = F.mse_loss if self.num_labels == 1 else F.nll_loss  # type: ignore

        # Model
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.gcn = OverallGCN(
            in_dim=self.encoder.config.hidden_size,
            hidden_dim=self.gcn_hidden_dim,
            out_dim=self.num_labels,
            dropout=dropout,
            pool=pool,
            layer_norm=layer_norm,
        )
        self.use_token_type_ids = is_token_type_ids_input(self.encoder)

    def forward(self, **inputs: dict) -> Any:
        token_embs = self.encoder(**inputs)[0]
        vvgs = get_vvgs(
            token_embs,
            vvg_type=self.vvg_type,
            timeline=self.timeline,
            weight_method=self.weight_method,
            penetrable_limit=self.penetrable_limit,
            directed=self.directed,
            degree_normalize=self.degree_normalize,
        )
        logits = self.gcn(vvgs, token_embs)
        return logits

    def training_step(self, batch: Mapping, batch_idx: int) -> dict:
        input_dict = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        logits = self(**input_dict)
        loss = self.loss_fn(logits, batch["labels"])  # type: ignore
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        if self.trainer.global_step == 0 and self.define_metric is not None and isinstance(self.logger, WandbLogger):
            self.logger.experiment.define_metric(self.define_metric, summary="max")

        input_dict = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        logits = self(**input_dict)
        loss = self.loss_fn(logits, batch["labels"])  # type: ignore
        preds = logits.squeeze() if self.num_labels == 1 else torch.argmax(logits, dim=-1)
        labels = batch["labels"]

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        if metric_dict is not None:
            log_dict = {"loss": loss} if metric_dict is None else {"loss": loss, **metric_dict}
            log_dict = {f"val/{k}": v for k, v in metric_dict.items()}
            self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            [
                {"params": self.encoder.parameters(), "lr": self.encoder_lr},
                {"params": self.gcn.parameters(), "lr": self.gcn_lr},
            ],
        )
        return optimizer
