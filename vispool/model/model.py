from concurrent.futures import ThreadPoolExecutor
from os import PathLike
from typing import Any, Mapping

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from transformers import AutoModel
from vector_vis_graph import WeightMethod, horizontal_vvg, natural_vvg  # noqa

from vispool import GLUE_NUM_LABELS, USE_THREADPOOL
from vispool.glue.transformer import get_glue_task_metric, is_token_type_ids_input


def get_vvg(tensor: torch.Tensor) -> torch.Tensor:
    device = tensor.device
    tensor_np = tensor.to("cpu").detach().numpy()
    vvg = natural_vvg(tensor_np).astype(np.float32)
    return torch.from_numpy(vvg).to(device)


def get_vvgs_parallel(tensor: torch.Tensor) -> torch.Tensor:
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(get_vvg, tensor))
    return torch.stack(results)


class GCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(128)
        self.linear1 = nn.Linear(in_dim, 128)
        self.linear2 = nn.Linear(128, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.linear1.weight)
        nn.init.kaiming_uniform_(self.linear2.weight)

    def forward(self, vvgs: torch.Tensor, token_embs: torch.Tensor) -> Any:
        cls_tokens = (vvgs @ token_embs)[:, 0, :]

        # Layer 1
        cls_tokens = self.ln1(cls_tokens)
        out = self.linear1(cls_tokens)
        out = F.relu(out)

        # Layer 2
        out = self.ln2(out)
        out = self.linear2(out)
        out = F.relu(out)

        out = self.dropout(out)
        if self.out_dim > 1:
            out = F.log_softmax(out, dim=-1)
        return out


class VVGTransformer(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str | PathLike[str],
        task_name: str,
        encoder_lr: float = 1e-5,
        gcn_lr: float = 1e-3,
        parameter_search: bool = False,
        define_metric: str | None = None,
    ) -> None:
        super().__init__()
        if not parameter_search:
            self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.encoder_lr = encoder_lr
        self.gcn_lr = gcn_lr
        self.define_metric = define_metric

        self.num_labels = GLUE_NUM_LABELS[task_name]
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.gcn = GCN(768, self.num_labels)

        self.metric = get_glue_task_metric(task_name)
        self.use_token_type_ids = is_token_type_ids_input(self.encoder)

        if self.num_labels == 1:
            self.loss_fn = F.mse_loss
        else:
            self.loss_fn = F.nll_loss  # type: ignore

    def forward(self, **inputs: dict) -> Any:
        token_embs = self.encoder(**inputs)[0]
        if USE_THREADPOOL:
            vvgs = get_vvgs_parallel(token_embs)
        else:
            vvgs = torch.stack([get_vvg(token_emb) for token_emb in token_embs])
        logits = self.gcn(vvgs, token_embs)
        return logits

    def training_step(self, batch: Mapping, batch_idx: int) -> dict:
        input_dict = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        logits = self(**input_dict)
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        if self.trainer.global_step == 0 and self.define_metric is not None and isinstance(self.logger, WandbLogger):
            self.logger.experiment.define_metric(self.define_metric, summary="max")

        input_dict = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        logits = self(**input_dict)
        val_loss = self.loss_fn(logits, batch["labels"])
        preds = logits.squeeze() if self.num_labels == 1 else torch.argmax(logits, dim=-1)
        labels = batch["labels"]

        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        if metric_dict is not None:
            log_dict = {"loss": val_loss} if metric_dict is None else {"loss": val_loss, **metric_dict}
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
