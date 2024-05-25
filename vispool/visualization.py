from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger

from vispool.model.model import VVGTransformer
from vispool.vvg import get_vvgs


class VisualVVGTransformer(VVGTransformer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.graphs: list[torch.Tensor] = []

    def forward(self, **inputs: dict) -> tuple[Any, torch.Tensor]:
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
        return logits, vvgs

    def training_step(self, batch: Mapping, batch_idx: int) -> dict:
        input_dict = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        logits, vvgs = self(**input_dict)
        if batch_idx == 0:
            self.graphs.append(vvgs)
        if self.num_labels == 1:
            logits = logits.squeeze()
        loss = self.loss_fn(logits, batch["labels"])  # type: ignore
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        if self.trainer.global_step == 0 and self.define_metric is not None and isinstance(self.logger, WandbLogger):
            self.logger.experiment.define_metric(self.define_metric, summary="max")

        input_dict = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        logits, _ = self(**input_dict)
        labels = batch["labels"]
        if self.num_labels == 1:
            logits = logits.squeeze()
            preds = logits
        else:
            preds = torch.argmax(logits, dim=-1)
        loss = self.loss_fn(logits, labels)  # type: ignore

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        if metric_dict is not None:
            log_dict = {"loss": loss} if metric_dict is None else {"loss": loss, **metric_dict}
            log_dict = {f"val/{k}": v for k, v in metric_dict.items()}
            self.log_dict(log_dict, prog_bar=True)


def visualize_graph_with_fixed_positions(adj_matrices: np.ndarray, out_dir: Path) -> None:
    num_nodes = len(adj_matrices[0])
    G_initial = nx.Graph()
    G_initial.add_nodes_from(range(num_nodes))
    pos = nx.circular_layout(G_initial)

    for i, adj_matrix in enumerate(adj_matrices):
        density = np.sum(adj_matrix != 0) / adj_matrix.size
        G = nx.from_numpy_array(adj_matrix)

        # Draw the graph with fixed positions
        plt.figure(figsize=(15, 15))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="skyblue",
            node_size=300,
            edge_color="gray",
            linewidths=0.01,
            font_size=10,
        )
        plt.title(f"Graph Visualization at Epoch {i+1}, Density: {density:.2%}")
        plt.savefig(out_dir.joinpath(f"graph-epoch-{i+1}.eps"), bbox_inches="tight")

        # Draw the sparsity map
        plt.figure(figsize=(15, 15))
        plt.imshow(adj_matrix, cmap="Greys", interpolation="nearest")
        plt.title(f"Sparsity Pattern at Epoch {i+1}, Density: {density:.2%}")
        plt.axis("off")
        plt.savefig(out_dir.joinpath(f"sparsity-epoch-{i+1}.eps"), bbox_inches="tight")
        plt.close()
