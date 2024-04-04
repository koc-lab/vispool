from enum import Enum
from pathlib import Path
from typing import Type, TypeVar

import click
import numpy as np
from datasets.utils.logging import disable_progress_bar
from dotenv import load_dotenv
from torch import set_float32_matmul_precision
from transformers import logging as transformers_logging
from vector_vis_graph import WeightMethod

from vispool.baseline import baseline_agent as base_agent
from vispool.baseline import baseline_sweep as base_sweep
from vispool.local import train_single_vispool_local
from vispool.model.gcn import PoolStrategy
from vispool.our import our_agent, our_sweep, single_our_agent, single_our_sweep
from vispool.visualization import VisualVVGTransformer, visualize_graph_with_fixed_positions
from vispool.vvg import VVGType

load_dotenv()
transformers_logging.set_verbosity_error()
disable_progress_bar()
set_float32_matmul_precision("high")


EnumType = TypeVar("EnumType", bound=Enum)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("model_checkpoint", type=click.STRING)
@click.argument("task_name", type=click.STRING)
def baseline_sweep(model_checkpoint: str, task_name: str) -> None:
    """Initialize a WandB sweep for fine-tuning a baseline transformer model
    generated from MODEL_CHECKPOINT on a GLUE task with name TASK_NAME."""
    sweep_id = base_sweep(model_checkpoint, task_name)
    print(f"Created sweep with id: {sweep_id}")


@cli.command()
@click.argument("sweep_id", type=click.STRING)
def baseline_agent(sweep_id: str) -> None:
    """Attach an agent to the created baseline sweep with the given SWEEP_ID."""
    base_agent(sweep_id)


@cli.command()
@click.argument("model_checkpoint", type=click.STRING)
@click.argument("task_name", type=click.STRING)
def vispool_sweep(model_checkpoint: str, task_name: str) -> None:
    """Initialize a WandB sweep for fine-tuning a vispool model
    generated from MODEL_CHECKPOINT on a GLUE task with name TASK_NAME."""
    sweep_id = our_sweep(model_checkpoint, task_name)
    print(f"Created sweep with id: {sweep_id}")


@cli.command()
@click.argument("sweep_id", type=click.STRING)
def vispool_agent(sweep_id: str) -> None:
    """Attach an agent to the created vispool sweep with the given SWEEP_ID."""
    our_agent(sweep_id)


@cli.command()
@click.argument("run_id", type=click.STRING)
def vispool_single_sweep(run_id: str) -> None:
    """Initialize a WandB grid sweep for different seeds with the hyperparameter values
    obtained in the given run with the specified ``run_id``."""
    sweep_id = single_our_sweep(run_id)
    print(f"Created sweep with id: {sweep_id}")


@cli.command()
@click.argument("sweep_id", type=click.STRING)
def vispool_single_agent(sweep_id: str) -> None:
    """Attach an agent to the created vispool single sweep with the given SWEEP_ID."""
    single_our_agent(sweep_id)


# Function to convert string inputs to Enum or Class types
def convert_to_enum(value: str, enum_type: Type[EnumType]) -> EnumType:
    try:
        return enum_type[value.upper()]
    except Exception:
        raise ValueError(f"Invalid value for {enum_type.__name__}: {value}")  # noqa


@cli.command()
@click.argument("model-checkpoint", type=click.STRING)
@click.argument("task-name", type=click.STRING)
@click.option("--visualize/--no-visualize", default=False, help="Visualise.")
@click.option("--seed", default=42, type=int, help="Seed.")
@click.option(
    "--pool",
    default="CLS",
    help="Pool strategy.",
    type=click.Choice([e.value for e in PoolStrategy], case_sensitive=False),
)
@click.option("--batch-size", default=32, type=int, help="Batch size.")
@click.option("--num-workers", default=4, type=int, help="Number of workers.")
@click.option("--enc-lr", default=1e-5, type=float, help="Encoder learning rate.")
@click.option("--gcn-lr", default=1e-2, type=float, help="GCN learning rate.")
@click.option("--hidden-dim", default=128, type=int, help="Hidden dimension.")
@click.option("--dropout", default=0.1, type=float, help="Dropout.")
@click.option("--penetrable-limit", default=0, type=int, help="Penetrable limit.")
@click.option(
    "--vvg-type",
    default="NATURAL",
    help="VVG type.",
    type=click.Choice([e.name for e in VVGType], case_sensitive=False),
)
@click.option(
    "--weight-method",
    default="UNWEIGHTED",
    help="Weight method.",
    type=click.Choice([e.name for e in WeightMethod], case_sensitive=False),
)
@click.option("--max-epochs", default=10, type=int, help="Max epochs.")
@click.option("--degree-normalize/--no-degree-normalize", default=False, help="Degree normalize.")
@click.option("--layer-norm/--no-layer-norm", default=False, help="Layer norm.")
@click.option("--directed/--no-directed", default=False, help="Directed.")
@click.option("--visualization-dir", type=click.Path(exists=True), default=".", help="Visualization directory.")
@click.option(
    "--save-visualization-graphs/--no-save-visualization-graphs",
    default=True,
    help="Save visualization graphs.",
)
def local_single(
    model_checkpoint: str,
    task_name: str,
    visualize: bool,
    seed: int,
    pool: str,
    batch_size: int,
    num_workers: int,
    enc_lr: float,
    gcn_lr: float,
    hidden_dim: int,
    dropout: float,
    penetrable_limit: int,
    vvg_type: str,
    weight_method: str,
    max_epochs: int,
    degree_normalize: bool,
    layer_norm: bool,
    directed: bool,
    visualization_dir: str,
    save_visualization_graphs: bool,
) -> None:
    """Run a single local run for vispool."""
    # Convert string inputs to their respective Enums
    pool_enum = convert_to_enum(pool, PoolStrategy)
    vvg_type_enum = convert_to_enum(vvg_type, VVGType)
    weight_method_enum = convert_to_enum(weight_method, WeightMethod)

    model = train_single_vispool_local(
        model_checkpoint,
        task_name,
        visualize=visualize,
        seed=seed,
        pool=pool_enum,
        batch_size=batch_size,
        num_workers=num_workers,
        enc_lr=enc_lr,
        gcn_lr=gcn_lr,
        hidden_dim=hidden_dim,
        dropout=dropout,
        penetrable_limit=penetrable_limit,
        vvg_type=vvg_type_enum,
        weight_method=weight_method_enum,
        max_epochs=max_epochs,
        degree_normalize=degree_normalize,
        layer_norm=layer_norm,
        directed=directed,
    )

    if visualize and isinstance(model, VisualVVGTransformer):
        document_graphs = np.array([batch[0].cpu().numpy() for batch in model.graphs])
        if save_visualization_graphs:
            np.savez(f"visualisation-graphs-list-{model_checkpoint}.npz", *document_graphs)
        visualize_graph_with_fixed_positions(adj_matrices=document_graphs, out_dir=Path(visualization_dir))


cli()
