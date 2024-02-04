import click
from datasets.utils.logging import disable_progress_bar
from dotenv import load_dotenv
from torch import set_float32_matmul_precision
from transformers import logging as transformers_logging

from vispool.baseline import baseline_agent as base_agent
from vispool.baseline import baseline_sweep as base_sweep
from vispool.our import our_agent, our_sweep, single_our_agent, single_our_sweep

load_dotenv()
transformers_logging.set_verbosity_error()
disable_progress_bar()
set_float32_matmul_precision("high")


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
def baseline_sweep_agent(model_checkpoint: str, task_name: str) -> None:
    """Initialize a WandB sweep for fine-tuning a baseline transformer model
    generated from MODEL_CHECKPOINT on a GLUE task with name TASK_NAME. Then,
    attach an agent to the created sweep."""
    sweep_id = base_sweep(model_checkpoint, task_name)
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
@click.argument("model_checkpoint", type=click.STRING)
@click.argument("task_name", type=click.STRING)
def vispool_sweep_agent(model_checkpoint: str, task_name: str) -> None:
    """Initialize a WandB sweep for fine-tuning a vispool model
    generated from MODEL_CHECKPOINT on a GLUE task with name TASK_NAME. Then,
    attach an agent to the created sweep."""
    sweep_id = our_sweep(model_checkpoint, task_name)
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


@cli.command()
@click.argument("run_id", type=click.STRING)
def vispool_single_sweep_agent(run_id: str) -> None:
    """Initialize a WandB grid sweep for different seeds with the hyperparameter values
    obtained in the given run with the specified ``run_id``. Then, attach an agent to
    the created sweep."""
    sweep_id = single_our_sweep(run_id)
    single_our_agent(sweep_id)


cli()
