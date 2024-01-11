import click
from datasets.utils.logging import disable_progress_bar
from dotenv import load_dotenv
from transformers import logging as transformers_logging

from vispool.baseline import baseline_agent as base_agent
from vispool.baseline import baseline_sweep as base_sweep

load_dotenv()
transformers_logging.set_verbosity_error()
disable_progress_bar()


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


cli()
