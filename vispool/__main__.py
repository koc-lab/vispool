import lightning as L
from datasets.utils.logging import disable_progress_bar
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformers import logging as transformers_logging

import wandb
from vispool import GLUE_TARGET_METRICS, GLUE_TASKS, WANDB_LOG_DIR
from vispool.glue.datamodule import GLUEDataModule
from vispool.glue.transformer import GLUETransformer

load_dotenv()
transformers_logging.set_verbosity_error()
disable_progress_bar()

MODEL_CHECKPOINT = "distilbert-base-uncased"
TASK_NAME = GLUE_TASKS[1]
MONITOR_METRIC = f"val/{GLUE_TARGET_METRICS[TASK_NAME]}"


def train() -> None:
    # Setup
    checkpoint_callback = ModelCheckpoint(monitor=MONITOR_METRIC, mode="max")
    early_stopping_callback = EarlyStopping(monitor=MONITOR_METRIC, mode="max", patience=3)
    logger = WandbLogger(project="vispool", save_dir=WANDB_LOG_DIR, tags=["baseline", TASK_NAME])
    batch_size = logger.experiment.config.batch_size
    learning_rate = logger.experiment.config.learning_rate

    # Train
    L.seed_everything(42)
    dm = GLUEDataModule(
        MODEL_CHECKPOINT,
        task_name=TASK_NAME,
        batch_size=batch_size,
        num_workers=8,
    )
    model = GLUETransformer(
        MODEL_CHECKPOINT,
        task_name=TASK_NAME,
        learning_rate=learning_rate,
        parameter_search=True,
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=3,
        logger=logger,  # type: ignore
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)


sweep_configuration = {
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": MONITOR_METRIC,
    },
    "parameters": {
        "batch_size": {"values": [32, 64]},
        "learning_rate": {"values": [2e-5, 3e-5, 4e-5, 5e-5]},
    },
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="vispool")
wandb.agent(sweep_id, function=train)
