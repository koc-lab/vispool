import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from vispool import GLUE_TARGET_METRICS, WANDB_LOG_DIR
from vispool.glue.datamodule import GLUEDataModule
from vispool.glue.transformer import GLUETransformer

MODEL_CHECKPOINT = None
TASK_NAME = None
MONITOR_METRIC = None


def baseline_agent(sweep_id: str) -> None:
    wandb.agent(sweep_id, function=train_baseline)


def baseline_sweep(model_checkpoint: str, task_name: str) -> str:
    global MODEL_CHECKPOINT, TASK_NAME, MONITOR_METRIC
    MODEL_CHECKPOINT = model_checkpoint
    TASK_NAME = task_name
    MONITOR_METRIC = f"val/{GLUE_TARGET_METRICS[TASK_NAME]}"

    sweep_configuration = {
        "method": "grid",
        "metric": {
            "goal": "maximize",
            "name": MONITOR_METRIC,
        },
        "parameters": {
            "num_workers": {"value": 8},
            "max_epochs": {"value": 10},
            "batch_size": {"value": 32},
            "max_seq_length": {"values": [128, 256]},
            "learning_rate": {"values": [2e-5, 3e-5, 4e-5, 5e-5]},
            "seed": {"values": [40, 41, 42, 43, 44]},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="vispool")
    return sweep_id


def train_baseline() -> None:
    if MODEL_CHECKPOINT is None or TASK_NAME is None or MONITOR_METRIC is None:
        raise ValueError("Must run `baseline_sweep`.")

    # Setup
    checkpoint_callback = ModelCheckpoint(monitor=MONITOR_METRIC, mode="max")
    early_stopping_callback = EarlyStopping(monitor=MONITOR_METRIC, mode="max", patience=3)
    logger = WandbLogger(project="vispool", save_dir=WANDB_LOG_DIR, tags=["baseline", TASK_NAME])

    seed = logger.experiment.config.get("seed", 42)
    batch_size = logger.experiment.config.get("batch_size", 32)
    max_seq_length = logger.experiment.config.get("max_seq_length", 128)
    num_workers = logger.experiment.config.get("num_workers", 4)
    learning_rate = logger.experiment.config.get("learning_rate", 2e-5)
    max_epochs = logger.experiment.config.get("max_epochs", 3)

    # Train
    L.seed_everything(seed)
    dm = GLUEDataModule(
        MODEL_CHECKPOINT,
        task_name=TASK_NAME,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        num_workers=num_workers,
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
        max_epochs=max_epochs,
        logger=logger,  # type: ignore
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
