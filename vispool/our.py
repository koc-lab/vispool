from os import cpu_count, getenv
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from vector_vis_graph import WeightMethod

import wandb
from vispool import GLUE_TARGET_METRICS, WANDB_LOG_DIR
from vispool.glue.datamodule import GLUEDataModule
from vispool.model.gcn import PoolStrategy
from vispool.model.model import VVGTransformer
from vispool.vvg import VVGType

MODEL_CHECKPOINT = None
TASK_NAME = None
MONITOR_METRIC = None


def our_agent(sweep_id: str, entity: Optional[str] = None, project: Optional[str] = None) -> None:
    global MODEL_CHECKPOINT, TASK_NAME, MONITOR_METRIC

    if entity is None:
        entity = getenv("WANDB_ENTITY")
    if project is None:
        project = getenv("WANDB_PROJECT")
    if entity is None or project is None:
        raise ValueError("Must specify entity and project.")

    tuner = wandb.controller(sweep_id, entity=entity, project=project)
    parameters = tuner.sweep_config.get("parameters")
    if parameters is not None:
        MODEL_CHECKPOINT = parameters.get("model_checkpoint")["value"]
        TASK_NAME = parameters.get("task_name")["value"]
        MONITOR_METRIC = parameters.get("monitor_metric")["value"]
    wandb.agent(sweep_id, function=train_our)


def our_sweep(model_checkpoint: str, task_name: str) -> str:
    global MODEL_CHECKPOINT, TASK_NAME, MONITOR_METRIC
    MODEL_CHECKPOINT = model_checkpoint
    TASK_NAME = task_name
    MONITOR_METRIC = f"val/{GLUE_TARGET_METRICS[TASK_NAME]}"

    sweep_configuration = {
        "name": f"vispool:{MODEL_CHECKPOINT}:{TASK_NAME}",
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": MONITOR_METRIC,
        },
        "parameters": {
            "model_checkpoint": {"value": MODEL_CHECKPOINT},
            "task_name": {"value": TASK_NAME},
            "monitor_metric": {"value": MONITOR_METRIC},
            "max_epochs": {"value": 15},
            "patience": {"value": 5},
            "batch_size": {"value": 32},
            "max_seq_length": {"value": 128},
            "seed": {"value": 42},
            "encoder_lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-4},
            "gcn_lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-2},
            "dropout": {"distribution": "uniform", "min": 0.1, "max": 0.5},
            "gcn_hidden_dim": {"values": [128, 512]},
            "pool": {"distribution": "categorical", "values": ["cls", "mean", "max"]},
            "vvg_type": {"distribution": "categorical", "values": ["natural", "horizontal"]},
            "penetrable_limit": {"distribution": "int_uniform", "min": 0, "max": 5},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="vispool")
    return sweep_id


def train_our() -> None:
    if MODEL_CHECKPOINT is None or TASK_NAME is None or MONITOR_METRIC is None:
        raise ValueError("Must run `our_sweep`.")

    # Setup
    logger = WandbLogger(
        project="vispool",
        save_dir=WANDB_LOG_DIR,
        tags=["vispool", MODEL_CHECKPOINT, TASK_NAME],
        resume=True,
    )
    seed = logger.experiment.config.get("seed")
    max_epochs = logger.experiment.config.get("max_epochs")
    patience = logger.experiment.config.get("patience")
    batch_size = logger.experiment.config.get("batch_size")
    max_seq_length = logger.experiment.config.get("max_seq_length")
    encoder_lr = logger.experiment.config.get("encoder_lr")
    gcn_lr = logger.experiment.config.get("gcn_lr")
    gcn_hidden_dim = logger.experiment.config.get("gcn_hidden_dim")
    dropout = logger.experiment.config.get("dropout")

    pool = PoolStrategy(logger.experiment.config.get("pool"))
    vvg_type = VVGType(logger.experiment.config.get("vvg_type"))
    penetrable_limit = logger.experiment.config.get("penetrable_limit")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor=MONITOR_METRIC, mode="max")
    early_stopping_callback = EarlyStopping(monitor=MONITOR_METRIC, mode="max", patience=patience)

    # Train
    L.seed_everything(seed)
    num_workers = cpu_count()
    num_workers = num_workers if num_workers is not None else 0
    dm = GLUEDataModule(
        MODEL_CHECKPOINT,
        task_name=TASK_NAME,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = VVGTransformer(
        MODEL_CHECKPOINT,
        task_name=TASK_NAME,
        encoder_lr=encoder_lr,
        gcn_lr=gcn_lr,
        gcn_hidden_dim=gcn_hidden_dim,
        dropout=dropout,
        pool=pool,
        vvg_type=vvg_type,
        weight_method=WeightMethod.UNWEIGHTED,
        penetrable_limit=penetrable_limit,
        directed=False,
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
