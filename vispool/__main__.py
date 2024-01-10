import lightning as L
import wandb
from datasets.utils.logging import disable_progress_bar
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
from transformers import logging as transformers_logging

from vispool import WANDB_LOG_DIR
from vispool.glue.datamodule import GLUEDataModule
from vispool.glue.transformer import GLUETransformer

load_dotenv()
transformers_logging.set_verbosity_error()
disable_progress_bar()
L.seed_everything(42)

MODEL_CHECKPOINT = "distilbert-base-uncased"
# MODEL_CHECKPOINT = "bert-base-uncased"
TASK_NAME = "mrpc"
# TASK_NAME = "cola"

dm = GLUEDataModule(
    MODEL_CHECKPOINT,
    task_name=TASK_NAME,
    batch_size=32,
    num_workers=8,
)

model = GLUETransformer(
    MODEL_CHECKPOINT,
    task_name=TASK_NAME,
)

logger = WandbLogger(project="vispool", save_dir=WANDB_LOG_DIR)
trainer = L.Trainer(
    accelerator="auto",
    devices="auto",
    max_epochs=3,
    logger=logger,  # type: ignore
    deterministic=True,
)

trainer.fit(model, datamodule=dm)
trainer.validate(model, datamodule=dm)
wandb.finish()
