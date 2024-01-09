import os

import lightning as L
from datasets.utils.logging import disable_progress_bar
from lightning.pytorch.loggers import CSVLogger
from transformers import logging as transformers_logging

from vispool.glue.datamodule import GLUEDataModule
from vispool.glue.transformer import GLUETransformer

transformers_logging.set_verbosity_error()
disable_progress_bar()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
L.seed_everything(42)

# MODEL_CHECKPOINT = "distilbert-base-uncased"
MODEL_CHECKPOINT = "bert-base-uncased"
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
    batch_size=32,
)

logger = CSVLogger("logs", name="my_exp_name")
trainer = L.Trainer(
    accelerator="auto",
    devices="auto",
    max_epochs=1,
    logger=logger,
    deterministic=True,
)

trainer.fit(model, datamodule=dm)
trainer.validate(model, datamodule=dm)
trainer.test(model, datamodule=dm)
