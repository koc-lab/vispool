import lightning as L
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger

from vispool.glue.datamodule import GLUEDataModule
from vispool.glue.transformer import GLUETransformer
from vispool.model.model import VVGTransformer

load_dotenv()

# TASK_NAME = "stsb"
TASK_NAME = "mrpc"

MODEL_CHECKPOINT = "distilbert-base-uncased"
# MODEL_CHECKPOINT = "bert-base-uncased"

dm = GLUEDataModule(
    MODEL_CHECKPOINT,
    task_name=TASK_NAME,
    batch_size=32,
    num_workers=4,
)

base_model = GLUETransformer(
    MODEL_CHECKPOINT,
    task_name=TASK_NAME,
    learning_rate=1e-5,
)

vvg_model = VVGTransformer(
    MODEL_CHECKPOINT,
    task_name=TASK_NAME,
    encoder_lr=1e-5,
    gcn_lr=1e-3,
)

logger = CSVLogger(save_dir="logs", name=TASK_NAME)


print("Training baseline on:", TASK_NAME)
trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=1,
    deterministic=True,
    logger=logger,
)
trainer.fit(base_model, datamodule=dm)
trainer.validate(base_model, datamodule=dm)

print("Training vvg on:", TASK_NAME)
trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=1,
    deterministic=True,
    logger=logger,
)
trainer.fit(vvg_model, datamodule=dm)
trainer.validate(vvg_model, datamodule=dm)
