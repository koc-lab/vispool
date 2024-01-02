import os

import lightning as L

from vispool.glue_datamodule import GLUEDataModule
from vispool.glue_transformer import GLUETransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "distilbert-base-uncased"
# MODEL_NAME = "albert-base-v2"

TASK_NAME = "cola"
# TASK_NAME = "mrpc"

MAX_EPOCHS = 2
SEED = 42

L.seed_everything(42)

dm = GLUEDataModule(
    model_name_or_path=MODEL_NAME,
    task_name=TASK_NAME,
)
dm.setup("fit")

model = GLUETransformer(
    model_name_or_path=MODEL_NAME,
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=dm.task_name,
    train_batch_size=64,
)

trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=1,
    logger=False,
)
trainer.fit(model, datamodule=dm)

print("Validation with validation dataloader...")
trainer.validate(model, dataloaders=dm.val_dataloader())

print("Validation with datamodule...")
trainer.validate(model, datamodule=dm)
