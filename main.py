import lightning as L
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger
from vector_vis_graph import WeightMethod

from vispool.glue.datamodule import GLUEDataModule
from vispool.glue.transformer import GLUETransformer
from vispool.model.gcn import PoolStrategy
from vispool.model.model import VVGTransformer
from vispool.vvg import VVGType

load_dotenv()

SEED = 42
TASK_NAME = "mrpc"
MODEL_CHECKPOINT = "distilbert-base-uncased"
POOL = PoolStrategy.CLS

logger = CSVLogger(save_dir="logs", name=TASK_NAME)

# Baseline
L.seed_everything(SEED)
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

print("Training baseline on:", TASK_NAME)
trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=1,
    deterministic=True,
    logger=logger,
)
# trainer.fit(base_model, datamodule=dm)

# Model
L.seed_everything(SEED)
dm = GLUEDataModule(
    MODEL_CHECKPOINT,
    task_name=TASK_NAME,
    batch_size=32,
    num_workers=4,
)

vvg_model = VVGTransformer(
    MODEL_CHECKPOINT,
    task_name=TASK_NAME,
    encoder_lr=1e-5,
    gcn_lr=1e-4,
    gcn_hidden_dim=256,
    dropout=0.1,
    pool=POOL,
    vvg_type=VVGType.NATURAL,
    weight_method=WeightMethod.UNWEIGHTED,
    penetrable_limit=0,
    directed=False,
    parameter_search=False,
)
print(vvg_model.gcn)

print("Training vvg on:", TASK_NAME)
trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=1,
    deterministic=True,
    logger=logger,
)
trainer.fit(vvg_model, datamodule=dm)
