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
TASK_NAME = "rte"
MODEL_CHECKPOINT = "distilbert-base-uncased"
POOL = PoolStrategy.CLS
BATCH_SIZE = 32
NUM_WORKERS = 8
ENC_LR = 1e-5
GCN_LR = 1e-2
HIDDEN_DIM = 128
DROPOUT = 0.1
PENETRABLE_LIMIT = 0
MAX_EPOCHS = 10
RUN_BASELINE = True
DEGREE_NORMALIZE = True
LAYER_NORM = False

logger = CSVLogger(save_dir="logs", name=TASK_NAME)

# Baseline
if RUN_BASELINE:
    L.seed_everything(SEED)
    dm = GLUEDataModule(
        MODEL_CHECKPOINT,
        task_name=TASK_NAME,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    base_model = GLUETransformer(
        MODEL_CHECKPOINT,
        task_name=TASK_NAME,
        learning_rate=ENC_LR,
    )

    print("Training baseline on:", TASK_NAME)
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCHS,
        deterministic=True,
        logger=logger,
    )
    trainer.fit(base_model, datamodule=dm)

# Model
L.seed_everything(SEED)
dm = GLUEDataModule(
    MODEL_CHECKPOINT,
    task_name=TASK_NAME,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

vvg_model = VVGTransformer(
    MODEL_CHECKPOINT,
    task_name=TASK_NAME,
    encoder_lr=ENC_LR,
    gcn_lr=GCN_LR,
    gcn_hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT,
    pool=POOL,
    vvg_type=VVGType.NATURAL,
    weight_method=WeightMethod.UNWEIGHTED,
    penetrable_limit=PENETRABLE_LIMIT,
    directed=False,
    degree_normalize=DEGREE_NORMALIZE,
    parameter_search=False,
    layer_norm=LAYER_NORM,
)

print("Training vvg on:", TASK_NAME)
trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=MAX_EPOCHS,
    deterministic=True,
    logger=logger,
)
trainer.fit(vvg_model, datamodule=dm)
