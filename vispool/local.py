import lightning as L
from lightning.pytorch.loggers import CSVLogger
from vector_vis_graph import WeightMethod

from vispool.glue.datamodule import GLUEDataModule
from vispool.glue.transformer import GLUETransformer
from vispool.model.gcn import PoolStrategy
from vispool.model.model import VVGTransformer
from vispool.visualization import VisualVVGTransformer
from vispool.vvg import VVGType


def train_single_baseline_local(
    model_checkpoint: str,
    task_name: str,
    *,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    enc_lr: float = 1e-5,
    max_epochs: int = 10,
) -> GLUETransformer:
    logger = CSVLogger(save_dir="logs", name=task_name)
    L.seed_everything(seed)
    dm = GLUEDataModule(
        model_checkpoint,
        task_name=task_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = GLUETransformer(
        model_checkpoint,
        task_name=task_name,
        learning_rate=enc_lr,
    )

    print("Training baseline on:", task_name)
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        deterministic=True,
        logger=logger,
    )
    trainer.fit(model, datamodule=dm)
    return model


def train_single_vispool_local(
    model_checkpoint: str = "distilbert-base-uncased",
    task_name: str = "rte",
    *,
    visualize: bool = False,
    seed: int = 42,
    pool: PoolStrategy = PoolStrategy.CLS,
    batch_size: int = 32,
    num_workers: int = 4,
    enc_lr: float = 1e-5,
    gcn_lr: float = 1e-2,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    penetrable_limit: int = 0,
    vvg_type: VVGType = VVGType.NATURAL,
    weight_method: WeightMethod = WeightMethod.UNWEIGHTED,
    max_epochs: int = 10,
    degree_normalize: bool = False,
    layer_norm: bool = False,
    directed: bool = False,
) -> VVGTransformer | VisualVVGTransformer:
    logger = CSVLogger(save_dir="logs", name=task_name)

    # Model
    L.seed_everything(seed)
    dm = GLUEDataModule(
        model_checkpoint,
        task_name=task_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model_type = VisualVVGTransformer if visualize else VVGTransformer
    model = model_type(
        model_checkpoint,
        task_name=task_name,
        encoder_lr=enc_lr,
        gcn_lr=gcn_lr,
        gcn_hidden_dim=hidden_dim,
        dropout=dropout,
        pool=pool,
        vvg_type=vvg_type,
        weight_method=weight_method,
        penetrable_limit=penetrable_limit,
        directed=directed,
        degree_normalize=degree_normalize,
        layer_norm=layer_norm,
        parameter_search=False,
    )

    print("Training vvg on:", task_name)
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        deterministic=True,
        logger=logger,
    )
    trainer.fit(model, datamodule=dm)
    return model
