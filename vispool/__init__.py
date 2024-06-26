import platform
from pathlib import Path

from numba import config

ROOT_DIR = Path(__file__).parent.parent.absolute()
WANDB_LOG_DIR = ROOT_DIR.joinpath("wandb_logs")
WANDB_LOG_DIR.mkdir(exist_ok=True)

__version__ = "0.1.0"


def is_threadpool_usable() -> bool:
    if platform.machine() == "x86_64":
        config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]  # type: ignore
        use_threadpool = True
    else:
        use_threadpool = False
    return use_threadpool


USE_THREADPOOL = is_threadpool_usable()


GLUE_TASKS = (
    "wnli",
    "rte",
    "mrpc",
    "stsb",
    "cola",
    "sst2",
    "qnli",
    "qqp",
    "mnli",
    "mnli-mm",
)

GLUE_OUTPUT_MODES = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "qnli": "classification",
    "qqp": "classification",
    "rte": "classification",
    "sst2": "classification",
    "stsb": "regression",
    "wnli": "classification",
}

GLUE_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mnli-mm": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "stsb": 1,
    "wnli": 2,
}

GLUE_TEXT_FIELDS = {
    "cola": ["sentence"],
    "mnli": ["premise", "hypothesis"],
    "mnli-mm": ["premise", "hypothesis"],
    "mrpc": ["sentence1", "sentence2"],
    "qnli": ["question", "sentence"],
    "qqp": ["question1", "question2"],
    "rte": ["sentence1", "sentence2"],
    "sst2": ["sentence"],
    "stsb": ["sentence1", "sentence2"],
    "wnli": ["sentence1", "sentence2"],
}

GLUE_LOADER_COLUMNS = [
    "datasets_idx",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "start_positions",
    "end_positions",
    "labels",
]

GLUE_TARGET_METRICS = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mnli-mm": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "wnli": "accuracy",
}
