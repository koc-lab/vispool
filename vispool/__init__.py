import platform
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
WANDB_LOG_DIR = ROOT_DIR.joinpath("wandb_logs")
WANDB_LOG_DIR.mkdir(exist_ok=True)

__version__ = "0.1.0"


def is_intel_linux() -> bool:
    is_x86_64_linux = platform.machine() == "x86_64" and platform.system() == "Linux"
    try:
        command_output = subprocess.check_output(["/usr/bin/lscpu"]).decode("utf-8")
        is_intel_cpu = "Intel" in command_output
    except Exception:
        is_intel_cpu = False

    print(f"Intel Linux: {is_x86_64_linux and is_intel_cpu}")
    return is_x86_64_linux and is_intel_cpu


USE_THREADPOOL = is_intel_linux()


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
