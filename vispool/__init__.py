__version__ = "0.1.0"

GLUE_TASKS = (
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
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
