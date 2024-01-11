from datasets.utils.logging import disable_progress_bar
from dotenv import load_dotenv
from transformers import logging as transformers_logging

from vispool.baseline import baseline_agent, baseline_sweep

load_dotenv()
transformers_logging.set_verbosity_error()
disable_progress_bar()

model_checkpoint = "distilbert-base-uncased"
task_name = "rte"
sweep_id = baseline_sweep(model_checkpoint, task_name)
baseline_agent(sweep_id)
