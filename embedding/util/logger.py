import json
from typing import Dict, Optional, Literal
from transformers.trainer import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class JSONLLoggerCallback(TrainerCallback):
    """
    A callback for logging training progress to a JSONL file.

    Args:
        filename (str): The name of the JSONL file to write logs to.
        mode (Literal["w", "w+", "a", "a+"], optional): The file open mode. Defaults to "w+".
            See Python's built-in `open` function for details on modes.
    """
    
    def __init__(self, filename: str, mode: Literal["w", "w+", "a", "a+"] = "w+") -> None:
        self._filename = filename
        self._mode = mode
        self._writer = None

    @property
    def writer(self):
        if self._writer is None:
            self._writer = open(self._filename, self._mode)
        return self._writer

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        log: Optional[Dict[str, float]] = None
        if "log" in kwargs and kwargs["log"]:
            log = kwargs["log"]
        elif state.log_history:
            log = state.log_history[-1]

        if log:
            line = json.dumps(log, ensure_ascii=False) + "\n"
            self.writer.write(line)

    def on_train_end(self, *pargs, **kwargs):
        self.writer.close()
