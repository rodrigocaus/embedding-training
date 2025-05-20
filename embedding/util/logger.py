import json
from typing import Dict, Optional, Literal
from transformers.trainer import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

LogEntry = Dict[str, float]


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

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[LogEntry] = None,
        **kwargs
    ):
        if not logs and state.log_history:
            logs = state.log_history[-1]

        if logs:
            logs["step"] = state.global_step
            line = json.dumps(logs, ensure_ascii=False) + "\n"
            self.writer.write(line)

    def on_train_end(self, *pargs, **kwargs):
        if self._writer is not None:
            self._writer.close()
