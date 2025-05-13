import os
import json
import psutil

from typing import Literal
from transformers.trainer import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


def bytes_to_mb(value: int, base_10: bool = False):
    if base_10:
        return round(value / 1000000, 2)
    return round(value / 1024 / 1024, 2)


class MemoryMonitor:
    def __init__(self):
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self.total_memory = psutil.virtual_memory().total

    def __call__(self):
        memory_info = self.process.memory_info()
        allocated_memory = memory_info.rss
        return {
            "used_memory_mb": bytes_to_mb(allocated_memory),
            "used_memory_percentage": round(allocated_memory / self.total_memory * 100, 2),
        }


class VRAMMonitor:
    def __init__(self, device="cuda"):
        from torch import cuda

        self.device = cuda.device(device)
        gpu_properties = cuda.get_device_properties(device)
        self.total_memory = gpu_properties.total_memory
        self.callback = cuda.memory_reserved

    def __call__(self):
        allocated_memory = self.callback(self.device)
        return {
            "used_vram_mb": bytes_to_mb(allocated_memory),
            "used_vram_percentage": round(allocated_memory / self.total_memory * 100, 2),
        }


class MemoryProfilerCallback(TrainerCallback):
    def __init__(self, filename: str, mode: Literal["w", "w+", "a", "a+"] = "w+", monitor_cuda=False) -> None:
        self._filename = filename
        self._mode = mode
        self._writer = None
        self._memory_monitor = MemoryMonitor()
        if monitor_cuda:
            self._vram_monitor = VRAMMonitor()
        else:
            self._vram_monitor = lambda: {}

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
        **kwargs
    ):
        entry = {
            "epoch": state.epoch,
            "step": state.global_step,
        }
        entry.update(self._memory_monitor())
        entry.update(self._vram_monitor())
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        self.writer.write(line)
