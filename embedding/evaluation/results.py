import pathlib
import pandas as pd
from collections import defaultdict


def normalize_model_name(name: str):
    model = pathlib.Path(name)
    if "checkpoint" in model.name:
        model = model.parent
    return model.name


class ResultsManager:
    def __init__(self, model_name: str, output_dir: pathlib.Path):
        self.model_name = model_name
        self.output_path = output_dir.joinpath(
            normalize_model_name(model_name)
        )
        self.results = defaultdict(list)

    def add(self, task: str, result: dict):
        self.results[task].append(result)

    def print(self):
        for task, results in self.results.items():
            print(f"{task}:")
            for result in results:
                pretty = ", ".join(
                    f"{key}={value}"
                    for key, value in result.items()
                )
                print("\t", pretty)

    def dump(self):
        """
        Dump the results to a file; 
        Each task is written in a single CSV file.
        """
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)

        for task, results in self.results.items():
            task_file = self.output_path.joinpath(f"{task}.csv")
            if task_file.exists():
                df = pd.read_csv(task_file)
                df = pd.concat([df, pd.DataFrame(results)])
            else:
                df = pd.DataFrame(results)
            df.to_csv(task_file, index=False)
