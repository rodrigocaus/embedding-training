import json
import yaml
import pathlib
from dataclasses import dataclass, is_dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List, Literal, Optional, Union, get_origin, get_args
from sentence_transformers import SentenceTransformerTrainingArguments


Object = Dict[str, Any]
ObjectiveType = Literal["contrastive", "similarity"]


@dataclass
class ModelConfig:
    """Configuration for a model.

    Attributes:
        name (str): The name or path of the model.
        args (dict): Additional arguments for loading the model.
            Defaults to an empty dictionary.
    """
    name: str
    args: Object = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """
    Configuration for a dataset.

    Attributes:
        name (str): The name or path of the dataset.
        args (dict): Additional arguments for loading the dataset.
        preprocess_args (dict): Arguments for preprocessing the dataset.
    """
    name: str
    args: Object = field(default_factory=dict)
    preprocess_args: Object = field(default_factory=dict)


@dataclass
class TrainingDatasets:
    train: DatasetConfig
    validation: Optional[DatasetConfig] = field(default=None)


@dataclass
class DistilConfig:
    alpha: float
    teacher: ModelConfig
    loss_args: Object = field(default_factory=dict)


@dataclass
class TrainingObjective:
    type: ObjectiveType
    datasets: TrainingDatasets
    loss_args: Object = field(default_factory=dict)
    matryoshka: Object = field(default_factory=dict)
    distillation: Optional[DistilConfig] = field(default=None)


@dataclass
class EvaluatorConfig:
    dataset: DatasetConfig
    args: Object = field(default_factory=dict)


@dataclass
class TrainingConfigV1:
    run_name: str
    base_model: ModelConfig
    output_dir: str
    objectives: List[TrainingObjective]
    args: SentenceTransformerTrainingArguments
    evaluator: EvaluatorConfig


def _is_optional_type(field) -> bool:
    origin_type = get_origin(field)
    args_type = get_args(field)

    return (origin_type is Union and
            len(args_type) == 2 and
            type(None) in args_type)


def to_dataclass(data: dict, dataclass_type: type):
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass")

    field_types = {
        field.name: field.type
        for field in dataclass_fields(dataclass_type)
    }
    fields = {}

    for key, value in data.items():
        if key not in field_types:
            raise ValueError(
                f"Key '{key}' not found in dataclass {dataclass_type} definition")

        field_type = field_types[key]
        if _is_optional_type(field_type):
            field_type = get_args(field_type)[0]

        if isinstance(value, dict) and is_dataclass(field_type):
            fields[key] = to_dataclass(value, field_type)
        elif isinstance(value, list) and field_type.__args__ and is_dataclass(field_type.__args__[0]):
            field_type = get_args(field_type)[0]
            fields[key] = [
                to_dataclass(item, field_type)
                if isinstance(item, dict) else item for item in value
            ]
        else:
            fields[key] = value

    return dataclass_type(**fields)


def load(filename: str) -> TrainingConfigV1:
    """
        Load a YAML or JSON file into a TrainingConfig object.
    """

    filepath = pathlib.Path(filename)
    if filepath.suffix.lower() == ".json":
        loader = json.load
    elif filepath.suffix.lower() in (".yml",  ".yaml"):
        loader = yaml.safe_load
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")

    with filepath.open() as fp:
        data = loader(fp)

    return to_dataclass(data, TrainingConfigV1)
