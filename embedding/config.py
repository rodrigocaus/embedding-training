import json
import yaml
import pathlib
from dataclasses import dataclass, is_dataclass, field
from typing import Any, Dict, List, Literal, Optional
from sentence_transformers import SentenceTransformerTrainingArguments


Object = Dict[str, Any]


@dataclass
class ModelConfig:
    name: str
    args: Object = field(default_factory=dict)


@dataclass
class TrainingObjective:
    type: Literal["contrastive", "similarity", "distillation"]
    loss_args: Object = field(default_factory=dict)
    matryoshka: Object = field(default_factory=dict)
    teacher: Optional[ModelConfig] = field(default=None)


@dataclass
class TrainingConfigV1:
    run_name: str
    base_model: ModelConfig
    output_dir: str
    objectives: List[TrainingObjective]
    args: SentenceTransformerTrainingArguments
    evaluator_args: Object = field(default_factory=dict)


def to_dataclass(data: dict, dataclass_type: type):
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass")

    field_types = {
        field.name: field.type
        for field in dataclass_type.__dataclass_fields__.values()
    }
    fields = {}

    for key, value in data.items():
        if key not in field_types:
            raise ValueError(f"Key '{key}' not found in dataclass definition")

        field_type = field_types[key]

        if isinstance(value, dict) and is_dataclass(field_type):
            fields[key] = to_dataclass(value, field_type)
        elif isinstance(value, list) and field_type.__args__ and is_dataclass(field_type.__args__[0]):
            fields[key] = [
                to_dataclass(item, field_type.__args__[0])
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
