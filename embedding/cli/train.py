import os
import config
import preprocess
from losses import kl
from util import logger, profiler

from torch import nn
from typing import NamedTuple, Optional
from datasets import load_dataset, Dataset
from sentence_transformers import (
    losses, evaluation,
    SentenceTransformer, SentenceTransformerTrainer
)


class Objective(NamedTuple):
    name: str
    loss: nn.Module
    train_dataset: Dataset
    validation_dataset: Optional[Dataset]


def map_objective(model: SentenceTransformer, spec: config.TrainingObjective) -> Objective:
    objective_type = spec.type
    if objective_type not in ("contrastive", "similarity", "distillation"):
        raise ValueError(f"Unknown objective type: {objective_type}")

    ## LOSS DEFINITION ##
    if objective_type == "contrastive":
        loss = losses.MultipleNegativesSymmetricRankingLoss(
            model=model, **spec.loss_args
        )
        dataset_preprocessor = preprocess.EFAQRankingGenerator

    elif objective_type == "similarity":
        loss = losses.CoSENTLoss(model=model, **spec.loss_args)
        dataset_preprocessor = preprocess.EFAQCosineSimilarityGenerator

    elif objective_type == "distillation":
        if not spec.teacher:
            raise ValueError(
                "Teacher model not specified for distillation loss")

        teacher = SentenceTransformer(spec.teacher.name, **spec.teacher.args)
        loss = kl.KLSimilarityLoss(
            student=model, teacher=teacher, **spec.loss_args
        )
        dataset_preprocessor = preprocess.EFAQRankingGenerator

    if spec.matryoshka:
        loss = losses.MatryoshkaLoss(
            model=model,
            loss=loss,
            **spec.matryoshka
        )

    ## DATASETS DEFINITION ##
    train_dataset_spec = spec.datasets.train
    validation_dataset_spec = spec.datasets.validation

    train_raw_dataset = load_dataset(
        train_dataset_spec.name, **train_dataset_spec.args
    )
    train_dataset_preprocess = dataset_preprocessor(
        train_raw_dataset, **train_dataset_spec.preprocess_args
    )
    train_dataset = Dataset.from_generator(train_dataset_preprocess)

    if validation_dataset_spec:
        validation_raw_dataset = load_dataset(
            validation_dataset_spec.name, **validation_dataset_spec.args
        )
        validation_dataset_preprocess = dataset_preprocessor(
            validation_raw_dataset, **validation_dataset_spec.preprocess_args
        )
        validation_dataset = Dataset.from_generator(
            validation_dataset_preprocess
        )
    else:
        validation_dataset = None

    return Objective(objective_type, loss, train_dataset, validation_dataset)


def load_evaluator(spec: config.EvaluatorConfig) -> evaluation.InformationRetrievalEvaluator:
    raw_dataset = load_dataset(spec.dataset.name, **spec.dataset.args)
    dataset_preprocessor = preprocess.EFAQRetrievalTransform(
        raw_dataset, **spec.dataset.preprocess_args
    )
    retrieval_data = dataset_preprocessor()

    return evaluation.InformationRetrievalEvaluator(
        **retrieval_data,
        **spec.args
    )


def main(filename: str):
    training_config = config.load(filename)

    training_args = training_config.args
    training_args.output_dir = os.path.join(
        training_config.output_dir, training_config.run_name
    )

    logs_dir = os.path.join(training_args.output_dir, "logs.jsonl")
    metrics_dir = os.path.join(training_args.output_dir, "resources.jsonl")

    ## Load Base Model ##
    base_model = SentenceTransformer(
        training_config.base_model.name, **training_config.base_model.args
    )

    ## Load Objectives ##
    train_dataset = {}
    validation_dataset = {}
    losses = {}
    for objective_spec in training_config.objectives:
        objective = map_objective(base_model, objective_spec)
        losses[objective.name] = objective.loss
        train_dataset[objective.name] = objective.train_dataset
        if objective.validation_dataset:
            validation_dataset[objective.name] = objective.validation_dataset

    if not validation_dataset:
        validation_dataset = None

    ## Load Evaluator ##
    evaluator = load_evaluator(training_config.evaluator)

    ## Trainer ##
    trainer = SentenceTransformerTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        loss=losses,
        evaluator=evaluator,
        callbacks=[
            logger.JSONLLoggerCallback(logs_dir),
            profiler.MemoryProfilerCallback(metrics_dir, monitor_cuda=True)
        ]
    )

    trainer.train()
