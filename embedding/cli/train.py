import os
import config
import logging
import preprocess
import similarity as custom_similarity
import losses as custom_losses
from util import logger, profiler

from torch import nn
from typing import NamedTuple, Optional, Tuple
from datasets import load_dataset, Dataset
from transformers import EarlyStoppingCallback
from sentence_transformers import (
    losses, evaluation,
    SimilarityFunction,
    SentenceTransformer, SentenceTransformerTrainer
)

_loss_objective_mapping = {
    "contrastive": {
        "ranking": losses.MultipleNegativesSymmetricRankingLoss,
        "default": losses.MultipleNegativesSymmetricRankingLoss,
    },
    "similarity": {
        "cosine": losses.CoSENTLoss,
        "augmented-cosine": custom_losses.AugmentedCoSENTLoss,
        "cross-entropy": custom_losses.BinaryCrossEntropyLoss,
        "augmented-cross-entropy": custom_losses.AugmentedBinaryCrossEntropyLoss,
        "cosine-cross-entropy": custom_losses.CosineCrossEntropyLoss,
        "default": losses.CoSENTLoss,
    }
}


class Objective(NamedTuple):
    name: str
    loss: nn.Module
    train_dataset: Dataset
    validation_dataset: Optional[Dataset]


def map_teacher(
    model: SentenceTransformer,
    spec: config.DistilConfig,
    objective: str,
    teacher_pool: dict
) -> Tuple[nn.Module, float]:
    if spec.teacher.name not in teacher_pool:
        teacher_pool[spec.teacher.name] = SentenceTransformer(
            spec.teacher.name, **spec.teacher.args
        )

    teacher = teacher_pool[spec.teacher.name]
    if objective == "contrastive":
        distil_loss = custom_losses.KLSimilarityLoss(
            student=model, teacher=teacher, **spec.loss_args
        )
    if objective == "similarity":
        distil_loss = custom_losses.SimilarityDistillationLoss(
            student=model, teacher=teacher, **spec.loss_args
        )

    return distil_loss, spec.alpha


def map_variation_penalty(
    model: SentenceTransformer,
    spec: config.VariationPenaltyConfig,
    objective: str
) -> Tuple[nn.Module, float]:
    if objective == "contrastive":
        variation_penalty_loss = custom_losses.InBatchNegativesVariancePenaltyLoss(
            model=model, **spec.loss_args
        )
    if objective == "similarity":
        variation_penalty_loss = custom_losses.LabeledNegativesVariancePenaltyLoss(
            model=model, **spec.loss_args
        )

    return variation_penalty_loss, spec.alpha


def map_objective(
    model: SentenceTransformer,
    spec: config.TrainingObjective,
    teacher_pool: dict
) -> Objective:
    objective_type = spec.type
    if objective_type not in _loss_objective_mapping.keys():
        raise ValueError(f"Unknown objective type: {objective_type}")

    ## SIMILARITY DEFINITION ##
    if spec.margin is not None:
        if objective_type == "contrastive":
            similarity_fn = custom_similarity.SimilarityWithMargin(
                margin=spec.margin, similarity=SimilarityFunction.COSINE
            )
        elif objective_type == "similarity":
            similarity_fn = custom_similarity.PairwiseSimilarityWithMargin(
                margin=spec.margin, similarity=SimilarityFunction.COSINE
            )

        spec.loss_args["similarity_fct"] = similarity_fn

    ## DATASET DEFINITION ##
    if objective_type == "contrastive":
        dataset_preprocessor = preprocess.EFAQRankingGenerator
    elif objective_type == "similarity":
        dataset_preprocessor = preprocess.EFAQCosineSimilarityGenerator

    ## LOSS DEFINITION ##
    try:
        loss_type = _loss_objective_mapping[objective_type][spec.loss]
    except KeyError:
        logging.warning("Invalid loss specified, using default")
        loss_type = _loss_objective_mapping[objective_type]["default"]

    loss = loss_type(model=model, **spec.loss_args)

    if spec.matryoshka:
        loss = losses.MatryoshkaLoss(
            model=model,
            loss=loss,
            **spec.matryoshka
        )

    extra_losses = []
    if spec.distillation:
        extra_losses.append(map_teacher(
            model, spec.distillation, objective_type, teacher_pool
        ))
    if spec.variation_penalty:
        extra_losses.append(map_variation_penalty(
            model, spec.variation_penalty, objective_type
        ))

    if extra_losses:
        extra_losses.insert(0, (loss, 1.0))
        loss_fn, weights = zip(*extra_losses)
        loss = custom_losses.CompositionLoss(
            losses=loss_fn, weights=weights
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
    train_dataset = train_dataset.shuffle()

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
        validation_dataset = validation_dataset.shuffle()
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
    teacher_pool = {}
    train_dataset = {}
    validation_dataset = {}
    losses = {}
    for objective_spec in training_config.objectives:
        objective = map_objective(base_model, objective_spec, teacher_pool)
        losses[objective.name] = objective.loss
        train_dataset[objective.name] = objective.train_dataset
        if objective.validation_dataset:
            validation_dataset[objective.name] = objective.validation_dataset

    if not validation_dataset:
        validation_dataset = None

    callbacks = [
        logger.JSONLLoggerCallback(logs_dir),
        profiler.MemoryProfilerCallback(metrics_dir, monitor_cuda=True)
    ]
    if training_config.early_stopping:
        args = {
            f"early_stopping_{key}": value
            for key, value in training_config.early_stopping.items()
        }
        callbacks.append(
            EarlyStoppingCallback(**args)
        )

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
        callbacks=callbacks
    )

    trainer.train()
