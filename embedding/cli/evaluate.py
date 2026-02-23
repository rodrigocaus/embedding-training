import pathlib
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SimilarityFunction,
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    InformationRetrievalEvaluator
)
from typing import Optional, Iterable

import config
from evaluation.results import ResultsManager
from evaluation.reranker import SentenceTransformerReranker, ServedReranker, RerankerInformationRetrievalEvaluator
from evaluation.bm25 import BM25InformationRetrievalEvaluator
from preprocess.retrieval import EFAQRetrievalTransform, STS2RetrievalTransform


def get_model_dimensions(base_dims: Optional[Iterable[int]], model: SentenceTransformer) -> list[int]:
    if base_dims is None:
        base_dims = (64, 128, 256, 384, 512, 768, 1024)

    sentence_embedding_dimension = model.get_sentence_embedding_dimension()
    return [dim for dim in base_dims if dim <= sentence_embedding_dimension]


def format_evaluation_name(*names) -> str:
    return "/".join(map(str, names))


def run_ir(
    model: Optional[SentenceTransformer],
    model_dimensions: Iterable[int],
    task: config.EvaluationTask,
    results: ResultsManager
):
    """
    Run Information Retrieval evaluation based on the given configuration.
    """
    dataset = load_dataset(task.dataset.name, **task.dataset.args)
    evaluation_name = format_evaluation_name(
        task.dataset.name,
        *task.dataset.args.values()
    )

    if "gosim" in task.dataset.name.lower():
        formatted_data = STS2RetrievalTransform(
            dataset=dataset,
            **task.dataset.preprocess_args
        )()
    else:
        formatted_data = EFAQRetrievalTransform(
            dataset=dataset,
            **task.dataset.preprocess_args
        )()

    if model is None:
        def evaluator_factory(dim): return BM25InformationRetrievalEvaluator(
            **formatted_data,
            **task.args
        )
        model_dimensions = [None]

    elif task.reranker is not None:
        if task.reranker.type == "sentence_transformer":
            reranker = SentenceTransformerReranker(
                task.reranker.model.name, **task.reranker.model.args
            )
        else:
            reranker = ServedReranker(task.reranker.model.name)

        def evaluator_factory(dim): return RerankerInformationRetrievalEvaluator(
            **formatted_data,
            reranker=reranker,
            **task.args,
            truncate_dim=dim
        )
    else:
        def evaluator_factory(dim): return InformationRetrievalEvaluator(
            **formatted_data,
            **task.args,
            truncate_dim=dim
        )

    for dim in model_dimensions:
        evaluator = evaluator_factory(dim)
        result = evaluator(model)
        result.pop("epoch", None)
        result.pop("step", None)
        result = {
            "name": evaluation_name,
            "dim": dim,
            **result
        }
        results.add("ir", result)


def run_sts(
    model: SentenceTransformer,
    model_dimensions: Iterable[int],
    task: config.EvaluationTask,
    results: ResultsManager
):
    """
    Run STS evaluation based on the given configuration.
    """
    # Load dataset
    label_to_score = {
        "similar": 1,
        "almost_similar": 0,
        "almost similar": 0,
        "dissimilar": -1,
    }
    dataset = load_dataset(task.dataset.name, **task.dataset.args)
    dataset = dataset.map(
        lambda example: {
            "score": label_to_score[example["label"]]
        },
        remove_columns=["label"]
    )

    evaluation_name = format_evaluation_name(
        task.dataset.name,
        *task.dataset.args.values()
    )

    # Run evaluation
    for dim in model_dimensions:
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=dataset["sentence1"],
            sentences2=dataset["sentence2"],
            scores=dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            truncate_dim=dim,
            **task.args
        )
        result = evaluator(model)
        result.pop("epoch", None)
        result.pop("step", None)
        result = {
            "name": evaluation_name,
            "dim": dim,
            **result
        }
        results.add("sts", result)


def run_evaluation(
    model: Optional[SentenceTransformer],
    model_dimensions: Optional[Iterable[int]],
    tasks: Iterable[config.EvaluationTask],
    results: ResultsManager
):
    """
    Run evaluation based on the given configuration.
    """
    for task in tasks:
        if task.type == "ir":
            run_ir(
                model,
                model_dimensions,
                task,
                results
            )
        elif task.type == "sts":
            run_sts(
                model,
                model_dimensions,
                task,
                results
            )

    results.print()
    results.dump()


def main(config_path: str = None):
    """
    Run evaluation based on a configuration file.
    """

    eval_config = config.load_eval(config_path)
    if eval_config.model.name.lower() != "bm25":
        model = SentenceTransformer(
            eval_config.model.name,
            **eval_config.model.args
        )
    else:
        model = None

    if model:
        model_dimensions = get_model_dimensions(eval_config.dimensions, model)
    else:
        model_dimensions = None

    output_dir = pathlib.Path(eval_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = ResultsManager(eval_config.model.name, output_dir)

    run_evaluation(
        model,
        model_dimensions,
        eval_config.tasks,
        results
    )
