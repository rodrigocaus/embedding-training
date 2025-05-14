from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)
import os
import pandas as pd
from utils.file_management import create_folder, format_evaluation_name
def sts_evaluator(
    model_path,
    model,
    test_dataset,
    col1="sentence1",
    col2="sentence2",
    score="score",
    dataset_name=None,
    original=False,
    dim=None,
):
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=test_dataset[col1],
        sentences2=test_dataset[col2],
        scores=test_dataset[score],
        main_similarity=SimilarityFunction.COSINE,
        truncate_dim=dim,
    )

    base_path = format_evaluation_name(
        model_path=model_path, original=original
    )

    create_folder(os.path.join("results", base_path))
    output_path = os.path.join("results", base_path, f"{dataset_name}.csv")
    result = evaluator(model)
    ### Saving the values in the summary file ###
    result.pop("epoch", None)
    result.pop("step", None)
    result = {
        "dim": dim,
        **result
    }

    df = pd.DataFrame([result])
    if os.path.isfile(output_path):
        df0 = pd.read_csv(output_path)
        df = pd.concat([df0, df])
        df = df.reset_index(drop=True)
    df.to_csv(output_path, index=False)
    ### Showing each results in the terminal###
    for key, value in result.items():
        print(f"{key}:{value}")