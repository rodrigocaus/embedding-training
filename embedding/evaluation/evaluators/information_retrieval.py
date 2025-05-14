from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
)
from utils.file_management import format_evaluation_name,create_folder
import os
import pandas as pd
def IR_evaluator(model, model_path, lang, formatted_data,dim=None, k=10, original=None):
    print(
        f"testing model in {model_path} for EFAQ for {lang} with dimension {dim}")

    evaluator = InformationRetrievalEvaluator(
        queries=formatted_data["queries"],
        corpus=formatted_data["corpus"],
        relevant_docs=formatted_data["relevant_docs"],
        map_at_k=[1, 3, k],
        ndcg_at_k=[1, 3, k],
        mrr_at_k=[1, 3, k],
        precision_recall_at_k=[1, 3, k],
        accuracy_at_k=[1, 3, k],
        truncate_dim=dim,
    )
    dataset_name = f"E_FAQ_{lang}"
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