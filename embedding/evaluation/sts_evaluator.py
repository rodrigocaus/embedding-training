import os

import pandas as pd
from typing import List
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


from utils.preprocess import format_retrieval_gosim3

from evaluators.information_retrieval import IR_evaluator
from evaluators.sts import sts_evaluator


def get_dims(hidden_size: int) -> List[int]:
    return [
        s for s in (64, 128, 256, 384, 512, 768, 1024)
        if s <= hidden_size
    ]

def map_label_sts(example):
    dict_to_value = {
        "similar":2,
        "almost similar": 1,
        "dissimilar": 0
    }
    example['score'] = dict_to_value[example["label"]]
    return example
 

if __name__ == "__main__":
    ### --- Model loading ---###
    ##MODEL_PATH = "Alibaba-NLP/gte-multilingual-base"
    MODEL_PATH = "embedding/evaluation/models/multilingual-e5-base-e-faq-c3"
    #MODEL_PATH = "GoBotsAI/multilingual-e5-large-finetuned-e-faq"
    original = "GoBotsAI" not in MODEL_PATH and "models" not in MODEL_PATH
    print(MODEL_PATH)
    dataset = load_dataset('GoBotsAI/GoSim-3')['test']
    dataset = dataset.map(map_label_sts)

    # GoSim-3 -> sentence1, sentence2, label [similar, dissimilar,almost similar]
    if "e5" in MODEL_PATH:
        model = SentenceTransformer(
            MODEL_PATH, prompts={"question": "query: "}, default_prompt_name="question"
        )
    else:
        model = SentenceTransformer(MODEL_PATH,trust_remote_code=True)
    ### --- Evaluation for IR and classification---#
    dimension = get_dims(model.get_sentence_embedding_dimension())
    if original:
        # just using full dimension
        dimension = [dimension[-1]]
    for dim in dimension:
        # compute_STS
        print("Computing STS ...")
        sts_evaluator(
            model_path=MODEL_PATH,
            model=model,
            test_dataset=dataset,
            col1="sentence1",
            col2="sentence2",
            score="score",
            dataset_name="GoSim3",
            original=original,
            dim=dim,
        )
        # compute_IR
        print("Computing IR ...")
        formatted_data = format_retrieval_gosim3(dataset=dataset)
        IR_evaluator(model=model,model_path=MODEL_PATH,lang='pt',formatted_data=formatted_data,dim=dim,original=original)
        