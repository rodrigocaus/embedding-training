import os

import pandas as pd
from typing import List
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from utils.preprocess import format_retrieval_evaluator

from evaluators.information_retrieval import IR_evaluator

def get_dims(hidden_size: int) -> List[int]:
    return [
        s for s in (64, 128, 256, 384, 512, 768, 1024)
        if s <= hidden_size
    ]



if __name__ == "__main__":
    ### --- Model loading ---###
    #MODEL_PATH = "Alibaba-NLP/gte-multilingual-base"
    MODEL_PATH = "embedding/evaluation/models/multilingual-e5-base-e-faq-c3"
    #MODEL_PATH = "GoBotsAI/multilingual-e5-large-finetuned-e-faq"
    original = "models" not in MODEL_PATH
    print(MODEL_PATH)
    dataset = load_dataset('GoBotsAI/e-faq')

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
        # compute_IR
        print("Computing IR ...")
        formatted_data = format_retrieval_evaluator(dataset=dataset['pt'])
        
        IR_evaluator(model=model,model_path=MODEL_PATH,lang='pt',formatted_data=formatted_data,dim=dim,original=original)
        formatted_data = format_retrieval_evaluator(dataset=dataset['es'])
        IR_evaluator(model=model,model_path=MODEL_PATH,lang='es',formatted_data=formatted_data,dim=dim,original=original)

        