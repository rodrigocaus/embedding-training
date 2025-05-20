from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from typing import List
from utils.preprocess import format_retrieval_evaluator
from utils.file_management import create_folder,format_evaluation_name
import sys
from score_functions import reranker_similarity
import json
import os
def get_dims(hidden_size: int) -> List[int]:
    return [
        s for s in (64, 128, 256, 384, 512, 768, 1024)
        if s <= hidden_size
    ]

def sematic_search(model,queries,corpus,k=20,dim=None):
    corpus_emb = model.encode(list(corpus.values()), convert_to_tensor=True)
    if dim is not None:
        corpus_emb = corpus_emb[:, :dim]
    initial_ranking = {}
    for qid, qtext in queries.items():
        q_emb = model.encode(qtext, convert_to_tensor=True)
        if dim is not None:
            q_emb = q_emb[:, :dim]
        hits = util.semantic_search(q_emb, corpus_emb, top_k=k)[0]
        initial_ranking[qid] = [list(corpus.keys())[h['corpus_id']] for h in hits]
    return initial_ranking
def compute_retrival_with_rerank(formatted_data,dim,host):
    queries = formatted_data['queries']
    corpus = formatted_data['corpus']
    relevant_docs = formatted_data['relevant_docs']
    print("Computing semantic search ...")
    initial_ranking = sematic_search(model,queries,corpus,k=20,dim=dim)
    print("Computing reranker ...")
    reranked = {
        qid: [doc for _, doc in sorted(
            zip(reranker_similarity(host = host,sentences1=[queries[qid]]*len(docs),sentences2=[corpus[x] for x in docs]), docs),
            reverse=True)]
        for qid, docs in initial_ranking.items()
       }
    result = [ value[0] in relevant_docs[qid] for qid, value in reranked.items()]
    acc = sum(result)/len(result)
    print(f"Accuracy@1 = {acc}")
    return acc


if __name__ == "__main__":
    host = sys.argv[1] # host of reranker 
    # --- DATASET --- #
    dataset = load_dataset('GoBotsAI/e-faq')
    # --- MODEL --- #
    #MODEL_PATH = "intfloat/multilingual-e5-large"
    MODEL_PATH = "embedding/evaluation/models/multilingual-e5-base-e-faq-c3"
    original = "models" not in MODEL_PATH
    print(MODEL_PATH)
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
    dimension = [dimension[-1]]
    results = {}
    for dim in dimension:
        # compute_IR
        print("Computing IR for pt ...")
        # For PT
        formatted_data = format_retrieval_evaluator(dataset=dataset['pt'])
        acc = compute_retrival_with_rerank(formatted_data=formatted_data,dim=None,host=host)
        res_pt={"pt":{"acc":acc}}
        # For ES
        print("Computing IR for es ...")
        formatted_data = format_retrieval_evaluator(dataset=dataset['es'])
        acc = compute_retrival_with_rerank(formatted_data=formatted_data,dim=None,host=host)
        res_es={"es":{"acc":acc}}
        results[dim] = res_pt | res_es
    create_folder("results/reranker")
    base_path = format_evaluation_name(
        model_path=MODEL_PATH, original=original
    )
    create_folder(os.path.join("results/reranker", base_path))
    output_path = os.path.join("results/reranker", base_path)
    output_file = os.path.join(output_path,"results.json")
    with open(output_file, "w+") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    
