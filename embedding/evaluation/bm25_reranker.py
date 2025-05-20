from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from typing import List
from utils.preprocess import format_retrieval_evaluator
from utils.file_management import create_folder,format_evaluation_name
import sys
from score_functions import reranker_similarity
import json
import os
import bm25s
def get_dims(hidden_size: int) -> List[int]:
    return [
        s for s in (64, 128, 256, 384, 512, 768, 1024)
        if s <= hidden_size
    ]

def bm25_search(retrieval_data,k=20):
    
    initial_ranking = {}
    ## --- RETRIEVER --- #
    corpus = bm25s.tokenize(
                    list(retrieval_data["corpus"].values())
                    )
    queries = bm25s.tokenize(
                    list(retrieval_data["queries"].values())
                )
    model = bm25s.BM25(method="lucene")
    model.index(corpus)

    ## --- SEARCH --- #
    queried_results, queried_scores = model.retrieve(
            queries, corpus=list(retrieval_data["corpus"].keys()), k=k, n_threads=4
    )

    for i,query_id in enumerate(retrieval_data["queries"].keys()):
        initial_ranking[query_id] = list(queried_results[i])
    return initial_ranking
def compute_retrival_with_rerank(retrieval_data,host):
    queries = retrieval_data['queries']
    corpus = retrieval_data['corpus']
    relevant_docs = retrieval_data['relevant_docs']
    print("Computing bm25 search ...")
    initial_ranking = bm25_search(retrieval_data,k=20)
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
    # --- BM25 --- #
    
    results = {}
    # compute_IR
    print("Computing IR for pt ...")
    # For PT
    formatted_data = format_retrieval_evaluator(dataset=dataset['pt'])
    acc = compute_retrival_with_rerank(retrieval_data=formatted_data,host=host)
    res_pt={"pt":{"acc":acc}}
    # For ES
    print("Computing IR for es ...")
    formatted_data = format_retrieval_evaluator(dataset=dataset['es'])
    acc = compute_retrival_with_rerank(retrieval_data=formatted_data,host=host)
    res_es={"es":{"acc":acc}}    
    create_folder(os.path.join("results/reranker", "BM25"))
    output_path = os.path.join("results/reranker", "BM25")
    output_file = os.path.join(output_path,"results.json")
    with open(output_file, "w+") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    
