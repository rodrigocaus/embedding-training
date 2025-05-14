import json
import bm25s
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from utils.preprocess import format_retrieval_evaluator
from utils.file_management import create_folder
from datasets import load_dataset
if __name__ == "__main__":
    # --- DATASET --- #
    dataset = load_dataset('GoBotsAI/e-faq')
    for data_split in ["pt","es"]:
        
        retrieval_data = format_retrieval_evaluator(dataset[data_split])

        # --- EVALUATOR --- #
        evaluator = InformationRetrievalEvaluator(
            **retrieval_data,
            mrr_at_k=[1, 10],
            ndcg_at_k=[1, 10],
            accuracy_at_k=[1, 10],
            precision_recall_at_k=[1, 10],
            map_at_k=[1, 10]
        )

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
            queries, corpus=list(retrieval_data["corpus"].keys()), k=100, n_threads=4
        )

        results = [
            [
                dict(corpus_id=doc, score=s)
                for doc, s in zip(documents, scores)
            ]
            for documents, scores in zip(queried_results, queried_scores)
        ]

        ## --- EVAL --- #
        create_folder("results/bm25")
        with open(f"results/bm25/{data_split}_results.json", "w+") as fp:
            metrics = evaluator.compute_metrics(results)
            json.dump(metrics, fp, ensure_ascii=False, indent=2)