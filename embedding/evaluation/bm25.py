import bm25s
from typing import Optional
from sentence_transformers.evaluation import InformationRetrievalEvaluator


class BM25InformationRetrievalEvaluator(InformationRetrievalEvaluator):
    def __call__(
        self,
        model: Optional[bm25s.BM25] = None,
        top_k: int = 100,
        n_threads: int = 4,
        **_
    ):
        if model is None:
            model = bm25s.BM25(method="lucene")

        corpus = bm25s.tokenize(self.corpus)
        queries = bm25s.tokenize(self.queries)
        model.index(corpus)

        queried_results, queried_scores = model.retrieve(
            queries, corpus=self.corpus_ids, k=top_k, n_threads=n_threads
        )

        results = [
            [
                dict(corpus_id=doc, score=s)
                for doc, s in zip(documents, scores)
            ]
            for documents, scores in zip(queried_results, queried_scores)
        ]

        return self.compute_metrics(results)
