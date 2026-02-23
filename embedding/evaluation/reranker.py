import requests
from typing import List, Optional
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import SentenceEvaluator


class Reranker(ABC):
    @abstractmethod
    def predict(self, sentences1: List[str], sentences2: List[str]) -> List[float]:
        """
        Predicts the similarity scores for pairs of sentences.
        """
        pass


class SentenceTransformerReranker(Reranker):
    def __init__(self, model: str, **kwargs):
        self.model = SentenceTransformer(model, **kwargs)

    def predict(self, sentences1: List[str], sentences2: List[str]) -> List[float]:
        embeddings1 = self.model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentences2, convert_to_tensor=True)
        similarities = self.model.similarity_pairwise(embeddings1, embeddings2)
        return similarities.view(-1).tolist()


class ServedReranker(Reranker):
    def __init__(self, host: str):
        self.host = host
        self.session = requests.Session()

    def predict(self, sentences1: List[str], sentences2: List[str]) -> List[float]:
        response = self.session.post(
            url=self.host,
            json={
                "inputs": {
                    "input1": sentences1,
                    "input2": sentences2
                }
            }
        )
        return response.json()['outputs']


class RerankerInformationRetrievalEvaluator(SentenceEvaluator):
    def __init__(
        self,
        queries: dict[str, str],  # qid => query
        corpus: dict[str, str],  # cid => doc
        relevant_docs: dict[str, set[str]],  # qid => Set[cid],
        reranker: Reranker,
        top_k: int = 20,
        truncate_dim: Optional[int] = None
    ):
        super().__init__()
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        self.relevant_docs = relevant_docs
        self.reranker = reranker
        self.top_k = top_k
        self.truncate_dim = truncate_dim

    def __call__(self, model: SentenceTransformer, **_):
        corpus_embeddings = model.encode(self.corpus, convert_to_tensor=True)
        queries_embeddings = model.encode(self.queries, convert_to_tensor=True)
        if self.truncate_dim is not None:
            corpus_embeddings = corpus_embeddings[:, :self.truncate_dim]
            queries_embeddings = queries_embeddings[:, :self.truncate_dim]

        hits = util.semantic_search(
            queries_embeddings, corpus_embeddings, top_k=self.top_k
        )

        reranked = {}
        for i, search_results in enumerate(hits):
            query = self.queries[i]
            sub_corpora = [self.corpus[h['corpus_id']] for h in search_results]
            reranking = self.reranker.predict(
                sentences1=[query]*len(sub_corpora),
                sentences2=sub_corpora
            )
            reordered_sub_corpora = [
                x for _, x in sorted(zip(reranking, sub_corpora), reverse=True)
            ]
            reranked[self.queries_ids[i]] = [
                self.corpus_ids[self.corpus.index(x)]
                for x in reordered_sub_corpora
            ]

        result = [
            value[0] in self.relevant_docs[qid]
            for qid, value in reranked.items()
        ]
        acc = sum(result)/len(result)
        return {
            "acc": acc
        }
