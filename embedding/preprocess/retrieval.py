import random
from collections import defaultdict
from datasets import Dataset


class EFAQRetrievalTransform:
    """
        Transform e-faq dataset in format:
        ```
            sentence: str
            similar: List[str]
            almost_similar: List[str]
            dissimilar: List[str]
        ```
        to a retrieval dict dataset: 
        ```
            queries: Dict[str, str]
                Map of query_id to query text
            corpus: Dict[str, str]
                Map of document_id to document text
            relevant_docs: Dict[str, Set[str]]
                Map of query_id to set of relevant document_ids
        ```
        Parameters
        ----------
        dataset: Dataset
            The input dataset.
        negative_samples: int
            The number of negative samples to include in corpus.
            If 0, no negatives are included
    """

    def __init__(self, dataset: Dataset, negative_samples: int):
        self.dataset = dataset
        self.negative_samples = negative_samples

    def __call__(self):
        queries = {}
        corpus = {}
        relevant_docs = defaultdict(set)
        for i, entry in enumerate(self.dataset, start=1):
            queries[f"q_{i}"] = entry["sentence"]
            for j, similar in enumerate(entry["similar"], start=1):
                corpus[f"s_{i}_{j}"] = similar
                relevant_docs[f"q_{i}"].add(f"s_{i}_{j}")

            negative_candidates = entry["almost_similar"] + entry["dissimilar"]
            random.shuffle(negative_candidates)
            negatives = negative_candidates[:self.negative_samples]
            for k, negative in enumerate(negatives, start=1):
                corpus[f"n_{i}_{k}"] = negative

        return {
            "queries": queries,
            "corpus": corpus,
            "relevant_docs": dict(relevant_docs),
        }
