import random
from collections import defaultdict
from datasets import Dataset
from typing import Any, Dict, Tuple, TypedDict


class InformationRetrievalDataset(TypedDict):
    """retrieval dict dataset: 
        ```
            queries: Dict[str, str]
                Map of query_id to query text
            corpus: Dict[str, str]
                Map of document_id to document text
            relevant_docs: Dict[str, Set[str]]
                Map of query_id to set of relevant document_ids
        ```
    """
    queries: Dict[str, str]
    corpus: Dict[str, str]
    relevant_docs: Dict[str, set[str]]


class EFAQRetrievalTransform:
    """
        Transform e-faq dataset in format:
        ```
            sentence: str
            similar: List[str]
            almost_similar: List[str]
            dissimilar: List[str]
        ```
        to `InformationRetrievalDataset`.
        Parameters
        ----------
        dataset: Dataset
            The input dataset.
        query_column: str
            The name of the column containing the query text.
        document_column: str
            The name of the column containing the document text.
        negative_columns: Tuple[str, ...]
            The names of the columns containing the negative samples.
        negative_samples: int = -1
            The number of negative samples to include in corpus.
            If -1, all negative samples are included.
            If 0, no negatives are included
    """

    def __init__(
        self,
        dataset: Dataset,
        query_column: str = "sentence",
        document_column: str = "similar",
        negative_columns: Tuple[str, ...] = ("almost_similar", "dissimilar"),
        negative_samples: int = -1
    ):
        self.dataset = dataset
        self.query_column = query_column
        self.document_column = document_column
        self.negative_columns = negative_columns
        self.negative_samples = negative_samples

    def __call__(self) -> InformationRetrievalDataset:
        queries = {}
        corpus = {}
        relevant_docs = defaultdict(set)

        for i, example in enumerate(self.dataset, start=1):
            query_id = f"q_{i}"
            queries[query_id] = example[self.query_column]
            for j, doc_text in enumerate(example[self.document_column], start=1):
                doc_id = f"doc_{i}_{j}"
                corpus[doc_id] = doc_text
                relevant_docs[query_id].add(doc_id)

            negative_samples = []
            for column in self.negative_columns:
                negative_samples.extend(example.get(column, []))

            random.shuffle(negative_samples)
            negative_samples = negative_samples[:self.negative_samples]

            for j, doc_text in enumerate(negative_samples, start=1):
                doc_id = f"negative_doc_{i}_{j}"
                corpus[doc_id] = doc_text

        return InformationRetrievalDataset(
            queries=queries,
            corpus=corpus,
            relevant_docs=dict(relevant_docs)
        )


class STS2RetrievalTransform:
    def __init__(
        self,
        dataset: Dataset,
        query_column: str = "sentence1",
        document_column: str = "sentence2",
        label_column: str = "label",
        similar_label_value: Any = "similar"
    ):
        self.dataset = dataset
        self.query_column = query_column
        self.document_column = document_column
        self.label_column = label_column
        self.similar_label_value = similar_label_value

    def __call__(self) -> InformationRetrievalDataset:
        queries = {}
        corpus = {}
        relevant = defaultdict(set)

        for i, item in enumerate(self.dataset):
            if item[self.label_column] == self.similar_label_value:
                queries[f"q_{i}"] = str(item[self.query_column])
                corpus[f"Q_{i}"] = str(item[self.document_column])
                relevant[f"q_{i}"].add(f"Q_{i}")
            else:
                corpus[f"c_{i}"] = str(item[self.query_column])

        return InformationRetrievalDataset(
            queries=queries,
            corpus=corpus,
            relevant_docs=dict(relevant)
        )
