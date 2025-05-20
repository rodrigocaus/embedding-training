import itertools
from datasets import Dataset
from typing import Iterable, List, Tuple, TypeVar

T = TypeVar("T")


def unique(items: Iterable[T]) -> List[T]:
    return list(set(items))


def filter_non_empty(items: Iterable[T]) -> List[T]:
    return [item for item in items if item]


class EFAQRankingGenerator:
    """
        Process a e-faq data in format:
        ```
            sentence: str
            similar: List[str]
            almost_similar: List[str]
            dissimilar: List[str]
        ```
        to a tuple of strings in format:
        ```
            (anchor, positive, negative_1, ..., negative_n)
        ```
        Parameters
        ----------
        dataset: Dataset
            The input dataset.
        negative_samples: int
            The number of negative samples to include in each tuple.
            If 0, no negatives are included. If > 0, and not enough
            negatives can be found for an (anchor, positive) pair,
            that pair is skipped.
    """

    def __init__(self, dataset: Dataset, negative_samples: int) -> None:
        if negative_samples < 0:
            raise ValueError("negative_samples must be >= 0")
        self.dataset = dataset
        self.negative_samples = negative_samples
        columns = ["anchor", "positive"]
        if self.negative_samples > 0:
            columns.extend([
                f"negative_{i+1}" for i in range(self.negative_samples)
            ])
        self.output_columns: Tuple[str, ...] = tuple(columns)

    def as_columns_dict(self, *args):
        return dict(zip(self.output_columns, args))

    def __iter__(self):
        for entry in self.dataset:
            anchor: str = entry["sentence"]
            positive_candidates: List[str] = filter_non_empty(
                unique(entry["similar"])
            )
            potential_negatives: List[str] = filter_non_empty(
                unique(entry["almost_similar"] + entry["dissimilar"])
            )
            if not positive_candidates:
                continue

            for positive in positive_candidates:
                if self.negative_samples == 0:
                    yield self.as_columns_dict(anchor, positive)
                elif len(potential_negatives) >= self.negative_samples:
                    for negatives in itertools.combinations(potential_negatives, self.negative_samples):
                        yield self.as_columns_dict(anchor, positive, *negatives)

    def __call__(self):
        return iter(self)
