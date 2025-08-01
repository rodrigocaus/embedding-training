import random
import itertools
from datasets import Dataset
from typing import Iterable, List, Set, Tuple, TypeVar

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

        negative_samples_pool: List[str] = []
        columns = ["anchor", "positive"]
        if self.negative_samples > 0:
            columns.extend([
                f"negative_{i+1}" for i in range(self.negative_samples)
            ])
            negative_samples_pool.extend(
                sentence
                for entry in dataset["almost_similar"]
                for sentence in filter_non_empty(entry)
            )
            negative_samples_pool.extend(
                sentence
                for entry in dataset["dissimilar"]
                for sentence in filter_non_empty(entry)
            )

        self.output_columns: Tuple[str, ...] = tuple(columns)
        self.negative_samples_pool = unique(negative_samples_pool)

    def as_columns_dict(self, *args):
        return dict(zip(self.output_columns, args))

    def __iter__(self):
        if self.negative_samples_pool and self.negative_samples > 0:
            random.shuffle(self.negative_samples_pool)
            negative_samples_pool = itertools.cycle(self.negative_samples_pool)
        else:
            negative_samples_pool = None

        for entry in self.dataset:
            anchor: str = entry["sentence"]
            positive_candidates: List[str] = filter_non_empty(
                unique(entry["similar"])
            )
            if not positive_candidates:
                continue

            negative_candidates: Set[str] = set(filter_non_empty(
                entry["dissimilar"] + entry["almost_similar"]
            ))

            while len(negative_candidates) < self.negative_samples:
                additional_negatives_needed = self.negative_samples - len(negative_candidates)
                new_negatives = itertools.islice(
                    negative_samples_pool, additional_negatives_needed
                )
                negative_candidates.update(new_negatives)

            for positive in positive_candidates:
                if self.negative_samples == 0:
                    yield self.as_columns_dict(anchor, positive)
                else:
                    negatives = random.sample(
                        list(negative_candidates), self.negative_samples
                    )
                    yield self.as_columns_dict(anchor, positive, *negatives)

    def __call__(self):
        return iter(self)
