import torch
from torch import Tensor
from typing import Callable
from sentence_transformers import SimilarityFunction


class SimilarityWithMargin(Callable[[Tensor, Tensor], Tensor]):
    """
        This similarity function is an adaptation of the `SimilarityFunction`.
        It computes similarity with additional magin between (anchor, positive) pairs
        and without the margin between (anchor, negative) pairs:
        ```
            sim*(a, b) = sim(a, b) - m, if a is similar to b
            sim*(a, b) = sim(a, b),     otherwise
        ```
        Is ideal for increasing separability in contrastive learning.

        Args:
            margin: additional margin
            similarity: similarity function between sentence
                embeddings. By default, cosine.
        """

    def __init__(
        self,
        margin: float = 0.01,
        similarity: SimilarityFunction = SimilarityFunction.COSINE
    ):
        self.margin = margin
        self.similarity_fn = SimilarityFunction.to_similarity_fn(similarity)

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        matrix = self.similarity_fn(a, b)
        n, m = matrix.shape
        eye = torch.eye(n, m, device=matrix.device)
        return matrix - self.margin * eye


class PairwiseSimilarityWithMargin(Callable[[Tensor, Tensor], Tensor]):
    """
        This similarity function is an adaptation of the `SimilarityFunction` pairwise mode.
        It computes similarity with additional magin between (sentence_a, sentence_b) pairs
        ```
            sim*(a, b) = sim(a, b) - m
        ```
        Is ideal for increasing separability in contrastive learning.

        Args:
            margin: additional margin
            similarity: similarity function between sentence
                embeddings. By default, cosine.
        """

    def __init__(
        self,
        margin: float = 0.01,
        similarity: SimilarityFunction = SimilarityFunction.COSINE
    ):
        self.margin = margin
        self.similarity_fn = SimilarityFunction.to_similarity_pairwise_fn(similarity)

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        matrix = self.similarity_fn(a, b)
        return matrix - self.margin
