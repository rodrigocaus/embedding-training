import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Iterable
from itertools import islice

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, **_) -> None:
        """
        This class implements a Binary Cross Entropy Loss for similarity scores.
        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair. The scores are expected to be in the range [-1, 1].

        Args:
            model: SentenceTransformerModel
            scale: Output of similarity function is multiplied by scale value.
                   This is typically used to scale the similarity scores before applying the sigmoid.

        Requirements:
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+
        """
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in islice(sentence_features, 2)
        ]

        scores = util.pairwise_cos_sim(embeddings[0], embeddings[1])
        scores = scores * self.scale

        labels = torch.sigmoid(labels * self.scale)

        return F.binary_cross_entropy_with_logits(
            scores, labels,
            reduction='mean'
        )


class AugmentedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, min_score: float = -1.0, **_) -> None:
        """
        This class implements a Binary Cross Entropy Loss for similarity scores.
        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair. The scores are expected to be in the range [-1, 1]. Additionally, each of the InputExamples are assumed unrelated
        within a mini-batch.

        Args:
            model: SentenceTransformerModel
            scale: Output of similarity function is multiplied by scale value.
                   This is typically used to scale the similarity scores before applying the sigmoid.

        Requirements:
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the sentence_A or sentence_B samples.
        """
        super().__init__()
        self.model = model
        self.scale = scale
        self.min_score = min_score

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in islice(sentence_features, 2)
        ]

        scores = util.cos_sim(embeddings[0], embeddings[1])
        scores = scores.view(-1) * self.scale
        m = len(labels)
        augmented_labels = torch.full((m, m), self.min_score).to(labels.device)
        # label matrix indicating which pairs are the given label
        augmented_labels = augmented_labels.diagonal_scatter(labels, 0)
        augmented_labels = torch.sigmoid(augmented_labels * self.scale)

        return F.binary_cross_entropy_with_logits(
            scores, augmented_labels.view(-1),
            reduction='mean'
        )


class CosineCrossEntropyLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, **_) -> None:
        """
        This class implements a Binary Cross Entropy Loss for similarity scores.
        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair. The scores are expected to be in the range [-1, 1].

        Instead of modeling the similarity scores as a sigmoid function, we model it as a cosine probability function.

        Args:
            model: SentenceTransformerModel
            scale: Score labels are multiplied by scale value before applying the sigmoid.

        Requirements:
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+
        """
        super().__init__()
        self.model = model
        self.scale = scale
        self._zero = torch.tensor(0.0)
        self._one = torch.tensor(1.0)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in islice(sentence_features, 2)
        ]

        scores = util.pairwise_cos_sim(embeddings[0], embeddings[1])
        scores = 0.5 * (scores + 1)
        scores = torch.clamp(scores, min=self._zero, max=self._one)

        labels = torch.sigmoid(labels * self.scale)

        return F.binary_cross_entropy(
            scores, labels,
            reduction='mean'
        )
