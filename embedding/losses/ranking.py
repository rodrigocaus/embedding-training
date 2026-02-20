from typing import Any, List, Iterable

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class AugmentedMultipleNegativesSymmetricRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim) -> None:
        """
        Given a list of (anchor, positive) pairs or (anchor, positive, negative) triplets, this loss optimizes the following:

        1. Forward loss: Given an anchor, find the sample with the highest similarity out of all positives in the batch.
           This is equivalent to :class:`MultipleNegativesRankingLoss`.
        2. Backward loss: Given a positive, find the sample with the highest similarity out of all anchors in the batch.

        For example with question-answer pairs, :class:`MultipleNegativesRankingLoss` just computes the loss to find
        the answer given a question, and :class:`MultipleNegativesSymmetricRankingLoss` additionally computes the
        loss to find the question given an answer. This class allows to pass negative samples for both anchor and positive pairs.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale
                value
            similarity_fct: similarity function between sentence
                embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative) triplets

        Inputs:
            +-------------------------------------------------+--------+
            | Texts                                           | Labels |
            +=================================================+========+
            | (anchor, positive) pairs                        | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative) triplets           | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) | none   |
            +-------------------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = AugmentedMultipleNegativesSymmetricRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def compute_loss(self, anchor: Tensor, positives: Tensor, negatives: Iterable[Tensor], labels: Tensor) -> Tensor:
        candidates = torch.cat((positives, *negatives))
        scores = self.similarity_fct(anchor, candidates) * self.scale
        return self.cross_entropy_loss(scores, labels)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings: List[Tensor] = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        anchor, positive, *negatives = embeddings
        # Example a[i] should match with b[i]
        labels = torch.arange(0, len(anchor), device=anchor.device)

        forward_loss = self.compute_loss(anchor, positive, negatives, labels)
        backward_loss = self.compute_loss(positive, anchor, negatives, labels)
        return (forward_loss + backward_loss) / 2.0

    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}
