import torch
from torch import nn, Tensor
from typing import Any, Dict, Iterable
from sentence_transformers import (
    SentenceTransformer,
    SimilarityFunction
)


class KLSimilarityLoss(nn.Module):
    """
        This loss is an adaptation of MultipleNegativesRankingLoss. MultipleNegativesRankingLoss computes the following loss:
        For a given anchor and a list of candidates, find the positive candidate.

        In KLSimilarityLoss, we consider MultipleNegativesRankingLoss for both Student and Teacher models,
        and calculate the softmax for each anchor. The final loss is the KLDivergence between the probability
        of anchor and positive examples.


        Args:
            student: SentenceTransformer model
            teacher: SentenceTransformer target model
            scale: Output of similarity function is multiplied by scale value
            similarity: Similarity function between sentence embeddings. By default, dot_product.

        Requirements:
            1. (anchor, positive) pairs

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
                from datasets import Dataset

                student = SentenceTransformer("microsoft/mpnet-base")
                teacher = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = KLSimilarityLoss(student, teacher)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """

    def __init__(
        self,
        student: SentenceTransformer,
        teacher: SentenceTransformer,
        scale: float = 20.0,
        reduction: str = "batchmean",
        similarity: SimilarityFunction = SimilarityFunction.COSINE
    ) -> None:
        super().__init__()

        self.teacher = teacher
        self.student = student
        self.scale = scale
        self.similarity = similarity
        self.similarity_fn = similarity.to_similarity_fn(similarity)
        self.loss_fn = nn.KLDivLoss(log_target=True, reduction=reduction)

    def scores(self, model: SentenceTransformer, sentence_features: Iterable[Dict[str, Tensor]]) -> Tensor:
        reps = [
            model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        anchor = reps[0]
        candidates = torch.cat(reps[1:])

        sims = self.scale * self.similarity_fn(anchor, candidates)
        return sims.log_softmax(dim=1)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tensor:
        with torch.no_grad():
            target = self.scores(self.teacher.eval(), sentence_features)
        logits = self.scores(self.student, sentence_features)

        return self.loss_fn(logits, target.detach())

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": self.similarity_fn.__name__,
            "similarity": self.similarity.name
        }
