import torch
from torch import nn, Tensor
from typing import Any, Dict, Iterable
from sentence_transformers import (
    SentenceTransformer,
    SimilarityFunction,
    losses
)


class SimilarityDistillationLoss(nn.Module):
    def __init__(
        self,
        student: SentenceTransformer,
        teacher: SentenceTransformer,
        scale: float = 20.0,
        similarity: SimilarityFunction = SimilarityFunction.DOT
    ) -> None:
        """
        This loss is an adaptation of CoSENTLoss for distillation. It uses a teacher model to generate similarity scores
        between sentence pairs, and then trains a student model to mimic these similarity scores using the CoSENTLoss.

        Args:
            student: SentenceTransformer model (student)
            teacher: SentenceTransformer model (teacher)
            scale: Output of similarity function is multiplied by scale value
            similarity: Similarity function between sentence embeddings. By default, dot_product.

        Requirements:
            1. Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
                from datasets import Dataset

                student = SentenceTransformer("microsoft/mpnet-base")
                teacher = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "sentence1": ["It's nice weather outside today.", "He drove to work."],
                    "sentence2": ["It's so sunny.", "She walked to the store."],
                })
                loss = SimilarityDistillationLoss(student, teacher)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()

        self.teacher = teacher
        self.similarity_fn = SimilarityFunction.to_similarity_pairwise_fn(
            SimilarityFunction.COSINE
        )
        self.loss_fn = losses.CoSENTLoss(
            student,
            scale=scale,
            similarity_fct=SimilarityFunction.to_similarity_pairwise_fn(
                similarity
            )
        )

    def similarity(self, model: SentenceTransformer, sentence_features: Iterable[Dict[str, Tensor]]) -> Tensor:
        reps = [
            model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        return self.similarity_fn(reps[0], reps[1])

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tensor:
        with torch.no_grad():
            target = self.similarity(self.teacher.eval(), sentence_features)

        return self.loss_fn(sentence_features, target.detach())

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "similarity_fct": self.similarity_fn.__name__,
        }
