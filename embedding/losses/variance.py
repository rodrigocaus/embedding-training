import torch
from torch import nn, Tensor
from typing import Any, Dict, Iterable
from sentence_transformers import (
    SentenceTransformer,
    util
)


_DOCSTRING_HEADER = """
        This loss an adaptation from Lee et al., to improve mini-batch
        contrastive learning by introducing an auxiliary loss term which
        explicitly reduces the variance of negative-pair similarities

        Args:
            model: SentenceTransformer model
            eps: margin value of the cosine similarity
"""


class NegativesVariancePenaltyLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.model = model
        self.eps = eps

    @property
    def citation(self) -> str:
        return """
@misc{lee-2025,
    title={On the Similarities of Embeddings in Contrastive Learning}, 
    author={Chungpa Lee and Sehee Lim and Kibok Lee and Jy-yong Sohn},
    year={2025},
    url={https://arxiv.org/abs/2506.09781}, 
}
"""


class InBatchNegativesVariancePenaltyLoss(NegativesVariancePenaltyLoss):
    f"""{_DOCSTRING_HEADER}

        Requirements:
            1. (anchor, positive) pairs

        Inputs:
            +--------------------------------------------------------+--------+
            | Texts                                                  | Labels |
            +========================================================+========+
            | (anchor, positive) pairs                               | none   |
            +--------------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) tuple  | none   |
            +--------------------------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
        "sentence1": ["It's nice weather outside today.", "He drove to work."],
                    "sentence2": ["It's so sunny.", "She walked to the store."],
                })
                loss = InBatchNegativesVariancePenaltyLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
    """

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tensor:
        reps = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        anchor = reps[0]
        candidates = torch.cat(reps[1:])
        scores = util.cos_sim(anchor, candidates)
        n, m = scores.shape
        # mask out positive pairs
        mask = 1.0 - torch.eye(n, m, device=scores.device)
        variances = torch.square((self.eps + scores)*mask)
        return (variances.sum()/n)/(n-1.0)


class LabeledNegativesVariancePenaltyLoss(NegativesVariancePenaltyLoss):
    f"""{_DOCSTRING_HEADER}

        Requirements:
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

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

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
        "sentence1": ["It's nice weather outside today.", "He drove to work."],
                    "sentence2": ["It's so sunny.", "She walked to the store."],
                    "score": [1.0, 0.3],
                })
                loss = LabeledNegativesVariancePenaltyLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
    """

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tensor:
        reps = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        scores = util.pairwise_cos_sim(reps[0], reps[1])
        n = len(scores)
        mask = (labels < 0).to(scores.device, torch.float32)
        variances = torch.square((self.eps + scores)*mask)
        return variances.sum()/(n-1.0)
