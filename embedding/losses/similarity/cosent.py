import torch
from torch import Tensor, nn
from typing import Iterable
from itertools import islice

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class AugmentedCoSENTLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, min_score: float = -1.0, **_) -> None:
        """
        This class implements an augmented version of CoSENT (Cosine Sentence) loss.
        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair. Additinally, each of the InputExamples are assumed unrelated
        within a mini-batch.

        It computes the following loss function:

        ``loss = logsum(1+exp(s(i,j)-s(k,l))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        pairs of input pairs in the batch that match this condition.

        Args:
            model: SentenceTransformerModel
            scale: Output of similarity function is multiplied by scale
                value. Represents the inverse temperature.
            min_score: Minimal value in range of the similarity function. Default is -1.

        References:
            - For further details, see: https://kexue.fm/archives/8847

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

        Relations:
            - :class:`CoSENTLoss` uses labeled samples only

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
                loss = AugmentedCoSENTLoss(model)

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
        self.min_score = min_score

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in islice(sentence_features, 2)
        ]

        scores = util.cos_sim(embeddings[0], embeddings[1])
        scores = scores.view(-1) * self.scale
        scores = scores[:, None] - scores[None, :]

        m = len(labels)
        augmented_labels = torch.full((m, m), self.min_score).to(labels.device)
        # label matrix indicating which pairs are relevant
        augmented_labels = augmented_labels.diagonal_scatter(labels, 0)
        augmented_labels = augmented_labels.view(-1)
        augmented_labels = (
            augmented_labels[:, None] < augmented_labels[None, :]
        )

        # mask out irrelevant pairs so they are negligible after exp()
        scores = scores - (1 - augmented_labels.float()) * 1e12
        # append a zero as e^0 = 1
        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        return torch.logsumexp(scores, dim=0)

    @property
    def citation(self) -> str:
        return """
@online{kexuefm-8847,
    title={CoSENT: A more efficient sentence vector scheme than Sentence-BERT},
    author={Su Jianlin},
    year={2022},
    month={Jan},
    url={https://kexue.fm/archives/8847},
}
"""
