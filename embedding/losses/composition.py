from torch import nn, Tensor
from typing import Sequence, Iterable, Dict


class CompositionLoss(nn.Module):
    """
        This class allows to combine multiple loss functions into a single loss function.
        The final loss is a weighted sum of the individual losses.

        Args:
            losses: A sequence of loss functions to be combined.
            weights: A sequence of weights for each loss function. If not provided, all losses will have a weight of 1.0.
                The length of weights should be equal to the length of losses.

        Requirements:
            The requirements for the input and labels depend on the individual loss functions being combined.
            Ensure that the input format is compatible with all the loss functions provided.

        Inputs:
            The input format depends on the individual loss functions being combined.
            It typically involves sentence features and labels.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses
                from embedding.losses import kl, composition

                student = SentenceTransformer("microsoft/mpnet-base")
                teacher = SentenceTransformer("microsoft/mpnet-base")
                loss1 = losses.MultipleNegativesSymmetricRankingLoss(model=student)
                loss2 = kl.KLSimilarityLoss(student=student, teacher=teacher)
                loss = composition.CompositionLoss(losses=[loss1, loss2], weights=[0.5, 0.5])

                # The 'loss' object can now be used in a SentenceTransformerTrainer
                # with a dataset that provides the required inputs for loss1 and loss2.
    """
    def __init__(
        self,
        losses: Sequence[nn.Module],
        weights: Sequence[float] = []
    ) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses)
        if not weights:
            weights = [1.0] * len(losses)
        if len(weights) != len(losses):
            raise ValueError(
                "'weights' should have the same size of 'losses'"
            )

        self.weights = weights

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tensor:
        loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            loss += weight * loss_fn(sentence_features, labels)

        return loss
