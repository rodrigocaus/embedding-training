from .cosent import AugmentedCoSENTLoss
from .cross_entropy import (
    BinaryCrossEntropyLoss, AugmentedBinaryCrossEntropyLoss,
    CosineCrossEntropyLoss,
)

__all__ = [
    "AugmentedCoSENTLoss",
    "BinaryCrossEntropyLoss",
    "AugmentedBinaryCrossEntropyLoss",
    "CosineCrossEntropyLoss",
]
