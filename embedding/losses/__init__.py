from .distillation import *
from .similarity import *
from .composition import CompositionLoss
from .variance import InBatchNegativesVariancePenaltyLoss, LabeledNegativesVariancePenaltyLoss


__all__ = [
    "KLSimilarityLoss",
    "SimilarityDistillationLoss",
    "CompositionLoss",
    "AugmentedCoSENTLoss",
]
