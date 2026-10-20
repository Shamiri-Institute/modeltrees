"""Node models for MOB and the fitting code they use.

The public names are for :func:`modeltrees.mob` and custom models
(docs/api-design.md, section 8.2). The ``NodeModel`` protocol and its
optional protocols are provisional in v0.1.
"""

from modeltrees.models.glm import GlmModel
from modeltrees.models.lm import LmModel
from modeltrees.models.multinom import MultinomModel
from modeltrees.models.negbin import NegBinModel
from modeltrees.models.safelogit import SafeLogitModel
from modeltrees.node_model import (
    BatchSplitModel,
    FitResult,
    FitStatus,
    LogLikModel,
    NodeModel,
    PredictingModel,
    ReportingModel,
    failed_fit,
)

__all__ = [
    "BatchSplitModel",
    "FitResult",
    "FitStatus",
    "GlmModel",
    "LmModel",
    "LogLikModel",
    "MultinomModel",
    "NegBinModel",
    "NodeModel",
    "PredictingModel",
    "ReportingModel",
    "SafeLogitModel",
    "failed_fit",
]
