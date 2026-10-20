from modeltrees._reference import reference_path
from modeltrees.errors import (
    DataError,
    FitError,
    FormulaError,
    GlmFitWarning,
    MissingDataWarning,
    ModelTreesError,
    ModelTreesWarning,
    NodeFitFailedWarning,
    TooFewClustersWarning,
    UnseenLevelError,
)

__all__ = [
    "DataError",
    "FitError",
    "FormulaError",
    "GlmFitWarning",
    "MissingDataWarning",
    "ModelTreesError",
    "ModelTreesWarning",
    "NodeFitFailedWarning",
    "TooFewClustersWarning",
    "UnseenLevelError",
    "reference_path",
]


def hello() -> str:
    return "Hello from modeltrees!"
