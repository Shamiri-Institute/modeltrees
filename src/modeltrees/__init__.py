"""Model-based recursive partitioning (MOB) in Python.

A port of R's partykit::mob, lmtree and glmtree, with negative binomial,
safe logit and multinomial node models (docs/api-design.md, section 8.1).
"""

from importlib.metadata import PackageNotFoundError, version

from modeltrees._api import (
    ModelTree,
    NodeInfo,
    RecordedWarning,
    SplitRule,
    TestResult,
    TreeSummary,
    glmtree,
    lmtree,
    mob,
    multinomtree,
    negbintree,
    safelogittree,
)
from modeltrees._loglik import LogLik
from modeltrees._reference import reference_path
from modeltrees._vcov import LeafVcov
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
from modeltrees.tree import MobControl

try:
    __version__ = version("modeltrees")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0+unknown"

# The public API (docs/api-design.md, section 8.1).
__all__ = [
    "DataError",
    "FitError",
    "FormulaError",
    "GlmFitWarning",
    "LeafVcov",
    "LogLik",
    "MissingDataWarning",
    "MobControl",
    "ModelTree",
    "ModelTreesError",
    "ModelTreesWarning",
    "NodeFitFailedWarning",
    "NodeInfo",
    "RecordedWarning",
    "SplitRule",
    "TestResult",
    "TooFewClustersWarning",
    "TreeSummary",
    "UnseenLevelError",
    "__version__",
    "glmtree",
    "lmtree",
    "mob",
    "multinomtree",
    "negbintree",
    "reference_path",
    "safelogittree",
]
