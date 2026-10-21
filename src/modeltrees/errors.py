"""Public exception and warning classes (docs/api-design.md, section 7).

Exceptions the core raises on purpose derive from a public class here, so
they reach the user with their own type, message and traceback. Every
exception that was a ``ValueError`` stays one. The internal control-flow
exceptions of the instability tests (``InstabilityTestError``,
``TooFewClustersError``) are not public: the core always catches them.
"""


class ModelTreesError(Exception):
    """Base class of the errors modeltrees raises on purpose."""


class FormulaError(ModelTreesError, ValueError):
    """A formula that cannot be parsed or uses an unsupported construct."""

    def __init__(self, message: str, position: int | None = None) -> None:
        super().__init__(message)
        self.position = position


class DataError(ModelTreesError, ValueError):
    """Input data that cannot be used: missing columns, wrong dtypes, Inf,
    negative weights, missing values with ``na_action="fail"``, lazy frames."""


class UnseenLevelError(DataError):
    """A level at prediction time that was not seen in training."""


class FitError(ModelTreesError, ValueError):
    """A node fit that cannot be completed and is not turned into a failed
    fit (lmtree, glmtree)."""


class ModelTreesWarning(UserWarning):
    """Base class of the warnings modeltrees emits."""


class NodeFitFailedWarning(ModelTreesWarning):
    """A node's model fit failed, so the node is terminal."""


class TooFewClustersWarning(ModelTreesWarning):
    """A node has too few clusters for its parameters.

    In the instability tests it is emitted before ``TooFewClustersError``,
    so the stop is not silent: with fewer clusters than parameters, or
    exactly as many when the cluster sums of the scaled scores sum to zero
    (no case weights, or equal ones). In ``vcov`` it is emitted for at most
    as many clusters as parameters, where the meat's cluster sums always
    sum to zero.
    """


class MissingDataWarning(ModelTreesWarning):
    """Rows were dropped for missing values, with the count per column."""


class GlmFitWarning(ModelTreesWarning):
    """A warning R's glm.fit or a family would give, with R's message."""
