"""The interface between MOB and the model fitted in each node.

Mirrors the type 1 fit interface of partykit::mob (modelparty.R, ``mob()``),
as specified in docs/model-interface.md, section 4. Core MOB code depends
only on :class:`NodeModel`; model implementations live in
:mod:`modeltrees.models`.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FitResult:
    coefficients: FloatArray  # shape (k,), NaN for aliased columns, as in R
    coef_names: tuple[str, ...]
    objfun: float  # additive objective, smaller is better
    estfun: FloatArray | None  # shape (n, k_nonaliased), weights included
    vcov: FloatArray | None = None  # only needed for vcov "info" or "sandwich"
    converged: bool = True


class NodeModel(Protocol):
    """A model that MOB fits in every node.

    Mirrors the type 1 fit interface of partykit::mob (modelparty.R, mob()).
    """

    def fit(
        self,
        y: NDArray[np.generic],
        x: FloatArray | None,
        *,
        start: FloatArray | None,
        weights: FloatArray,
        offset: FloatArray | None,
        cluster: NDArray[np.generic] | None,
        caseweights: bool,
        estfun: bool,
        vcov: bool,
    ) -> FitResult: ...

    def n_response_columns(self, y: NDArray[np.generic]) -> int:
        """NCOL(y), used for the default minsize (mob_grow, modelparty.R:533)."""
        ...


class PredictingModel(NodeModel, Protocol):
    """Optional: prediction from a fitted node (predict.modelparty)."""

    def predict(
        self,
        result: FitResult,
        x: FloatArray | None,
        offset: FloatArray | None,
        kind: str,
    ) -> FloatArray: ...


class LogLikModel(NodeModel, Protocol):
    """Optional: log-likelihood and df (logLik.modelparty)."""

    def loglik(self, result: FitResult) -> tuple[float, int]: ...
