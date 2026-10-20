"""The interface between MOB and the model fitted in each node.

Mirrors the type 1 fit interface of partykit::mob (modelparty.R, ``mob()``),
as specified in docs/model-interface.md, section 4. Core MOB code depends
only on :class:`NodeModel`; model implementations live in
:mod:`modeltrees.models`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class FitStatus(Enum):
    """How a fit ended (docs/model-interface.md, section 4.2)."""

    OK = "ok"
    # No usable estimate: in split search the candidate is inadmissible, at a
    # node the node becomes terminal with a warning.
    FAILED = "failed"
    # An estimate, but no scores (partykit's estfun = NULL): the node is
    # terminal without tests. In split search the fit counts as usual.
    NO_ESTFUN = "no_estfun"


@dataclass(frozen=True)
class FitResult:
    coefficients: FloatArray  # shape (k,), NaN for aliased columns, as in R
    coef_names: tuple[str, ...]
    objfun: float  # additive objective, smaller is better
    estfun: FloatArray | None  # shape (n, k_nonaliased), weights included
    vcov: FloatArray | None = None  # only needed for vcov "info" or "sandwich"
    converged: bool = True
    status: FitStatus = FitStatus.OK
    reason: str | None = None  # why the fit failed or has no scores
    # Model-specific diagnostics, for example theta, warnings or boundary
    # counts. The core stores them and does not read them.
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status is FitStatus.FAILED


def failed_fit(coef_names: tuple[str, ...], reason: str, **extras: Any) -> FitResult:
    """A failed fit: NaN coefficients, objective Inf and no scores."""
    return FitResult(
        coefficients=np.full(len(coef_names), np.nan),
        coef_names=coef_names,
        objfun=float("inf"),
        estfun=None,
        converged=False,
        status=FitStatus.FAILED,
        reason=reason,
        extras=extras,
    )


def extra_df(model: object) -> int:
    """Parameters a model estimates per node beyond its coefficients.

    They count in the pruning df, as theta does for the negative binomial
    model. Read from the optional attribute ``extra_df`` (default 0).
    """
    return int(getattr(model, "extra_df", 0))


def uses_batch_split(model: object) -> bool:
    """Whether split search may use the model's ``split_objectives``.

    A model opts out with ``batch_split = False``, for example when a
    candidate's fit can fail, which approximate objectives cannot detect.
    """
    return callable(getattr(model, "split_objectives", None)) and bool(
        getattr(model, "batch_split", True)
    )


class NodeModel(Protocol):
    """A model that MOB fits in every node.

    Mirrors the type 1 fit interface of partykit::mob (modelparty.R, mob()).
    A fit that cannot produce an estimate returns ``status = FAILED``
    (see :func:`failed_fit`) instead of raising; one without scores returns
    ``status = NO_ESTFUN``. Optional attributes: ``extra_df`` (int, default
    0) and ``batch_split`` (bool, default True).
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


# A model report: fixed keys per model (docs/api-design.md, section 3.5).
Report = Mapping[str, float | int | str | bool | None]


class PredictingModel(NodeModel, Protocol):
    """Optional: prediction from a fitted node (predict.modelparty).

    ``kind`` is "link" (the linear predictor ``x beta + offset``),
    "response" (its inverse link), "prob" or "class" (multinomial: the
    class probabilities, and the 0-based code of the most probable class
    as floats). A kind the model does not define raises ``ValueError``.
    Rows of a failed fit are NaN (docs/api-design.md, section 4.2).
    """

    def predict(
        self,
        result: FitResult,
        x: FloatArray | None,
        offset: FloatArray | None,
        kind: str,
    ) -> FloatArray: ...


class LogLikModel(NodeModel, Protocol):
    """Optional: log-likelihood and df (logLik.modelparty).

    The value is NaN for a failed fit (docs/api-design.md, section 3.6).
    """

    def loglik(self, result: FitResult) -> tuple[float, int]: ...


class ReportingModel(NodeModel, Protocol):
    """Optional: the model's extras for the coefficient table, with the
    same keys for every fit (docs/api-design.md, section 3.5)."""

    def report(self, result: FitResult) -> Report: ...


def rank(result: FitResult) -> int:
    """The number of estimated (non-NaN) coefficients."""
    return int(np.sum(~np.isnan(result.coefficients)))


def linear_predictor(
    result: FitResult, x: FloatArray | None, offset: FloatArray | None
) -> FloatArray:
    """``x beta + offset``, aliased (NaN) coefficients counting as 0.

    predict.lm and predict.glm drop aliased columns the same way. ``x =
    None`` is an intercept-only model, which needs ``offset`` for the number
    of rows. A failed fit gives NaN.
    """
    coef = np.asarray(result.coefficients, dtype=np.float64)
    if x is None:
        if offset is None:
            raise ValueError("x = None needs an offset to give the number of rows")
        xm = np.ones((np.shape(offset)[0], 1))
    else:
        xm = np.asarray(x, dtype=np.float64)
        if xm.ndim == 1:
            xm = xm[:, None]
    if xm.shape[1] != coef.shape[0]:
        raise ValueError(
            f"x has {xm.shape[1]} columns for {coef.shape[0]} coefficients"
        )
    if result.failed:
        return np.full(xm.shape[0], np.nan)
    keep = ~np.isnan(coef)
    eta: FloatArray = xm[:, keep] @ coef[keep]
    if offset is not None:
        eta = eta + np.asarray(offset, dtype=np.float64)
    return eta
