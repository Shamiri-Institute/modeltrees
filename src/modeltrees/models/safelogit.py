"""Safe binary logistic regression, the node model ``safelogitfit``.

Port of r/extensions/safelogit.R, specified in docs/extensions-spec.md,
section 4 (provisional: the thesis names safelogitModel but gives no code).
The fit is glm.fit with ``binomial("logit")``; "safe" adds three things:

1. glm.fit's warnings are recorded in ``extras["warnings"]``, not emitted;
2. a glm.fit error is a failed fit instead of an exception;
3. separation is classified on the positive-weight rows from the fitted
   values. A row is at the boundary if ``|y - p| < separation_tol``. If every
   row is (``complete``), the ML estimator does not exist and the fit has no
   scores, so the node is terminal; ``quasi`` (some rows) and ``none`` keep
   the fit and its scores.

Without complete separation the result is that of
``GlmModel("binomial")``.
"""

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from modeltrees._reference import reference_active
from modeltrees.models._common import (
    numeric_response,
    regressor_matrix,
    weighted_crossprod,
)
from modeltrees.models.families import Binomial
from modeltrees.models.glm_fit import GlmFitError, glm_fit, glm_objective
from modeltrees.node_model import (
    FitResult,
    FitStatus,
    Report,
    failed_fit,
    linear_predictor,
    rank,
)

FloatArray = NDArray[np.float64]

Separation = Literal["none", "quasi", "complete"]


def classify_separation(
    y: FloatArray, fitted: FloatArray, tol: float
) -> tuple[Separation, int]:
    """The separation class and the number of rows at the boundary.

    ``y`` and ``fitted`` are the positive-weight rows only.
    """
    at_bound = np.abs(y - fitted) < tol
    n_boundary = int(np.sum(at_bound))
    if n_boundary == y.shape[0]:
        return "complete", n_boundary
    return ("quasi" if n_boundary > 0 else "none"), n_boundary


def _unique(messages: Sequence[str]) -> tuple[str, ...]:
    """R's ``unique()``: first occurrences, in order."""
    return tuple(dict.fromkeys(messages))


class SafeLogitModel:
    """Binary logistic regression that tolerates separation.

    The response is 0/1; convert a categorical response with
    :func:`modeltrees.models.glm.binomial_response` (first level 0, the rest
    1). Proportions raise ``ValueError``, as safelogitfit stops.

    ``objfun`` is the negative log-likelihood ``aic / 2 - rank`` and the
    scores are ``estfun.glm``'s, as in :class:`GlmModel` with
    ``"binomial"``. Rows with zero weight are left out of the fit and get
    zero scores. With ``vcov=True`` the Hessian is binomial GlmModel's,
    ``X' diag(working weights) X``; under quasi-separation it is near
    singular. ``extras`` holds ``status`` ("ok" or "error"),
    ``separation``, ``n_boundary``, ``converged``, ``boundary`` and
    ``warnings``, as the R fit function returns them, plus ``error`` for a
    failed fit.

    A completely separated fit has status ``NO_ESTFUN`` and keeps glm.fit's
    coefficients and objective; in split search its objective counts as
    usual, so a separated subgroup can be split off.
    """

    def __init__(
        self,
        *,
        epsilon: float = 1e-8,
        maxit: int = 25,
        separation_tol: float = 1e-6,
        coef_names: Sequence[str] | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.maxit = maxit
        self.separation_tol = separation_tol
        self.coef_names = None if coef_names is None else tuple(coef_names)
        self.family = Binomial()

    def fit(
        self,
        y: NDArray[np.generic],
        x: FloatArray | None,
        *,
        start: FloatArray | None = None,
        weights: FloatArray,
        offset: FloatArray | None = None,
        cluster: NDArray[np.generic] | None = None,
        caseweights: bool = True,
        estfun: bool = False,
        vcov: bool = False,
    ) -> FitResult:
        yv = numeric_response(y)
        if not np.all((yv == 0) | (yv == 1)):
            raise ValueError("safelogitfit needs a binary response")
        n = yv.shape[0]
        xm, names = regressor_matrix(n, x, self.coef_names)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        off = None if offset is None else np.asarray(offset, dtype=np.float64)
        keep = w > 0
        if not np.all(keep):
            xm_fit, y_fit, w_fit = xm[keep], yv[keep], w[keep]
            off_fit = None if off is None else off[keep]
        else:
            xm_fit, y_fit, w_fit, off_fit = xm, yv, w, off
        args: dict[str, Any] = {
            "weights": w_fit,
            "start": start,
            "offset": off_fit,
            "epsilon": self.epsilon,
            "maxit": self.maxit,
        }
        objective_only = not estfun and not vcov and not reference_active()
        try:
            if objective_only:
                # Split search: the same arithmetic as glm_fit, without the
                # null deviance; the separation is still classified.
                o = glm_objective(
                    xm_fit, y_fit, self.family, record_warnings=True, **args
                )
                coef, aic, rank, mu = o.coefficients, o.aic, o.rank, o.fitted_values
                converged, boundary, messages = o.converged, o.boundary, o.warnings
            else:
                z = glm_fit(xm_fit, y_fit, self.family, emit_warnings=False, **args)
                coef, aic, rank, mu = z.coefficients, z.aic, z.rank, z.fitted_values
                converged, boundary, messages = z.converged, z.boundary, z.warnings
        except GlmFitError as exc:
            return failed_fit(
                names,
                str(exc),
                status="error",
                error=str(exc),
                separation=None,
                n_boundary=None,
                converged=False,
                boundary=None,
                warnings=_unique(exc.warnings),
            )
        separation, n_boundary = classify_separation(y_fit, mu, self.separation_tol)
        extras = {
            "status": "ok",
            "separation": separation,
            "n_boundary": n_boundary,
            "converged": converged,
            "boundary": boundary,
            "warnings": _unique(messages),
        }
        objfun = float(aic / 2 - rank)
        if separation == "complete":
            return FitResult(
                coefficients=coef,
                coef_names=names,
                objfun=objfun,
                estfun=None,
                converged=converged,
                status=FitStatus.NO_ESTFUN,
                reason="complete separation: the ML estimator does not exist",
                extras=extras,
            )
        cols = ~np.isnan(coef)
        scores = None
        if estfun:
            wres = z.residuals * z.weights
            scores = np.zeros((n, int(cols.sum())))
            scores[keep] = wres[:, None] * xm_fit[:, cols]
        hessian = None
        if vcov:
            hessian = weighted_crossprod(xm_fit[:, cols], z.weights)
        return FitResult(
            coefficients=coef,
            coef_names=names,
            objfun=objfun,
            estfun=scores,
            converged=converged,
            extras=extras,
            hessian=hessian,
        )

    def n_response_columns(self, y: NDArray[np.generic]) -> int:
        return 1

    def loglik(self, result: FitResult) -> tuple[float, int]:
        """``logLik.glm`` of the binomial fit: minus the objective, df the
        rank. A completely separated fit keeps its coefficients, so its df
        is the rank too."""
        if result.failed:
            return math.nan, len(result.coefficients)
        return -result.objfun, rank(result)

    def report(self, result: FitResult) -> Report:
        e = result.extras
        messages = e.get("warnings")
        return {
            "separation": e.get("separation"),
            "n_boundary": e.get("n_boundary"),
            "converged": result.converged,
            "n_warnings": None if messages is None else len(messages),
        }

    def predict(
        self,
        result: FitResult,
        x: FloatArray | None,
        offset: FloatArray | None,
        kind: str,
    ) -> FloatArray:
        """The linear predictor for "link", the success probability with
        R's logit bounds for "response"."""
        if kind not in ("link", "response"):
            raise ValueError(
                f"safelogittree predicts 'link' or 'response', not {kind!r}"
            )
        eta = linear_predictor(result, x, offset)
        return eta if kind == "link" else self.family.linkinv(eta)
