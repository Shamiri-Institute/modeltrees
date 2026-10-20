"""The GLM node model of ``glmtree``.

Port of ``glmfit`` from partykit 1.3-0 (reference/partykit/R/glmtree.R:57-133)
on top of the glm.fit port in :mod:`modeltrees.models.glm_fit`.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from modeltrees._reference import reference_active
from modeltrees.models._common import check_vcov, numeric_response, regressor_matrix
from modeltrees.models.families import Family, get_family
from modeltrees.models.glm_fit import glm_fit, glm_objective
from modeltrees.node_model import FitResult

FloatArray = NDArray[np.float64]

# glmtree.R:84: families whose df counts the dispersion parameter.
_DISPERSION_IN_DF = ("gaussian", "Gamma", "inverse.gaussian")
# glmtree.R:99: families scored with dispersion 1.
_UNIT_DISPERSION = ("poisson", "binomial", "Negative Binomial")


class GlmModel:
    """A GLM in each node, as glmtree fits it.

    ``objfun`` is ``aic / 2 - df`` with ``df`` the rank, plus one for
    gaussian (glmtree.R:83-86): the negative log-likelihood. The scores are
    ``working residuals * working weights * x`` over the non-aliased columns,
    divided by the dispersion: 1 for binomial and poisson, otherwise
    ``sum(wres^2 / weights) / sum(working weights)`` with case weights and
    ``sum(wres^2) / sum(working weights)`` without (glmtree.R:96-110).

    Rows with zero weight are left out of the fit and get zero scores. For
    binomial and poisson this is exactly R's result. For gaussian R's AIC
    counts those rows and adds ``-sum(log(weights))``, so partykit's
    objective is Inf for any node with a zero weight; here it is the
    objective of the data with those rows deleted (docs/model-fitting.md).
    """

    def __init__(
        self,
        family: str | Family = "gaussian",
        *,
        epsilon: float = 1e-8,
        maxit: int = 25,
        coef_names: Sequence[str] | None = None,
    ) -> None:
        self.family = get_family(family) if isinstance(family, str) else family
        self.epsilon = epsilon
        self.maxit = maxit
        self.coef_names = None if coef_names is None else tuple(coef_names)

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
        check_vcov(vcov)
        yv = numeric_response(y)
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
        if not estfun and not reference_active():
            # Split search needs only the objective (and the coefficients for
            # restart = FALSE): no warnings, null deviance or residuals.
            o = glm_objective(
                xm_fit,
                y_fit,
                self.family,
                weights=w_fit,
                start=start,
                offset=off_fit,
                epsilon=self.epsilon,
                maxit=self.maxit,
            )
            df = o.rank + (1 if self.family.family in _DISPERSION_IN_DF else 0)
            return FitResult(
                coefficients=o.coefficients,
                coef_names=names,
                objfun=float(o.aic / 2 - df),
                estfun=None,
                converged=o.converged,
            )
        z = glm_fit(
            xm_fit,
            y_fit,
            self.family,
            weights=w_fit,
            start=start,
            offset=off_fit,
            epsilon=self.epsilon,
            maxit=self.maxit,
        )
        df = z.rank + (1 if z.family.family in _DISPERSION_IN_DF else 0)
        objfun = z.aic / 2 - df
        scores = None
        if estfun:
            wres = z.residuals * z.weights
            if z.family.family in _UNIT_DISPERSION:
                dispersion = 1.0
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    num = wres**2 / w_fit if caseweights else wres**2
                dispersion = float(np.nansum(num) / np.nansum(z.weights))
            cols = ~np.isnan(z.coefficients)
            scores = np.zeros((n, int(cols.sum())))
            scores[keep] = wres[:, None] * xm_fit[:, cols] / dispersion
        return FitResult(
            coefficients=z.coefficients,
            coef_names=names,
            objfun=float(objfun),
            estfun=scores,
            converged=z.converged,
        )

    def n_response_columns(self, y: NDArray[np.generic]) -> int:
        return 1 if np.ndim(y) <= 1 else int(np.shape(y)[1])


def binomial_response(
    y: ArrayLike | Any, levels: Sequence[Any] | None = None
) -> FloatArray:
    """Encode a categorical response for a binomial GLM, as R does.

    The first level is failure (0) and every other level is success (1),
    R's rule in ``binomial()$initialize`` (``y != levels(y)[1]``). The level
    order is ``levels`` if given, else the categories of a pandas-style
    categorical (``.cat.categories`` or ``.categories``), else the sorted
    unique values. R sorts strings by the locale's collation when it makes
    factor levels, which can differ from Python's sort (for example for
    mixed-case strings), so callers should pass the level order explicitly
    as ``levels`` when it matters.
    """
    if levels is None:
        holder = getattr(y, "cat", y)
        categories = getattr(holder, "categories", None)
        if categories is not None:
            levels = list(categories)
    values = np.asarray(y)
    if values.dtype.kind == "f" and np.any(np.isnan(values)):
        raise ValueError("the response has missing values")
    if values.dtype.kind == "O" and any(v is None for v in values.ravel()):
        raise ValueError("the response has missing values")
    if levels is None:
        levels = list(np.unique(values))
    if len(levels) == 0:
        raise ValueError("the response has no levels")
    known = set(levels)
    unknown = {v for v in values.ravel().tolist() if v not in known}
    if unknown:
        raise ValueError(
            f"response values not among the levels: {sorted(map(str, unknown))}"
        )
    return (values != levels[0]).astype(np.float64).reshape(-1)
