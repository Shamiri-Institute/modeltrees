"""The linear regression node model of ``lmtree``.

Port of ``lmfit`` from partykit 1.3-0 (reference/partykit/R/lmtree.R:59-104).
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from modeltrees.models._common import check_vcov, numeric_response, regressor_matrix
from modeltrees.models.lm_fit import lm_fit, lm_wfit
from modeltrees.node_model import FitResult
from modeltrees.pruning import PruneRule, lm_prune_rule

FloatArray = NDArray[np.float64]


class LmModel:
    """Least squares in each node, as lmtree fits it.

    ``objfun`` is the weighted residual sum of squares (lmtree.R:78) and the
    scores are ``residuals * weights * x`` over the non-aliased columns
    (lmtree.R:85). ``start``, ``cluster`` and ``caseweights`` do not change
    the fit, as in R.
    """

    def __init__(self, coef_names: Sequence[str] | None = None) -> None:
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
        xm, names = regressor_matrix(yv.shape[0], x, self.coef_names)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        # lmtree.R:68-73: lm.fit when every weight is exactly 1, else lm.wfit.
        if np.all(w == 1):
            z = lm_fit(xm, yv, offset)
            w = np.ones_like(yv)
        else:
            z = lm_wfit(xm, yv, w, offset)
        scores = None
        if estfun:
            keep = ~np.isnan(z.coefficients)
            scores = (z.residuals * w)[:, None] * xm[:, keep]
        return FitResult(
            coefficients=z.coefficients,
            coef_names=names,
            objfun=float(np.sum(w * z.residuals**2)),
            estfun=scores,
        )

    def n_response_columns(self, y: NDArray[np.generic]) -> int:
        return 1 if np.ndim(y) <= 1 else int(np.shape(y)[1])

    def prune_rule(self, name: str) -> PruneRule | None:
        """lmtree's pruning rules, for an RSS objective (lmtree.R:12-29)."""
        return lm_prune_rule(name)
