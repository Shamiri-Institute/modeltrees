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
from modeltrees.split_search import SplitObjectives

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

DBL_EPSILON = float(np.finfo(np.float64).eps)
# A side whose scaled cross-product matrix has a larger condition number is
# fitted exactly instead of scored (cond(X) above 1e4: far from R's 1e-7
# rank tolerance, but the normal equations lose that much accuracy).
MAX_SCORE_CONDITION = 1e8


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

    def split_objectives(
        self,
        y: NDArray[np.generic],
        x: FloatArray | None,
        *,
        weights: FloatArray,
        offset: FloatArray | None,
        cluster: NDArray[np.generic] | None,
        caseweights: bool,
        order: IntArray,
        cuts: IntArray,
    ) -> SplitObjectives:
        """Residual sums of squares of every numeric candidate at once.

        Candidate ``i`` sends the rows ``order[:cuts[i]]`` left. Each side's
        RSS is ``r'Wr - b' A^-1 b`` from prefix (left) and suffix (right)
        sums of the weighted cross-products ``A = X'WX``, ``b = X'Wr`` and
        ``r'Wr``. ``r`` is the residual of the node's own least-squares fit,
        not ``y``: ``y - r`` lies in the column space of ``X`` on every
        side, so each side's RSS is unchanged, and the sums stay on the
        scale of the RSS instead of ``y'Wy``, which avoids cancellation.
        ``A`` and ``b`` are scaled by ``A``'s diagonal and solved through a
        symmetric eigendecomposition, which also gives the condition number.

        A side with a zero diagonal entry or a condition number above
        ``MAX_SCORE_CONDITION`` (rank-deficient, collinear or tiny) gets
        NaN, and split search fits that candidate exactly. ``error`` is a
        conservative bound on the rounding error of the summation and the
        solve, and of the exact fit it is compared with, per side
        ``64 eps (n + p) (cond r'Wr + sqrt(r'Wr y'Wy))``. The second term
        covers residuals that are small next to ``y``, where both paths
        lose digits.
        """
        yv = numeric_response(y)
        if offset is not None:
            yv = yv - np.asarray(offset, dtype=np.float64)
        xm, _ = regressor_matrix(yv.shape[0], x, self.coef_names)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        n, p = xm.shape
        node = lm_wfit(xm, yv, w) if not np.all(w == 1) else lm_fit(xm, yv)
        coef0 = np.where(np.isnan(node.coefficients), 0.0, node.coefficients)
        r = yv - xm @ coef0

        xo, ro, wo = xm[order], r[order], w[order]
        wx = wo[:, None] * xo
        iu, ju = np.triu_indices(p)
        terms = np.empty((n, len(iu) + p + 3))
        terms[:, : len(iu)] = wx[:, iu] * xo[:, ju]
        terms[:, len(iu) : len(iu) + p] = wx * ro[:, None]
        terms[:, -3] = wo * ro**2
        terms[:, -2] = wo * yv[order] ** 2
        terms[:, -1] = wo > 0
        prefix = np.cumsum(terms, axis=0)
        suffix = np.cumsum(terms[::-1], axis=0)[::-1]
        zero = np.zeros(terms.shape[1])
        left = np.vstack([zero, prefix])[cuts]
        right = np.vstack([suffix, zero])[cuts]
        rss_l, err_l = _side_rss(left, iu, ju, p)
        rss_r, err_r = _side_rss(right, iu, ju, p)
        return SplitObjectives(rss_l, rss_r, err_l + err_r)

    def n_response_columns(self, y: NDArray[np.generic]) -> int:
        return 1 if np.ndim(y) <= 1 else int(np.shape(y)[1])

    def prune_rule(self, name: str) -> PruneRule | None:
        """lmtree's pruning rules, for an RSS objective (lmtree.R:12-29)."""
        return lm_prune_rule(name)


def _side_rss(
    sums: FloatArray, iu: IntArray, ju: IntArray, p: int
) -> tuple[FloatArray, FloatArray]:
    """RSS and its error bound per side from the summed cross-products.

    ``sums`` has the upper triangle of ``A``, then ``b``, ``r'Wr``,
    ``y'Wy`` and the count of positive-weight rows, one row per side. NaN marks sides to fit
    exactly.
    """
    m = sums.shape[0]
    a = np.empty((m, p, p))
    a[:, iu, ju] = sums[:, : len(iu)]
    a[:, ju, iu] = sums[:, : len(iu)]
    b = sums[:, len(iu) : len(iu) + p]
    rr = sums[:, -3]
    yy = sums[:, -2]
    count = sums[:, -1]
    d = np.diagonal(a, axis1=1, axis2=2)
    ok = np.all(d > 0, axis=1)
    rss = np.full(m, np.nan)
    err = np.full(m, np.nan)
    if not np.any(ok):
        return rss, err
    s = 1 / np.sqrt(d[ok])
    scaled = a[ok] * s[:, :, None] * s[:, None, :]
    lam, vec = np.linalg.eigh(scaled)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = lam[:, -1] / lam[:, 0]
        proj = np.einsum("mji,mj->mi", vec, b[ok] * s)
        q = np.sum(proj**2 / lam, axis=1)
    good = (lam[:, 0] > 0) & (cond <= MAX_SCORE_CONDITION)
    idx = np.flatnonzero(ok)[good]
    rss[idx] = rr[ok][good] - q[good]
    rr_g, yy_g = rr[ok][good], yy[ok][good]
    err[idx] = (
        64
        * DBL_EPSILON
        * (count[ok][good] + p)
        * (cond[good] * rr_g + np.sqrt(np.abs(rr_g * yy_g)))
    )
    return rss, err
