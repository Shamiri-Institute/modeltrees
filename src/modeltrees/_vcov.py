# Formulas derived from sandwich 3.1-3 (GPL-2 | GPL-3), R/vcovCL.R (meatCL).
# See THIRD_PARTY_NOTICES.md.
"""The cluster-robust sandwich of a leaf fit (docs/api-design.md, section 5.1).

Pure linear algebra on a node model's Hessian and scores, so that it can be
tested on its own. The adjustments are those of ``sandwich::meatCL``
(reference/sandwich/R/vcovCL.R:130-176) for one clustering variable:

    V = H^-1 (c sum_g s_g s_g') H^-1,
    c = [G / (G - 1) if cadjust] [(n - 1) / (n - k) if HC1],

with ``s_g`` the scores summed within cluster ``g``. sandwich's ``crossprod /
n`` in the meat and its bread ``n H^-1`` cancel the ``n``. The meat follows
the instability tests (``mob_grow_fluctests``, modelparty.R:221-246) rather
than sandwich where the two differ, which they do only under weights: rows
with zero weight are removed, ``n`` is the leaf's ``nobs`` (the sum of the
case weights under case weights) and, without clusters, each row is its own
cluster with its score divided by its weight, the meat of the data with
each row repeated ``w_i`` times.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# A scaled Hessian with a larger condition number counts as singular even
# when its Cholesky factorization succeeds: its inverse has no correct digit.
MAX_CONDITION = 1 / float(np.finfo(np.float64).eps)

VcovType = Literal["HC0", "HC1"]
VcovStatus = Literal["ok", "singular", "unavailable"]


@dataclass(frozen=True)
class Sandwich:
    """A sandwich covariance over the non-aliased coefficients."""

    matrix: FloatArray  # (k, k); NaN unless status is "ok"
    status: VcovStatus
    reason: str | None
    condition: float  # of the Hessian scaled by its diagonal; NaN without one
    n_clusters: float  # G: clusters with positive weight (rows, or nobs, without)
    too_few_clusters: bool  # G <= k: the meat has rank below k


def sandwich(
    hessian: FloatArray | None,
    estfun: FloatArray | None,
    *,
    weights: FloatArray,
    cluster: NDArray[np.generic] | None,
    caseweights: bool,
    type: VcovType = "HC1",
    cadjust: bool = True,
) -> Sandwich:
    """The sandwich ``H^-1 (c sum_g s_g s_g') H^-1``.

    ``hessian`` is minus the derivative of the summed scores
    (``FitResult.hessian``), ``estfun`` the (n, k) scores with the weights
    inside, ``weights`` the n case or precision weights and ``cluster`` the
    n cluster ids, or None for each row its own cluster.

    The Hessian is scaled by its diagonal and factored by Cholesky; if that
    fails, or the scaled condition number exceeds ``MAX_CONDITION`` (1 /
    eps), the status is "singular" and the matrix NaN. Without a Hessian or
    scores the status is "unavailable". So it is with a single cluster
    (G = 1): the summed scores are zero at the estimate, so the meat is
    empty, and ``G / (G - 1)`` is undefined; sandwich 3.1-3 stops there.
    ``too_few_clusters`` is set when ``G <= k``; the caller warns.
    """
    if type not in ("HC0", "HC1"):
        raise ValueError(f"type must be 'HC0' or 'HC1', not {type!r}")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if hessian is None or estfun is None:
        known = hessian if hessian is not None else estfun
        k0 = 0 if known is None else int(np.shape(known)[1])
        return Sandwich(
            np.full((k0, k0), np.nan),
            "unavailable",
            "the fit has no Hessian or no scores",
            np.nan,
            np.nan,
            False,
        )
    h = np.asarray(hessian, dtype=np.float64)
    ef = np.asarray(estfun, dtype=np.float64)
    n_rows, k = ef.shape
    if h.shape != (k, k):
        raise ValueError(f"the Hessian is {h.shape} for {k} score columns")
    if w.shape[0] != n_rows:
        raise ValueError(f"{w.shape[0]} weights for {n_rows} rows of scores")
    if cluster is not None and np.shape(cluster)[0] != n_rows:
        raise ValueError(f"{np.shape(cluster)[0]} cluster ids for {n_rows} rows")

    # Zero-weight rows contribute nothing and are not clusters.
    pos = w > 0
    ef, w = ef[pos], w[pos]
    nobs = float(np.sum(w)) if caseweights else float(ef.shape[0])
    if cluster is None:
        # Each row its own cluster; under case weights the meat of the rows
        # repeated w_i times, sum_i estfun_i estfun_i' / w_i.
        scaled = ef / np.sqrt(w)[:, None] if caseweights else ef
        meat = scaled.T @ scaled
        g = nobs
    else:
        _, codes = np.unique(np.asarray(cluster)[pos], return_inverse=True)
        summed = np.zeros((int(codes.max(initial=-1)) + 1, k))
        np.add.at(summed, codes, ef)
        meat = summed.T @ summed
        g = float(summed.shape[0])
    too_few = g <= k

    condition, factor = _factor(h)
    if g <= 1:
        return Sandwich(
            np.full((k, k), np.nan),
            "unavailable",
            "a single cluster: the meat is empty and G / (G - 1) undefined",
            condition,
            g,
            too_few,
        )
    if type == "HC1" and nobs <= k:
        return Sandwich(
            np.full((k, k), np.nan),
            "unavailable",
            f"HC1 needs n > k, with n = {nobs:g} and k = {k}",
            condition,
            g,
            too_few,
        )
    if factor is None:
        return Sandwich(
            np.full((k, k), np.nan),
            "singular",
            "the Hessian scaled by its diagonal is not positive definite",
            condition,
            g,
            too_few,
        )
    c = g / (g - 1) if cadjust else 1.0
    if type == "HC1":
        c *= (nobs - 1) / (nobs - k)
    chol, d = factor
    # H^-1 = D^-1 S^-1 D^-1 with S = D^-1 H D^-1 the scaled Hessian.
    hinv = scipy.linalg.cho_solve((chol, False), np.diag(1 / d)) / d[:, None]
    v = hinv @ (c * meat) @ hinv
    return Sandwich((v + v.T) / 2, "ok", None, condition, g, too_few)


def _factor(
    h: FloatArray,
) -> tuple[float, tuple[FloatArray, FloatArray] | None]:
    """The condition number of ``h`` scaled by its diagonal and the Cholesky
    factor of the scaled matrix with the scale, or None if it fails."""
    diag = np.diag(h)
    if h.shape[0] == 0:
        return 1.0, (np.zeros((0, 0)), np.zeros(0))
    if not np.all(np.isfinite(h)) or np.any(diag <= 0):
        return float("inf"), None
    d = np.sqrt(diag)
    scaled = h / d[:, None] / d[None, :]
    lam = np.linalg.eigvalsh(scaled)
    condition = float(lam[-1] / lam[0]) if lam[0] > 0 else float("inf")
    if not condition <= MAX_CONDITION:
        return condition, None
    try:
        chol = scipy.linalg.cholesky(scaled, lower=False)
    except np.linalg.LinAlgError:
        return condition, None
    return condition, (chol, d)


@dataclass(frozen=True)
class LeafVcov:
    """The cluster-robust covariance of one node's coefficients
    (docs/api-design.md, section 5.1), from :meth:`ModelTree.vcov`.

    ``matrix`` covers every coefficient in ``names`` order, with NaN rows
    and columns for aliased or unidentified ones, and is NaN throughout
    unless ``status`` is "ok". ``n_clusters`` is G, the clusters with
    positive weight in the node, None without clusters. ``condition`` is
    the condition number of the Hessian scaled by its diagonal (NaN
    without a Hessian).
    """

    node: int
    names: tuple[str, ...]
    matrix: FloatArray
    n_clusters: int | None
    status: VcovStatus
    reason: str | None
    condition: float

    @property
    def std_errors(self) -> dict[str, float]:
        """The square roots of the diagonal, by coefficient name."""
        d = np.sqrt(np.diag(self.matrix))
        return {name: float(v) for name, v in zip(self.names, d, strict=True)}


def expand(s: Sandwich, keep: NDArray[np.bool_]) -> FloatArray:
    """The sandwich over every coefficient: NaN rows and columns where
    ``keep`` is False, and everywhere unless the status is "ok"."""
    k_all = keep.shape[0]
    out = np.full((k_all, k_all), np.nan)
    if s.status == "ok":
        out[np.ix_(keep, keep)] = s.matrix
    out.setflags(write=False)
    return out
