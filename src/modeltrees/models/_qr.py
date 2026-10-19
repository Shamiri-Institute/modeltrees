"""R's least-squares QR: Householder with limited column pivoting.

Port of ``.Call(C_Cdqrls, x, y, tol)`` from R 4.6.1, which lm.fit, lm.wfit
and glm.fit all use. Cdqrls (src/library/stats/src/lm.c) calls LINPACK's
``dqrls`` (src/appl/dqrls.f), which factors with ``dqrdc2``
(src/appl/dqrdc2.f) and solves with ``dqrsl`` (job 1110). R's choice of
aliased columns comes from dqrdc2's rule: a column whose remaining norm has
fallen below ``tol`` times its original norm is moved to the end, so later
columns that depend on earlier ones are the ones dropped. numpy's and
scipy's QR use different pivoting, which is why this is ported.

Column operations are vectorised over the remaining columns; the result is
the same factorisation, with rounding that can differ from the Fortran
loops in the last bits.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


@dataclass(frozen=True)
class QRLeastSquares:
    """The pieces of Cdqrls's result that lm.fit and glm.fit use."""

    coefficients: FloatArray  # pivoted order, 0 beyond rank as in dqrls
    residuals: FloatArray
    effects: FloatArray  # Q'y
    rank: int
    pivot: IntArray  # 0-based; R's pivot minus 1
    qr: FloatArray
    qraux: FloatArray

    @property
    def pivoted(self) -> bool:
        return bool(np.any(self.pivot != np.arange(self.pivot.shape[0])))


def dqrdc2(x: FloatArray, tol: float) -> tuple[FloatArray, int, FloatArray, IntArray]:
    """Factor ``x`` as dqrdc2.f does. Returns (qr, rank, qraux, pivot)."""
    a = np.array(x, dtype=np.float64, order="F", copy=True)
    n, p = a.shape
    pivot = np.arange(p, dtype=np.intp)
    qraux = np.linalg.norm(a, axis=0) if n > 0 else np.zeros(p)
    work1 = qraux.copy()
    work2 = qraux.copy()
    work2[work2 == 0.0] = 1.0
    k = p + 1  # 1-based in the Fortran; columns k..p are the negligible ones
    for l in range(min(n, p)):  # the Fortran's name
        # Cycle negligible columns to the end. l + 1 >= k stops the cycling.
        while not (l + 1 >= k or qraux[l] >= work2[l] * tol):
            order = np.r_[np.arange(l + 1, p), l]
            a[:, l:] = a[:, order]
            pivot[l:] = pivot[order]
            qraux[l:] = qraux[order]
            work1[l:] = work1[order]
            work2[l:] = work2[order]
            k -= 1
        if l == n - 1:
            continue
        col = a[l:, l]
        nrmxl = float(np.linalg.norm(col))
        if nrmxl == 0.0:
            continue
        if a[l, l] != 0.0:
            nrmxl = float(np.copysign(nrmxl, a[l, l]))
        col *= 1.0 / nrmxl
        col[0] = 1.0 + col[0]
        if l + 1 < p:
            rest = a[l:, l + 1 :]
            t = -(col @ rest) / col[0]
            rest += np.outer(col, t)
            for j in range(l + 1, p):
                if qraux[j] == 0.0:
                    continue
                tt = max(1.0 - (abs(a[l, j]) / qraux[j]) ** 2, 0.0)
                # "modified 9/99 by doug bates": recompute instead of
                # downdating when the norm has dropped too far.
                if abs(tt) >= 1e-6:
                    qraux[j] = qraux[j] * np.sqrt(tt)
                else:
                    qraux[j] = np.linalg.norm(a[l + 1 :, j])
                    work1[j] = qraux[j]
        qraux[l] = col[0]
        a[l, l] = -nrmxl
    rank = min(k - 1, n)
    return a, rank, qraux, pivot


def _apply_householder(
    qr: FloatArray, qraux: FloatArray, j: int, v: FloatArray
) -> None:
    """Apply the j-th Householder reflection to ``v`` in place (dqrsl)."""
    h = qr[j:, j].copy()
    h[0] = qraux[j]
    t = -(h @ v[j:]) / h[0]
    v[j:] += t * h


def dqrls(x: FloatArray, y: FloatArray, tol: float) -> QRLeastSquares:
    """Least squares as Cdqrls does: dqrdc2, then dqrsl with job 1110.

    Raises ValueError for non-finite input, as Cdqrls does.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("NA/NaN/Inf in 'x'")
    if not np.all(np.isfinite(y)):
        raise ValueError("NA/NaN/Inf in 'y'")
    n, p = x.shape
    qr, k, qraux, pivot = dqrdc2(x, tol)
    coef = np.zeros(p)
    qty = y.copy()
    rsd = y.copy()
    if k > 0:
        ju = min(k, n - 1)
        for j in range(ju):
            if qraux[j] != 0.0:
                _apply_householder(qr, qraux, j, qty)
        b = qty[:k].copy()
        # Back substitution with R; dqrsl stops at a zero diagonal (info > 0),
        # which dqrls ignores.
        for j in range(k - 1, -1, -1):
            if qr[j, j] == 0.0:
                break
            b[j] /= qr[j, j]
            if j > 0:
                b[:j] -= b[j] * qr[:j, j]
        coef[:k] = b
        rsd = np.zeros(n)
        rsd[k:] = qty[k:]
        for j in range(ju - 1, -1, -1):
            if qraux[j] != 0.0:
                _apply_householder(qr, qraux, j, rsd)
    return QRLeastSquares(
        coefficients=coef,
        residuals=rsd,
        effects=qty,
        rank=k,
        pivot=pivot,
        qr=qr,
        qraux=qraux,
    )
