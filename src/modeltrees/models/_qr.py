# Ported from R 4.6.1 (GPL-2 or later), src/library/stats/src/lm.c (Cdqrls) and the
# LINPACK-derived src/appl/dqrls.f, dqrdc2.f, dqrsl.f.
# See THIRD_PARTY_NOTICES.md.
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

``dqrls`` first tries a LAPACK fast path (``dgeqrf``, ``dormqr``,
``dtrtrs``) and keeps it only when dqrdc2 would certainly not pivot: every
column must keep at least ``FAST_PATH_RATIO`` (or 1000 times ``tol``, if
larger) of its norm after the earlier columns are projected out, which is
1000 times stricter than R's ``1e-7``.
Without pivoting dqrdc2 is plain Householder QR with LAPACK's sign
convention, so both paths compute the same factorisation and differ only in
rounding. Anything doubtful (rank deficiency, near-collinearity, zero
columns, ``n <= p``) goes to the port. ``reference_path()`` turns the fast
path off. docs/performance.md, Step 1d, has the measurements.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lapack

from modeltrees._reference import reference_active

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
    qr: FloatArray | None  # None from the LAPACK fast path
    qraux: FloatArray | None

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


# A column must keep this share of its norm after projecting out the earlier
# columns for the fast path. dqrdc2 pivots below tol <= 1e-7 (lm.fit) or
# 1e-11 (glm.fit), so a ratio of 1e-4 leaves a margin of 1000 or more, far
# beyond the rounding in dqrdc2's downdated column norms.
FAST_PATH_RATIO = 1e-4


def dqrls(x: FloatArray, y: FloatArray, tol: float) -> QRLeastSquares:
    """Least squares as Cdqrls does: dqrdc2, then dqrsl with job 1110.

    Uses LAPACK when the matrix is clearly of full rank (module docstring)
    and the port otherwise. Raises ValueError for non-finite input, as
    Cdqrls does.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("NA/NaN/Inf in 'x'")
    if not np.all(np.isfinite(y)):
        raise ValueError("NA/NaN/Inf in 'y'")
    if not reference_active():
        fast = dqrls_lapack(x, y, max(FAST_PATH_RATIO, 1000 * tol))
        if fast is not None:
            return fast
    return dqrls_reference(x, y, tol)


def dqrls_lapack(
    x: FloatArray, y: FloatArray, ratio: float = FAST_PATH_RATIO
) -> QRLeastSquares | None:
    """Householder least squares through LAPACK, or None if rank is doubtful.

    Returns None unless ``n > p > 0`` and every ``|R[j, j]|`` is at least
    ``ratio`` times the norm of column ``j``. That ratio is the
    remaining column norm dqrdc2 compares with ``tol``, so a None-free
    result is one where dqrdc2 keeps the column order and full rank.
    Residuals are ``Q (0, Q'y[p:])`` as dqrsl forms them.
    """
    n, p = x.shape
    if not n > p > 0:
        return None
    norms = np.sqrt(np.einsum("ij,ij->j", x, x))
    if not np.all(norms > 0):
        return None
    qr, tau, _, info = lapack.dgeqrf(np.asfortranarray(x))
    if info != 0:
        return None
    diag = np.abs(np.diagonal(qr)[:p])
    if not np.all(diag >= ratio * norms):
        return None
    qty_c, _, info = lapack.dormqr(
        "L", "T", qr, tau, y[:, None].copy(), n, overwrite_c=1
    )
    if info != 0:
        return None
    qty = qty_c[:, 0]
    coef, info = lapack.dtrtrs(qr[:p, :p], qty[:p, None])
    if info != 0:
        return None
    coef = coef[:, 0]
    c = np.zeros((n, 1))
    c[p:, 0] = qty[p:]
    rsd_c, _, info = lapack.dormqr("L", "N", qr, tau, c, n, overwrite_c=1)
    if info != 0:
        return None
    return QRLeastSquares(
        coefficients=coef,
        residuals=rsd_c[:, 0],
        effects=qty,
        rank=p,
        pivot=np.arange(p, dtype=np.intp),
        qr=None,
        qraux=None,
    )


def dqrls_reference(x: FloatArray, y: FloatArray, tol: float) -> QRLeastSquares:
    """The dqrdc2 and dqrsl port, without the fast path. Inputs are finite."""
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
