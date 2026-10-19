"""Ports of R's ``stats::lm.fit`` and ``stats::lm.wfit`` (R 4.6.1).

Both solve through R's pivoting QR (:mod:`modeltrees.models._qr`) with
``tol = 1e-7``. Aliased coefficients are NaN. ``lm.wfit`` fits on the
positive-weight rows and gives zero-weight rows the residual
``y - x %*% coef`` with aliased coefficients taken as 0, as R does.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from modeltrees.models._qr import dqrls

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


@dataclass(frozen=True)
class LmFit:
    coefficients: FloatArray  # NaN for aliased columns
    residuals: FloatArray
    fitted_values: FloatArray
    rank: int
    pivot: IntArray  # 0-based
    df_residual: int


def _prepare(
    x: ArrayLike, y: ArrayLike, offset: ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    xm = np.asarray(x, dtype=np.float64)
    if xm.ndim != 2:
        raise ValueError("'x' must be a matrix")
    if xm.shape[0] == 0:
        raise ValueError("0 (non-NA) cases")
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    off = None if offset is None else np.asarray(offset, dtype=np.float64)
    if off is not None:
        yv = yv - off
    if yv.shape[0] != xm.shape[0]:
        raise ValueError("incompatible dimensions")
    return xm, yv, off


def _unpivot(coef: FloatArray, rank: int, pivot: IntArray) -> FloatArray:
    # R: coef[r2] <- NA; coef[pivot] <- coef
    c = coef.copy()
    c[rank:] = np.nan
    out = np.empty_like(c)
    out[pivot] = c
    return out


def lm_fit(
    x: ArrayLike, y: ArrayLike, offset: ArrayLike | None = None, tol: float = 1e-7
) -> LmFit:
    """``lm.fit(x, y, offset = offset, tol = tol)``."""
    xm, yv, off = _prepare(x, y, offset)
    n, p = xm.shape
    if p == 0:
        return LmFit(np.zeros(0), yv, 0 * yv, 0, np.zeros(0, dtype=np.intp), n)
    z = dqrls(xm, yv, tol)
    coef = _unpivot(z.coefficients, z.rank, z.pivot)
    fitted = yv - z.residuals
    if off is not None:
        fitted = fitted + off
    return LmFit(coef, z.residuals, fitted, z.rank, z.pivot, n - z.rank)


def lm_wfit(
    x: ArrayLike,
    y: ArrayLike,
    w: ArrayLike,
    offset: ArrayLike | None = None,
    tol: float = 1e-7,
) -> LmFit:
    """``lm.wfit(x, y, w, offset = offset, tol = tol)``."""
    xm, yv, off = _prepare(x, y, offset)
    wv = np.asarray(w, dtype=np.float64).reshape(-1)
    if wv.shape[0] != xm.shape[0]:
        raise ValueError("incompatible dimensions")
    if np.any((wv < 0) | np.isnan(wv)):
        raise ValueError("missing or negative weights not allowed")
    ok = wv != 0
    zero_weights = not np.all(ok)
    xk, yk, wk = xm[ok], yv[ok], wv[ok]
    n, p = xk.shape
    if p == 0:
        return LmFit(np.zeros(0), yv, 0 * yv, 0, np.zeros(0, dtype=np.intp), n)
    if n == 0:
        return LmFit(np.full(p, np.nan), yv, 0 * yv, 0, np.arange(p, dtype=np.intp), 0)
    wts = np.sqrt(wk)
    z = dqrls(xk * wts[:, None], yk * wts, tol)
    coef = _unpivot(z.coefficients, z.rank, z.pivot)
    res_ok = z.residuals / wts
    residuals = res_ok
    fitted = yk - res_ok
    if zero_weights:
        f0 = xm[~ok] @ np.where(np.isnan(coef), 0.0, coef)
        residuals = yv.copy()
        residuals[ok] = res_ok
        residuals[~ok] = yv[~ok] - f0
        fitted_all = yv.copy()
        fitted_all[ok] = fitted
        fitted_all[~ok] = f0
        fitted = fitted_all
    if off is not None:
        fitted = fitted + off
    return LmFit(coef, residuals, fitted, z.rank, z.pivot, n - z.rank)
