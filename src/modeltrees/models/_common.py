"""Helpers shared by the node models."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def regressor_matrix(
    n: int, x: FloatArray | None, coef_names: Sequence[str] | None
) -> tuple[FloatArray, tuple[str, ...]]:
    """The regressor matrix and coefficient names.

    ``x = None`` becomes a one-column intercept matrix named "(Intercept)",
    as in glmfit and lmfit (glmtree.R:74-75, lmtree.R:64-65). Without names
    the columns are named x1, x2, ... as lm.fit does.
    """
    if x is None:
        xm = np.ones((n, 1))
        default: tuple[str, ...] = ("(Intercept)",)
    else:
        xm = np.asarray(x, dtype=np.float64)
        if xm.ndim == 1:
            xm = xm[:, None]
        if xm.shape[0] != n:
            raise ValueError(f"x has {xm.shape[0]} rows but y has {n} values")
        default = tuple(f"x{j + 1}" for j in range(xm.shape[1]))
    if coef_names is None:
        return xm, default
    names = tuple(coef_names)
    if len(names) != xm.shape[1]:
        raise ValueError(f"{len(names)} coefficient names for {xm.shape[1]} columns")
    return xm, names


def numeric_response(y: NDArray[np.generic]) -> FloatArray:
    values = np.asarray(y)
    if values.dtype.kind not in "biuf":
        raise TypeError(
            f"the response must be numeric or boolean, got dtype {values.dtype}; "
            "convert a categorical binomial response with binomial_response()"
        )
    return values.astype(np.float64).reshape(-1)


def check_vcov(vcov: bool) -> None:
    if vcov:
        raise NotImplementedError(
            'vcov = "info" and "sandwich" are not in the v0.1 scope '
            '(docs/decisions.md); only the default "opg" is supported'
        )
