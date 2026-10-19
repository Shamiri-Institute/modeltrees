"""GLM families with canonical links, as R's stats package defines them.

Ports of ``gaussian()``, ``binomial()`` and ``poisson()`` from R 4.6.1
(src/library/stats/R/family.R) with their canonical links. The logit link
functions and the binomial deviance residuals are R's C code
(src/library/stats/src/family.c), including the +-30 thresholds on the
linear predictor, which change fitted values near separation.

Only a vector response is supported. R's two-column binomial response
(successes, failures) is out of scope.
"""

import math
import warnings
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, xlog1py, xlogy

FloatArray = NDArray[np.float64]

DBL_EPSILON = float(np.finfo(np.float64).eps)


class GlmFitWarning(UserWarning):
    """A warning R's glm.fit or a family would give, with R's message."""


@dataclass(frozen=True)
class Initialized:
    """What ``family$initialize`` sets in glm.fit's frame."""

    y: FloatArray
    n: FloatArray
    mustart: FloatArray


class Family(Protocol):
    @property
    def family(self) -> str: ...
    @property
    def link(self) -> str: ...

    def linkfun(self, mu: FloatArray) -> FloatArray: ...
    def linkinv(self, eta: FloatArray) -> FloatArray: ...
    def mu_eta(self, eta: FloatArray) -> FloatArray: ...
    def variance(self, mu: FloatArray) -> FloatArray: ...
    def dev_resids(
        self, y: FloatArray, mu: FloatArray, wt: FloatArray
    ) -> FloatArray: ...
    def aic(
        self, y: FloatArray, n: FloatArray, mu: FloatArray, wt: FloatArray, dev: float
    ) -> float: ...
    def initialize(self, y: FloatArray, weights: FloatArray) -> Initialized: ...
    def validmu(self, mu: FloatArray) -> bool: ...
    def valideta(self, eta: FloatArray) -> bool: ...


class Gaussian:
    """``gaussian(link = "identity")``."""

    family: ClassVar[str] = "gaussian"
    link: ClassVar[str] = "identity"

    def linkfun(self, mu: FloatArray) -> FloatArray:
        return mu

    def linkinv(self, eta: FloatArray) -> FloatArray:
        return eta

    def mu_eta(self, eta: FloatArray) -> FloatArray:
        return np.ones_like(eta)

    def variance(self, mu: FloatArray) -> FloatArray:
        return np.ones_like(mu)

    def dev_resids(self, y: FloatArray, mu: FloatArray, wt: FloatArray) -> FloatArray:
        return wt * (y - mu) ** 2

    def aic(
        self, y: FloatArray, n: FloatArray, mu: FloatArray, wt: FloatArray, dev: float
    ) -> float:
        # R: nobs * (log(dev/nobs * 2*pi) + 1) + 2 - sum(log(wt)). nobs counts
        # zero-weight rows and log(0) makes the result Inf, as in R.
        nobs = y.shape[0]
        with np.errstate(divide="ignore"):
            logw = float(np.sum(np.log(wt)))
        return nobs * (math.log(dev / nobs * 2 * math.pi) + 1) + 2 - logw

    def initialize(self, y: FloatArray, weights: FloatArray) -> Initialized:
        return Initialized(y=y, n=np.ones_like(y), mustart=y.copy())

    def validmu(self, mu: FloatArray) -> bool:
        return True

    def valideta(self, eta: FloatArray) -> bool:
        return True


_THRESH = 30.0
_MTHRESH = -30.0
_INVEPS = 1.0 / DBL_EPSILON


class Binomial:
    """``binomial(link = "logit")`` with a vector response."""

    family: ClassVar[str] = "binomial"
    link: ClassVar[str] = "logit"

    def linkfun(self, mu: FloatArray) -> FloatArray:
        if np.any((mu < 0) | (mu > 1)):
            raise ValueError("Value out of range (0, 1)")
        with np.errstate(divide="ignore"):
            return np.log(mu / (1 - mu))

    def linkinv(self, eta: FloatArray) -> FloatArray:
        # family.c logit_linkinv: eps and 1/eps outside +-30.
        with np.errstate(over="ignore"):
            tmp = np.where(
                eta < _MTHRESH,
                DBL_EPSILON,
                np.where(eta > _THRESH, _INVEPS, np.exp(eta)),
            )
        return tmp / (1 + tmp)

    def mu_eta(self, eta: FloatArray) -> FloatArray:
        # family.c logit_mu_eta: DBL_EPSILON outside +-30.
        with np.errstate(over="ignore", invalid="ignore"):
            e = np.exp(eta)
            opexp = 1 + e
            inside = e / (opexp * opexp)
        return np.where((eta > _THRESH) | (eta < _MTHRESH), DBL_EPSILON, inside)

    def variance(self, mu: FloatArray) -> FloatArray:
        return mu * (1 - mu)

    def dev_resids(self, y: FloatArray, mu: FloatArray, wt: FloatArray) -> FloatArray:
        # family.c binomial_dev_resids with y_log_y(y, mu) = y * log(y/mu), 0 at y = 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(y != 0, y * np.log(y / mu), 0.0)
            b = np.where(1 - y != 0, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
        return 2 * wt * (a + b)

    def aic(
        self, y: FloatArray, n: FloatArray, mu: FloatArray, wt: FloatArray, dev: float
    ) -> float:
        m = n if np.any(n > 1) else wt
        k = np.round(m * y)
        size = np.round(m)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(m > 0, wt / m, 0.0)
        logd = _log_dbinom(k, size, mu)
        return float(-2 * np.sum(np.where(factor != 0, factor * logd, 0.0)))

    def initialize(self, y: FloatArray, weights: FloatArray) -> Initialized:
        y = y.copy()
        y[weights == 0] = 0
        if np.any((y < 0) | (y > 1)):
            raise ValueError("y values must be 0 <= y <= 1")
        mustart = (weights * y + 0.5) / (weights + 1)
        m = weights * y
        if np.any(np.abs(m - np.round(m)) > 1e-3):
            warnings.warn(
                "non-integer #successes in a binomial glm!", GlmFitWarning, stacklevel=4
            )
        return Initialized(y=y, n=np.ones_like(y), mustart=mustart)

    def validmu(self, mu: FloatArray) -> bool:
        return bool(np.all(np.isfinite(mu)) and np.all((mu > 0) & (mu < 1)))

    def valideta(self, eta: FloatArray) -> bool:
        return True


class Poisson:
    """``poisson(link = "log")``."""

    family: ClassVar[str] = "poisson"
    link: ClassVar[str] = "log"

    def linkfun(self, mu: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore"):
            return np.log(mu)

    def linkinv(self, eta: FloatArray) -> FloatArray:
        # R: pmax(exp(eta), .Machine$double.eps)
        with np.errstate(over="ignore"):
            return np.maximum(np.exp(eta), DBL_EPSILON)

    def mu_eta(self, eta: FloatArray) -> FloatArray:
        with np.errstate(over="ignore"):
            return np.maximum(np.exp(eta), DBL_EPSILON)

    def variance(self, mu: FloatArray) -> FloatArray:
        return mu

    def dev_resids(self, y: FloatArray, mu: FloatArray, wt: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = np.where(y > 0, wt * (y * np.log(y / mu) - (y - mu)), mu * wt)
        return 2 * r

    def aic(
        self, y: FloatArray, n: FloatArray, mu: FloatArray, wt: FloatArray, dev: float
    ) -> float:
        return float(-2 * np.sum(_log_dpois(y, mu) * wt))

    def initialize(self, y: FloatArray, weights: FloatArray) -> Initialized:
        if np.any(y < 0):
            raise ValueError("negative values not allowed for the 'Poisson' family")
        return Initialized(y=y, n=np.ones_like(y), mustart=y + 0.1)

    def validmu(self, mu: FloatArray) -> bool:
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def valideta(self, eta: FloatArray) -> bool:
        return True


def _log_dbinom(k: FloatArray, size: FloatArray, p: FloatArray) -> FloatArray:
    """``dbinom(k, size, p, log = TRUE)`` for integer k and size.

    R uses Loader's saddle-point expansion; this closed form agrees to
    rounding error, far inside the tolerances the fixtures are checked at.
    """
    return (
        gammaln(size + 1)
        - gammaln(k + 1)
        - gammaln(size - k + 1)
        + xlogy(k, p)
        + xlog1py(size - k, -p)
    )


def _log_dpois(y: FloatArray, mu: FloatArray) -> FloatArray:
    """``dpois(y, mu, log = TRUE)``; -Inf for non-integer y, as in R."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = xlogy(y, mu) - mu - gammaln(y + 1)
    nonint = np.abs(y - np.round(y)) > 1e-7 * np.maximum(1.0, np.abs(y))
    return np.where(nonint, -np.inf, out)


def get_family(name: str) -> Family:
    """The family by R's name, with its canonical link."""
    families: dict[str, Family] = {
        "gaussian": Gaussian(),
        "binomial": Binomial(),
        "poisson": Poisson(),
    }
    try:
        return families[name]
    except KeyError:
        raise ValueError(
            f"family {name!r} is not supported; use one of {sorted(families)}"
        ) from None
