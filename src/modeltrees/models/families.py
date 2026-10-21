# Ported from R 4.6.1 (GPL-2 or later), src/library/stats/R/family.R and src/family.c;
# negative.binomial from MASS 7.3-66 (GPL-2 | GPL-3).
# See THIRD_PARTY_NOTICES.md.
"""GLM families with canonical links, as R's stats package defines them.

Ports of ``gaussian()``, ``binomial()`` and ``poisson()`` from R 4.6.1
(src/library/stats/R/family.R) with their canonical links. The logit link
functions and the binomial deviance residuals are R's C code
(src/library/stats/src/family.c), including the +-30 thresholds on the
linear predictor, which change fitted values near separation.

Only a vector response is supported. R's two-column binomial response
(successes, failures) is out of scope.

The binomial and poisson deviance residuals, inverse links, link
derivatives and AIC have specialised versions that skip work R's general
formulas do and then discard: the second logarithm of a 0/1 response, the
clamping when no linear predictor is beyond +-30, ``gammaln`` of repeated
small counts (docs/performance.md, Step 3, option 1). They give R's values
to rounding. Under :func:`modeltrees.reference_path` the general formulas
run, and the tests compare both.
"""

import math
import warnings
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, xlog1py, xlogy

from modeltrees._reference import reference_active
from modeltrees.errors import GlmFitWarning

FloatArray = NDArray[np.float64]

DBL_EPSILON = float(np.finfo(np.float64).eps)


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
        if reference_active() or _beyond(eta, _THRESH):
            return self.linkinv_reference(eta)
        tmp = np.exp(eta)
        return tmp / (1 + tmp)

    def linkinv_reference(self, eta: FloatArray) -> FloatArray:
        # family.c logit_linkinv: eps and 1/eps outside +-30.
        with np.errstate(over="ignore"):
            tmp = np.where(
                eta < _MTHRESH,
                DBL_EPSILON,
                np.where(eta > _THRESH, _INVEPS, np.exp(eta)),
            )
        return tmp / (1 + tmp)

    def mu_eta(self, eta: FloatArray) -> FloatArray:
        if reference_active() or _beyond(eta, _THRESH):
            return self.mu_eta_reference(eta)
        e = np.exp(eta)
        opexp = 1 + e
        return e / (opexp * opexp)

    def mu_eta_reference(self, eta: FloatArray) -> FloatArray:
        # family.c logit_mu_eta: DBL_EPSILON outside +-30.
        with np.errstate(over="ignore", invalid="ignore"):
            e = np.exp(eta)
            opexp = 1 + e
            inside = e / (opexp * opexp)
        return np.where((eta > _THRESH) | (eta < _MTHRESH), DBL_EPSILON, inside)

    def variance(self, mu: FloatArray) -> FloatArray:
        return mu * (1 - mu)

    def dev_resids(self, y: FloatArray, mu: FloatArray, wt: FloatArray) -> FloatArray:
        if reference_active() or not _binary(y):
            return self.dev_resids_reference(y, mu, wt)
        # For y in {0, 1} one of R's two terms is 0 and the other is
        # log(1 / mu) or log(1 / (1 - mu)): the same arithmetic on the kept
        # term, without the discarded logarithm and division.
        t = np.where(y == 1, mu, 1 - mu)
        with np.errstate(divide="ignore"):
            return 2 * wt * np.log(1 / t)

    def dev_resids_reference(
        self, y: FloatArray, mu: FloatArray, wt: FloatArray
    ) -> FloatArray:
        # family.c binomial_dev_resids with y_log_y(y, mu) = y * log(y/mu), 0 at y = 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(y != 0, y * np.log(y / mu), 0.0)
            b = np.where(1 - y != 0, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
        return 2 * wt * (a + b)

    def aic(
        self, y: FloatArray, n: FloatArray, mu: FloatArray, wt: FloatArray, dev: float
    ) -> float:
        if reference_active() or np.any(n > 1) or not (_ones(wt) and _binary(y)):
            return self.aic_reference(y, n, mu, wt, dev)
        # One trial per row with unit weight: dbinom(y, 1, mu, log = TRUE) is
        # log(mu) or log1p(-mu), and the lgamma terms are lgamma(1) = 0 and
        # lgamma(2) = 0.
        with np.errstate(divide="ignore"):
            logd = np.where(y == 1, np.log(mu), np.log1p(-mu))
        return float(-2 * np.sum(logd))

    def aic_reference(
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
            out = np.exp(eta)
        return np.maximum(out, DBL_EPSILON, out=out)

    def mu_eta(self, eta: FloatArray) -> FloatArray:
        return self.linkinv(eta)

    def variance(self, mu: FloatArray) -> FloatArray:
        return mu

    def dev_resids(self, y: FloatArray, mu: FloatArray, wt: FloatArray) -> FloatArray:
        if reference_active():
            return self.dev_resids_reference(y, mu, wt)
        # R's two branches in one expression: at y = 0 the logarithm is
        # skipped and wt * (0 - (0 - mu)) is exactly mu * wt.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = y / mu
            np.log(r, out=r, where=y > 0)
            r[y <= 0] = 0.0
            return 2 * (wt * (y * r - (y - mu)))

    def dev_resids_reference(
        self, y: FloatArray, mu: FloatArray, wt: FloatArray
    ) -> FloatArray:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = np.where(y > 0, wt * (y * np.log(y / mu) - (y - mu)), mu * wt)
        return 2 * r

    def aic(
        self, y: FloatArray, n: FloatArray, mu: FloatArray, wt: FloatArray, dev: float
    ) -> float:
        if reference_active() or not bool(np.all(y == np.round(y))):
            return self.aic_reference(y, n, mu, wt, dev)
        # Integer counts: R's non-integer check cannot fire, and lgamma(y + 1)
        # comes from a table over 0..max(y) when that is no longer than y.
        top = float(np.max(y)) if y.shape[0] else 0.0
        if top <= y.shape[0]:
            lgam = gammaln(np.arange(int(top) + 1) + 1.0)[y.astype(np.intp)]
        else:
            lgam = gammaln(y + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = xlogy(y, mu) - mu - lgam
        return float(-2 * np.sum(out * wt))

    def aic_reference(
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


class NegativeBinomial:
    """``MASS::negative.binomial(theta, link = "log")`` (MASS 7.3-66).

    The family glm.nb refits with at each alternation, for a fixed theta.
    The log link is ``make.link("log")``: ``pmax(exp(eta), DBL_EPSILON)``
    for the mean and its derivative, every eta valid. ``validmu`` is
    ``all(mu > 0)``, which unlike poisson's admits Inf; a NaN mean makes R
    stop with an error and is invalid here.
    """

    link: ClassVar[str] = "log"

    def __init__(self, theta: float) -> None:
        self.theta = float(theta)
        # R: paste("Negative Binomial(", format(round(theta, 4)), ")").
        # Only the prefix matters, for the unit dispersion of the scores.
        self.family = f"Negative Binomial({round(self.theta, 4):.7g})"

    def linkfun(self, mu: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore"):
            return np.log(mu)

    def linkinv(self, eta: FloatArray) -> FloatArray:
        with np.errstate(over="ignore"):
            out = np.exp(eta)
        return np.maximum(out, DBL_EPSILON, out=out)

    def mu_eta(self, eta: FloatArray) -> FloatArray:
        return self.linkinv(eta)

    def variance(self, mu: FloatArray) -> FloatArray:
        return mu + mu**2 / self.theta

    def dev_resids(self, y: FloatArray, mu: FloatArray, wt: FloatArray) -> FloatArray:
        th = self.theta
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                2
                * wt
                * (
                    y * np.log(np.maximum(1, y) / mu)
                    - (y + th) * np.log((y + th) / (mu + th))
                )
            )

    def aic(
        self, y: FloatArray, n: FloatArray, mu: FloatArray, wt: FloatArray, dev: float
    ) -> float:
        th = np.float64(self.theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = (
                (y + th) * np.log(mu + th)
                - y * np.log(mu)
                + gammaln(y + 1)
                - th * np.log(th)
                + gammaln(th)
                - gammaln(th + y)
            )
        return float(2 * np.sum(term * wt))

    def initialize(self, y: FloatArray, weights: FloatArray) -> Initialized:
        if np.any(y < 0):
            raise ValueError(
                "negative values not allowed for the negative binomial family"
            )
        return Initialized(y=y, n=np.ones_like(y), mustart=y + (y == 0) / 6)

    def validmu(self, mu: FloatArray) -> bool:
        return bool(np.all(mu > 0))

    def valideta(self, eta: FloatArray) -> bool:
        return True


def _beyond(eta: FloatArray, bound: float) -> bool:
    """Whether any linear predictor is outside +-bound (or NaN)."""
    if eta.shape[0] == 0:
        return False
    lo, hi = float(np.min(eta)), float(np.max(eta))
    return not (-bound <= lo and hi <= bound)


def _binary(y: FloatArray) -> bool:
    return bool(np.all((y == 0) | (y == 1)))


def _ones(w: FloatArray) -> bool:
    return bool(np.all(w == 1))


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
