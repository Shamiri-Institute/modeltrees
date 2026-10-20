"""Negative binomial regression with unknown theta, as ``MASS::glm.nb``.

Ports ``glm.nb`` and ``theta.ml`` from MASS 7.3-66
(reference/MASS/R/negbin.R:76-211 and :328-366) on top of the glm.fit port,
and the node model ``negbinfit`` of r/extensions/negbin.R, specified in
docs/extensions-spec.md, section 3. Line numbers below refer to negbin.R.

The glm.nb loop is copied step by step, because the scores come from its
last glm.fit call and theta from the theta.ml call after it:

1. a Poisson fit from ``start``;
2. theta from ``theta.ml`` at the Poisson means;
3. alternation: refit beta with ``negative.binomial(theta)`` from
   ``etastart = log(mu)``, re-estimate theta at the previous means, update
   the means, until ``|Lm0 - Lm| / d1 + |theta_old - theta| <= epsilon``.
"""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln, zeta

from modeltrees.errors import FitError
from modeltrees.models._common import check_vcov, numeric_response, regressor_matrix
from modeltrees.models.families import (
    GlmFitWarning,
    NegativeBinomial,
    Poisson,
)
from modeltrees.models.glm_fit import GlmFit, GlmFitError, glm_fit
from modeltrees.node_model import FitResult, failed_fit

FloatArray = NDArray[np.float64]

# .Machine$double.eps^0.25, theta.ml's tolerance on the Newton step.
THETA_EPS = float(np.finfo(np.float64).eps) ** 0.25


class NegBinError(FitError):
    """An error R's glm.nb or theta.ml would stop with."""


@dataclass(frozen=True)
class ThetaML:
    """``theta.ml``'s value with its attributes."""

    theta: float
    se: float  # attr "SE": sqrt(1 / info) at the last Newton step
    warning: str | None  # attr "warn": the last warning, as R keeps it
    warnings: tuple[str, ...]  # every warning, in R's order
    iterations: int


def theta_ml(
    y: FloatArray,
    mu: FloatArray,
    n: float,
    weights: FloatArray,
    limit: int = 10,
    eps: float = THETA_EPS,
) -> ThetaML:
    """ML estimate of theta for fixed means, as ``MASS::theta.ml`` (:328-366).

    Newton steps on the score from ``n / sum(w (y / mu - 1)^2)``, while
    ``it < limit`` and the last step is above ``eps``. R's ``while`` stops
    with an error on a NaN step; that raises :class:`NegBinError`.
    """

    # digamma and trigamma of th + y are evaluated once per distinct count:
    # the same values, at a fraction of the cost. trigamma is zeta(2, .),
    # which is what scipy's polygamma(1, .) computes.
    uy, inv = np.unique(y, return_inverse=True)

    # Scalars are numpy floats so that division by zero and log(0) give
    # R's Inf and NaN instead of raising.
    def score(th: np.float64) -> float:
        return float(
            np.sum(
                weights
                * (
                    digamma(th + uy)[inv]
                    - digamma(th)
                    + np.log(th)
                    + 1
                    - np.log(th + mu)
                    - (y + th) / (mu + th)
                )
            )
        )

    def info(th: np.float64) -> float:
        return float(
            np.sum(
                weights
                * (
                    -zeta(2.0, th + uy)[inv]
                    + zeta(2.0, th)
                    - 1 / th
                    + 2 / (mu + th)
                    - (y + th) / (mu + th) ** 2
                )
            )
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t0 = float(n / np.sum(weights * (y / mu - 1) ** 2))
        it = 0
        delta = 1.0
        i = math.nan
        while True:
            it += 1
            if not it < limit:
                break
            if math.isnan(delta):
                raise NegBinError("theta.ml: missing value where TRUE/FALSE needed")
            if not abs(delta) > eps:
                break
            th = np.float64(abs(t0))
            i = info(th)
            delta = float(np.float64(score(th)) / np.float64(i))
            t0 = float(th + delta)
    if math.isnan(t0):
        raise NegBinError("theta.ml: missing value where TRUE/FALSE needed")
    messages: list[str] = []
    warn: str | None = None
    if t0 < 0:
        t0 = 0.0
        messages.append("estimate truncated at zero")
        warn = "estimate truncated at zero"
    if it == limit:
        messages.append("iteration limit reached")
        warn = "iteration limit reached"
    with np.errstate(divide="ignore", invalid="ignore"):
        se = float(np.sqrt(np.float64(1) / np.float64(i)))
    return ThetaML(t0, se, warn, tuple(messages), it)


def negbin_loglik(theta: float, mu: FloatArray, y: FloatArray, w: FloatArray) -> float:
    """glm.nb's internal ``loglik`` (:82-84), the weighted log-likelihood."""
    th = np.float64(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(
            np.sum(
                w
                * (
                    gammaln(th + y)
                    - gammaln(th)
                    - gammaln(y + 1)
                    + th * np.log(th)
                    + y * np.log(mu + (y == 0))
                    - (th + y) * np.log(th + mu)
                )
            )
        )


@dataclass(frozen=True)
class NegBinFit:
    """What glm.nb returns, under R's names where they exist."""

    fit: GlmFit  # the last glm.fit call: coefficients, scores, deviance
    theta: float  # glm.nb's theta, from the theta.ml call after that fit
    theta_se: float | None  # SE.theta; None if the alternation never ran
    theta_irls: float  # the theta the last glm.fit call used
    twologlik: float  # 2 * Lm at (fit's means, theta)
    th_warn: str | None
    alternations: int  # glm.nb's iter (the loop counter after the loop)
    warnings: tuple[str, ...]  # every warning, in R's order

    @property
    def coefficients(self) -> FloatArray:
        return self.fit.coefficients

    @property
    def deviance(self) -> float:
        return self.fit.deviance

    @property
    def converged(self) -> bool:
        return self.fit.converged

    @property
    def iter(self) -> int:
        return self.fit.iter


def glm_nb(
    x: FloatArray,
    y: FloatArray,
    *,
    weights: FloatArray | None = None,
    offset: FloatArray | None = None,
    start: FloatArray | None = None,
    epsilon: float = 1e-8,
    maxit: int = 25,
) -> NegBinFit:
    """``MASS::glm.nb(y ~ 0 + x, weights, offset, start, control)``.

    The model has no intercept term, so glm.fit runs with
    ``intercept = FALSE``, as in negbinfit. Warnings are collected in the
    result, not emitted.
    """
    xm = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.ones(yv.shape[0]) if weights is None else np.asarray(weights, np.float64)
    if np.any(w < 0):
        raise NegBinError("negative weights not allowed")
    messages: list[str] = []

    def fitter(**kw: object) -> GlmFit:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GlmFitWarning)
            f = glm_fit(
                xm,
                yv,
                weights=w,
                offset=offset,
                epsilon=epsilon,
                maxit=maxit,
                intercept=False,
                **kw,  # type: ignore[arg-type]
            )
        messages.extend(f.warnings)
        return f

    def theta_of(mu: FloatArray) -> ThetaML:
        t = theta_ml(yv, mu, float(np.sum(w)), w, limit=maxit)
        messages.extend(t.warnings)
        return t

    # :122-132. as.vector() drops the attributes of the first theta.
    fit = fitter(family=Poisson(), start=start)
    mu = fit.fitted_values
    th = theta_of(mu)
    theta = th.theta
    th_attrs: ThetaML | None = None
    fam = NegativeBinomial(theta)
    # :137-142
    it = 0
    d1 = math.sqrt(2 * max(1, fit.df_residual))
    d2 = delta = 1.0
    lm = negbin_loglik(theta, mu, yv, w)
    lm0 = lm + 2 * d1
    theta_irls = math.nan
    # :143-166
    while True:
        it += 1
        if not (it <= maxit and abs(lm0 - lm) / d1 + abs(delta) / d2 > epsilon):
            break
        with np.errstate(divide="ignore"):
            eta = np.log(mu)  # g(mu), the link function
        fit = fitter(family=fam, etastart=eta)
        theta_irls = fam.theta
        t0 = theta
        # theta.ml at the means before this fit's update, as in R.
        th_attrs = theta_of(mu)
        theta = th_attrs.theta
        fam = NegativeBinomial(theta)
        mu = fit.fitted_values
        delta = t0 - theta
        lm0 = lm
        lm = negbin_loglik(theta, mu, yv, w)
    # :167-171
    th_warn = None if th_attrs is None else th_attrs.warning
    if it > maxit:
        messages.append("alternation limit reached")
        th_warn = "alternation limit reached"
    return NegBinFit(
        fit=fit,
        theta=theta,
        theta_se=None if th_attrs is None else th_attrs.se,
        theta_irls=theta_irls,
        twologlik=2 * lm,
        th_warn=th_warn,
        alternations=it,
        warnings=tuple(messages),
    )


class NegBinModel:
    """Negative binomial regression with its own theta in every node.

    Port of ``negbinfit`` (r/extensions/negbin.R). ``coefficients`` is beta
    only; theta is not tested and has no score column
    (docs/extensions-spec.md, section 3). The objective is the negative
    log-likelihood ``-twologlik / 2`` at (beta, theta), or with
    ``objective = "deviance"`` the residual deviance of the last glm.fit
    call, as party 1.0-1 compared splits. The scores are sandwich's
    ``estfun.glm`` of that call: working residuals times working weights
    times x, dispersion 1.

    Rows with zero weight are left out of the fit and get zero scores, as in
    GlmModel; they contribute nothing to glm.nb's sums. An error in glm.fit
    or theta.ml is a failed fit (docs/model-interface.md, section 4.2).
    ``extras`` holds theta, theta_se, theta_irls, theta_warning, converged
    and iter, as the R fit function returns them, and every warning.
    """

    # theta counts in the pruning df: k + 1 parameters per node.
    extra_df = 1

    def __init__(
        self,
        *,
        objective: Literal["loglik", "deviance"] = "loglik",
        epsilon: float = 1e-8,
        maxit: int = 25,
        coef_names: Sequence[str] | None = None,
    ) -> None:
        if objective not in ("loglik", "deviance"):
            raise ValueError(
                f"objective must be 'loglik' or 'deviance', not {objective!r}"
            )
        self.objective = objective
        self.epsilon = epsilon
        self.maxit = maxit
        self.coef_names = None if coef_names is None else tuple(coef_names)

    def fit_nb(
        self,
        y: NDArray[np.generic],
        x: FloatArray | None,
        *,
        start: FloatArray | None = None,
        weights: FloatArray,
        offset: FloatArray | None = None,
    ) -> tuple[NegBinFit, FloatArray, tuple[str, ...], NDArray[np.bool_]]:
        """The glm.nb fit on the positive-weight rows, with the full design."""
        yv = numeric_response(y)
        n = yv.shape[0]
        xm, names = regressor_matrix(n, x, self.coef_names)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        off = None if offset is None else np.asarray(offset, dtype=np.float64)
        keep = w > 0
        if np.all(keep):
            z = glm_nb(
                xm,
                yv,
                weights=w,
                offset=off,
                start=start,
                epsilon=self.epsilon,
                maxit=self.maxit,
            )
        else:
            z = glm_nb(
                xm[keep],
                yv[keep],
                weights=w[keep],
                offset=None if off is None else off[keep],
                start=start,
                epsilon=self.epsilon,
                maxit=self.maxit,
            )
        return z, xm, names, keep

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
        try:
            z, xm, names, keep = self.fit_nb(
                y, x, start=start, weights=weights, offset=offset
            )
        except (GlmFitError, NegBinError) as exc:
            _, names = regressor_matrix(np.asarray(y).shape[0], x, self.coef_names)
            return failed_fit(names, str(exc))
        if estfun:
            for message in z.warnings:
                warnings.warn(message, GlmFitWarning, stacklevel=2)
        objfun = -z.twologlik / 2 if self.objective == "loglik" else z.deviance
        scores = None
        if estfun:
            wres = z.fit.residuals * z.fit.weights
            cols = ~np.isnan(z.coefficients)
            scores = np.zeros((xm.shape[0], int(cols.sum())))
            scores[keep] = wres[:, None] * xm[keep][:, cols]
        return FitResult(
            coefficients=z.coefficients,
            coef_names=names,
            objfun=float(objfun),
            estfun=scores,
            converged=z.converged,
            extras={
                "theta": z.theta,
                "theta_se": z.theta_se,
                "theta_irls": z.theta_irls,
                "theta_warning": z.th_warn,
                "converged": z.converged,
                "iter": z.iter,
                "warnings": z.warnings,
            },
        )

    def n_response_columns(self, y: NDArray[np.generic]) -> int:
        return 1
