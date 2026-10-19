"""Port of R's ``stats::glm.fit`` (R 4.6.1) for the families in
:mod:`modeltrees.models.families`.

The IRLS loop follows R line by line: starting values from
``family$initialize`` (or ``start``), a weighted least-squares step through
R's pivoting QR (:mod:`modeltrees.models._qr`) with tolerance
``min(1e-7, epsilon / 1000)``, deviance-based convergence
``|dev - devold| / (0.1 + |dev|) < epsilon``, step halving towards the
previous coefficients when the deviance is not finite or the fitted values
are invalid, and NaN for aliased coefficients. R's warnings are emitted as
:class:`GlmFitWarning` with R's messages and recorded in the result. R's
errors (``stop()``) raise :class:`GlmFitError`.

``etastart``, ``mustart`` and ``trace`` are not ported.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from modeltrees.models._qr import QRLeastSquares, dqrls
from modeltrees.models.families import Family, GlmFitWarning

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

DBL_EPSILON = float(np.finfo(np.float64).eps)


class GlmFitError(ValueError):
    """An error R's glm.fit raises with ``stop()``, with R's message."""


@dataclass(frozen=True)
class GlmFit:
    """The components of glm.fit's return value, under R's names."""

    coefficients: FloatArray  # NaN for aliased columns
    residuals: FloatArray  # working residuals (y - mu) / mu.eta(eta)
    fitted_values: FloatArray
    linear_predictors: FloatArray
    deviance: float
    aic: float
    null_deviance: float
    iter: int
    weights: FloatArray  # working weights from the last iteration, 0 if not good
    prior_weights: FloatArray
    df_residual: int
    df_null: int
    y: FloatArray  # after family initialize
    converged: bool
    boundary: bool
    rank: int
    pivot: IntArray  # 0-based
    family: Family
    warnings: tuple[str, ...]


def _warn(messages: list[str], message: str) -> None:
    messages.append(message)
    warnings.warn(message, GlmFitWarning, stacklevel=3)


def glm_fit(
    x: ArrayLike,
    y: ArrayLike,
    family: Family,
    *,
    weights: ArrayLike | None = None,
    start: ArrayLike | None = None,
    offset: ArrayLike | None = None,
    epsilon: float = 1e-8,
    maxit: int = 25,
    intercept: bool = True,
) -> GlmFit:
    """Fit a GLM by IRLS exactly as ``glm.fit`` does."""
    xm = np.asarray(x, dtype=np.float64)
    if xm.ndim == 1:
        xm = xm[:, None]
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    nobs = yv.shape[0]
    nvars = xm.shape[1]
    if xm.shape[0] != nobs:
        raise ValueError(f"x has {xm.shape[0]} rows but y has {nobs} values")
    empty = nvars == 0
    prior = (
        np.ones(nobs)
        if weights is None
        else np.asarray(weights, dtype=np.float64).copy()
    )
    off = np.zeros(nobs) if offset is None else np.asarray(offset, dtype=np.float64)
    if prior.shape != (nobs,) or off.shape != (nobs,):
        raise ValueError("weights and offset need one value per observation")
    messages: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", GlmFitWarning)
        init = family.initialize(yv, prior)
    for w_ in caught:
        _warn(messages, str(w_.message))
    yv, n_trials, mustart = init.y, init.n, init.mustart

    linkinv, mu_eta, variance = family.linkinv, family.mu_eta, family.variance

    def deviance(mu: FloatArray) -> float:
        return float(np.sum(family.dev_resids(yv, mu, prior)))

    good = np.ones(nobs, dtype=bool)
    w = np.zeros(0)
    fit: QRLeastSquares | None = None
    coef: FloatArray | None = None
    if empty:
        eta = np.zeros(nobs) + off
        if not family.valideta(eta):
            raise GlmFitError("invalid linear predictor values in empty model")
        mu = linkinv(eta)
        if not family.validmu(mu):
            raise GlmFitError("invalid fitted means in empty model")
        dev = deviance(mu)
        w = np.sqrt((prior * mu_eta(eta) ** 2) / variance(mu))
        residuals = (yv - mu) / mu_eta(eta)
        good = np.ones(residuals.shape[0], dtype=bool)
        boundary = conv = True
        coef = np.zeros(0)
        iteration = 0
    else:
        coefold: FloatArray | None = None
        coef_cur: FloatArray | None = None
        if start is not None:
            start_v = np.asarray(start, dtype=np.float64).reshape(-1)
            if start_v.shape[0] != nvars:
                raise GlmFitError(
                    f"length of 'start' should equal {nvars} and correspond to "
                    "initial coefs"
                )
            coefold = start_v.copy()
            coef_cur = start_v.copy()
            eta = off + xm @ start_v
        else:
            eta = family.linkfun(mustart)
        mu = linkinv(eta)
        if not (family.validmu(mu) and family.valideta(eta)):
            raise GlmFitError("cannot find valid starting values: please specify some")
        devold = deviance(mu)
        boundary = conv = False
        tol = min(1e-7, epsilon / 1000)
        iteration = 0
        for iteration in range(1, maxit + 1):
            good = prior > 0
            varmu = variance(mu)[good]
            if np.any(np.isnan(varmu)):
                raise GlmFitError("NAs in V(mu)")
            if np.any(varmu == 0):
                raise GlmFitError("0s in V(mu)")
            mu_eta_val = mu_eta(eta)
            if np.any(np.isnan(mu_eta_val[good])):
                raise GlmFitError("NAs in d(mu)/d(eta)")
            good = (prior > 0) & (mu_eta_val != 0)
            if not np.any(good):
                conv = False
                _warn(messages, f"no observations informative at iteration {iteration}")
                break
            z = (eta - off)[good] + (yv - mu)[good] / mu_eta_val[good]
            w = np.sqrt((prior[good] * mu_eta_val[good] ** 2) / variance(mu)[good])
            fit = dqrls(xm[good] * w[:, None], z * w, tol)
            if not np.all(np.isfinite(fit.coefficients)):
                conv = False
                _warn(messages, f"non-finite coefficients at iteration {iteration}")
                break
            if nobs < fit.rank:
                raise GlmFitError(
                    f"X matrix has rank {fit.rank}, but only {nobs} observations"
                )
            # R: start[fit$pivot] <- fit$coefficients
            new = np.zeros(nvars) if coef_cur is None else coef_cur.copy()
            new[fit.pivot] = fit.coefficients
            coef_cur = new
            eta = xm @ coef_cur + off
            mu = linkinv(eta)
            dev = deviance(mu)
            boundary = False
            if not np.isfinite(dev):
                if coefold is None:
                    raise GlmFitError(
                        "no valid set of coefficients has been found: please "
                        "supply starting values"
                    )
                _warn(messages, "step size truncated due to divergence")
                ii = 1
                while not np.isfinite(dev):
                    if ii > maxit:
                        raise GlmFitError("inner loop 1; cannot correct step size")
                    ii += 1
                    coef_cur = (coef_cur + coefold) / 2
                    eta = xm @ coef_cur + off
                    mu = linkinv(eta)
                    dev = deviance(mu)
                boundary = True
            if not (family.valideta(eta) and family.validmu(mu)):
                if coefold is None:
                    raise GlmFitError(
                        "no valid set of coefficients has been found: please "
                        "supply starting values"
                    )
                _warn(messages, "step size truncated: out of bounds")
                ii = 1
                while not (family.valideta(eta) and family.validmu(mu)):
                    if ii > maxit:
                        raise GlmFitError("inner loop 2; cannot correct step size")
                    ii += 1
                    coef_cur = (coef_cur + coefold) / 2
                    eta = xm @ coef_cur + off
                    mu = linkinv(eta)
                boundary = True
                dev = deviance(mu)
            if abs(dev - devold) / (0.1 + abs(dev)) < epsilon:
                conv = True
                coef = coef_cur
                break
            devold = dev
            coef = coefold = coef_cur
        if coef is None or fit is None:
            # R fails here with "object 'coef' not found".
            raise GlmFitError("no coefficients: the first iteration failed")
        if not conv:
            _warn(messages, "glm.fit: algorithm did not converge")
        if boundary:
            _warn(messages, "glm.fit: algorithm stopped at boundary value")
        eps = 10 * DBL_EPSILON
        if family.family == "binomial" and (np.any(mu > 1 - eps) or np.any(mu < eps)):
            _warn(messages, "glm.fit: fitted probabilities numerically 0 or 1 occurred")
        if family.family == "poisson" and np.any(mu < eps):
            _warn(messages, "glm.fit: fitted rates numerically 0 occurred")
        coef = coef.copy()
        if fit.rank < nvars:
            coef[fit.pivot[fit.rank :]] = np.nan
        residuals = (yv - mu) / mu_eta(eta)

    if coef is None:
        raise GlmFitError("no coefficients")  # unreachable; narrows the type
    wt = np.zeros(nobs)
    wt[good] = w**2
    if intercept:
        wtdmu: float | FloatArray = float(np.sum(prior * yv) / np.sum(prior))
    else:
        wtdmu = linkinv(off)
    nulldev = float(
        np.sum(family.dev_resids(yv, np.broadcast_to(wtdmu, yv.shape), prior))
    )
    n_ok = nobs - int(np.sum(prior == 0))
    nulldf = n_ok - int(intercept)
    rank = 0 if empty or fit is None else fit.rank
    resdf = n_ok - rank
    aic_model = family.aic(yv, n_trials, mu, prior, dev) + 2 * rank
    return GlmFit(
        coefficients=coef,
        residuals=residuals,
        fitted_values=mu,
        linear_predictors=eta,
        deviance=float(dev),
        aic=float(aic_model),
        null_deviance=nulldev,
        iter=iteration,
        weights=wt,
        prior_weights=prior,
        df_residual=resdf,
        df_null=nulldf,
        y=yv,
        converged=conv,
        boundary=boundary,
        rank=rank,
        pivot=np.arange(nvars, dtype=np.intp) if fit is None else fit.pivot,
        family=family,
        warnings=tuple(messages),
    )
