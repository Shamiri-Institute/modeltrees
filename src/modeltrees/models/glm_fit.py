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

``etastart`` is ported for glm.nb's alternation (models/negbin.py).
``mustart`` and ``trace`` are not ported.
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
    """An error R's glm.fit raises with ``stop()``, with R's message.

    ``warnings`` holds the warnings glm.fit gave before the error, when the
    caller recorded them.
    """

    warnings: tuple[str, ...] = ()


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


def _warn(messages: list[str], message: str, emit: bool = True) -> None:
    messages.append(message)
    if emit:
        warnings.warn(message, GlmFitWarning, stacklevel=3)


@dataclass
class _Irls:
    """The state glm.fit has after its IRLS loop and final checks."""

    coef: FloatArray
    fit: QRLeastSquares | None
    eta: FloatArray
    mu: FloatArray
    dev: float
    w: FloatArray
    good: NDArray[np.bool_]
    residuals: FloatArray | None  # None when skipped (objective only)
    conv: bool
    boundary: bool
    iteration: int
    y: FloatArray
    n_trials: FloatArray


def _prepare(
    x: ArrayLike,
    y: ArrayLike,
    weights: ArrayLike | None,
    offset: ArrayLike | None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    xm = np.asarray(x, dtype=np.float64)
    if xm.ndim == 1:
        xm = xm[:, None]
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    nobs = yv.shape[0]
    if xm.shape[0] != nobs:
        raise ValueError(f"x has {xm.shape[0]} rows but y has {nobs} values")
    prior = (
        np.ones(nobs)
        if weights is None
        else np.asarray(weights, dtype=np.float64).copy()
    )
    off = np.zeros(nobs) if offset is None else np.asarray(offset, dtype=np.float64)
    if prior.shape != (nobs,) or off.shape != (nobs,):
        raise ValueError("weights and offset need one value per observation")
    return xm, yv, prior, off


def _irls(
    xm: FloatArray,
    yv: FloatArray,
    family: Family,
    prior: FloatArray,
    start: ArrayLike | None,
    off: FloatArray,
    epsilon: float,
    maxit: int,
    messages: list[str] | None,
    etastart: ArrayLike | None = None,
    emit: bool = True,
) -> _Irls:
    """glm.fit up to its final warnings. ``messages = None`` is quiet mode:
    no warnings and no working residuals, with the same arithmetic otherwise.
    ``emit = False`` records the warnings in ``messages`` without emitting
    them.
    """
    quiet = messages is None
    nobs = yv.shape[0]
    nvars = xm.shape[1]
    empty = nvars == 0

    if quiet:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GlmFitWarning)
            init = family.initialize(yv, prior)
    else:
        assert messages is not None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", GlmFitWarning)
            init = family.initialize(yv, prior)
        for w_ in caught:
            _warn(messages, str(w_.message), emit)
    yv, n_trials, mustart = init.y, init.n, init.mustart

    def warn(message: str) -> None:
        if messages is not None:
            _warn(messages, message, emit)

    linkinv, mu_eta, variance = family.linkinv, family.mu_eta, family.variance

    def deviance(mu: FloatArray) -> float:
        return float(np.sum(family.dev_resids(yv, mu, prior)))

    good = np.ones(nobs, dtype=bool)
    w = np.zeros(0)
    fit: QRLeastSquares | None = None
    coef: FloatArray | None = None
    residuals: FloatArray | None = None
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
        if etastart is not None:
            # R: eta <- etastart %||% {...}; start is then neither checked
            # nor used as coefold, only as the base of the coefficient update.
            eta = np.asarray(etastart, dtype=np.float64).reshape(-1).copy()
            if eta.shape[0] != nobs:
                raise ValueError("etastart needs one value per observation")
            if start is not None:
                coef_cur = np.asarray(start, dtype=np.float64).reshape(-1).copy()
        elif start is not None:
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
        positive = prior > 0
        all_positive = bool(np.all(positive))
        for iteration in range(1, maxit + 1):
            good = positive
            var_all = variance(mu)
            varmu = var_all if all_positive else var_all[good]
            if np.any(np.isnan(varmu)):
                raise GlmFitError("NAs in V(mu)")
            if np.any(varmu == 0):
                raise GlmFitError("0s in V(mu)")
            mu_eta_val = mu_eta(eta)
            if np.any(np.isnan(mu_eta_val if all_positive else mu_eta_val[good])):
                raise GlmFitError("NAs in d(mu)/d(eta)")
            good = positive & (mu_eta_val != 0)
            if not np.any(good):
                conv = False
                warn(f"no observations informative at iteration {iteration}")
                break
            # Indexing with an all-true mask copies without changing a value,
            # so it is skipped when every row is good.
            if bool(np.all(good)):
                z = (eta - off) + (yv - mu) / mu_eta_val
                w = np.sqrt((prior * mu_eta_val**2) / var_all)
                fit = dqrls(xm * w[:, None], z * w, tol)
            else:
                z = (eta - off)[good] + (yv - mu)[good] / mu_eta_val[good]
                w = np.sqrt((prior[good] * mu_eta_val[good] ** 2) / var_all[good])
                fit = dqrls(xm[good] * w[:, None], z * w, tol)
            if not np.all(np.isfinite(fit.coefficients)):
                conv = False
                warn(f"non-finite coefficients at iteration {iteration}")
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
                warn("step size truncated due to divergence")
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
                warn("step size truncated: out of bounds")
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
        if not quiet:
            if not conv:
                warn("glm.fit: algorithm did not converge")
            if boundary:
                warn("glm.fit: algorithm stopped at boundary value")
            eps = 10 * DBL_EPSILON
            if family.family == "binomial" and (
                np.any(mu > 1 - eps) or np.any(mu < eps)
            ):
                warn("glm.fit: fitted probabilities numerically 0 or 1 occurred")
            if family.family == "poisson" and np.any(mu < eps):
                warn("glm.fit: fitted rates numerically 0 occurred")
        coef = coef.copy()
        if fit.rank < nvars:
            coef[fit.pivot[fit.rank :]] = np.nan
        if not quiet:
            residuals = (yv - mu) / mu_eta(eta)

    if coef is None:
        raise GlmFitError("no coefficients")  # unreachable; narrows the type
    return _Irls(
        coef,
        fit,
        eta,
        mu,
        dev,
        w,
        good,
        residuals,
        conv,
        boundary,
        iteration,
        yv,
        n_trials,
    )


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
    etastart: ArrayLike | None = None,
    emit_warnings: bool = True,
) -> GlmFit:
    """Fit a GLM by IRLS exactly as ``glm.fit`` does.

    ``emit_warnings = False`` keeps R's warnings in ``warnings`` without
    emitting them, as ``withCallingHandlers`` with ``muffleWarning`` does.
    """
    xm, yv, prior, off = _prepare(x, y, weights, offset)
    nobs = yv.shape[0]
    nvars = xm.shape[1]
    messages: list[str] = []
    try:
        r = _irls(
            xm,
            yv,
            family,
            prior,
            start,
            off,
            epsilon,
            maxit,
            messages,
            etastart,
            emit_warnings,
        )
    except GlmFitError as exc:
        exc.warnings = tuple(messages)
        raise
    yv, mu, fit = r.y, r.mu, r.fit
    assert r.residuals is not None
    linkinv = family.linkinv
    wt = np.zeros(nobs)
    wt[r.good] = r.w**2
    if intercept:
        wtdmu: float | FloatArray = float(np.sum(prior * yv) / np.sum(prior))
    else:
        wtdmu = linkinv(off)
    nulldev = float(
        np.sum(family.dev_resids(yv, np.broadcast_to(wtdmu, yv.shape), prior))
    )
    n_ok = nobs - int(np.sum(prior == 0))
    nulldf = n_ok - int(intercept)
    rank = 0 if nvars == 0 or fit is None else fit.rank
    resdf = n_ok - rank
    aic_model = family.aic(yv, r.n_trials, mu, prior, r.dev) + 2 * rank
    return GlmFit(
        coefficients=r.coef,
        residuals=r.residuals,
        fitted_values=mu,
        linear_predictors=r.eta,
        deviance=float(r.dev),
        aic=float(aic_model),
        null_deviance=nulldev,
        iter=r.iteration,
        weights=wt,
        prior_weights=prior,
        df_residual=resdf,
        df_null=nulldf,
        y=yv,
        converged=r.conv,
        boundary=r.boundary,
        rank=rank,
        pivot=np.arange(nvars, dtype=np.intp) if fit is None else fit.pivot,
        family=family,
        warnings=tuple(messages),
    )


@dataclass(frozen=True)
class GlmObjective:
    """What split search needs from glm.fit: the fit's AIC and rank."""

    coefficients: FloatArray  # NaN for aliased columns
    aic: float
    rank: int
    converged: bool
    fitted_values: FloatArray
    boundary: bool
    warnings: tuple[str, ...]  # empty unless record_warnings


def glm_objective(
    x: ArrayLike,
    y: ArrayLike,
    family: Family,
    *,
    weights: ArrayLike | None = None,
    start: ArrayLike | None = None,
    offset: ArrayLike | None = None,
    epsilon: float = 1e-8,
    maxit: int = 25,
    etastart: ArrayLike | None = None,
    record_warnings: bool = False,
) -> GlmObjective:
    """``glm_fit`` without warnings, null deviance, residuals and weights.

    The coefficients, AIC, rank and fitted values are computed by the same
    arithmetic as in :func:`glm_fit`, so they are identical to its result.
    R's errors still raise :class:`GlmFitError`. ``record_warnings = True``
    collects R's warnings in the result without emitting them (and computes
    the working residuals, which are then discarded).
    """
    xm, yv, prior, off = _prepare(x, y, weights, offset)
    nvars = xm.shape[1]
    messages: list[str] | None = [] if record_warnings else None
    try:
        r = _irls(
            xm,
            yv,
            family,
            prior,
            start,
            off,
            epsilon,
            maxit,
            messages,
            etastart,
            emit=False,
        )
    except GlmFitError as exc:
        exc.warnings = () if messages is None else tuple(messages)
        raise
    rank = 0 if nvars == 0 or r.fit is None else r.fit.rank
    aic_model = family.aic(r.y, r.n_trials, r.mu, prior, r.dev) + 2 * rank
    return GlmObjective(
        r.coef,
        float(aic_model),
        rank,
        r.conv,
        r.mu,
        r.boundary,
        () if messages is None else tuple(messages),
    )
