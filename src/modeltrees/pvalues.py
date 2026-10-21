# Ported from partykit 1.3-0 (GPL-2 | GPL-3), R/modelparty.R (logp.supLM, p-value
# adjustment).
# See THIRD_PARTY_NOTICES.md.
"""P-values for the MOB parameter instability tests.

Ports the p-value pieces of partykit 1.3-0 ``mob_grow_fluctests`` and
``mob_grow`` (reference/partykit/R/modelparty.R). Line numbers below refer to
that file. See docs/instability-tests.md for the full map.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

from modeltrees._hansen_table import MOB_BETA_SUPLM

FloatArray = NDArray[np.float64]

# Rows per k in the Hansen table and the largest k it covers.
_HANSEN_ROWS_PER_K = 25
_HANSEN_MAX_K = 40

# Below this, log(gammaincc) loses accuracy or underflows, so the continued
# fraction is evaluated in log space instead.
_TINY_TAIL = 1e-280


def log_chisq_sf(x: float, df: float) -> float:
    """log P(X > x) for X ~ chi-squared(df).

    Equivalent to R's ``pchisq(x, df, lower.tail = FALSE, log.p = TRUE)``,
    including for tails far below the smallest double, where R stays finite.
    """
    if math.isnan(x) or math.isnan(df):
        return math.nan
    if x <= 0.0:
        return 0.0
    if math.isinf(x):
        return -math.inf
    a = 0.5 * df
    h = 0.5 * x
    lower = float(special.gammainc(a, h))
    if lower < 0.5:
        return math.log1p(-lower)
    upper = float(special.gammaincc(a, h))
    if upper > _TINY_TAIL:
        return math.log(upper)
    return _log_gammaincc_cf(a, h)


def _log_gammaincc_cf(a: float, x: float) -> float:
    """log Q(a, x) from the continued fraction, valid for x > a + 1.

    Modified Lentz evaluation as in Numerical Recipes ``gcf``. Only reached
    when Q(a, x) < 1e-280, which implies x is far above a + 1.
    """
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    hh = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        hh *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return -x + a * math.log(x) - math.lgamma(a) + math.log(hh)


@dataclass(frozen=True)
class Trimming:
    """Trimming quantities for supLM, as in modelparty.R:262-265."""

    start: float  # R's `from`: first admissible value of the time index
    end: float  # R's `to`: last admissible value of the time index
    lam: float  # R's `lambda`, the ratio passed to logp.supLM


def suplm_trimming(nobs: float, trim: float, minsize: int) -> Trimming:
    """Trimming for the supLM statistic (modelparty.R:262-265).

    ``trim`` above 1 is an absolute number of observations, otherwise a
    fraction of ``nobs``. ``minsize`` is the tree's minimal node size; partykit
    raises the trimming to at least that.
    """
    start = float(trim) if trim > 1 else float(math.ceil(nobs * trim))
    start = max(start, float(minsize))
    end = nobs - start
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = float(np.float64((nobs - start) * end) / np.float64(start * (nobs - end)))
    return Trimming(start=start, end=end, lam=lam)


def suplm_tau(lam: float) -> float:
    """Map lambda to Hansen's trimming parameter tau (modelparty.R:281)."""
    return lam if lam < 1 else 1.0 / (1.0 + math.sqrt(lam))


def _log_add_naive(
    log_w1: float, lp1: float, log_w2: float, lp2: float, scale: float = 1.0
) -> float:
    """R's ``log((exp(log_w1 + lp1) + exp(log_w2 + lp2)) * scale)``.

    partykit computes this naively, without log-sum-exp, and falls back to a
    plain average when the result is not finite, so the underflow must be
    reproduced, not avoided.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        e1 = np.exp(np.float64(log_w1) + lp1)
        e2 = np.exp(np.float64(log_w2) + lp2)
        return float(np.log((e1 + e2) * scale))


def _safe_log(x: float) -> float:
    """R's ``log``: log(0) is -Inf instead of an error."""
    return math.log(x) if x > 0 else (-math.inf if x == 0 else math.nan)


def _logp_estrella(x: float, k: int, lam: float) -> float:
    """Estrella (2003) approximation, modelparty.R:272-273."""
    return (
        -math.lgamma(k / 2)
        + k / 2 * _safe_log(x / 2)
        - x / 2
        + _safe_log(abs(_safe_log(lam) * (1 - k / x) + 2 / x))
    )


def logp_suplm(statistic: float, k: int, lam: float) -> float:
    """Log p-value of the supLM statistic.

    Port of the closure ``logp.supLM`` in partykit 1.3-0
    ``mob_grow_fluctests`` (reference/partykit/R/modelparty.R:268-303).
    For k <= 40 it interpolates Hansen's (1997) response surface,
    ``mob_beta_suplm`` (R/mob-pvalue.R), in tau. For k > 40 it uses the
    Estrella (2003) approximation with partykit's ad hoc power interpolation
    below 1.5 * k. The result is not clipped to (-Inf, 0], as in R.
    """
    x = float(statistic)
    if k > _HANSEN_MAX_K:
        # modelparty.R:277. R's ifelse() evaluates both branches; only the
        # selected one matters.
        if x <= 1.5 * k:
            with np.errstate(invalid="ignore"):
                ratio = float(np.float64(x / (1.5 * k)) ** math.sqrt(k))
            return ratio * _logp_estrella(1.5 * k, k, lam)
        return _logp_estrella(x, k, lam)

    # modelparty.R:280-285
    if math.isnan(lam):
        # R fails at `if(lambda < 1)` with a missing value (trim and minsize 0).
        raise ValueError("supLM lambda is NaN")
    tau = suplm_tau(lam)
    beta = MOB_BETA_SUPLM[(k - 1) * _HANSEN_ROWS_PER_K : k * _HANSEN_ROWS_PER_K]
    dummy = beta[:, 0] + beta[:, 1] * x
    dummy = dummy * (dummy > 0)
    pp = [log_chisq_sf(float(q), float(df)) for q, df in zip(dummy, beta[:, 2])]

    # modelparty.R:286-300. pp is 1-based in R; pp[i] in R is pp[i - 1] here.
    if tau == 0.5:
        return log_chisq_sf(x, k)
    if tau <= 0.01:
        return pp[24]
    if tau >= 0.49:
        p = _log_add_naive(
            _safe_log(0.5 - tau),
            pp[0],
            _safe_log(tau - 0.49),
            log_chisq_sf(x, k),
            100.0,
        )
        if not math.isfinite(p):
            p = (pp[0] + log_chisq_sf(x, k)) / 2
        return p
    taua = (0.51 - tau) * 50
    tau1 = math.floor(taua)
    p = _log_add_naive(
        _safe_log(tau1 + 1 - taua), pp[tau1 - 1], _safe_log(taua - tau1), pp[tau1]
    )
    if not math.isfinite(p):
        p = (pp[tau1 - 1] + pp[tau1]) / 2
    return p


def logp_chisq(statistic: float, df: int) -> float:
    """Log p-value of the factor test, modelparty.R:346."""
    return log_chisq_sf(float(statistic), float(df))


def adjust_pvalues(p_raw: ArrayLike) -> FloatArray:
    """partykit's Bonferroni option (mob_grow, modelparty.R:553-557).

    With m the number of non-NaN p-values, returns ``1 - (1 - p)^m`` (Sidak)
    where p > 0.001 and ``min(1, m * p)`` (Bonferroni) otherwise. NaN marks an
    untested variable and is kept.
    """
    p = np.asarray(p_raw, dtype=np.float64)
    m = int(np.sum(~np.isnan(p)))
    bonferroni = np.minimum(1.0, m * p)
    sidak = 1.0 - (1.0 - p) ** m
    with np.errstate(invalid="ignore"):
        use_sidak = ~np.isnan(p) & (p > 0.001)
    return np.where(use_sidak, sidak, bonferroni)
