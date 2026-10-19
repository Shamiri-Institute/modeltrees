"""Parameter instability tests for MOB.

Port of ``mob_grow_fluctests`` and ``root.matrix`` from partykit 1.3-0
(reference/partykit/R/modelparty.R:195-381) for the default ``mob_control()``
settings: ``vcov = "opg"``, ``ordinal = "chisq"``, ``caseweights = TRUE``, no
clusters, ``mtry = Inf``, ``breakties = FALSE`` and ``parm = NULL``. Line
numbers below refer to modelparty.R. docs/instability-tests.md lists what is
ported and what is out of scope.

The statistics are computed by :func:`fluctuation_statistics`, the p-values
by the functions in :mod:`modeltrees.pvalues`, and :func:`fluctuation_tests`
combines them as partykit does.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

from modeltrees.pvalues import (
    Trimming,
    adjust_pvalues,
    logp_chisq,
    logp_suplm,
    suplm_trimming,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


class InstabilityTestError(Exception):
    """The tests failed. partykit catches this with try() and stops splitting."""


class VariableKind(Enum):
    NUMERIC = "numeric"
    NOMINAL = "nominal"  # unordered factor
    ORDERED = "ordered"  # ordered factor


@dataclass(frozen=True)
class PartitioningVariable:
    """One partitioning variable restricted to a node's rows.

    ``values`` holds floats for numeric variables and integer level codes for
    factors, with codes increasing in R's level order.
    """

    name: str
    kind: VariableKind
    values: NDArray[np.generic]


@dataclass(frozen=True)
class FluctuationStatistics:
    """Test statistics before conversion to p-values."""

    statistic: FloatArray  # 0.0 where a variable is not tested, as in R
    # False where R leaves the p-value NA: fewer than two distinct values, or
    # no supLM index inside the trimming window.
    computed: NDArray[np.bool_]
    df: NDArray[np.intp]  # chi-squared df for factors, 0 otherwise
    nobs: float
    k: int
    trimming: Trimming


@dataclass(frozen=True)
class FluctuationTests:
    """Result of mob_grow_fluctests, plus partykit's multiplicity adjustment."""

    variables: tuple[str, ...]
    statistic: FloatArray
    log_p_value: FloatArray  # raw log p-values, NaN where not tested
    p_value: FloatArray  # raw p-values, exp(log_p_value)
    p_value_adjusted: FloatArray  # after adjust_pvalues (mob_grow, :553-557)
    best: int | None  # 0-based index of the smallest raw p-value
    nobs: float
    k: int
    trimming: Trimming


def root_matrix(x: FloatArray) -> FloatArray:
    """Symmetric square root, as ``root.matrix`` in modelparty.R:195-203.

    partykit copies this from strucchange. It errors on any negative
    eigenvalue, however small.
    """
    if x.shape == (1, 1):
        return np.sqrt(x)
    values, vectors = np.linalg.eigh(x)
    if np.any(values < 0):
        raise InstabilityTestError("Matrix is not positive semidefinite")
    return (vectors * np.sqrt(values)) @ vectors.T


def decorrelated_process(
    estfun: FloatArray, weights: FloatArray
) -> tuple[FloatArray, NDArray[np.bool_], float]:
    """Scale and decorrelate the scores, modelparty.R:220-252, vcov "opg".

    Zero-weight rows are dropped from the scores and the weights before
    anything else, so the result equals R's on the data with those rows
    deleted. partykit does not do this consistently (it divides the filtered
    scores by the unfiltered weights, :241, and then fails), which
    docs/decisions.md rules out replicating.

    Returns the process for the rows with positive weight, the mask of those
    rows, and ``nobs``. Assumes ``caseweights = TRUE`` and no clusters.
    """
    process = np.asarray(estfun, dtype=np.float64)
    if process.ndim == 1:
        process = process[:, None]
    keep = weights > 0
    process = process[keep]
    w = weights[keep]
    n = process.shape[0]
    nobs = float(np.sum(w)) if np.any(w != 1) else float(n)

    process = process / math.sqrt(nobs)
    # Weights are already inside the scores, so the meat divides them out.
    scaled = process / np.sqrt(w)[:, None]
    meat = scaled.T @ scaled
    try:
        # chol2inv(chol(meat)). check_finite makes a non-finite meat raise, as
        # R's chol does.
        factor = scipy.linalg.cho_factor(meat, lower=False, check_finite=True)
        meat_inv = scipy.linalg.cho_solve(factor, np.eye(meat.shape[0]))
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise InstabilityTestError(f"chol failed: {exc}") from exc
    j12 = root_matrix(meat_inv)
    return process @ j12.T, keep, nobs


def _chisq_statistic(
    process: FloatArray, codes: IntArray, weights: FloatArray, nobs: float
) -> tuple[float, int]:
    """Factor statistic and df, modelparty.R:310-326 and :344-346.

    ``process``, ``codes`` and ``weights`` all hold the positive-weight rows.
    """
    oi = np.argsort(codes, kind="stable")
    zi = codes[oi]
    proci = process[oi]
    levels, seg = np.unique(zi, return_inverse=True)
    if levels.shape[0] < 2:
        return 0.0, 0
    segweights = np.bincount(seg, weights=weights[oi]) / nobs
    sums = np.zeros((levels.shape[0], process.shape[1]))
    np.add.at(sums, seg, proci)
    stat = float(np.sum(sums**2 / segweights[:, None]))
    return stat, process.shape[1] * (levels.shape[0] - 1)


def _suplm_statistic(
    process: FloatArray,
    values: FloatArray,
    weights: FloatArray,
    nobs: float,
    trimming: Trimming,
    use_weights: bool,
) -> float | None:
    """supLM statistic, modelparty.R:349-367. None when no index is in range."""
    oi = np.argsort(values, kind="stable")  # R's order() is stable
    cumproc = np.cumsum(process[oi], axis=0)
    n = process.shape[0]
    tt0 = (
        np.cumsum(weights[oi]) if use_weights else np.arange(1, n + 1, dtype=np.float64)
    )
    in_range = (tt0 >= trimming.start) & (tt0 <= trimming.end)
    if not np.any(in_range):
        return None
    xx = np.sum(cumproc**2, axis=1)[in_range]
    tt = tt0[in_range] / nobs
    return float(np.max(xx / (tt * (1 - tt))))


def fluctuation_statistics(
    estfun: ArrayLike,
    z: Sequence[PartitioningVariable],
    weights: ArrayLike,
    *,
    minsize: int,
    trim: float = 0.1,
) -> FluctuationStatistics:
    """Instability test statistics for each partitioning variable.

    ``estfun`` is the node's n by k score matrix with weights included and
    ``weights`` the node's case weights (length n). ``minsize`` is the tree's
    minimal node size, which raises the supLM trimming. Rows with zero weight
    are dropped from the scores, weights and partitioning variables before any
    computation, so a node tests like the data with those rows deleted. A
    variable with fewer than two distinct values among the positive-weight
    rows is not tested.
    """
    w = np.asarray(weights, dtype=np.float64)
    process, keep, nobs = decorrelated_process(np.asarray(estfun, dtype=np.float64), w)
    k = process.shape[1]
    trimming = suplm_trimming(nobs, trim, minsize)
    w = w[keep]
    use_weights = bool(np.any(w != 1))

    m = len(z)
    stat = np.zeros(m)
    computed = np.zeros(m, dtype=bool)
    df = np.zeros(m, dtype=np.intp)
    for i, var in enumerate(z):
        zi = np.asarray(var.values)[keep]
        if np.unique(zi).shape[0] < 2:
            continue
        if var.kind is VariableKind.NUMERIC:
            s = _suplm_statistic(
                process, zi.astype(np.float64), w, nobs, trimming, use_weights
            )
            if s is not None:
                stat[i] = s
                computed[i] = True
        else:
            # Ordered factors use the same test under ordinal = "chisq"
            # (modelparty.R:313).
            s_f, df_f = _chisq_statistic(process, zi.astype(np.intp), w, nobs)
            stat[i] = s_f
            computed[i] = df_f > 0
            df[i] = df_f
    return FluctuationStatistics(
        statistic=stat, computed=computed, df=df, nobs=nobs, k=k, trimming=trimming
    )


def statistics_to_log_pvalues(
    stats: FluctuationStatistics, kinds: Sequence[VariableKind]
) -> FloatArray:
    """Raw log p-values for each statistic, NaN where not tested."""
    logp = np.full(stats.statistic.shape[0], np.nan)
    for i, kind in enumerate(kinds):
        if not stats.computed[i]:
            continue
        if kind is VariableKind.NUMERIC:
            logp[i] = logp_suplm(float(stats.statistic[i]), stats.k, stats.trimming.lam)
        else:
            logp[i] = logp_chisq(float(stats.statistic[i]), int(stats.df[i]))
    return logp


def best_variable(log_p_value: FloatArray) -> int | None:
    """R's ``which.min``: NaN ignored, first index on ties (modelparty.R:373)."""
    if np.all(np.isnan(log_p_value)):
        return None
    return int(np.nanargmin(log_p_value))


def fluctuation_tests(
    estfun: ArrayLike,
    z: Sequence[PartitioningVariable],
    weights: ArrayLike,
    *,
    minsize: int,
    trim: float = 0.1,
    bonferroni: bool = True,
) -> FluctuationTests:
    """All parameter instability tests at a node, as partykit runs them.

    Raises InstabilityTestError where partykit's tests raise an error, which
    mob_grow catches to make the node terminal (modelparty.R:550, :569-573).
    """
    stats = fluctuation_statistics(estfun, z, weights, minsize=minsize, trim=trim)
    logp = statistics_to_log_pvalues(stats, [v.kind for v in z])
    p = np.exp(logp)
    return FluctuationTests(
        variables=tuple(v.name for v in z),
        statistic=stats.statistic,
        log_p_value=logp,
        p_value=p,
        p_value_adjusted=adjust_pvalues(p) if bonferroni else p,
        best=best_variable(logp),
        nobs=stats.nobs,
        k=stats.k,
        trimming=stats.trimming,
    )
