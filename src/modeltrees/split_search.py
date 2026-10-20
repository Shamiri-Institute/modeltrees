"""Split search in the selected partitioning variable.

Port of ``mob_grow_findsplit`` and ``mob_grow_getlevels`` from partykit 1.3-0
(reference/partykit/R/modelparty.R:384-511 and :701-723) for binary splits.
Line numbers below refer to modelparty.R. The search is model-agnostic: the
caller passes a function that fits the model on a subset of the node's rows.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from modeltrees.instability import VariableKind
from modeltrees.node_model import FitResult

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]

# Fits the model on the node rows selected by the mask, from the given start.
SideFit = Callable[[BoolArray, FloatArray | None], FitResult]


@dataclass(frozen=True)
class SplitObjectives:
    """Approximate objectives of numeric split candidates, from a model.

    ``left[i] + right[i]`` approximates the summed objective of candidate
    ``i`` and ``error[i]`` bounds its absolute error. A NaN in any of the
    three marks a candidate the model cannot score reliably (for example a
    rank-deficient side); split search fits it exactly.
    """

    left: FloatArray
    right: FloatArray
    error: FloatArray


# Scores numeric candidates at once: ``order`` sorts the node's rows by the
# variable (stable) and candidate i sends the rows order[:cuts[i]] left. None
# means the model declines and every candidate is fitted.
BatchObjectives = Callable[[IntArray, IntArray], SplitObjectives | None]

# Candidates refitted exactly after screening: every one that could be within
# REFIT_MARGIN (relative) of the best given the error bounds, and at least
# REFIT_MIN of the smallest approximations.
REFIT_MARGIN = 1e-6
REFIT_MIN = 5


@dataclass(frozen=True)
class Split:
    """A binary split, as partykit's ``partysplit`` with ``right = TRUE``.

    Numeric: ``z <= breaks`` goes to kid 1. Ordered: levels at 1-based
    positions up to ``breaks`` go to kid 1. Nominal: ``index`` maps each
    original level to kid 1 or 2, or None for levels absent from the node.
    """

    variable: int  # 0-based index into the partitioning variables
    kind: VariableKind
    breaks: float | None
    index: tuple[int | None, ...] | None


@dataclass(frozen=True)
class SplitSearch:
    """Every candidate split of one variable and its objective."""

    variable: int
    kind: VariableKind
    minsize: float  # minsize after the local reinterpretation (:386-388)
    # Numeric: the sorted unique values, candidate i sends z <= values[i] left.
    values: FloatArray | None
    # Factors: per candidate, the level codes sent left, and R's row label.
    left_levels: tuple[tuple[int, ...], ...] | None
    labels: tuple[str, ...] | None
    objective: FloatArray  # inf where a side has fewer than minsize
    best: int | None  # first minimum, None if no candidate is admissible
    split: Split | None


def w2n(weights: FloatArray, caseweights: bool) -> float:
    """Number of observations: the weight sum, or the positive count (:176)."""
    return float(np.sum(weights)) if caseweights else float(np.sum(weights > 0))


def getlevels(n_levels: int, ordered: bool) -> BoolArray:
    """Candidate partitions of a factor's levels, as ``mob_grow_getlevels``.

    Each row is a candidate and marks the levels sent to kid 1. Ordered:
    the first ``i`` levels, for ``i < n_levels``. Nominal: the
    ``2^(n_levels - 1) - 1`` subsets given by the bits of the row number, so
    the last level always goes right (:709-718).
    """
    if ordered:
        return np.tril(np.ones((n_levels, n_levels), dtype=bool))[:-1]
    rows = 2 ** (n_levels - 1) - 1
    bits = np.arange(1, rows + 1)[:, None] >> np.arange(n_levels)[None, :]
    return (bits % 2).astype(bool)


def _which_min(values: FloatArray) -> int:
    """R's ``which.min``: NaN ignored, first index on ties."""
    return int(np.nanargmin(values))


def _effective_minsize(minsize: float, nobs: float) -> float:
    # :386-388. minsize reaches here as an integer, so only 0 changes.
    if 0.5 < minsize < 1:
        minsize = 1 - minsize
    if minsize < 0.5:
        minsize = math.ceil(nobs * minsize)
    return minsize


def find_split_reference(
    variable: int,
    kind: VariableKind,
    z: NDArray[np.generic],
    level_names: Sequence[str] | None,
    weights: FloatArray,
    fit: SideFit,
    *,
    minsize: float,
    caseweights: bool = True,
    restart: bool = True,
    numsplit: str = "left",
) -> SplitSearch:
    """Search the best binary split of the node in one variable.

    This is the exact reference implementation: every admissible candidate
    is fitted on both sides, with ``w2n`` summed over each side's mask.
    :func:`find_split` gives the same result faster and is compared with
    this function in the tests (docs/decisions.md).

    ``z`` holds the node's values of the variable: floats for numeric
    variables, 0-based level codes for factors. ``level_names`` are the
    original factor levels (None for numeric). Candidate values and levels
    come from rows with positive weight only (docs/decisions.md); rows with
    zero weight ride along in the masks passed to ``fit`` but cannot create
    a candidate.
    """
    minsize = _effective_minsize(minsize, w2n(weights, caseweights))
    positive = weights > 0

    def admissible(left: BoolArray) -> bool:
        return not (
            w2n(weights[left], caseweights) < minsize
            or w2n(weights[~left], caseweights) < minsize
        )

    if kind is VariableKind.NUMERIC:
        zv = np.asarray(z, dtype=np.float64)
        uz = np.unique(zv[positive])
        if uz.shape[0] == 0:
            raise ValueError(
                "Cannot find admissible split point in partitioning variable"
            )
        dev = np.full(uz.shape[0], np.inf)
        start_left: FloatArray | None = None
        start_right: FloatArray | None = None
        for i, value in enumerate(uz):
            left = zv <= value
            if not admissible(left):
                continue
            # With restart = FALSE the previous admissible candidate's
            # coefficients are the starting values (:417-438). The fits here
            # always have the same names and length, so R's reset never fires.
            fit_left = fit(left, None if restart else start_left)
            fit_right = fit(~left, None if restart else start_right)
            start_left, start_right = fit_left.coefficients, fit_right.coefficients
            dev[i] = fit_left.objfun + fit_right.objfun
        if not np.any(np.isfinite(dev)):
            return SplitSearch(variable, kind, minsize, uz, None, None, dev, None, None)
        best = _which_min(dev)
        if numsplit == "center":
            # mean(uz[best + 0:1]); NaN past the last value, as R's NA.
            upper = uz[best + 1] if best + 1 < uz.shape[0] else np.nan
            breaks = float((uz[best] + upper) / 2)
        else:
            breaks = float(uz[best])
        split = Split(variable, kind, breaks, None)
        return SplitSearch(variable, kind, minsize, uz, None, None, dev, best, split)

    if level_names is None:
        raise ValueError("a factor needs its level names")
    codes = np.asarray(z, dtype=np.intp)
    # factor(zselect) drops unused levels (:466); here also levels that only
    # zero-weight rows have.
    present = np.unique(codes[positive])
    al = getlevels(present.shape[0], kind is VariableKind.ORDERED)
    if kind is VariableKind.ORDERED:
        labels = tuple(level_names[c] for c in present[:-1])
    else:
        labels = tuple("+".join(level_names[c] for c in present[row]) for row in al)
    left_levels = tuple(tuple(int(c) for c in present[row]) for row in al)
    dev = np.full(al.shape[0], np.inf)
    for i, row in enumerate(al):
        left = np.isin(codes, present[row])
        if not admissible(left):
            continue
        if al.shape[0] == 1:
            # R quirk, replicated: with a single candidate partition the
            # objective is set to 1 without fitting (:475).
            dev[i] = 1.0
            continue
        # partykit passes cluster[zs] to the right-hand fit (:479), a bug that
        # docs/decisions.md says not to replicate; fit() always gets the
        # complement's rows, cluster included.
        dev[i] = fit(left, None).objfun + fit(~left, None).objfun
    if not np.any(np.isfinite(dev)):
        return SplitSearch(
            variable, kind, minsize, None, left_levels, labels, dev, None, None
        )
    best = _which_min(dev)
    if kind is VariableKind.ORDERED:
        # Position of the chosen level among the original levels (:492-497).
        split = Split(variable, kind, float(present[best] + 1), None)
    else:
        index: list[int | None] = [None] * len(level_names)
        for j, c in enumerate(present):
            index[c] = 1 if al[best, j] else 2
        split = Split(variable, kind, None, tuple(index))
    return SplitSearch(
        variable, kind, minsize, None, left_levels, labels, dev, best, split
    )


def find_split(
    variable: int,
    kind: VariableKind,
    z: NDArray[np.generic],
    level_names: Sequence[str] | None,
    weights: FloatArray,
    fit: SideFit,
    *,
    minsize: float,
    caseweights: bool = True,
    restart: bool = True,
    numsplit: str = "left",
    batch: BatchObjectives | None = None,
) -> SplitSearch:
    """Search the best binary split, as :func:`find_split_reference`, faster.

    Factors have few candidates and use the reference code. For a numeric
    variable, admissibility is computed for all candidates at once from
    cumulative weights. Candidates within rounding of ``minsize`` are
    rechecked with the reference's masked sums, so the admissible set is
    the reference's.

    With ``batch`` (a model's ``split_objectives``) and ``restart = True``,
    every admissible candidate is first scored approximately. Candidates
    that could be the best given the error bounds, within ``REFIT_MARGIN``,
    and at least the ``REFIT_MIN`` smallest, are refitted with ``fit``, as
    are candidates the model could not score. The winner is the first
    minimum of those exact values, which no unrefitted candidate can reach,
    so the chosen split is the reference's. ``objective`` holds exact values
    for refitted candidates and the approximations for the others.
    """
    if kind is not VariableKind.NUMERIC:
        return find_split_reference(
            variable,
            kind,
            z,
            level_names,
            weights,
            fit,
            minsize=minsize,
            caseweights=caseweights,
            restart=restart,
            numsplit=numsplit,
        )
    minsize = _effective_minsize(minsize, w2n(weights, caseweights))
    zv = np.asarray(z, dtype=np.float64)
    positive = weights > 0
    uz, inv = np.unique(zv[positive], return_inverse=True)
    if uz.shape[0] == 0:
        raise ValueError("Cannot find admissible split point in partitioning variable")
    candidates = np.flatnonzero(
        _admissible_numeric(zv, uz, inv, weights, minsize, caseweights)
    )
    dev = np.full(uz.shape[0], np.inf)

    def exact(
        i: int, start: tuple[FloatArray | None, FloatArray | None]
    ) -> tuple[FitResult, FitResult]:
        left = zv <= uz[i]
        return fit(left, start[0]), fit(~left, start[1])

    scores = None
    if batch is not None and restart and candidates.shape[0] > 0:
        order = np.argsort(zv, kind="stable")
        cuts = np.searchsorted(zv[order], uz[candidates], side="right")
        scores = batch(order, cuts.astype(np.intp))
    if scores is not None:
        approx = scores.left + scores.right
        refit = _refit_set(approx, scores.error)
        dev[candidates] = approx
        for i in candidates[refit]:
            fl, fr = exact(int(i), (None, None))
            dev[i] = fl.objfun + fr.objfun
    else:
        starts: tuple[FloatArray | None, FloatArray | None] = (None, None)
        for i in candidates:
            # restart = FALSE: the previous admissible candidate's
            # coefficients are the starting values (:417-438).
            fl, fr = exact(int(i), (None, None) if restart else starts)
            starts = (fl.coefficients, fr.coefficients)
            dev[i] = fl.objfun + fr.objfun
    if not np.any(np.isfinite(dev)):
        return SplitSearch(variable, kind, minsize, uz, None, None, dev, None, None)
    best = _which_min(dev)
    if numsplit == "center":
        # mean(uz[best + 0:1]); NaN past the last value, as R's NA.
        upper = uz[best + 1] if best + 1 < uz.shape[0] else np.nan
        breaks = float((uz[best] + upper) / 2)
    else:
        breaks = float(uz[best])
    split = Split(variable, kind, breaks, None)
    return SplitSearch(variable, kind, minsize, uz, None, None, dev, best, split)


def _admissible_numeric(
    zv: FloatArray,
    uz: FloatArray,
    inv: IntArray,
    weights: FloatArray,
    minsize: float,
    caseweights: bool,
) -> BoolArray:
    """Both sides have ``w2n >= minsize``, for every candidate ``z <= uz[i]``.

    Without case weights the counts are exact. With them, the cumulative
    sums can differ from the reference's masked sums in the last bits, so
    candidates within 1e-8 of the total of ``minsize`` are rechecked with
    the reference's sums.
    """
    positive = weights > 0
    per = np.bincount(
        inv, weights=weights[positive] if caseweights else None, minlength=uz.shape[0]
    )
    total = w2n(weights, caseweights)
    n_left = np.cumsum(per)
    n_right = total - n_left
    ok = (n_left >= minsize) & (n_right >= minsize)
    if caseweights:
        tol = 1e-8 * abs(total)
        near = (np.abs(n_left - minsize) <= tol) | (np.abs(n_right - minsize) <= tol)
        for i in np.flatnonzero(near):
            left = zv <= uz[i]
            ok[i] = not (
                w2n(weights[left], caseweights) < minsize
                or w2n(weights[~left], caseweights) < minsize
            )
    return ok


def _refit_set(approx: FloatArray, error: FloatArray) -> BoolArray:
    """Candidates to fit exactly after screening (see :func:`find_split`)."""
    scored = np.isfinite(approx) & np.isfinite(error)
    refit = ~scored
    if not np.any(scored):
        return refit
    a, e = approx[scored], error[scored]
    best_hi = float(np.min(a + e))
    close = a - e <= best_hi + REFIT_MARGIN * abs(float(np.min(a)))
    k = min(REFIT_MIN, a.shape[0])
    close[np.argpartition(a, k - 1)[:k]] = True
    refit[np.flatnonzero(scored)[close]] = True
    return refit


def kid_ids(split: Split, z: NDArray[np.generic]) -> NDArray[np.intp]:
    """Kid (1 or 2) of each row, as ``kidids_split`` (split.R:132-160).

    Numeric breaks use ``.bincode(right = TRUE)``, so ``z <= breaks`` goes to
    kid 1. Ordered factors compare 1-based level positions with the break.
    Nominal levels map through ``index``; a level without a kid gives 0.
    """
    if split.kind is VariableKind.NUMERIC:
        assert split.breaks is not None
        return np.where(np.asarray(z, dtype=np.float64) <= split.breaks, 1, 2).astype(
            np.intp
        )
    codes = np.asarray(z, dtype=np.intp)
    if split.kind is VariableKind.ORDERED:
        assert split.breaks is not None
        return np.where(codes + 1 <= split.breaks, 1, 2).astype(np.intp)
    assert split.index is not None
    lookup = np.array([0 if k is None else k for k in split.index], dtype=np.intp)
    return lookup[codes]
