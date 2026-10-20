"""Multinomial (baseline-category) logit, the node model ``multinomfit``.

The model of r/extensions/multinom.R, specified in docs/extensions-spec.md,
section 5 (provisional: the thesis names mlogitModel but gives no code).
R fits it with nnet's BFGS; here it is fitted by Newton's method on the
log-likelihood, with step halving and convergence on the gradient
(docs/decisions.md, Q10). The MLE is unique, so both reach the same
estimate; R's is accurate to about the size of its gradient
(docs/extensions-port.md, section 7).

The response has levels l_1, ..., l_K, coded 0, ..., K - 1, and l_1 is the
reference class:

    P(Y = l_k | x) = exp(x'beta_k) / sum_j exp(x'beta_j), beta_1 = 0.

The (K - 1) p coefficients are named ``"<level>:<regressor>"`` in
class-major order (all regressors of l_2, then l_3, ...), the order of
``vcov(multinom)`` and sandwich's ``estfun.multinom``.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from modeltrees.models._common import check_vcov, regressor_matrix
from modeltrees.models._qr import dqrdc2
from modeltrees.node_model import FitResult, FitStatus, Report, failed_fit

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

# lm.fit's tolerance for the rank of the design.
RANK_TOL = 1e-7
# Relative rounding allowance on the objective in step halving.
ROUNDING = 1e-12


def multinom_response(
    y: ArrayLike | Any, levels: Sequence[Any], *, codes: bool = False
) -> IntArray:
    """Encode a categorical response as 0-based codes in ``levels`` order.

    ``levels`` is the full level order; its first level is the reference
    class. With ``codes = False`` the values of ``y`` are labels among
    ``levels``. With ``codes = True`` they are already integer codes
    ``0, ..., len(levels) - 1`` and are only checked. The order must be
    given explicitly: R sorts factor levels by the locale's collation, which
    can differ from Python's sort.
    """
    levels = list(levels)
    if len(levels) < 2:
        raise ValueError("a multinomial response needs at least two levels")
    if len(set(levels)) != len(levels):
        raise ValueError("the levels are not unique")
    values = np.asarray(y).reshape(-1)
    if values.dtype.kind == "f" and np.any(np.isnan(values)):
        raise ValueError("the response has missing values")
    if values.dtype.kind == "O" and any(v is None for v in values):
        raise ValueError("the response has missing values")
    if codes:
        if values.dtype.kind not in "iuf" or np.any(values != np.round(values)):
            raise ValueError("codes must be integers")
        out = values.astype(np.intp)
        if np.any((out < 0) | (out >= len(levels))):
            raise ValueError(f"codes must be in 0, ..., {len(levels) - 1}")
        return out
    index = {level: i for i, level in enumerate(levels)}
    unknown = {v for v in values.tolist() if v not in index}
    if unknown:
        raise ValueError(
            f"response values not among the levels: {sorted(map(str, unknown))}"
        )
    return np.array([index[v] for v in values.tolist()], dtype=np.intp)


@dataclass(frozen=True)
class MultinomFit:
    """A Newton fit of the multinomial logit on the present classes."""

    coefficients: FloatArray  # (K' - 1, p), relative to the first class
    value: float  # minus the weighted log-likelihood
    probabilities: FloatArray  # (n, K')
    gradient: FloatArray  # (K' - 1, p), of the log-likelihood
    converged: bool
    iterations: int


def multinom_evaluate(
    beta: FloatArray, x: FloatArray, y: IntArray, w: FloatArray
) -> tuple[float, FloatArray]:
    """Minus the weighted log-likelihood and the class probabilities.

    ``beta`` is (K - 1, p), relative to class 0; ``y`` holds codes.
    """
    eta = np.zeros((x.shape[0], beta.shape[0] + 1))
    eta[:, 1:] = x @ beta.T
    lse = logsumexp(eta, axis=1)
    rows = np.arange(x.shape[0])
    value = -float(np.sum(w * (eta[rows, y] - lse)))
    return value, np.exp(eta - lse[:, None])


def multinom_scores(
    prob: FloatArray, x: FloatArray, y: IntArray, w: FloatArray
) -> FloatArray:
    """Per-row scores ``w_i (1{y_i = k} - p_ik) x_ij``, class-major columns
    for the classes 1, ..., K - 1."""
    n, n_classes = prob.shape
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), y] = 1.0
    resid = w[:, None] * (onehot[:, 1:] - prob[:, 1:])
    return (resid[:, :, None] * x[:, None, :]).reshape(n, -1)


def multinom_hessian(prob: FloatArray, x: FloatArray, w: FloatArray) -> FloatArray:
    """Hessian of minus the log-likelihood, in class-major order:
    ``sum_i w_i p_ik (delta_kl - p_il) x_i x_i'`` for k, l in 1, ..., K - 1."""
    p = x.shape[1]
    pk = prob[:, 1:]
    m = pk.shape[1]
    hess = np.empty((m * p, m * p))
    for k in range(m):
        for j in range(k, m):
            c = w * pk[:, k] * ((k == j) - pk[:, j])
            block = x.T @ (c[:, None] * x)
            hess[k * p : (k + 1) * p, j * p : (j + 1) * p] = block
            hess[j * p : (j + 1) * p, k * p : (k + 1) * p] = block.T
    return hess


def multinom_newton(
    x: FloatArray,
    y: IntArray,
    n_classes: int,
    weights: FloatArray,
    *,
    start: FloatArray | None = None,
    gtol: float = 1e-12,
    maxit: int = 100,
    max_halvings: int = 50,
) -> MultinomFit:
    """Maximise the multinomial log-likelihood by Newton's method.

    ``y`` holds codes ``0, ..., n_classes - 1``, class 0 is the reference,
    and every weight is positive. Each iteration solves with the Hessian,
    halving the step until minus the log-likelihood does not increase
    (beyond rounding, ``ROUNDING`` relative).
    Converged means that every gradient entry is at most ``gtol`` times the
    sum of ``w_i |x_ij|`` over the rows, the largest the entry can be.
    """
    p = x.shape[1]
    m = n_classes - 1
    beta: FloatArray = (
        np.zeros((m, p)) if start is None else np.array(start, dtype=np.float64)
    )
    scale = np.abs(x).T @ weights  # (p,)
    scale[scale == 0] = 1.0
    value, prob = multinom_evaluate(beta, x, y, weights)
    converged = False
    iterations = 0
    while True:
        gradient = multinom_scores(prob, x, y, weights).sum(axis=0).reshape(m, p)
        if np.all(np.abs(gradient) <= gtol * scale):
            converged = True
            break
        if iterations >= maxit:
            break
        iterations += 1
        hess = multinom_hessian(prob, x, weights)
        try:
            delta = scipy.linalg.solve(hess, gradient.reshape(-1), assume_a="pos")
        except (np.linalg.LinAlgError, ValueError):
            break
        step = delta.reshape(m, p)
        t = 1.0
        for _ in range(max_halvings + 1):
            trial = beta + t * step
            trial_value, trial_prob = multinom_evaluate(trial, x, y, weights)
            # Near the optimum the objective changes below its rounding
            # error, so an increase within rounding does not halve the step.
            if np.isfinite(trial_value) and trial_value <= value + ROUNDING * (
                1 + abs(value)
            ):
                break
            t /= 2
        else:
            break  # no step decreases the objective: stop where we are
        beta, value, prob = trial, trial_value, trial_prob
    return MultinomFit(beta, value, prob, gradient, converged, iterations)


class MultinomModel:
    """Multinomial logit with its own coefficients in every node.

    ``levels`` is the full level order of the response, whose first level
    is the reference class; ``y`` holds 0-based codes
    (:func:`multinom_response`). ``coef_names`` names the regressors.

    Degenerate nodes (docs/extensions-spec.md, section 5, and
    docs/decisions.md). A class is present if it has positive total weight.

    - A non-reference class is absent: its coefficients are NaN, it has no
      score columns, and the fit uses the present classes.
    - The reference class is absent: NaN coefficients and no scores
      (``NO_ESTFUN``), with the objective R computes, fitted with the first
      present class as reference, in ``objfun`` and ``extras["value"]``.
    - One class present: objective 0, NaN coefficients, no scores
      (``NO_ESTFUN``).
    - The positive-weight design is rank deficient (``dqrdc2`` with lm.fit's
      tolerance 1e-7): a failed fit (decisions Q9).
    - An offset raises ``ValueError``.

    Rows with zero weight are left out of the fit and get zero scores.
    ``extras`` holds ``status``, ``classes_present`` (labels),
    ``converged``, ``iterations``, ``value`` and ``max_rel_gradient``.
    """

    def __init__(
        self,
        levels: Sequence[Any],
        *,
        coef_names: Sequence[str] | None = None,
        gtol: float = 1e-12,
        maxit: int = 100,
    ) -> None:
        self.levels = tuple(levels)
        if len(self.levels) < 2:
            raise ValueError("a multinomial response needs at least two levels")
        self.coef_names = None if coef_names is None else tuple(coef_names)
        self.gtol = gtol
        self.maxit = maxit

    def names(self, regressors: Sequence[str]) -> tuple[str, ...]:
        """Coefficient names, class-major: ``"<level>:<regressor>"``."""
        return tuple(f"{lev}:{r}" for lev in self.levels[1:] for r in regressors)

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
        if offset is not None:
            raise ValueError("multinomfit does not support an offset")
        n_levels = len(self.levels)
        yv = multinom_response(y, self.levels, codes=True)
        n = yv.shape[0]
        xm, regressors = regressor_matrix(n, x, self.coef_names)
        p = xm.shape[1]
        names = self.names(regressors)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        keep = w > 0
        y_fit, x_fit, w_fit = yv[keep], xm[keep], w[keep]
        present = np.flatnonzero(np.bincount(y_fit, minlength=n_levels) > 0)
        labels = tuple(self.levels[i] for i in present)
        extras: dict[str, Any] = {
            "status": "ok",
            "classes_present": labels,
            "converged": None,
            "iterations": None,
            "value": None,
            "max_rel_gradient": None,
        }
        if present.shape[0] < 2:
            # Every observation is in one class: the log-likelihood supremum
            # is 0 and no coefficient is identified.
            extras["status"] = "single_class"
            return FitResult(
                coefficients=np.full(len(names), np.nan),
                coef_names=names,
                objfun=0.0,
                estfun=None,
                status=FitStatus.NO_ESTFUN,
                reason="only one class is present",
                extras=extras,
            )
        reference_absent = present[0] != 0
        if reference_absent:
            extras["status"] = "reference_absent"
        rank = dqrdc2(x_fit * np.sqrt(w_fit)[:, None], RANK_TOL)[1]
        if rank < p:
            extras["status"] = "rank_deficient"
            return failed_fit(
                names, f"the design has rank {rank} < {p} columns", **extras
            )
        # Codes among the present classes; the first present class is the
        # reference of the fit.
        recode = np.full(n_levels, -1, dtype=np.intp)
        recode[present] = np.arange(present.shape[0])
        beta0 = None
        if start is not None and not reference_absent:
            s = np.asarray(start, dtype=np.float64).reshape(n_levels - 1, p)
            beta0 = np.nan_to_num(s[present[1:] - 1], nan=0.0)
        z = multinom_newton(
            x_fit,
            recode[y_fit],
            present.shape[0],
            w_fit,
            start=beta0,
            gtol=self.gtol,
            maxit=self.maxit,
        )
        scale = np.abs(x_fit).T @ w_fit
        scale[scale == 0] = 1.0
        extras["converged"] = z.converged
        extras["iterations"] = z.iterations
        extras["value"] = z.value
        extras["max_rel_gradient"] = float(np.max(np.abs(z.gradient) / scale))
        if reference_absent:
            # The coefficients cannot be expressed relative to the reference
            # class, but the objective is R's, so the fit stays admissible in
            # split search and the node is terminal.
            return FitResult(
                coefficients=np.full(len(names), np.nan),
                coef_names=names,
                objfun=z.value,
                estfun=None,
                converged=z.converged,
                status=FitStatus.NO_ESTFUN,
                reason=f"the reference class {self.levels[0]!r} is absent",
                extras=extras,
            )
        coef = np.full((n_levels - 1, p), np.nan)
        coef[present[1:] - 1] = z.coefficients
        scores = None
        if estfun:
            scores = np.zeros((n, (present.shape[0] - 1) * p))
            scores[keep] = multinom_scores(z.probabilities, x_fit, recode[y_fit], w_fit)
        return FitResult(
            coefficients=coef.reshape(-1),
            coef_names=names,
            objfun=z.value,
            estfun=scores,
            converged=z.converged,
            extras=extras,
        )

    def n_response_columns(self, y: NDArray[np.generic]) -> int:
        # NCOL of a factor is 1: minsize defaults to ceiling(10 (K - 1) p).
        return 1

    def loglik(self, result: FitResult) -> tuple[float, int]:
        """``logLik.multinom``: minus the objective, with df the number of
        estimated coefficients, ``(K' - 1) p`` for the K' present classes.

        A fit without the reference class has R's objective and R's df
        (fitted with the first present class as reference), though its
        coefficients are NaN here. One class present gives 0 and 0.
        """
        if result.failed:
            return math.nan, len(result.coefficients)
        p = len(result.coefficients) // (len(self.levels) - 1)
        present = len(result.extras["classes_present"])
        return -result.objfun, (present - 1) * p

    def report(self, result: FitResult) -> Report:
        e = result.extras
        present = e.get("classes_present")
        return {
            "fit_status": e.get("status"),
            "classes_present": None
            if present is None
            else ", ".join(str(c) for c in present),
            "converged": e.get("converged"),
        }

    def predict(
        self,
        result: FitResult,
        x: FloatArray | None,
        offset: FloatArray | None,
        kind: str,
    ) -> FloatArray:
        """Predictions from the fitted multinomial logit.

        "link" is the ``(n, K - 1)`` matrix of linear predictors of the
        non-reference classes, "prob" the ``(n, K)`` class probabilities,
        "class" the 0-based code of the most probable class (ties to the
        first) as floats. "response" is undefined and raises. An absent
        class has probability 0 and NaN link. A fit with one class present
        gives that class probability 1 and a NaN link; a failed fit, or
        one without the reference class (no coefficients), gives NaN.
        """
        if kind == "response":
            raise ValueError(
                "type 'response' is undefined for a multinomial tree; use "
                "'prob' for the class probabilities or 'class' for the labels"
            )
        if kind not in ("link", "prob", "class"):
            raise ValueError(
                f"multinomtree predicts 'link', 'prob' or 'class', not {kind!r}"
            )
        if offset is not None:
            raise ValueError("multinomfit does not support an offset")
        if x is None:
            raise ValueError("x is needed, with its intercept column")
        xm = np.asarray(x, dtype=np.float64)
        if xm.ndim == 1:
            xm = xm[:, None]
        n, p = xm.shape
        n_levels = len(self.levels)
        beta = np.asarray(result.coefficients, dtype=np.float64)
        if beta.shape[0] != (n_levels - 1) * p:
            raise ValueError(
                f"x has {p} columns for {beta.shape[0]} coefficients "
                f"and {n_levels} classes"
            )
        beta = beta.reshape(n_levels - 1, p)
        status = result.extras.get("status")
        absent = np.all(np.isnan(beta), axis=1)
        link = np.full((n, n_levels - 1), np.nan)
        prob = np.full((n, n_levels), np.nan)
        if status == "single_class" and not result.failed:
            prob[:] = 0.0
            only = self.levels.index(result.extras["classes_present"][0])
            prob[:, only] = 1.0
        elif status != "reference_absent" and not result.failed:
            eta = np.zeros((n, n_levels))
            eta[:, 1:] = xm @ np.where(np.isnan(beta), 0.0, beta).T
            link[:, ~absent] = eta[:, 1:][:, ~absent]
            eta[:, 1:][:, absent] = -np.inf
            expo = np.exp(eta - np.max(eta, axis=1, keepdims=True))
            prob = expo / np.sum(expo, axis=1, keepdims=True)
        if kind == "link":
            return link
        if kind == "prob":
            return prob
        codes = np.full(n, np.nan)
        defined = ~np.any(np.isnan(prob), axis=1)
        codes[defined] = np.argmax(prob[defined], axis=1)
        return codes
