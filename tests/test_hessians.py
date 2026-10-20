"""The node models' Hessians (docs/api-design.md, sections 5.2 and 5.3).

``FitResult.hessian`` is minus the derivative of the summed scores over the
non-aliased columns, in estfun's column order. Every model is checked
against a central finite difference of the summed scores written out from
the model's formula, on the node fit fixtures (tests/fixtures/node_fits/,
and tests/fixtures/glm/ for lm and glm), and its inverse against R's
model-based covariance (``vcov_model`` in tests/fixtures/vcovcl/, the leaf
refits of docs/sandwich-fixtures.md) and statsmodels' ``cov_params()``, at
the dispersion each uses.

Tolerances. The central difference with relative step ``FD_STEP = 1e-5``
has truncation error of order ``FD_STEP^2 = 1e-10`` and rounding error of
order ``eps / FD_STEP = 2e-11`` relative to the largest entry, so
``FD_RTOL = 1e-7`` of the largest entry leaves a margin of 1000. Comparisons
with R use the suite's ``RTOL = 1e-8``.
"""

from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
import pytest
import statsmodels.api as sm  # type: ignore[import-untyped]
from fixtures_io import FIXTURES, load_dataset, load_json, nan_array
from grown_trees import fixture, model, tree_inputs

from modeltrees.models.lm import LmModel
from modeltrees.node_model import FitResult

FloatArray = np.ndarray
IntArray = np.ndarray

RTOL = 1e-8
FD_STEP = 1e-5
FD_RTOL = 1e-7


def central_difference(
    score_sum: Callable[[FloatArray], FloatArray], theta: FloatArray
) -> FloatArray:
    """``-d score_sum / d theta`` by central differences, column by column."""
    k = theta.shape[0]
    out = np.empty((k, k))
    for j in range(k):
        h = FD_STEP * max(1.0, abs(float(theta[j])))
        e = np.zeros(k)
        e[j] = h
        out[:, j] = -(score_sum(theta + e) - score_sum(theta - e)) / (2 * h)
    return out


def assert_fd(got: FloatArray, fd: FloatArray, extra: FloatArray | float = 0.0) -> None:
    """``got`` equals the finite difference within FD_RTOL of its largest
    entry, plus ``extra`` (a derived bound, elementwise)."""
    assert got.shape == fd.shape
    tol = FD_RTOL * np.max(np.abs(fd)) + extra
    assert np.all(np.abs(got - fd) <= tol), np.max(np.abs(got - fd) - tol)


def symmetric(h: FloatArray) -> None:
    np.testing.assert_array_equal(h, h.T)


# The glm.fit fixtures as node fits for lm and glm.

GLM_FILES = sorted(p.stem for p in (FIXTURES / "glm").glob("*.json"))


@cache
def _glm_fixture(name: str) -> dict[str, Any]:
    f: dict[str, Any] = load_json(f"glm/{name}.json")
    return f


def _glm_inputs(
    name: str,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None]:
    """y, x, weights (ones without) and offset of a glm.fit fixture."""
    g = _glm_fixture(name)
    d, _ = load_dataset(g["data_csv"])
    x = np.column_stack([d[c] for c in g["x_columns"]])
    y = d["y"]
    w = d[g["weights_column"]] if g["weights_column"] else np.ones(y.shape[0])
    off = d[g["offset_column"]] if g["offset_column"] else None
    return y, x, w, off


# The leaves of the vcovcl fixtures, refitted with the tree's model.

VCOVCL_FILES = sorted(p.stem for p in (FIXTURES / "vcovcl").glob("*.json"))


@cache
def _vcovcl(name: str) -> dict[str, Any]:
    f: dict[str, Any] = load_json(f"vcovcl/{name}.json")
    return f


def _tree_of(name: str) -> tuple[str, str]:
    directory, file = _vcovcl(name)["tree_fixture"].split("/")
    return directory, file.removesuffix(".json")


def vcovcl_leaves(*models: str) -> list[Any]:
    return [
        pytest.param(name, leaf["id"], id=f"{name}-{leaf['id']}")
        for name in VCOVCL_FILES
        if _vcovcl(name)["model"] in models
        for leaf in _vcovcl(name)["leaves"]
    ]


def _leaf(name: str, node_id: int) -> dict[str, Any]:
    return next(lf for lf in _vcovcl(name)["leaves"] if lf["id"] == node_id)


@cache
def leaf_fit(name: str, node_id: int) -> FitResult:
    f = fixture(*_tree_of(name))
    y, x, w, off = tree_inputs(f)
    rows = np.array(_leaf(name, node_id)["rows"], dtype=np.intp)
    return model(f).fit(
        y[rows],
        None if x is None else x[rows],
        start=None,
        weights=w[rows],
        offset=None if off is None else off[rows],
        cluster=None,
        caseweights=f["control"]["caseweights"],
        estfun=True,
        vcov=True,
    )


def leaf_data(
    name: str, node_id: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None]:
    f = fixture(*_tree_of(name))
    y, x, w, off = tree_inputs(f)
    rows = np.array(_leaf(name, node_id)["rows"], dtype=np.intp)
    xm = np.ones((rows.shape[0], 1)) if x is None else x
    return (
        y[rows],
        xm if x is None else xm[rows],
        w[rows],
        None if off is None else off[rows],
    )


def test_fixture_lists() -> None:
    assert VCOVCL_FILES == [
        "sim_cluster_few_lmtree_cluster",
        "sim_cluster_glmtree_binomial_cluster",
        "sim_cluster_lmtree_cluster",
        "sim_cluster_models_glmtree_poisson_cluster",
        "sim_cluster_models_multinom_cluster",
        "sim_cluster_models_negbin_cluster",
        "sim_cluster_models_safelogit_cluster",
    ]


# 1. LmModel: X'WX.

LM_FILES = [n for n in GLM_FILES if _glm_fixture(n)["partykit_lmfit"] is not None]


def _lm_fit(name: str) -> FitResult:
    y, x, w, off = _glm_inputs(name)
    return LmModel().fit(y, x, weights=w, offset=off, estfun=True, vcov=True)


def test_lm_files() -> None:
    assert LM_FILES == [
        "glm_gaussian_identity",
        "glm_gaussian_rank_deficient",
        "glm_gaussian_weighted",
    ]


@pytest.mark.parametrize("name", LM_FILES)
def test_lm_hessian_is_minus_the_score_derivative(name: str) -> None:
    y, x, w, off = _glm_inputs(name)
    got = _lm_fit(name)
    assert got.hessian is not None and got.estfun is not None
    keep = ~np.isnan(got.coefficients)
    assert got.hessian.shape == (int(keep.sum()),) * 2 == (got.estfun.shape[1],) * 2
    symmetric(got.hessian)
    xk = x[:, keep]
    yo = y if off is None else y - off

    def score_sum(beta: FloatArray) -> FloatArray:
        out: FloatArray = (w * (yo - xk @ beta)) @ xk
        return out

    assert_fd(got.hessian, central_difference(score_sum, got.coefficients[keep]))
    # The summed scores of the fit itself are zero, so the Hessian is the
    # derivative at the estimate.
    np.testing.assert_allclose(
        got.estfun.sum(axis=0), 0, atol=1e-9 * np.abs(got.estfun).sum()
    )


@pytest.mark.parametrize("name", LM_FILES)
def test_lm_inverse_against_statsmodels(name: str) -> None:
    # statsmodels' WLS covariance is sigma^2 (X'WX)^-1 with sigma^2 =
    # RSS / (n - rank); the scores carry no 1 / sigma^2, so neither does H.
    y, x, w, off = _glm_inputs(name)
    got = _lm_fit(name)
    assert got.hessian is not None
    keep = ~np.isnan(got.coefficients)
    yo = y if off is None else y - off
    ref = sm.WLS(yo, x[:, keep], weights=w).fit()
    sigma2 = got.objfun / (y.shape[0] - int(keep.sum()))
    np.testing.assert_allclose(
        np.linalg.inv(got.hessian) * sigma2, ref.cov_params(), rtol=RTOL
    )


@pytest.mark.parametrize(("name", "node_id"), vcovcl_leaves("lm"))
def test_lm_hessian_against_r(name: str, node_id: int) -> None:
    leaf = _leaf(name, node_id)
    got = leaf_fit(name, node_id)
    assert got.hessian is not None
    np.testing.assert_allclose(got.hessian, np.array(leaf["hessian"]), rtol=RTOL)
    # vcov(lm) is sigma^2 (X'X)^-1 with sigma^2 = RSS / (n - k).
    sigma2 = got.objfun / (leaf["n"] - leaf["k"])
    np.testing.assert_allclose(
        np.linalg.inv(got.hessian) * sigma2, np.array(leaf["vcov_model"]), rtol=RTOL
    )
    np.testing.assert_allclose(
        got.coefficients, nan_array(leaf["coefficients"]), rtol=RTOL
    )


def test_lm_hessian_only_on_request() -> None:
    y, x, w, _ = _glm_inputs("glm_gaussian_identity")
    assert LmModel().fit(y, x, weights=w, estfun=True).hessian is None


def test_lm_zero_weights_add_nothing() -> None:
    y, x, w, _ = _glm_inputs("glm_gaussian_weighted")
    w0 = w.copy()
    w0[::5] = 0.0
    full = LmModel().fit(y, x, weights=w0, vcov=True)
    kept = LmModel().fit(y[w0 > 0], x[w0 > 0], weights=w0[w0 > 0], vcov=True)
    assert full.hessian is not None and kept.hessian is not None
    np.testing.assert_allclose(full.hessian, kept.hessian, rtol=1e-13)
