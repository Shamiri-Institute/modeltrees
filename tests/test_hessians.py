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

import warnings
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
import pytest
import statsmodels.api as sm  # type: ignore[import-untyped]
from fixtures_io import FIXTURES, load_dataset, load_json, nan_array
from grown_trees import fixture, model, tree_inputs

from modeltrees.models.families import (
    Family,
    GlmFitWarning,
    NegativeBinomial,
)
from modeltrees.models.glm import GlmModel
from modeltrees.models.lm import LmModel
from modeltrees.models.negbin import NegBinModel, glm_nb
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


# 2. GlmModel: X' diag(working weights) X / dispersion.


def _glm_model(name: str) -> GlmModel:
    g = _glm_fixture(name)
    return GlmModel(g["family"]["family"], **g["control"])


@cache
def _glm_fit(name: str) -> FitResult:
    y, x, w, off = _glm_inputs(name)
    start = _glm_fixture(name)["start"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlmFitWarning)
        return _glm_model(name).fit(
            y,
            x,
            start=None if start is None else np.array(start, np.float64),
            weights=w,
            offset=off,
            estfun=True,
            vcov=True,
        )


def _glm_dispersion(name: str) -> float:
    """The scores' dispersion from R's working residuals and weights."""
    g = _glm_fixture(name)
    if g["family"]["family"] in ("binomial", "poisson"):
        return 1.0
    _, _, w, _ = _glm_inputs(name)
    ww = np.array(g["result"]["working_weights"])
    wres = np.array(g["result"]["working_residuals"]) * ww
    keep = w > 0
    return float(np.sum(wres[keep] ** 2 / w[keep]) / np.sum(ww))


def glm_score_sum(
    family: Family,
    y: FloatArray,
    x: FloatArray,
    w: FloatArray,
    off: FloatArray | None,
    dispersion: float,
) -> Callable[[FloatArray], FloatArray]:
    """``sum_i w_i (y_i - mu_i) mu'(eta_i) / V(mu_i) x_i / dispersion``, the
    exact score of a GLM, as a function of the coefficients."""

    def score_sum(beta: FloatArray) -> FloatArray:
        eta = x @ beta + (0 if off is None else off)
        mu = family.linkinv(eta)
        out: FloatArray = (w * (y - mu) * family.mu_eta(eta) / family.variance(mu)) @ x
        return out / dispersion

    return score_sum


def exact_weights(
    family: Family,
    beta: FloatArray,
    x: FloatArray,
    w: FloatArray,
    off: FloatArray | None,
) -> FloatArray:
    """The working weights at ``beta``, ``w mu'(eta)^2 / V(mu)``."""
    eta = x @ beta + (0 if off is None else off)
    out: FloatArray = w * family.mu_eta(eta) ** 2 / family.variance(family.linkinv(eta))
    return out


def lag_bound(
    early: FloatArray, exact: FloatArray, x: FloatArray, dispersion: float
) -> FloatArray:
    """``|X|' |W_early - W_exact| |X| / dispersion``: how far a Hessian at
    glm.fit's working weights (from the start of the last iteration) can be
    from the derivative at the estimate."""
    out: FloatArray = np.abs(x).T @ (np.abs(early - exact)[:, None] * np.abs(x))
    return out / dispersion


@pytest.mark.parametrize("name", GLM_FILES)
def test_glm_hessian_by_hand_with_r_working_weights(name: str) -> None:
    g = _glm_fixture(name)
    _, x, _, _ = _glm_inputs(name)
    got = _glm_fit(name)
    assert got.hessian is not None and got.estfun is not None
    keep = ~np.isnan(got.coefficients)
    np.testing.assert_array_equal(
        keep, ~np.isnan(nan_array(g["result"]["coefficients"]))
    )
    assert got.hessian.shape == (got.estfun.shape[1],) * 2
    symmetric(got.hessian)
    xk = x[:, keep]
    ww = np.array(g["result"]["working_weights"])
    expected = xk.T @ (ww[:, None] * xk) / _glm_dispersion(name)
    np.testing.assert_allclose(got.hessian, expected, rtol=RTOL)


@pytest.mark.parametrize("name", GLM_FILES)
def test_glm_hessian_is_minus_the_score_derivative(name: str) -> None:
    y, x, w, off = _glm_inputs(name)
    got = _glm_fit(name)
    assert got.hessian is not None
    family = _glm_model(name).family
    keep = ~np.isnan(got.coefficients)
    pos = w > 0
    xk, yk, wk = x[pos][:, keep], y[pos], w[pos]
    ok = None if off is None else off[pos]
    beta = got.coefficients[keep]
    dispersion = _glm_dispersion(name)
    fd = central_difference(glm_score_sum(family, yk, xk, wk, ok, dispersion), beta)
    early = np.array(_glm_fixture(name)["result"]["working_weights"])[pos]
    lag = lag_bound(early, exact_weights(family, beta, xk, wk, ok), xk, dispersion)
    assert_fd(got.hessian, fd, lag)
    if got.converged:
        # The deviance criterion lets the last step be about sqrt(epsilon),
        # so a converged fit's lag is small (observed up to 4e-5).
        assert np.max(lag) <= 1e-4 * np.max(np.abs(fd))


CONVERGED_GLM = [n for n in GLM_FILES if _glm_fixture(n)["result"]["converged"]]


@pytest.mark.parametrize("name", CONVERGED_GLM)
def test_glm_inverse_against_statsmodels(name: str) -> None:
    """statsmodels' covariance is ``scale (X'WX)^-1`` at its own (tight)
    estimate; ours is ``dispersion H^-1`` at glm.fit's working weights. The
    tolerance is twice the first-order change of the inverse by the lag
    bound, plus RTOL."""
    y, x, w, off = _glm_inputs(name)
    got = _glm_fit(name)
    assert got.hessian is not None
    family = _glm_model(name).family
    keep = ~np.isnan(got.coefficients)
    pos = w > 0
    xk = x[pos][:, keep]
    sm_family = {
        "gaussian": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
    }[family.family]
    ref = sm.GLM(
        y[pos],
        xk,
        family=sm_family,
        freq_weights=w[pos],
        offset=None if off is None else off[pos],
    ).fit(tol=1e-13)
    # H = X'WX / dispersion, statsmodels' covariance scale (X'WX)^-1 (scale
    # 1 for binomial and poisson, its own sigma^2 for gaussian).
    v = np.linalg.inv(got.hessian) * ref.scale / _glm_dispersion(name)
    beta = got.coefficients[keep]
    early = np.array(_glm_fixture(name)["result"]["working_weights"])[pos]
    off_k = None if off is None else off[pos]
    lag = lag_bound(early, exact_weights(family, beta, xk, w[pos], off_k), xk, 1.0)
    # V = scale A^-1 with A = X'WX moves by about V dA V / scale.
    tol = 2 * np.abs(v) @ lag @ np.abs(v) / ref.scale + RTOL * np.max(np.abs(v))
    expected = ref.cov_params()
    assert np.all(np.abs(v - expected) <= tol)


@pytest.mark.parametrize(
    ("name", "node_id"), vcovcl_leaves("glm_binomial", "glm_poisson")
)
def test_glm_hessian_against_r(name: str, node_id: int) -> None:
    leaf = _leaf(name, node_id)
    got = leaf_fit(name, node_id)
    assert got.hessian is not None
    np.testing.assert_allclose(got.hessian, np.array(leaf["hessian"]), rtol=RTOL)
    # Dispersion 1 for binomial and poisson.
    np.testing.assert_allclose(
        np.linalg.inv(got.hessian), np.array(leaf["vcov_model"]), rtol=RTOL
    )


def test_glm_gaussian_hessian_uses_the_score_dispersion() -> None:
    # Gaussian scores are divided by the dispersion, so H is X'WX over it,
    # with and without case weights.
    y, x, w, _ = _glm_inputs("glm_gaussian_weighted")
    for caseweights in (True, False):
        got = GlmModel("gaussian").fit(
            y, x, weights=w, caseweights=caseweights, estfun=True, vcov=True
        )
        assert got.hessian is not None and got.estfun is not None
        resid = got.estfun[:, 0] / x[:, 0]  # wres / dispersion
        wres = y - x @ got.coefficients
        dispersion = float(np.median((w * wres) / resid))
        np.testing.assert_allclose(
            got.hessian * dispersion, x.T @ (w[:, None] * x), rtol=1e-12
        )


def test_glm_vcov_without_estfun() -> None:
    # vcov=True alone takes the full glm.fit path and gives the same Hessian.
    y, x, w, off = _glm_inputs("glm_poisson_log")
    a = GlmModel("poisson").fit(y, x, weights=w, offset=off, estfun=True, vcov=True)
    b = GlmModel("poisson").fit(y, x, weights=w, offset=off, vcov=True)
    assert a.hessian is not None and b.hessian is not None and b.estfun is None
    np.testing.assert_array_equal(a.hessian, b.hessian)


# 3. NegBinModel: as glm at theta_irls, dispersion 1.

NEGBIN_FITS = sorted(p.stem for p in (FIXTURES / "node_fits").glob("negbin_*.json"))


@cache
def _node_fixture(name: str) -> dict[str, Any]:
    f: dict[str, Any] = load_json(f"node_fits/{name}.json")
    return f


def _node_arrays(
    name: str,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None, FloatArray | None]:
    """y, x, weights, offset and start of a node fit fixture (y as given)."""
    f = _node_fixture(name)
    y = np.array(f["y"]) if f["response_levels"] else np.array(f["y"], np.float64)
    return (
        y,
        np.array(f["x"], np.float64),
        np.array(f["weights"], np.float64),
        None if f["offset"] is None else np.array(f["offset"], np.float64),
        None if f["start"] is None else np.array(f["start"], np.float64),
    )


@cache
def _negbin_fit(name: str) -> FitResult:
    f = _node_fixture(name)
    y, x, w, off, start = _node_arrays(name)
    return NegBinModel(coef_names=f["x_names"], **f["fit_args"]).fit(
        y, x, start=start, weights=w, offset=off, estfun=True, vcov=True
    )


def _negbin_working_weights(name: str) -> FloatArray:
    """The working weights of glm.nb's last glm.fit call."""
    f = _node_fixture(name)
    y, x, w, off, start = _node_arrays(name)
    keep = w > 0
    z = glm_nb(
        x[keep],
        y[keep],
        weights=w[keep],
        offset=None if off is None else off[keep],
        start=start,
        epsilon=f["fit_args"]["epsilon"],
        maxit=f["fit_args"]["maxit"],
    )
    return z.fit.weights


def test_negbin_files() -> None:
    assert NEGBIN_FITS == [
        "negbin_quine_deviance",
        "negbin_quine_eth_a_intercept",
        "negbin_quine_full",
        "negbin_sim_offset_weights",
        "negbin_sim_start",
    ]


@pytest.mark.parametrize("name", NEGBIN_FITS)
def test_negbin_hessian_is_minus_the_score_derivative(name: str) -> None:
    """The log link is not canonical for the negative binomial, so the
    expected information (vcov(glm.nb)'s bread) is not the derivative of the
    scores at the data. It is the derivative with the response held at the
    fitted means; the derivative at the data adds the observed term
    ``sum_i w_i (y_i - mu_i) (mu_i / theta) / (1 + mu_i / theta)^2 x_i x_i'``.
    Both are checked, within the lag bound of glm.fit's working weights."""
    y, x, w, off, _ = _node_arrays(name)
    got = _negbin_fit(name)
    assert got.hessian is not None and got.estfun is not None
    assert got.hessian.shape == (got.estfun.shape[1],) * 2
    symmetric(got.hessian)
    theta = got.extras["theta_irls"]
    family = NegativeBinomial(theta)
    keep = w > 0
    xk, yk, wk = x[keep], y[keep], w[keep]
    ok = None if off is None else off[keep]
    beta = got.coefficients
    mu = family.linkinv(xk @ beta + (0 if ok is None else ok))
    lag = lag_bound(
        _negbin_working_weights(name), exact_weights(family, beta, xk, wk, ok), xk, 1.0
    )
    expected = central_difference(glm_score_sum(family, mu, xk, wk, ok, 1.0), beta)
    assert_fd(got.hessian, expected, lag)
    c = wk * (yk - mu) * (mu / theta) / (1 + mu / theta) ** 2
    observed = central_difference(glm_score_sum(family, yk, xk, wk, ok, 1.0), beta)
    assert_fd(got.hessian + xk.T @ (c[:, None] * xk), observed, lag)
    assert np.max(lag) <= 1e-4 * np.max(np.abs(expected))


@pytest.mark.parametrize("name", NEGBIN_FITS)
def test_negbin_inverse_against_statsmodels(name: str) -> None:
    # statsmodels' GLM with NegativeBinomial(alpha = 1 / theta) at our
    # theta_irls: scale 1, its own tight estimate.
    y, x, w, off, _ = _node_arrays(name)
    got = _negbin_fit(name)
    assert got.hessian is not None
    theta = got.extras["theta_irls"]
    keep = w > 0
    xk = x[keep]
    ok = None if off is None else off[keep]
    ref = sm.GLM(
        y[keep],
        xk,
        family=sm.families.NegativeBinomial(alpha=1 / theta),
        freq_weights=w[keep],
        offset=ok,
    ).fit(tol=1e-13)
    v = np.linalg.inv(got.hessian)
    family = NegativeBinomial(theta)
    lag = lag_bound(
        _negbin_working_weights(name),
        exact_weights(family, got.coefficients, xk, w[keep], ok),
        xk,
        1.0,
    )
    tol = 2 * np.abs(v) @ lag @ np.abs(v) + RTOL * np.max(np.abs(v))
    assert np.all(np.abs(v - ref.cov_params()) <= tol)


@pytest.mark.parametrize(("name", "node_id"), vcovcl_leaves("negbin"))
def test_negbin_hessian_against_r(name: str, node_id: int) -> None:
    leaf = _leaf(name, node_id)
    got = leaf_fit(name, node_id)
    assert got.hessian is not None
    np.testing.assert_allclose(got.hessian, np.array(leaf["hessian"]), rtol=RTOL)
    # vcov(glm.nb): dispersion 1, conditional on theta.
    np.testing.assert_allclose(
        np.linalg.inv(got.hessian), np.array(leaf["vcov_model"]), rtol=RTOL
    )


def test_negbin_failed_fit_has_no_hessian() -> None:
    # y = 0 everywhere: theta.ml fails, and so does the fit.
    got = NegBinModel().fit(np.zeros(20), None, weights=np.ones(20), vcov=True)
    assert got.failed and got.hessian is None
