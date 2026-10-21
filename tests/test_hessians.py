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
from test_multinom import _r_offset

from modeltrees.models._common import weighted_crossprod
from modeltrees.models.families import (
    Binomial,
    Family,
    GlmFitWarning,
    NegativeBinomial,
)
from modeltrees.models.glm import GlmModel
from modeltrees.models.glm_fit import glm_fit
from modeltrees.models.lm import LmModel
from modeltrees.models.multinom import (
    MultinomModel,
    multinom_evaluate,
    multinom_hessian,
    multinom_response,
)
from modeltrees.models.negbin import NegBinModel, glm_nb
from modeltrees.models.safelogit import SafeLogitModel
from modeltrees.node_model import FitResult, FitStatus

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


# 4. SafeLogitModel: binomial, None under complete separation.

SAFELOGIT_FITS = sorted(
    p.stem for p in (FIXTURES / "node_fits").glob("safelogit_*.json")
)


@cache
def _safelogit_fit(name: str) -> FitResult:
    f = _node_fixture(name)
    y, x, w, off, start = _node_arrays(name)
    return SafeLogitModel(coef_names=f["x_names"], **f["fit_args"]).fit(
        y, x, start=start, weights=w, offset=off, estfun=True, vcov=True
    )


def test_safelogit_files() -> None:
    assert SAFELOGIT_FITS == [
        "safelogit_complete",
        "safelogit_constant",
        "safelogit_none",
        "safelogit_quasi",
    ]


@pytest.mark.parametrize("name", ["safelogit_none", "safelogit_quasi"])
def test_safelogit_hessian_is_minus_the_score_derivative(name: str) -> None:
    f = _node_fixture(name)
    y, x, w, off, start = _node_arrays(name)
    got = _safelogit_fit(name)
    assert got.status is FitStatus.OK
    assert got.hessian is not None and got.estfun is not None
    assert got.hessian.shape == (got.estfun.shape[1],) * 2
    symmetric(got.hessian)
    family = Binomial()
    keep = w > 0
    xk, yk, wk = x[keep], y[keep], w[keep]
    ok = None if off is None else off[keep]
    z = glm_fit(
        xk,
        yk,
        family,
        weights=wk,
        start=start,
        offset=ok,
        epsilon=f["fit_args"]["epsilon"],
        maxit=f["fit_args"]["maxit"],
        emit_warnings=False,
    )
    np.testing.assert_array_equal(z.coefficients, got.coefficients)
    # The model passes weighted_crossprod a Fortran-ordered x (xm_fit[:, cols])
    # and the test a C-ordered one, so dgemm runs with other transpose flags
    # and may round differently on some CPUs (1.4e-14 on ubuntu CI).
    expected = weighted_crossprod(xk, z.weights)
    np.testing.assert_allclose(
        got.hessian, expected, rtol=1e-12, atol=1e-12 * np.max(np.abs(expected))
    )
    beta = got.coefficients
    lag = lag_bound(z.weights, exact_weights(family, beta, xk, wk, ok), xk, 1.0)
    fd = central_difference(glm_score_sum(family, yk, xk, wk, ok, 1.0), beta)
    assert_fd(got.hessian, fd, lag)
    assert np.max(lag) <= 1e-4 * np.max(np.abs(fd))


def test_safelogit_quasi_separation_is_near_singular() -> None:
    # The d4 coefficient diverges, so its information is near zero: the
    # Hessian is near singular, but only through its scale. Its d4 row is
    # nearly uncorrelated with the others, so the Hessian scaled by its
    # diagonal is well conditioned (docs/api-port.md, section 15).
    got = _safelogit_fit("safelogit_quasi")
    assert got.extras["separation"] == "quasi" and got.hessian is not None
    h = got.hessian
    assert h[2, 2] < 1e-6 * min(h[0, 0], h[1, 1])
    assert np.linalg.cond(h) > 1e7
    d = np.sqrt(np.diag(h))
    assert np.linalg.cond(h / np.outer(d, d)) < 2


def test_safelogit_inverse_against_statsmodels() -> None:
    y, x, w, off, _ = _node_arrays("safelogit_none")
    assert off is None and np.all(w == 1)
    got = _safelogit_fit("safelogit_none")
    assert got.hessian is not None
    ref = sm.Logit(y, x).fit(method="newton", tol=1e-13, disp=False)
    v = np.linalg.inv(got.hessian)
    f = _node_fixture("safelogit_none")
    z = glm_fit(
        x,
        y,
        Binomial(),
        weights=w,
        epsilon=f["fit_args"]["epsilon"],
        maxit=f["fit_args"]["maxit"],
    )
    lag = lag_bound(
        z.weights, exact_weights(Binomial(), got.coefficients, x, w, None), x, 1.0
    )
    tol = 2 * np.abs(v) @ lag @ np.abs(v) + RTOL * np.max(np.abs(v))
    assert np.all(np.abs(v - ref.cov_params()) <= tol)


@pytest.mark.parametrize("name", ["safelogit_complete", "safelogit_constant"])
def test_safelogit_complete_separation_has_no_hessian(name: str) -> None:
    got = _safelogit_fit(name)
    assert got.status is FitStatus.NO_ESTFUN
    assert got.estfun is None and got.hessian is None


@pytest.mark.parametrize(("name", "node_id"), vcovcl_leaves("safelogit"))
def test_safelogit_hessian_against_r(name: str, node_id: int) -> None:
    leaf = _leaf(name, node_id)
    got = leaf_fit(name, node_id)
    assert got.extras["separation"] == "none" and got.hessian is not None
    np.testing.assert_allclose(got.hessian, np.array(leaf["hessian"]), rtol=RTOL)
    np.testing.assert_allclose(
        np.linalg.inv(got.hessian), np.array(leaf["vcov_model"]), rtol=RTOL
    )


def test_safelogit_failed_fit_has_no_hessian() -> None:
    # glm.fit stops on a start of the wrong length: a failed fit.
    got = SafeLogitModel().fit(
        np.array([0.0, 1.0, 1.0, 0.0]),
        None,
        start=np.zeros(3),
        weights=np.ones(4),
        vcov=True,
    )
    assert got.failed and got.hessian is None


# 5. MultinomModel: Newton's Hessian over present classes.

MULTINOM_FITS = sorted(p.stem for p in (FIXTURES / "node_fits").glob("multinom_*.json"))


@cache
def _multinom_fit(name: str) -> FitResult:
    f = _node_fixture(name)
    y, x, w, _, start = _node_arrays(name)
    model = MultinomModel(f["response_levels"], coef_names=f["x_names"])
    return model.fit(
        multinom_response(y, f["response_levels"]),
        x,
        start=start,
        weights=w,
        estfun=True,
        vcov=True,
    )


def multinom_score_sum(
    x: FloatArray, y: IntArray, w: FloatArray, n_classes: int
) -> Callable[[FloatArray], FloatArray]:
    """The summed multinomial scores as a function of the flat coefficients
    of the classes 1, ..., n_classes - 1 (class-major), written out."""
    onehot = np.eye(n_classes)[y]

    def score_sum(beta: FloatArray) -> FloatArray:
        eta = np.zeros((x.shape[0], n_classes))
        eta[:, 1:] = x @ beta.reshape(n_classes - 1, -1).T
        prob = np.exp(eta - eta.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        out: FloatArray = ((w[:, None] * (onehot - prob))[:, 1:].T @ x).reshape(-1)
        return out

    return score_sum


def test_multinom_files() -> None:
    assert MULTINOM_FITS == [
        "multinom_absent_class",
        "multinom_full",
        "multinom_reference_absent",
        "multinom_single_class",
        "multinom_start",
        "multinom_weights_zero",
    ]


OK_MULTINOM = [
    "multinom_absent_class",
    "multinom_full",
    "multinom_start",
    "multinom_weights_zero",
]


@pytest.mark.parametrize("name", OK_MULTINOM)
def test_multinom_hessian_is_minus_the_score_derivative(name: str) -> None:
    f = _node_fixture(name)
    y, x, w, _, _ = _node_arrays(name)
    got = _multinom_fit(name)
    assert got.status is FitStatus.OK
    assert got.hessian is not None and got.estfun is not None
    assert got.hessian.shape == (got.estfun.shape[1],) * 2
    symmetric(got.hessian)
    # Over the reference and the present classes, in the scores' order.
    codes = multinom_response(y, f["response_levels"])
    keep = w > 0
    p = x.shape[1]
    beta = got.coefficients.reshape(-1, p)
    present = ~np.isnan(beta[:, 0])
    classes = np.r_[0, np.flatnonzero(present) + 1]
    recode = np.full(len(f["response_levels"]), -1)
    recode[classes] = np.arange(classes.shape[0])
    score_sum = multinom_score_sum(
        x[keep], recode[codes[keep]], w[keep], classes.shape[0]
    )
    fd = central_difference(score_sum, beta[present].reshape(-1))
    assert_fd(got.hessian, fd)


@pytest.mark.parametrize("name", ["multinom_reference_absent", "multinom_single_class"])
def test_multinom_no_estfun_has_no_hessian(name: str) -> None:
    got = _multinom_fit(name)
    assert got.status is FitStatus.NO_ESTFUN
    assert got.estfun is None and got.hessian is None


def test_multinom_inverse_against_statsmodels() -> None:
    f = _node_fixture("multinom_full")
    y, x, w, _, _ = _node_arrays("multinom_full")
    assert np.all(w == 1)
    got = _multinom_fit("multinom_full")
    assert got.hessian is not None
    ref = sm.MNLogit(multinom_response(y, f["response_levels"]), x).fit(
        method="newton", maxiter=100, tol=1e-14, disp=False
    )
    # statsmodels' params are p by K - 1 and its covariance follows
    # params.ravel(order="F"): class-major, as ours.
    expected = np.asarray(ref.cov_params())
    np.testing.assert_allclose(
        np.asarray(ref.params).T.reshape(-1), got.coefficients, rtol=1e-10
    )
    np.testing.assert_allclose(np.linalg.inv(got.hessian), expected, rtol=RTOL)


def test_multinom_rank_deficient_has_no_hessian() -> None:
    x = np.column_stack([np.ones(6), np.arange(6.0), 2 * np.arange(6.0)])
    y = np.array([0, 1, 2, 0, 1, 2])
    got = MultinomModel(("a", "b", "c")).fit(y, x, weights=np.ones(6), vcov=True)
    assert got.failed and got.hessian is None


@pytest.mark.parametrize(("name", "node_id"), vcovcl_leaves("multinom"))
def test_multinom_hessian_against_r(name: str, node_id: int) -> None:
    """R's Hessian and ``vcov(multinom)`` are at nnet's coefficients, which
    differ from the MLE by ``-H^-1 g_R`` to first order (g_R the column sums
    of R's scores; docs/extensions-port.md, section 7). The tolerance is
    twice the change of our Hessian, and of its inverse, when our
    coefficients move by that offset, plus RTOL."""
    leaf = _leaf(name, node_id)
    got = leaf_fit(name, node_id)
    assert got.hessian is not None
    tree = fixture(*_tree_of(name))
    node = next(n for n in tree["nodes"] if n["id"] == node_id)
    y, x, w, _ = leaf_data(name, node_id)
    offset = _r_offset(got, x, y, w, np.array(node["estfun"]))
    p = x.shape[1]
    beta = got.coefficients.reshape(-1, p) + offset.reshape(-1, p)
    _, prob = multinom_evaluate(beta, x, y, w)
    moved = multinom_hessian(prob, x, w)
    h_r = np.array(leaf["hessian"])
    tol = 2 * np.abs(moved - got.hessian) + RTOL * np.max(np.abs(h_r))
    assert np.all(np.abs(got.hessian - h_r) <= tol)
    v, v_moved = np.linalg.inv(got.hessian), np.linalg.inv(moved)
    v_r = np.array(leaf["vcov_model"])
    tol = 2 * np.abs(v_moved - v) + RTOL * np.max(np.abs(v_r))
    assert np.all(np.abs(v - v_r) <= tol)
    # And at R's coefficients the formula reproduces R's Hessian.
    _, prob_r = multinom_evaluate(
        nan_array(leaf["coefficients"]).reshape(-1, p), x, y, w
    )
    np.testing.assert_allclose(multinom_hessian(prob_r, x, w), h_r, rtol=RTOL)
