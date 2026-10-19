"""glm.fit, lm.fit and the QR port against the glm.fit fixtures.

Each fixture holds R's ``glm.fit`` result on its own data, plus partykit's
``glmfit`` (and ``lmfit`` for gaussian cases) objective and scores on the
same data.
"""

import warnings
from functools import cache
from typing import Any

import numpy as np
import pytest
import statsmodels.api as sm  # type: ignore[import-untyped]
from fixtures_io import FIXTURES, load_dataset, load_json, nan_array

from modeltrees.models._qr import dqrdc2, dqrls
from modeltrees.models.families import (
    Binomial,
    Gaussian,
    GlmFitWarning,
    Poisson,
    get_family,
)
from modeltrees.models.glm import GlmModel
from modeltrees.models.glm_fit import GlmFit, GlmFitError, glm_fit
from modeltrees.models.lm import LmModel
from modeltrees.models.lm_fit import lm_fit, lm_wfit

RTOL = 1e-8
ATOL = 1e-10

GLM_FILES = sorted(p.stem for p in (FIXTURES / "glm").glob("*.json"))


@cache
def _fixture(name: str) -> dict[str, Any]:
    fixture: dict[str, Any] = load_json(f"glm/{name}.json")
    return fixture


def _inputs(
    g: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    data, _ = load_dataset(g["data_csv"])
    x = np.column_stack([data[c] for c in g["x_columns"]])
    w = data[g["weights_column"]] if g["weights_column"] else None
    off = data[g["offset_column"]] if g["offset_column"] else None
    return x, data["y"], w, off


@cache
def _fit(name: str) -> GlmFit:
    g = _fixture(name)
    x, y, w, off = _inputs(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlmFitWarning)
        return glm_fit(
            x,
            y,
            get_family(g["family"]["family"]),
            weights=w,
            start=g["start"],
            offset=off,
            epsilon=g["control"]["epsilon"],
            maxit=g["control"]["maxit"],
        )


def test_fixtures_cover_the_required_cases() -> None:
    assert len(GLM_FILES) == 11
    results = [_fixture(n)["result"] for n in GLM_FILES]
    assert any(not r["converged"] for r in results)  # maxit
    assert any(None in r["coefficients"] for r in results)  # rank deficient
    assert any(_fixture(n)["start"] is not None for n in GLM_FILES)  # user start
    assert {_fixture(n)["family"]["family"] for n in GLM_FILES} == {
        "gaussian",
        "binomial",
        "poisson",
    }


@pytest.mark.parametrize("name", GLM_FILES)
def test_flags_and_counts(name: str) -> None:
    r = _fixture(name)["result"]
    got = _fit(name)
    assert got.iter == r["iter"]
    assert got.converged is r["converged"]
    assert got.boundary is r["boundary"]
    assert got.rank == r["rank"]
    assert got.df_residual == r["df_residual"]
    assert got.df_null == r["df_null"]


@pytest.mark.parametrize("name", GLM_FILES)
def test_coefficients(name: str) -> None:
    r = _fixture(name)["result"]
    np.testing.assert_allclose(
        _fit(name).coefficients, nan_array(r["coefficients"]), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("name", GLM_FILES)
def test_deviance_and_aic(name: str) -> None:
    r = _fixture(name)["result"]
    got = _fit(name)
    assert got.deviance == pytest.approx(r["deviance"], rel=RTOL, abs=ATOL)
    assert got.null_deviance == pytest.approx(r["null_deviance"], rel=RTOL, abs=ATOL)
    assert got.aic == pytest.approx(r["aic"], rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("name", GLM_FILES)
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        ("fitted_values", "fitted_values"),
        ("linear_predictors", "linear_predictors"),
        ("weights", "working_weights"),
        ("residuals", "working_residuals"),
        ("prior_weights", "prior_weights"),
    ],
)
def test_per_observation_vectors(name: str, ours: str, theirs: str) -> None:
    r = _fixture(name)["result"]
    np.testing.assert_allclose(
        getattr(_fit(name), ours), nan_array(r[theirs]), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("name", GLM_FILES)
def test_warnings(name: str) -> None:
    g = _fixture(name)
    assert list(_fit(name).warnings) == g["warnings"]
    x, y, w, off = _inputs(g)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        glm_fit(
            x,
            y,
            get_family(g["family"]["family"]),
            weights=w,
            start=g["start"],
            offset=off,
            maxit=g["control"]["maxit"],
        )
    assert [str(c.message) for c in caught] == g["warnings"]
    assert all(c.category is GlmFitWarning for c in caught)


@pytest.mark.parametrize("name", GLM_FILES)
def test_partykit_glmfit(name: str) -> None:
    """partykit's glmfit on the fixture data: objfun and estfun."""
    g = _fixture(name)
    x, y, w, off = _inputs(g)
    model = GlmModel(
        g["family"]["family"],
        epsilon=g["control"]["epsilon"],
        maxit=g["control"]["maxit"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlmFitWarning)
        got = model.fit(
            y,
            x,
            start=None if g["start"] is None else np.array(g["start"]),
            weights=np.ones_like(y) if w is None else w,
            offset=off,
            cluster=None,
            caseweights=True,
            estfun=True,
            vcov=False,
        )
    expected = g["partykit_glmfit"]
    assert got.objfun == pytest.approx(expected["objfun"], rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(
        got.coefficients, _fit(name).coefficients, rtol=0, atol=0
    )
    assert got.estfun is not None
    np.testing.assert_allclose(
        got.estfun, np.array(expected["estfun"]), rtol=RTOL, atol=ATOL
    )
    assert got.converged is g["result"]["converged"]


LM_FILES = [n for n in GLM_FILES if _fixture(n).get("partykit_lmfit")]


def test_lmfit_fixtures_exist() -> None:
    assert "glm_gaussian_rank_deficient" in LM_FILES
    assert "glm_gaussian_weighted" in LM_FILES


@pytest.mark.parametrize("name", LM_FILES)
def test_partykit_lmfit(name: str) -> None:
    g = _fixture(name)
    x, y, w, off = _inputs(g)
    expected = g["partykit_lmfit"]
    got = LmModel().fit(
        y,
        x,
        start=None,
        weights=np.ones_like(y) if w is None else w,
        offset=off,
        cluster=None,
        caseweights=True,
        estfun=True,
        vcov=False,
    )
    np.testing.assert_allclose(
        got.coefficients, nan_array(expected["coefficients"]), rtol=RTOL, atol=ATOL
    )
    assert got.objfun == pytest.approx(expected["objfun"], rel=RTOL, abs=ATOL)
    assert got.estfun is not None
    np.testing.assert_allclose(
        got.estfun, np.array(expected["estfun"]), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("name", LM_FILES)
def test_lm_fit_matches_glm_fit_for_gaussian(name: str) -> None:
    g = _fixture(name)
    x, y, w, off = _inputs(g)
    lm = lm_fit(x, y, off) if w is None else lm_wfit(x, y, w, off)
    glm = _fit(name)
    np.testing.assert_allclose(lm.coefficients, glm.coefficients, rtol=1e-10)
    assert lm.rank == glm.rank


def test_rank_deficient_aliases_the_last_dependent_column() -> None:
    """x3 = x1 + x2: R keeps x0, x1, x2 and aliases x3."""
    got = _fit("glm_gaussian_rank_deficient")
    assert np.isnan(got.coefficients).tolist() == [False, False, False, True]
    assert got.pivot.tolist() == [0, 1, 2, 3]


# Step halving. No R fixture covers it (docs/decisions.md), so the converged
# result is cross-checked against statsmodels' MLE. The data and start were
# chosen so that the first IRLS step overflows exp() and glm.fit halves it.
# The same call in R 4.6.1 (glm.fit, poisson, start = c(-5, -2),
# maxit = 1000) also takes 267 iterations with one "step size truncated due
# to divergence" warning; that was checked by hand and is not a fixture.


def _step_halving_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261030)
    x1 = np.linspace(0, 100, 40)
    y = rng.poisson(np.exp(0.5 + 0.05 * x1)).astype(np.float64)
    return np.column_stack([np.ones(40), x1]), y


def test_step_halving_converges_to_the_mle() -> None:
    x, y = _step_halving_data()
    with pytest.warns(GlmFitWarning, match="step size truncated due to divergence"):
        got = glm_fit(x, y, Poisson(), start=np.array([-5.0, -2.0]), maxit=1000)
    assert got.converged
    assert got.iter == 267
    assert got.warnings == ("step size truncated due to divergence",)
    # boundary is reset each iteration, so it reports only the last one.
    assert got.boundary is False
    reference = sm.GLM(y, x, family=sm.families.Poisson()).fit(tol=1e-12)
    np.testing.assert_allclose(got.coefficients, reference.params, rtol=1e-8)
    assert got.deviance == pytest.approx(reference.deviance, rel=1e-8)
    # The default start needs no halving and reaches the same MLE.
    default = glm_fit(x, y, Poisson())
    assert default.warnings == ()
    np.testing.assert_allclose(got.coefficients, default.coefficients, rtol=1e-8)


def test_step_halving_is_bounded_by_maxit() -> None:
    """The first full step from the start makes exp() overflow, so the
    deviance is Inf and glm.fit halves back towards the start until it is
    finite. Its inner loop gives up after maxit halvings. Here that takes 26
    halvings, so maxit = 26 fails and maxit = 27 gets through, as in R
    4.6.1 (checked by hand: "inner loop 1; cannot correct step size" at 26,
    and at 27 coefficients -31.0700693395219787, 2.7323154255466497)."""
    x, y = _step_halving_data()
    start = np.array([-5.0, -2.0])
    with (
        warnings.catch_warnings(),
        pytest.raises(GlmFitError, match="inner loop 1; cannot correct step size"),
    ):
        warnings.simplefilter("ignore", GlmFitWarning)
        glm_fit(x, y, Poisson(), start=start, maxit=26)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlmFitWarning)
        got = glm_fit(x, y, Poisson(), start=start, maxit=27)
    assert not got.converged
    assert got.warnings == (
        "step size truncated due to divergence",
        "glm.fit: algorithm did not converge",
    )
    np.testing.assert_allclose(
        got.coefficients, [-31.0700693395219787, 2.7323154255466497], rtol=1e-8
    )


def test_wrong_start_length_raises() -> None:
    x, y = _step_halving_data()
    with pytest.raises(GlmFitError, match="length of 'start' should equal 2"):
        glm_fit(x, y, Poisson(), start=np.zeros(3))


def test_family_input_checks() -> None:
    x = np.ones((3, 1))
    with pytest.raises(ValueError, match="0 <= y <= 1"):
        glm_fit(x, np.array([0.0, 2.0, 1.0]), Binomial())
    with pytest.raises(ValueError, match="negative values"):
        glm_fit(x, np.array([0.0, -1.0, 1.0]), Poisson())
    with pytest.raises(ValueError, match="not supported"):
        get_family("Gamma")


def test_binomial_zero_weight_rows_are_set_to_zero() -> None:
    """binomial()$initialize sets y to 0 where the weight is 0, so an
    out-of-range y there is accepted."""
    x = np.column_stack([np.ones(6), np.arange(6.0)])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 5.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    got = glm_fit(x, y, Binomial(), weights=w)
    assert got.y[-1] == 0.0
    ref = glm_fit(x[:5], y[:5], Binomial())
    np.testing.assert_allclose(got.coefficients, ref.coefficients, rtol=1e-12)


def test_gaussian_aic_with_zero_weight_is_inf_as_in_r() -> None:
    """R's gaussian()$aic subtracts sum(log(wt)), so a zero weight gives Inf.
    glm_fit keeps this; GlmModel does not (see test_node_models.py)."""
    rng = np.random.default_rng(3)
    x = np.column_stack([np.ones(20), rng.standard_normal(20)])
    y = rng.standard_normal(20)
    w = np.r_[0.0, np.ones(19)]
    assert glm_fit(x, y, Gaussian(), weights=w).aic == np.inf


# The QR port. Aliasing and pivoting follow dqrdc2.


def test_dqrdc2_moves_a_dependent_column_to_the_end() -> None:
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal((2, 30))
    x = np.column_stack([np.ones(30), a, 2 * a, b])
    _, rank, _, pivot = dqrdc2(x, 1e-7)
    assert rank == 3
    assert pivot.tolist() == [0, 1, 3, 2]
    fit = lm_fit(x, rng.standard_normal(30))
    assert np.isnan(fit.coefficients).tolist() == [False, False, True, False]


def test_dqrdc2_zero_column() -> None:
    rng = np.random.default_rng(5)
    x = np.column_stack([np.zeros(10), rng.standard_normal((10, 2))])
    _, rank, _, pivot = dqrdc2(x, 1e-7)
    assert rank == 2
    assert pivot.tolist() == [1, 2, 0]


def test_dqrls_full_rank_matches_lstsq() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal((50, 4))
    y = rng.standard_normal(50)
    fit = dqrls(x, y, 1e-7)
    expected = np.linalg.lstsq(x, y, rcond=None)[0]
    assert fit.rank == 4
    assert not fit.pivoted
    np.testing.assert_allclose(fit.coefficients, expected, rtol=1e-12)
    np.testing.assert_allclose(fit.residuals, y - x @ expected, atol=1e-12)


def test_dqrls_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError, match="NA/NaN/Inf in 'y'"):
        dqrls(np.ones((2, 1)), np.array([1.0, np.inf]), 1e-7)
    with pytest.raises(ValueError, match="NA/NaN/Inf in 'x'"):
        dqrls(np.array([[1.0], [np.nan]]), np.ones(2), 1e-7)


def test_lm_wfit_zero_weight_residuals() -> None:
    """lm.wfit fits on positive weights and gives zero-weight rows the
    residual y - x b."""
    rng = np.random.default_rng(7)
    x = np.column_stack([np.ones(12), rng.standard_normal(12)])
    y = rng.standard_normal(12)
    w = np.r_[np.zeros(3), rng.uniform(0.5, 2, 9)]
    got = lm_wfit(x, y, w)
    ref = lm_wfit(x[3:], y[3:], w[3:])
    np.testing.assert_allclose(got.coefficients, ref.coefficients, rtol=1e-12)
    np.testing.assert_allclose(got.residuals[:3], y[:3] - x[:3] @ ref.coefficients)
    assert got.df_residual == 7
