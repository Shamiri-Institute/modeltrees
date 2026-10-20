"""The lmtree and glmtree node models against the tree fixtures.

Every node's rows are refit from the data CSV, and the coefficients,
objective and scores are compared with what partykit stored for that node.
statsmodels gives an independent check of the same fits; it never replaces
the R fixtures.
"""

import warnings
from functools import cache
from typing import Any

import numpy as np
import pytest
import statsmodels.api as sm  # type: ignore[import-untyped]
from fixtures_io import nan_array
from tree_fixtures import TREE_FILES
from tree_fixtures import data as _data
from tree_fixtures import inputs as _inputs
from tree_fixtures import model as _model
from tree_fixtures import tree as _tree

from modeltrees.models.families import GlmFitWarning
from modeltrees.models.glm import GlmModel, binomial_response
from modeltrees.models.glm_fit import glm_fit
from modeltrees.models.lm import LmModel
from modeltrees.node_model import FitResult

RTOL = 1e-8
ATOL = 1e-10


def _nodes() -> list[Any]:
    return [
        pytest.param(name, node["id"], id=f"{name}-{node['id']}")
        for name in TREE_FILES
        for node in _tree(name)["nodes"]
    ]


NODES = _nodes()


@cache
def _node_fit(name: str, node_id: int) -> FitResult:
    tree = _tree(name)
    node = next(n for n in tree["nodes"] if n["id"] == node_id)
    y, x, w, off = _inputs(tree)
    rows = np.array(node["rows"], dtype=np.intp)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GlmFitWarning)  # no fixture node warns
        return _model(tree).fit(
            y[rows],
            None if x is None else x[rows],
            start=None,
            weights=w[rows],
            offset=None if off is None else off[rows],
            cluster=None,
            caseweights=tree["control"]["caseweights"],
            estfun=True,
            vcov=False,
        )


def _node(name: str, node_id: int) -> dict[str, Any]:
    return next(n for n in _tree(name)["nodes"] if n["id"] == node_id)


def test_every_tree_and_node_is_covered() -> None:
    assert len(TREE_FILES) == 32
    assert len(NODES) == 280
    assert all(_node(*p.values)["estfun"] is not None for p in NODES)


@pytest.mark.parametrize(("name", "node_id"), NODES)
def test_coefficients(name: str, node_id: int) -> None:
    node = _node(name, node_id)
    got = _node_fit(name, node_id)
    assert got.coef_names == tuple(node["coefficient_names"])
    np.testing.assert_allclose(
        got.coefficients, nan_array(node["coefficients"]), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(("name", "node_id"), NODES)
def test_objfun(name: str, node_id: int) -> None:
    node = _node(name, node_id)
    assert _node_fit(name, node_id).objfun == pytest.approx(
        node["objfun"], rel=RTOL, abs=ATOL
    )


@pytest.mark.parametrize(("name", "node_id"), NODES)
def test_estfun(name: str, node_id: int) -> None:
    node = _node(name, node_id)
    got = _node_fit(name, node_id).estfun
    assert got is not None
    expected = np.array(node["estfun"])
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


def test_glmtree_gaussian_matches_lmtree_coefficients() -> None:
    """The same tree fitted by lmtree and by glmtree(gaussian)."""
    for lm, glm in [
        ("sim_gaussian_lmtree_multisplit", "sim_gaussian_glmtree_gaussian_multisplit"),
    ]:
        for node in _tree(lm)["nodes"]:
            np.testing.assert_allclose(
                _node_fit(lm, node["id"]).coefficients,
                _node_fit(glm, node["id"]).coefficients,
                rtol=1e-10,
            )


# Independent cross-check with statsmodels. Not a substitute for R.


@pytest.mark.parametrize(("name", "node_id"), NODES)
def test_statsmodels_cross_check(name: str, node_id: int) -> None:
    tree = _tree(name)
    node = _node(name, node_id)
    y, x, w, off = _inputs(tree)
    rows = np.array(node["rows"], dtype=np.intp)
    xr = np.ones((rows.shape[0], 1)) if x is None else x[rows]
    yr, wr = y[rows], w[rows]
    offr = None if off is None else off[rows]
    got = _node_fit(name, node_id)
    family = tree["family"]["family"] if tree["family"] else "gaussian"
    if family == "gaussian":
        ref = sm.WLS(yr, xr, weights=wr).fit()
        np.testing.assert_allclose(got.coefficients, ref.params, rtol=1e-7, atol=1e-9)
        if tree["function_name"] == "lmtree":
            assert got.objfun == pytest.approx(ref.ssr, rel=1e-7)
        else:
            # glmtree's objective is the negative gaussian log-likelihood.
            assert got.objfun == pytest.approx(-ref.llf, rel=1e-7)
        return
    sm_family = {"binomial": sm.families.Binomial(), "poisson": sm.families.Poisson()}
    ref_glm = sm.GLM(
        yr, xr, family=sm_family[family], freq_weights=wr, offset=offr
    ).fit(tol=1e-12)
    np.testing.assert_allclose(got.coefficients, ref_glm.params, rtol=1e-7, atol=1e-9)
    # With frequency weights the log-likelihood is the weighted sum, which is
    # glmtree's objective for these families.
    assert got.objfun == pytest.approx(-ref_glm.llf, rel=1e-7)
    # statsmodels' score_obs is the exact score w * (y - mu) * x, frequency
    # weights included. glmtree's scores are not quite that: glm.fit returns
    # the working weights from the start of its last iteration with the
    # residuals at the final fit. Near the optimum the deviance changes
    # quadratically, so the deviance criterion lets the last step be about
    # sqrt(epsilon), and at epsilon = 1e-8 the two scores differ by up to
    # about 1e-4 (docs/model-fitting.md). The formula (weights, dispersion,
    # columns) is therefore checked on a refit with epsilon = 1e-13, where
    # the gap is below 1e-5. R's own scores are matched at 1e-8 above.
    tight = GlmModel(family, epsilon=1e-13).fit(
        yr, xr, weights=wr, offset=offr, estfun=True
    )
    assert tight.estfun is not None
    expected = ref_glm.model.score_obs(tight.coefficients)
    np.testing.assert_allclose(
        tight.estfun, expected, rtol=1e-5, atol=1e-8 * np.abs(expected).max()
    )


# Behaviour not covered by a fixture.


def _small_data(seed: int, n: int = 40) -> tuple[np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    return x, rng


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
def test_glm_zero_weights_equal_deleted_rows(family: str) -> None:
    x, rng = _small_data(1)
    n = x.shape[0]
    eta = 0.3 + 0.8 * x[:, 1]
    y = {
        "gaussian": eta + rng.standard_normal(n),
        "binomial": (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float),
        "poisson": rng.poisson(np.exp(eta)).astype(float),
    }[family]
    w = rng.integers(1, 4, size=n).astype(float)
    w[[0, 5, 9]] = 0.0
    keep = w > 0
    model = GlmModel(family)
    kw: dict[str, Any] = {
        "start": None,
        "offset": None,
        "cluster": None,
        "caseweights": True,
        "estfun": True,
        "vcov": False,
    }
    got = model.fit(y, x, weights=w, **kw)
    ref = model.fit(y[keep], x[keep], weights=w[keep], **kw)
    np.testing.assert_allclose(got.coefficients, ref.coefficients, rtol=1e-12)
    assert got.objfun == pytest.approx(ref.objfun, rel=1e-12)
    assert got.estfun is not None and ref.estfun is not None
    assert np.all(got.estfun[~keep] == 0)
    np.testing.assert_allclose(got.estfun[keep], ref.estfun, rtol=1e-12)
    # partykit's objective is R's aic/2 - df on all rows. That is the same
    # for binomial and poisson, and Inf for gaussian (docs/model-fitting.md).
    full = glm_fit(x, y, model.family, weights=w)
    r_objfun = full.aic / 2 - (full.rank + (family == "gaussian"))
    if family == "gaussian":
        assert r_objfun == np.inf
    else:
        assert got.objfun == pytest.approx(r_objfun, rel=1e-12)


def test_lm_zero_weights_equal_deleted_rows() -> None:
    x, rng = _small_data(2)
    y = x @ [1.0, 2.0] + rng.standard_normal(x.shape[0])
    w = np.ones(x.shape[0])
    w[[1, 2]] = 0.0
    kw: dict[str, Any] = {
        "start": None,
        "offset": None,
        "cluster": None,
        "caseweights": True,
        "estfun": True,
        "vcov": False,
    }
    got = LmModel().fit(y, x, weights=w, **kw)
    ref = LmModel().fit(y[w > 0], x[w > 0], weights=w[w > 0], **kw)
    np.testing.assert_allclose(got.coefficients, ref.coefficients, rtol=1e-12)
    assert got.objfun == pytest.approx(ref.objfun, rel=1e-12)
    assert got.estfun is not None
    assert np.all(got.estfun[w == 0] == 0)


def test_gaussian_dispersion_with_and_without_case_weights() -> None:
    """glmtree.R:100-107: with case weights the dispersion divides each
    squared working score by its weight."""
    x, rng = _small_data(3)
    n = x.shape[0]
    y = x @ [0.5, -1.0] + rng.standard_normal(n)
    w = rng.integers(1, 5, size=n).astype(float)
    kw: dict[str, Any] = {
        "start": None,
        "offset": None,
        "cluster": None,
        "estfun": True,
        "vcov": False,
    }
    with_cw = GlmModel("gaussian").fit(y, x, weights=w, caseweights=True, **kw)
    without = GlmModel("gaussian").fit(y, x, weights=w, caseweights=False, **kw)
    z = glm_fit(x, y, GlmModel("gaussian").family, weights=w)
    wres = z.residuals * z.weights
    raw = wres[:, None] * x
    assert with_cw.estfun is not None and without.estfun is not None
    np.testing.assert_allclose(
        with_cw.estfun, raw / (np.sum(wres**2 / w) / np.sum(z.weights)), rtol=1e-12
    )
    np.testing.assert_allclose(
        without.estfun, raw / (np.sum(wres**2) / np.sum(z.weights)), rtol=1e-12
    )
    assert with_cw.objfun == without.objfun


def test_aliased_columns_are_nan_and_left_out_of_estfun() -> None:
    x, rng = _small_data(4)
    x = np.column_stack([x, 2 * x[:, 1]])
    y = x[:, :2] @ [1.0, 1.0] + rng.standard_normal(x.shape[0])
    kw: dict[str, Any] = {
        "start": None,
        "offset": None,
        "cluster": None,
        "caseweights": True,
        "estfun": True,
        "vcov": False,
    }
    for model in (LmModel(), GlmModel("gaussian")):
        got = model.fit(y, x, weights=np.ones(x.shape[0]), **kw)
        assert np.isnan(got.coefficients).tolist() == [False, False, True]
        assert got.estfun is not None and got.estfun.shape == (x.shape[0], 2)
        assert got.coef_names == ("x1", "x2", "x3")


def test_intercept_only_when_x_is_none() -> None:
    y = np.array([1.0, 2.0, 4.0])
    kw: dict[str, Any] = {
        "start": None,
        "offset": None,
        "cluster": None,
        "caseweights": True,
        "estfun": True,
        "vcov": False,
    }
    for model in (LmModel(), GlmModel("gaussian")):
        got = model.fit(y, None, weights=np.ones(3), **kw)
        assert got.coef_names == ("(Intercept)",)
        np.testing.assert_allclose(got.coefficients, [7 / 3])


def test_vcov_is_out_of_scope() -> None:
    kw: dict[str, Any] = {
        "start": None,
        "offset": None,
        "cluster": None,
        "caseweights": True,
        "estfun": False,
    }
    for model in (GlmModel("poisson"),):
        with pytest.raises(NotImplementedError, match="v0.1"):
            model.fit(np.ones(3), None, weights=np.ones(3), vcov=True, **kw)


def test_n_response_columns() -> None:
    assert LmModel().n_response_columns(np.ones(4)) == 1
    assert GlmModel().n_response_columns(np.ones(4)) == 1
    assert LmModel().n_response_columns(np.ones((4, 2))) == 2


def test_wrong_number_of_names_raises() -> None:
    with pytest.raises(ValueError, match="2 coefficient names for 1 columns"):
        LmModel(coef_names=["a", "b"]).fit(
            np.ones(3), None, weights=np.ones(3), estfun=False
        )


# The binomial response at the boundary (docs/decisions.md).


def test_binomial_response_first_level_is_failure() -> None:
    y = np.array(["pos", "neg", "neg", "pos"])
    np.testing.assert_array_equal(binomial_response(y), [1, 0, 0, 1])
    np.testing.assert_array_equal(
        binomial_response(y, levels=["pos", "neg"]), [0, 1, 1, 0]
    )
    three = np.array(["a", "b", "c", "a"])
    np.testing.assert_array_equal(binomial_response(three), [0, 1, 1, 0])
    np.testing.assert_array_equal(binomial_response(np.array([True, False])), [1, 0])


def test_binomial_response_uses_categorical_levels() -> None:
    class Categorical:  # the attributes pandas.Categorical exposes
        categories = ("yes", "no")

        def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
            return np.array(["no", "yes", "no"], dtype=object)

    np.testing.assert_array_equal(binomial_response(Categorical()), [1, 0, 1])


def test_binomial_response_rejects_unknown_and_missing_values() -> None:
    with pytest.raises(ValueError, match="not among the levels"):
        binomial_response(np.array(["a", "b", "z"]), levels=["a", "b"])
    with pytest.raises(ValueError, match="missing"):
        binomial_response(np.array(["a", None], dtype=object))


def test_string_response_needs_conversion() -> None:
    with pytest.raises(TypeError, match="binomial_response"):
        GlmModel("binomial").fit(
            np.array(["a", "b"]), None, weights=np.ones(2), estfun=False
        )


def test_synth_diabetes_response_encoding() -> None:
    """R's glm.fit encodes the diabetes factor (levels neg, pos) as pos = 1."""
    tree = _tree("synth_diabetes_glmtree_binomial")
    y, _, _, _ = _inputs(tree)
    data, columns = _data(tree["data_csv"])
    assert columns["diabetes"]["levels"] == ["neg", "pos"]
    np.testing.assert_array_equal(y, data["diabetes"] == 1)
