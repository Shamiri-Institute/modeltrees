"""The safe logit model against the R fixtures.

In the order of CLAUDE.md: single fits (tests/fixtures/node_fits/safelogit_*),
then split search and whole trees (tests/fixtures/extension_trees/). The
oracle is r/extensions/safelogit.R (docs/extensions-spec.md, section 4,
provisional).

Under complete separation glm.fit stops where its deviance criterion
happens to be met, not at an estimate (the MLE does not exist), so the
coefficients of completely separated nodes are compared loosely. Their
objective, separation class and warnings are compared as usual.
"""

import math
import warnings
from functools import cache
from typing import Any

import numpy as np
import pytest
from fixtures_io import FIXTURES, load_json, nan_array
from tree_fixtures import inputs, variables

from modeltrees import reference_path
from modeltrees.instability import VariableKind
from modeltrees.models.families import GlmFitWarning
from modeltrees.models.glm import GlmModel
from modeltrees.models.safelogit import SafeLogitModel, classify_separation
from modeltrees.node_model import FitResult, FitStatus
from modeltrees.split_search import find_split
from modeltrees.tree import GrownTree, MobControl, StopReason, mob

RTOL = 1e-8
ATOL = 1e-10
# Coefficients of completely separated fits: where IRLS stopped.
SEPARATED_RTOL = 1e-3

NODE_FITS = sorted(p.stem for p in (FIXTURES / "node_fits").glob("safelogit_*.json"))
TREES = sorted(
    p.stem
    for p in (FIXTURES / "extension_trees").glob("*.json")
    if load_json(f"extension_trees/{p.name}")["fit_function"] == "safelogitfit"
)


def test_fixture_lists() -> None:
    assert NODE_FITS == [
        "safelogit_complete",
        "safelogit_constant",
        "safelogit_none",
        "safelogit_quasi",
    ]
    assert TREES == ["sim_separation_safelogit", "sim_separation_safelogit_plain"]


def _warnings(value: str | list[str]) -> tuple[str, ...]:
    # jsonlite writes a length-one character vector as a string.
    return (value,) if isinstance(value, str) else tuple(value)


def _assert_model_info(extras: Any, info: dict[str, Any]) -> None:
    assert extras["status"] == info["status"]
    assert extras["separation"] == info["separation"]
    assert extras["n_boundary"] == info["n_boundary"]
    assert extras["converged"] == info["converged"]
    assert extras["boundary"] == info["boundary"]
    assert extras["warnings"] == _warnings(info["warnings"])


def _assert_coefficients(got: FitResult, expected: list[float | None]) -> None:
    complete = got.extras["separation"] == "complete"
    np.testing.assert_allclose(
        got.coefficients,
        nan_array(expected),
        rtol=SEPARATED_RTOL if complete else RTOL,
        atol=ATOL,
    )


# 1. Single fits.


@cache
def _node_fixture(name: str) -> dict[str, Any]:
    fixture: dict[str, Any] = load_json(f"node_fits/{name}.json")
    return fixture


def _node_fit(name: str, estfun: bool = True) -> FitResult:
    f = _node_fixture(name)
    model = SafeLogitModel(coef_names=f["x_names"], **f["fit_args"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # safe logit never emits warnings
        return model.fit(
            np.array(f["y"], dtype=np.float64),
            np.array(f["x"], dtype=np.float64),
            start=None if f["start"] is None else np.array(f["start"], np.float64),
            weights=np.array(f["weights"], dtype=np.float64),
            offset=None if f["offset"] is None else np.array(f["offset"], np.float64),
            estfun=estfun,
        )


@pytest.mark.parametrize("name", NODE_FITS)
def test_node_fit(name: str) -> None:
    expected = _node_fixture(name)["result"]
    got = _node_fit(name)
    assert got.coef_names == tuple(expected["coefficient_names"])
    _assert_coefficients(got, expected["coefficients"])
    assert got.objfun == pytest.approx(expected["objfun"], rel=RTOL, abs=ATOL)
    _assert_model_info(got.extras, expected["model_info"])
    if expected["estfun"] is None:
        assert got.status is FitStatus.NO_ESTFUN
        assert got.estfun is None
    else:
        assert got.status is FitStatus.OK
        assert got.estfun is not None
        np.testing.assert_allclose(
            got.estfun, np.array(expected["estfun"]), rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("name", NODE_FITS)
def test_objective_only_fit_classifies_separation(name: str) -> None:
    # Split search's fits: the same coefficients, objective, status and
    # extras as the node fit, without scores.
    full = _node_fit(name)
    quick = _node_fit(name, estfun=False)
    np.testing.assert_array_equal(quick.coefficients, full.coefficients)
    assert quick.objfun == full.objfun
    assert quick.status is full.status
    assert quick.estfun is None
    assert quick.extras == full.extras
    with reference_path():
        ref = _node_fit(name, estfun=False)
    assert ref.status is full.status and ref.extras == full.extras


def test_without_separation_equals_glm_binomial() -> None:
    f = _node_fixture("safelogit_none")
    assert f["checks"]["identical_to_partykit_glmfit"] is True
    y = np.array(f["y"], np.float64)
    x = np.array(f["x"], np.float64)
    w = np.array(f["weights"], np.float64)
    glm = GlmModel("binomial", coef_names=f["x_names"])
    safe = SafeLogitModel(coef_names=f["x_names"])
    for estfun in (True, False):
        a = glm.fit(y, x, weights=w, estfun=estfun)
        b = safe.fit(y, x, weights=w, estfun=estfun)
        np.testing.assert_array_equal(b.coefficients, a.coefficients)
        assert b.objfun == a.objfun
        assert b.converged == a.converged
        if estfun:
            assert a.estfun is not None and b.estfun is not None
            np.testing.assert_array_equal(b.estfun, a.estfun)


def test_quasi_separation_scores() -> None:
    # docs/extensions-spec.md, section 4: boundary rows have numerically
    # zero scores, and the diverging coefficient's score sums to about -2e-6.
    f = _node_fixture("safelogit_quasi")
    got = _node_fit("safelogit_quasi")
    assert got.estfun is not None
    y = np.array(f["y"], np.float64)
    x = np.array(f["x"], np.float64)
    at_bound = np.abs(y - 1 / (1 + np.exp(-(x @ got.coefficients)))) < 1e-6
    assert int(at_bound.sum()) == 83
    assert np.all(np.abs(got.estfun[at_bound]) < 1e-6 * np.abs(x[at_bound]) + 1e-15)
    np.testing.assert_allclose(
        got.estfun.sum(axis=0), f["checks"]["estfun_colsums"], rtol=1e-6, atol=1e-12
    )


# 2. Split search and trees.


@cache
def _tree_fixture(name: str) -> dict[str, Any]:
    fixture: dict[str, Any] = load_json(f"extension_trees/{name}.json")
    return fixture


def _model(fixture: dict[str, Any]) -> SafeLogitModel:
    return SafeLogitModel(
        coef_names=fixture["coefficient_names"], **fixture["fit_args"]
    )


def _control(fixture: dict[str, Any]) -> MobControl:
    c = fixture["control"]
    assert c["maxdepth"] == "Inf" and c["prune"] is None
    return MobControl(
        alpha=c["alpha"],
        bonferroni=c["bonferroni"],
        minsize=c["minsize"],
        trim=c["trim"],
        dfsplit=c["dfsplit"],
        restart=c["restart"],
        caseweights=c["caseweights"],
        numsplit=c["numsplit"],
        catsplit=c["catsplit"],
    )


def _grow(
    fixture: dict[str, Any], model: Any, response: str | None = None
) -> GrownTree:
    if response is not None:
        fixture = {**fixture, "response": response}
    y, x, w, off = inputs(fixture)
    return mob(
        model,
        y,
        x,
        variables(fixture),
        weights=w,
        offset=off,
        control=_control(fixture),
    )


@cache
def _grown(name: str) -> GrownTree:
    fixture = _tree_fixture(name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warnings, as in R
        return _grow(fixture, _model(fixture))


def _objective(values: list[float | None]) -> np.ndarray:
    return np.array([np.inf if v is None else v for v in values], dtype=np.float64)


SEARCH_NODES = [
    pytest.param(name, node["id"], id=f"{name}-{node['id']}")
    for name in TREES
    for node in _tree_fixture(name)["nodes"]
    if node["split_search"] is not None
]


@pytest.mark.parametrize(("name", "node_id"), SEARCH_NODES)
def test_split_search(name: str, node_id: int) -> None:
    """Split search on the fixture node's rows, before growing any tree."""
    fixture = _tree_fixture(name)
    node = next(n for n in fixture["nodes"] if n["id"] == node_id)
    y, x, w, off = inputs(fixture)
    var = variables(fixture)[node["split_search"]["variable"]]
    rows = np.array(node["rows"], dtype=np.intp)
    m = _model(fixture)

    def fit(mask: np.ndarray, start: np.ndarray | None) -> FitResult:
        r = rows[mask]
        return m.fit(
            y[r],
            None if x is None else x[r],
            start=start,
            weights=w[r],
            offset=None if off is None else off[r],
            estfun=False,
        )

    got = find_split(
        fixture["partitioning"].index(var.name),
        var.kind,
        var.values[rows],
        var.levels,
        w[rows],
        fit,
        minsize=node["minsize"],
    )
    expected = node["split_search"]
    assert got.minsize == expected["minsize"]
    assert got.n_failed == 0
    if var.kind is VariableKind.NUMERIC:
        assert got.values is not None
        np.testing.assert_array_equal(got.values, np.array(expected["candidates"]))
    else:
        assert got.labels == tuple(expected["candidates"])
    exp = _objective(expected["objective"])
    np.testing.assert_array_equal(np.isinf(got.objective), np.isinf(exp))
    finite = np.isfinite(exp)
    np.testing.assert_allclose(got.objective[finite], exp[finite], rtol=RTOL, atol=ATOL)
    _assert_split(got.split, node["split"])


def _assert_split(got: Any, expected: dict[str, Any] | None) -> None:
    if expected is None:
        assert got is None
        return
    assert got is not None
    assert got.variable == expected["variable_index"]
    assert got.kind.value == expected["kind"]
    if expected["kind"] == "nominal":
        assert got.index == tuple(expected["index"])
    else:
        assert got.breaks == expected["breaks"]


@pytest.mark.parametrize("name", TREES)
def test_tree_nodes(name: str) -> None:
    fixture = _tree_fixture(name)
    got = _grown(name)
    assert got.minsize == fixture["root_minsize"]
    assert len(got.nodes) == len(fixture["nodes"])
    parents = {k.id: n.id for n in got.nodes for k in n.kids}
    for g, e in zip(got.nodes, fixture["nodes"], strict=True):
        assert g.id == e["id"]
        assert g.depth == e["depth"]
        assert parents.get(g.id) == e["parent"]
        assert [k.id for k in g.kids] == e["kids"]
        assert g.stop_reason.value == e["stop_reason"]
        np.testing.assert_array_equal(g.rows, e["rows"])
        assert g.nobs == e["nobs"]
        _assert_split(g.split, e["split"])
        assert g.fit.coef_names == tuple(e["coefficient_names"])
        _assert_coefficients(g.fit, e["coefficients"])
        assert g.fit.objfun == pytest.approx(e["objfun"], rel=RTOL, abs=ATOL)
        _assert_model_info(g.fit.extras, e["model_info"])
        if e["estfun"] is None:
            assert g.fit.status is FitStatus.NO_ESTFUN and g.fit.estfun is None
        else:
            assert g.fit.estfun is not None
            np.testing.assert_allclose(
                g.fit.estfun, np.array(e["estfun"]), rtol=RTOL, atol=ATOL
            )


@pytest.mark.parametrize("name", TREES)
def test_tree_tests_and_searches(name: str) -> None:
    for g, e in zip(_grown(name).nodes, _tree_fixture(name)["nodes"], strict=True):
        status = e["test"]["status"] if e["test"] else "not_run"
        if status == "not_run":
            assert g.test is None and not g.test_failed and g.p_value is None
        elif status == "failed":
            assert g.test is None and g.test_failed and g.p_value is None
        else:
            assert status == "ok"
            t = e["test"]
            assert g.test is not None
            np.testing.assert_allclose(g.test.statistic, t["statistic"], rtol=RTOL)
            np.testing.assert_allclose(
                g.test.p_value_adjusted, nan_array(t["p_value_adjusted"]), rtol=RTOL
            )
            assert g.test.best == t["best"]
            assert g.p_value == pytest.approx(e["p_value"], rel=RTOL)
        if e["split_search"] is None:
            assert g.split_search is None
            continue
        assert g.split_search is not None and g.split_search.n_failed == 0
        exp = _objective(e["split_search"]["objective"])
        got = g.split_search.objective
        np.testing.assert_array_equal(np.isinf(got), np.isinf(exp))
        finite = np.isfinite(exp)
        np.testing.assert_allclose(got[finite], exp[finite], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("name", TREES)
def test_tree_summary(name: str) -> None:
    fixture = _tree_fixture(name)
    got = _grown(name)
    np.testing.assert_array_equal(got.fitted_node, fixture["fitted_node"])
    summary = fixture["summary"]
    leaves = [n for n in got.nodes if n.terminal]
    assert len(got.nodes) == summary["n_nodes"]
    assert len(leaves) == summary["n_terminal"]
    assert max(n.depth for n in got.nodes) == summary["depth"]
    assert sum(n.fit.objfun for n in leaves) == pytest.approx(
        summary["objfun"], rel=RTOL
    )


def test_separated_nodes_are_terminal_without_tests() -> None:
    got = _grown("sim_separation_safelogit")
    separated = [n for n in got.nodes if n.stop_reason is StopReason.NO_ESTFUN]
    assert [n.id for n in separated] == [4, 5]
    assert all(n.fit.extras["separation"] == "complete" for n in separated)


def test_plain_tree_equals_glmtree() -> None:
    # Without separation the whole tree is glmtree's with binomial(), as the
    # generator checked in R.
    fixture = _tree_fixture("sim_separation_safelogit_plain")
    safe = _grown("sim_separation_safelogit_plain")
    glm = _grow(
        fixture,
        GlmModel("binomial", coef_names=fixture["coefficient_names"]),
    )
    assert len(safe.nodes) == len(glm.nodes)
    for a, b in zip(safe.nodes, glm.nodes, strict=True):
        assert a.id == b.id and a.stop_reason is b.stop_reason
        np.testing.assert_array_equal(a.rows, b.rows)
        assert a.split == b.split
        np.testing.assert_array_equal(a.fit.coefficients, b.fit.coefficients)
        assert a.fit.objfun == b.fit.objfun
        assert a.fit.estfun is not None and b.fit.estfun is not None
        np.testing.assert_array_equal(a.fit.estfun, b.fit.estfun)
        assert (a.test is None) == (b.test is None)
        if a.test is not None and b.test is not None:
            np.testing.assert_array_equal(a.test.statistic, b.test.statistic)
            np.testing.assert_array_equal(a.test.log_p_value, b.test.log_p_value)
        if a.split_search is not None and b.split_search is not None:
            np.testing.assert_array_equal(
                a.split_search.objective, b.split_search.objective
            )


def test_plain_glmtree_splits_the_separated_group_further() -> None:
    # docs/extensions-spec.md, section 4: glmtree grows the same first two
    # splits, then splits the constant-response group (z2 = g4) on noise.
    fixture = _tree_fixture("sim_separation_safelogit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlmFitWarning)
        glm = _grow(
            fixture, GlmModel("binomial", coef_names=fixture["coefficient_names"])
        )
    safe = _grown("sim_separation_safelogit")
    assert glm.node(1).split == safe.node(1).split
    assert glm.node(2).split == safe.node(2).split
    assert len(glm.nodes) > len(safe.nodes)


# 3. Unit tests without an R fixture.


def _sim(seed: int, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    p = 1 / (1 + np.exp(-(x @ np.array([0.2, 0.8]))))
    return x, (rng.random(n) < p).astype(np.float64)


def test_glm_fit_error_is_a_failed_fit() -> None:
    # glm.fit stops on a start of the wrong length; safelogitfit catches it.
    x, y = _sim(1)
    r = SafeLogitModel().fit(
        y, x, start=np.zeros(3), weights=np.ones(y.shape[0]), estfun=True
    )
    assert r.status is FitStatus.FAILED and math.isinf(r.objfun)
    assert r.estfun is None and np.all(np.isnan(r.coefficients))
    assert r.extras["status"] == "error"
    assert r.reason is not None and "length of 'start'" in r.reason
    assert r.extras["separation"] is None and r.extras["n_boundary"] is None


def test_non_binary_response_raises() -> None:
    with pytest.raises(ValueError, match="binary response"):
        SafeLogitModel().fit(np.array([0.0, 0.5, 1.0]), None, weights=np.ones(3))


def test_zero_weights_equal_deleted_rows() -> None:
    x, y = _sim(2)
    n = int(y.shape[0])
    w = np.random.default_rng(3).integers(0, 3, size=(n,)).astype(np.float64)
    keep = w > 0
    m = SafeLogitModel()
    full = m.fit(y, x, weights=w, estfun=True)
    kept = m.fit(y[keep], x[keep], weights=w[keep], estfun=True)
    np.testing.assert_array_equal(full.coefficients, kept.coefficients)
    assert full.objfun == kept.objfun and full.extras == kept.extras
    assert full.estfun is not None and kept.estfun is not None
    np.testing.assert_array_equal(full.estfun[keep], kept.estfun)
    assert np.all(full.estfun[~keep] == 0)


def test_separation_ignores_zero_weight_rows() -> None:
    # A zero-weight row that breaks the separation does not count.
    x = np.column_stack([np.ones(40), np.linspace(-1, 1, 40)])
    y = (x[:, 1] > 0).astype(np.float64)
    y[0] = 1.0
    w = np.ones(40)
    w[0] = 0.0
    r = SafeLogitModel().fit(y, x, weights=w, estfun=True)
    assert r.extras["separation"] == "complete"
    assert r.status is FitStatus.NO_ESTFUN
    assert r.extras["n_boundary"] == 39


def test_classify_separation() -> None:
    y = np.array([0.0, 1.0, 1.0])
    assert classify_separation(y, np.array([1e-9, 1 - 1e-9, 0.5]), 1e-6) == (
        "quasi",
        2,
    )
    assert classify_separation(y, np.array([1e-9, 1 - 1e-9, 1.0]), 1e-6) == (
        "complete",
        3,
    )
    assert classify_separation(y, np.array([0.2, 0.7, 0.5]), 1e-6) == ("none", 0)
