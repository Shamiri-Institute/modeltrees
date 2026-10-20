"""The multinomial logit model against the R fixtures.

In the order of CLAUDE.md: single fits (tests/fixtures/node_fits/multinom_*),
then split search and whole trees (tests/fixtures/extension_trees/). The
oracle is r/extensions/multinom.R on nnet 7.3-21 (docs/extensions-spec.md,
section 5, provisional).

R's BFGS stops with a small but nonzero gradient g_R, recorded in every
fixture as the column sums of R's scores. Newton's method here converges to
the MLE, so R's coefficients differ from it by -H^-1 g_R to first order (H
the Hessian of minus the log-likelihood). The coefficient, score and test
statistic tolerances are derived from that offset, fixture by fixture
(docs/extensions-port.md, section 7); objectives, which are flat at the
optimum, and split-search objectives use the usual tolerance.
"""

import math
from functools import cache
from typing import Any

import numpy as np
import pytest
from fixtures_io import FIXTURES, load_json, nan_array
from tree_fixtures import cluster, data, variables

from modeltrees.instability import (
    FluctuationTests,
    PartitioningVariable,
    VariableKind,
    fluctuation_tests,
)
from modeltrees.models.multinom import (
    MultinomModel,
    multinom_evaluate,
    multinom_hessian,
    multinom_newton,
    multinom_response,
    multinom_scores,
)
from modeltrees.node_model import FitResult, FitStatus
from modeltrees.split_search import find_split
from modeltrees.tree import GrownTree, MobControl, StopReason, mob

RTOL = 1e-8
ATOL = 1e-10
# The offset of R's coefficients from the MLE, -H^-1 g_R, is a first-order
# estimate; the tolerances allow twice its size.
OFFSET_FACTOR = 2.0
# A fixture whose gradient implies a larger coefficient error than this is
# too loosely converged to test against.
MAX_COEF_TOL = 1e-6

NODE_FITS = sorted(p.stem for p in (FIXTURES / "node_fits").glob("multinom_*.json"))
TREES = sorted(
    p.stem
    for p in (FIXTURES / "extension_trees").glob("*.json")
    if load_json(f"extension_trees/{p.name}")["fit_function"] == "multinomfit"
)


def test_fixture_lists() -> None:
    assert NODE_FITS == [
        "multinom_absent_class",
        "multinom_full",
        "multinom_reference_absent",
        "multinom_single_class",
        "multinom_start",
        "multinom_weights_zero",
    ]
    assert TREES == [
        "sim_multinom_cluster",
        "sim_multinom_noref_onesplit",
        "sim_multinom_nosplit",
        "sim_multinom_onesplit",
    ]


def _labels(value: str | list[str]) -> tuple[str, ...]:
    # jsonlite writes a length-one character vector as a string.
    return (value,) if isinstance(value, str) else tuple(value)


# Tolerances from R's gradient.


def _r_offset(
    got: FitResult, x: np.ndarray, y: np.ndarray, w: np.ndarray, r_estfun: np.ndarray
) -> np.ndarray:
    """-H^-1 g_R over the non-NaN coefficients: R's coefficients minus the
    MLE, to first order. g_R is the column sums of R's scores."""
    p = x.shape[1]
    beta = got.coefficients.reshape(-1, p)
    present = ~np.isnan(beta[:, 0])
    keep = w > 0
    # Codes among the reference and the present classes.
    classes = np.r_[0, np.flatnonzero(present) + 1]
    recode = np.full(beta.shape[0] + 1, -1)
    recode[classes] = np.arange(classes.shape[0])
    _, prob = multinom_evaluate(beta[present], x[keep], recode[y[keep]], w[keep])
    hess = multinom_hessian(prob, x[keep], w[keep])
    offset: np.ndarray = -np.linalg.solve(hess, r_estfun.sum(axis=0))
    return offset


def _coef_tol(offset: np.ndarray) -> float:
    tol = OFFSET_FACTOR * float(np.max(np.abs(offset))) + ATOL
    assert tol < MAX_COEF_TOL
    return tol


def _assert_scores(
    got: np.ndarray, expected: np.ndarray, x: np.ndarray, w: np.ndarray, coef_tol: float
) -> None:
    """Scores within the change a coefficient error of coef_tol can cause:
    |d p_k| <= 2 p_k (1 - p_k) max_l |d eta_l| <= ||x_i||_1 coef_tol / 2."""
    k = got.shape[1] // x.shape[1]
    xi = np.tile(np.abs(x), k)
    bound = w[:, None] * xi * np.abs(x).sum(axis=1)[:, None] * coef_tol / 2
    assert got.shape == expected.shape
    assert np.all(np.abs(got - expected) <= bound + ATOL)


# 1. Single fits.


@cache
def _node_fixture(name: str) -> dict[str, Any]:
    fixture: dict[str, Any] = load_json(f"node_fits/{name}.json")
    return fixture


def _node_inputs(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = _node_fixture(name)
    y = multinom_response(f["y"], f["response_levels"])
    return y, np.array(f["x"], np.float64), np.array(f["weights"], np.float64)


def _node_fit(name: str, estfun: bool = True) -> FitResult:
    f = _node_fixture(name)
    y, x, w = _node_inputs(name)
    model = MultinomModel(f["response_levels"], coef_names=f["x_names"])
    return model.fit(
        y,
        x,
        start=None if f["start"] is None else np.array(f["start"], np.float64),
        weights=w,
        estfun=estfun,
    )


OK_NODE_FITS = [n for n in NODE_FITS if _node_fixture(n)["result"]["estfun"]]


@pytest.mark.parametrize("name", OK_NODE_FITS)
def test_node_fit(name: str) -> None:
    f = _node_fixture(name)
    expected = f["result"]
    got = _node_fit(name)
    y, x, w = _node_inputs(name)
    assert got.status is FitStatus.OK and got.converged
    assert got.coef_names == tuple(expected["coefficient_names"])
    r_estfun = np.array(expected["estfun"])
    tol = _coef_tol(_r_offset(got, x, y, w, r_estfun))
    np.testing.assert_allclose(
        got.coefficients, nan_array(expected["coefficients"]), rtol=0, atol=tol
    )
    assert got.objfun == pytest.approx(expected["objfun"], rel=RTOL)
    assert got.estfun is not None
    _assert_scores(got.estfun, r_estfun, x, w, tol)
    info = expected["model_info"]
    assert got.extras["status"] == info["status"]
    assert got.extras["classes_present"] == _labels(info["classes_present"])
    assert got.extras["value"] == pytest.approx(info["value"], rel=RTOL)


@pytest.mark.parametrize("name", OK_NODE_FITS)
def test_r_offset_explains_the_difference(name: str) -> None:
    # The first-order offset predicts R's coefficients from the MLE far
    # more closely than the tolerance: the tolerance is not slack.
    f = _node_fixture(name)
    got = _node_fit(name)
    y, x, w = _node_inputs(name)
    offset = _r_offset(got, x, y, w, np.array(f["result"]["estfun"]))
    cols = ~np.isnan(got.coefficients)
    residual = nan_array(f["result"]["coefficients"])[cols] - (
        got.coefficients[cols] + offset
    )
    assert np.max(np.abs(residual)) < 0.05 * np.max(np.abs(offset)) + ATOL


def test_node_fit_absent_class() -> None:
    got = _node_fit("multinom_absent_class")
    assert np.all(np.isnan(got.coefficients[2:]))
    assert got.estfun is not None and got.estfun.shape[1] == 2
    assert got.extras["classes_present"] == ("lo", "mid")


def test_node_fit_reference_absent_has_no_estfun() -> None:
    expected = _node_fixture("multinom_reference_absent")["result"]
    got = _node_fit("multinom_reference_absent")
    assert got.status is FitStatus.NO_ESTFUN and got.converged
    assert got.estfun is None and np.all(np.isnan(got.coefficients))
    assert got.coef_names == tuple(expected["coefficient_names"])
    # R's objective, in objfun and in extras["value"].
    assert got.objfun == pytest.approx(expected["objfun"], rel=RTOL)
    info = expected["model_info"]
    assert got.extras["value"] == got.objfun
    assert got.extras["value"] == pytest.approx(info["value"], rel=RTOL)
    assert got.extras["status"] == info["status"] == "reference_absent"
    assert got.extras["classes_present"] == _labels(info["classes_present"])


def test_node_fit_single_class() -> None:
    expected = _node_fixture("multinom_single_class")["result"]
    got = _node_fit("multinom_single_class")
    assert got.status is FitStatus.NO_ESTFUN
    assert got.objfun == expected["objfun"] == 0
    assert got.estfun is None and np.all(np.isnan(got.coefficients))
    assert got.extras["status"] == "single_class"
    assert got.extras["classes_present"] == ("mid",)


def test_start_does_not_change_the_estimate() -> None:
    # multinom_start and multinom_full are the same fit from rep(0.1, 4)
    # and from zero.
    a = _node_fit("multinom_full")
    b = _node_fit("multinom_start")
    np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=1e-12, atol=1e-13)
    assert a.objfun == pytest.approx(b.objfun, rel=1e-14)


def test_objective_only_fit_equals_node_fit() -> None:
    for name in NODE_FITS:
        a = _node_fit(name)
        b = _node_fit(name, estfun=False)
        np.testing.assert_array_equal(a.coefficients, b.coefficients)
        assert a.objfun == b.objfun and a.status is b.status
        assert b.estfun is None


# 2. Split search and trees.


@cache
def _tree_fixture(name: str) -> dict[str, Any]:
    fixture: dict[str, Any] = load_json(f"extension_trees/{name}.json")
    return fixture


def _model(fixture: dict[str, Any]) -> MultinomModel:
    regressors = fixture["regressors"]
    names = [n.split(":", 1)[1] for n in fixture["coefficient_names"]]
    assert names[: len(regressors) + 1] == ["(Intercept)", *regressors]
    return MultinomModel(
        fixture["response_levels"], coef_names=["(Intercept)", *regressors]
    )


def _inputs(fixture: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Response codes, regressors with intercept, weights."""
    assert fixture["offset"] is None
    d, columns = data(fixture["data_csv"])
    assert columns[fixture["response"]]["levels"] == fixture["response_levels"]
    y = d[fixture["response"]]
    x = np.column_stack([np.ones(y.shape[0])] + [d[r] for r in fixture["regressors"]])
    w = d[fixture["weights"]] if fixture["weights"] else np.ones(y.shape[0])
    return y, x, w


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


@cache
def _grown(name: str) -> GrownTree:
    fixture = _tree_fixture(name)
    y, x, w = _inputs(fixture)
    return mob(
        _model(fixture),
        y,
        x,
        variables(fixture),
        weights=w,
        cluster=cluster(fixture),
        control=_control(fixture),
    )


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
    y, x, w = _inputs(fixture)
    var = variables(fixture)[node["split_search"]["variable"]]
    rows = np.array(node["rows"], dtype=np.intp)
    m = _model(fixture)

    def fit(mask: np.ndarray, start: np.ndarray | None) -> FitResult:
        r = rows[mask]
        return m.fit(y[r], x[r], start=start, weights=w[r], estfun=False)

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


def _node_data(
    name: str, e: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y, x, w = _inputs(_tree_fixture(name))
    rows = np.array(e["rows"], dtype=np.intp)
    return y[rows], x[rows], w[rows], np.array(e["estfun"])


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
        assert g.fit.objfun == pytest.approx(e["objfun"], rel=RTOL)
        info = e["model_info"]
        if e["estfun"] is None:
            # A leaf without the reference class: R's objective, no
            # coefficients and no scores.
            assert g.fit.status is FitStatus.NO_ESTFUN
            assert e["stop_reason"] == "no_estfun"
            assert g.fit.estfun is None
            assert np.all(np.isnan(g.fit.coefficients))
            assert np.all(np.isnan(nan_array(e["coefficients"])))
            assert g.test is None and g.test_failed
        else:
            y, x, w, r_estfun = _node_data(name, e)
            tol = _coef_tol(_r_offset(g.fit, x, y, w, r_estfun))
            np.testing.assert_allclose(
                g.fit.coefficients, nan_array(e["coefficients"]), rtol=0, atol=tol
            )
            assert g.fit.estfun is not None
            _assert_scores(g.fit.estfun, r_estfun, x, w, tol)
        assert g.fit.extras["status"] == info["status"]
        assert g.fit.extras["classes_present"] == _labels(info["classes_present"])


def _tests(
    name: str, e: dict[str, Any], estfun: np.ndarray, w: np.ndarray
) -> FluctuationTests:
    fixture = _tree_fixture(name)
    rows = np.array(e["rows"], dtype=np.intp)
    z = [
        PartitioningVariable(v.name, v.kind, v.values[rows])
        for v in variables(fixture).values()
    ]
    cl = cluster(fixture)
    c = fixture["control"]
    return fluctuation_tests(
        estfun,
        z,
        w,
        minsize=e["minsize"],
        trim=c["trim"],
        bonferroni=c["bonferroni"],
        cluster=None if cl is None else cl[rows],
        caseweights=c["caseweights"],
    )


TESTED_NODES = [
    pytest.param(name, node["id"], id=f"{name}-{node['id']}")
    for name in TREES
    for node in _tree_fixture(name)["nodes"]
    if node["test"] and node["test"]["status"] == "ok"
]


@pytest.mark.parametrize(("name", "node_id"), TESTED_NODES)
def test_tests_on_r_scores(name: str, node_id: int) -> None:
    """The instability tests on R's own scores reproduce R's statistics at
    the usual tolerance: the test code does not add to the differences."""
    e = next(n for n in _tree_fixture(name)["nodes"] if n["id"] == node_id)
    _, _, w, r_estfun = _node_data(name, e)
    t = _tests(name, e, r_estfun, w)
    np.testing.assert_allclose(t.statistic, e["test"]["statistic"], rtol=RTOL)
    np.testing.assert_allclose(
        t.p_value_adjusted, nan_array(e["test"]["p_value_adjusted"]), rtol=RTOL
    )
    assert t.best == e["test"]["best"]


@pytest.mark.parametrize(("name", "node_id"), TESTED_NODES)
def test_tree_tests(name: str, node_id: int) -> None:
    """The grown tree's statistics, within their change when the
    coefficients move from the MLE by R's offset -H^-1 g_R."""
    e = next(n for n in _tree_fixture(name)["nodes"] if n["id"] == node_id)
    g = _grown(name).node(node_id)
    assert g.test is not None
    y, x, w, r_estfun = _node_data(name, e)
    offset = _r_offset(g.fit, x, y, w, r_estfun)
    p = x.shape[1]
    beta = g.fit.coefficients.reshape(-1, p) + offset.reshape(-1, p)
    _, prob = multinom_evaluate(beta, x, y, w)
    shifted = _tests(name, e, multinom_scores(prob, x, y, w), w)
    t = e["test"]
    for got, moved, expected in (
        (g.test.statistic, shifted.statistic, np.array(t["statistic"])),
        (
            g.test.p_value_adjusted,
            shifted.p_value_adjusted,
            nan_array(t["p_value_adjusted"]),
        ),
    ):
        tol = OFFSET_FACTOR * np.abs(moved - got) + RTOL * np.abs(expected)
        assert np.all(np.abs(got - expected) <= tol)
    assert g.test.best == t["best"]
    assert g.p_value == g.test.p_value_adjusted[g.test.best]


@pytest.mark.parametrize("name", TREES)
def test_tree_split_searches(name: str) -> None:
    for g, e in zip(_grown(name).nodes, _tree_fixture(name)["nodes"], strict=True):
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


def test_reference_absent_sides() -> None:
    """Every admissible root candidate of the reference-absent tree, each
    side fitted on its own: R's classes, status and objective per side."""
    name = "sim_multinom_noref_onesplit"
    fixture = _tree_fixture(name)
    check = fixture["reference_absent"]
    y, x, w = _inputs(fixture)
    z = variables(fixture)[check["variable"]].values
    m = _model(fixture)
    admissible = [c for c in check["candidates"] if c["objective"] is not None]
    assert len(admissible) == check["n_admissible"]
    n_absent = 0
    for c in admissible:
        left = z <= c["candidate"]
        for side, mask in (("left", left), ("right", ~left)):
            got = m.fit(y[mask], x[mask], weights=w[mask], estfun=False)
            assert int(mask.sum()) == c[f"n_{side}"]
            assert got.extras["status"] == c[f"status_{side}"]
            assert got.extras["classes_present"] == _labels(c[f"classes_{side}"])
            assert got.objfun == pytest.approx(c[f"objective_{side}"], rel=RTOL)
            if c[f"status_{side}"] == "reference_absent":
                assert got.status is FitStatus.NO_ESTFUN
                n_absent += 1
    assert n_absent == check["n_reference_absent_candidates"]
    leaf = check["leaves_without_reference"][0]
    g = _grown(name).node(leaf["id"])
    assert g.stop_reason is StopReason.NO_ESTFUN
    assert g.fit.extras["status"] == "reference_absent"
    assert g.fit.objfun == pytest.approx(leaf["objfun"], rel=RTOL)
    split = _grown(name).node(1).split
    assert split is not None and split.breaks == check["chosen_break"]


# 3. Unit tests without an R fixture.


LEVELS = ("a", "b", "c")


def _sim(
    seed: int, n: int = 300, beta: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.random(n)])
    b = np.array([[0.3, 0.8, -0.5], [-0.2, -0.6, 1.0]]) if beta is None else beta
    eta = np.column_stack([np.zeros(n), x @ b.T])
    prob = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=pi) for pi in prob], dtype=np.intp)
    return x, y


def test_against_statsmodels_mnlogit() -> None:
    sm = pytest.importorskip("statsmodels.api")
    for seed in (1, 2, 3):
        x, y = _sim(seed)
        ours = MultinomModel(LEVELS).fit(y, x, weights=np.ones(y.shape[0]))
        theirs = sm.MNLogit(y, x).fit(method="newton", tol=1e-14, maxiter=100, disp=0)
        # statsmodels' params are (p, K - 1): regressor-major.
        np.testing.assert_allclose(
            ours.coefficients, theirs.params.T.reshape(-1), rtol=1e-9, atol=1e-11
        )
        assert ours.objfun == pytest.approx(-theirs.llf, rel=1e-12)
        # score_obs is class-major, as estfun.
        sm_scores = theirs.model.score_obs(theirs.params)
        r = MultinomModel(LEVELS).fit(y, x, weights=np.ones(y.shape[0]), estfun=True)
        assert r.estfun is not None
        np.testing.assert_allclose(r.estfun, sm_scores, rtol=1e-9, atol=1e-11)


def test_frequency_weights_equal_replicated_rows() -> None:
    x, y = _sim(4, n=150)
    w = np.random.default_rng(5).integers(1, 4, size=(150,)).astype(np.float64)
    rep = np.repeat(np.arange(150), w.astype(int))
    m = MultinomModel(LEVELS)
    a = m.fit(y, x, weights=w)
    b = m.fit(y[rep], x[rep], weights=np.ones(rep.shape[0]))
    np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=1e-10)
    assert a.objfun == pytest.approx(b.objfun, rel=1e-12)


def test_zero_weights_equal_deleted_rows() -> None:
    x, y = _sim(6)
    n = int(y.shape[0])
    w = np.random.default_rng(7).integers(0, 3, size=(n,)).astype(np.float64)
    keep = w > 0
    m = MultinomModel(LEVELS)
    full = m.fit(y, x, weights=w, estfun=True)
    kept = m.fit(y[keep], x[keep], weights=w[keep], estfun=True)
    np.testing.assert_array_equal(full.coefficients, kept.coefficients)
    assert full.objfun == kept.objfun
    assert full.estfun is not None and kept.estfun is not None
    np.testing.assert_array_equal(full.estfun[keep], kept.estfun)
    assert np.all(full.estfun[~keep] == 0)


def test_class_present_only_with_zero_weight_is_absent() -> None:
    x, y = _sim(8)
    w = np.ones(y.shape[0])
    w[y == 2] = 0.0
    r = MultinomModel(LEVELS).fit(y, x, weights=w, estfun=True)
    assert r.status is FitStatus.OK
    assert r.extras["classes_present"] == ("a", "b")
    assert np.all(np.isnan(r.coefficients[3:]))
    assert r.estfun is not None and r.estfun.shape == (y.shape[0], 3)


def test_rank_deficient_design_is_a_failed_fit() -> None:
    # docs/decisions.md, Q9: R has no oracle (nnet fits it anyway).
    x, y = _sim(9)
    xd = np.column_stack([x, 2 * x[:, 1] - x[:, 2]])
    m = MultinomModel(LEVELS)
    for estfun in (True, False):
        r = m.fit(y, xd, weights=np.ones(y.shape[0]), estfun=estfun)
        assert r.status is FitStatus.FAILED and math.isinf(r.objfun)
        assert r.estfun is None and np.all(np.isnan(r.coefficients))
        assert r.coef_names == m.names(("x1", "x2", "x3", "x4"))
        assert r.extras["status"] == "rank_deficient"
        assert r.reason is not None and "rank 3 < 4" in r.reason
    # Rank deficient only on the positive-weight rows.
    w = np.ones(y.shape[0])
    xz = x.copy()
    xz[:, 2] = 0.0
    xz[0, 2] = 1.0
    w[0] = 0.0
    assert m.fit(y, xz, weights=w).status is FitStatus.FAILED
    assert m.fit(y, xz, weights=np.ones(y.shape[0])).status is FitStatus.OK


def _search(x: np.ndarray, y: np.ndarray) -> tuple[Any, np.ndarray]:
    """Split search on z = 0, ..., n - 1 with minsize 30, and the mask of
    candidates that minsize admits (30 rows or more on each side)."""
    n = y.shape[0]
    m = MultinomModel(LEVELS)

    def fit(mask: np.ndarray, start: np.ndarray | None) -> FitResult:
        return m.fit(y[mask], x[mask], start=start, weights=np.ones(int(mask.sum())))

    z = np.arange(n, dtype=np.float64)
    got = find_split(0, VariableKind.NUMERIC, z, None, np.ones(n), fit, minsize=30)
    assert got.values is not None
    admissible = (got.values >= 29) & (got.values <= n - 31)
    return got, admissible


def test_rank_deficient_candidates_are_inadmissible() -> None:
    # Rows 0 to 99 have a regressor equal to the intercept, so a left side
    # inside that block is rank deficient.
    x, y = _sim(10, n=200)
    x[:100, 2] = 1.0
    got, admissible = _search(x, y)
    inside = admissible & (got.values <= 99)
    assert np.any(inside) and np.all(np.isinf(got.objective[inside]))
    assert np.all(np.isfinite(got.objective[admissible & ~inside]))
    assert got.n_failed == int(np.sum(inside))


def test_reference_absent_candidates_stay_admissible() -> None:
    # Rows sorted by class: a cut at or after the last reference row leaves
    # the right side without the reference class, a no_estfun fit with R's
    # objective. Once the right side holds only class c, it is a
    # single-class fit with objective 0. Both are admissible.
    x, y = _sim(11, n=200)
    order = np.argsort(y, kind="stable")
    x, y = x[order], y[order]
    n_ref, n_mid = int(np.sum(y == 0)), int(np.sum(y == 1))
    got, admissible = _search(x, y)
    no_ref = admissible & (got.values >= n_ref - 1) & (got.values < n_ref + n_mid - 1)
    only_c = admissible & (got.values >= n_ref + n_mid - 1)
    assert np.any(no_ref) and np.any(only_c)
    assert np.all(np.isfinite(got.objective[admissible]))
    assert got.n_failed == 0
    m = MultinomModel(LEVELS)
    for i in np.flatnonzero(no_ref):
        cut = int(got.values[i]) + 1
        left = m.fit(y[:cut], x[:cut], weights=np.ones(cut))
        right = m.fit(y[cut:], x[cut:], weights=np.ones(200 - cut))
        assert right.status is FitStatus.NO_ESTFUN
        assert right.extras["status"] == "reference_absent"
        assert got.objective[i] == pytest.approx(left.objfun + right.objfun, rel=RTOL)


def _reference_absent_block(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows 0 to 99 from all three classes; rows 100 to 199 only b or c,
    by the sign of x1 with 10 percent noise."""
    x, y = _sim(seed, n=200)
    rng = np.random.default_rng(seed + 100)
    flip = rng.random(100) < 0.1
    y = y.copy()
    y[100:] = np.where((x[100:, 1] > 0) ^ flip, 2, 1)
    return x, y


def test_reference_absent_split_competes_and_wins() -> None:
    # The best split leaves the right child without the reference class.
    # Its objective is the sum of both sides' true objectives, so it beats
    # every other cut and is chosen, as R's would be.
    x, y = _reference_absent_block(19)
    got, admissible = _search(x, y)
    assert got.n_failed == 0
    assert np.all(np.isfinite(got.objective[admissible]))
    assert got.best is not None and got.values[got.best] == 99
    m = MultinomModel(LEVELS)
    left = m.fit(y[:100], x[:100], weights=np.ones(100))
    right = m.fit(y[100:], x[100:], weights=np.ones(100))
    assert left.status is FitStatus.OK
    assert right.status is FitStatus.NO_ESTFUN
    assert right.extras["status"] == "reference_absent"
    assert right.objfun == right.extras["value"] > 0
    assert got.objective[got.best] == pytest.approx(
        left.objfun + right.objfun, rel=RTOL
    )
    others = admissible & (got.values != 99)
    assert np.all(got.objective[others] > got.objective[got.best])


def test_reference_absent_node_is_terminal() -> None:
    x, y = _reference_absent_block(19)
    z = np.r_[np.zeros(100), np.ones(100)]
    # minsize 30 lets the children pass the minsize rule, so the right
    # child stops for its missing scores.
    tree = mob(
        MultinomModel(LEVELS),
        y,
        x,
        {"z": _variable(z)},
        weights=np.ones(200),
        control=MobControl(minsize=30),
    )
    assert len(tree.nodes) == 3
    assert tree.node(2).stop_reason is not StopReason.NO_ESTFUN
    right = tree.node(3)
    assert right.stop_reason is StopReason.NO_ESTFUN
    assert right.fit.extras["status"] == "reference_absent"
    assert np.all(np.isnan(right.fit.coefficients))
    assert math.isfinite(right.fit.objfun) and right.fit.objfun > 0


def test_single_class_side_counts_in_split_search() -> None:
    # Objective 0 for a one-class side, as in R; the split is admissible.
    x, y = _sim(12, n=200)
    y = y.copy()
    y[:60] = 0
    m = MultinomModel(LEVELS)
    left = m.fit(y[:60], x[:60], weights=np.ones(60))
    right = m.fit(y[60:], x[60:], weights=np.ones(140))
    assert left.status is FitStatus.NO_ESTFUN and left.objfun == 0.0
    assert right.status is FitStatus.OK
    got, _ = _search(x, y)
    assert got.objective[59] == right.objfun


def test_offset_raises() -> None:
    x, y = _sim(13)
    with pytest.raises(ValueError, match="offset"):
        MultinomModel(LEVELS).fit(
            y, x, weights=np.ones(y.shape[0]), offset=np.zeros(y.shape[0])
        )


def test_default_minsize_and_pruning_df_count_all_coefficients() -> None:
    # minsize = ceiling(10 (K - 1) p); a node with an absent class still
    # counts its NaN coefficients, in minsize and in the pruning df.
    x, y = _sim(14, n=400)
    z = {"z": _variable(np.random.default_rng(15).random(400))}
    seen: list[tuple[int, int]] = []

    def rule(
        objfun: tuple[float, float], df: tuple[int, int], nobs: tuple[float, float]
    ) -> bool:
        seen.append(df)
        return False

    y = y.copy()
    y[:200][y[:200] == 2] = 1  # class c absent from the left half
    zz = {"z": _variable(np.r_[np.zeros(200), np.ones(200)])}
    tree = mob(
        MultinomModel(LEVELS),
        y,
        x,
        zz,
        weights=np.ones(400),
        control=MobControl(prune=rule),
    )
    assert tree.minsize == math.ceil(10 * 2 * 3)
    assert len(tree.nodes) == 3
    left = tree.node(2)
    assert left.fit.extras["classes_present"] == ("a", "b")
    assert np.isnan(left.fit.coefficients[3:]).all()
    assert seen == [(6, 2 * 6 + 1)]
    tree_z = mob(MultinomModel(LEVELS), y, x, z, weights=np.ones(400))
    assert tree_z.minsize == 60


def _variable(values: np.ndarray) -> Any:
    from modeltrees.tree import Variable

    return Variable("z", VariableKind.NUMERIC, values, None)


def test_single_class_node_stops_without_tests() -> None:
    x, y = _sim(16, n=400)
    y = y.copy()
    y[:200] = 1
    tree = mob(
        MultinomModel(LEVELS),
        y,
        x,
        {"z": _variable(np.r_[np.zeros(200), np.ones(200)])},
        weights=np.ones(400),
    )
    assert tree.node(2).stop_reason is StopReason.NO_ESTFUN
    assert tree.node(2).fit.objfun == 0.0


def test_newton_converges_from_far_start() -> None:
    x, y = _sim(17)
    w = np.ones(y.shape[0])
    ref = multinom_newton(x, y, 3, w)
    far = multinom_newton(x, y, 3, w, start=np.full((2, 3), 8.0))
    assert ref.converged and far.converged
    np.testing.assert_allclose(far.coefficients, ref.coefficients, rtol=1e-9)


def test_separated_classes_stop_with_large_coefficients() -> None:
    # Complete separation: no MLE. Newton must stop, without error.
    x = np.column_stack([np.ones(90), np.linspace(-1, 1, 90)])
    y = np.repeat(np.arange(3), 30)
    r = MultinomModel(LEVELS).fit(y, x, weights=np.ones(90), estfun=True)
    assert r.status is FitStatus.OK
    assert np.max(np.abs(r.coefficients)) > 10
    assert r.objfun < 1e-3


def test_multinom_response() -> None:
    levels = ["lo", "mid", "hi"]
    np.testing.assert_array_equal(
        multinom_response(["hi", "lo", "mid", "hi"], levels), [2, 0, 1, 2]
    )
    np.testing.assert_array_equal(
        multinom_response(np.array([2, 0, 1]), levels, codes=True), [2, 0, 1]
    )
    np.testing.assert_array_equal(
        multinom_response(np.array([2.0, 0.0]), levels, codes=True), [2, 0]
    )
    with pytest.raises(ValueError, match="not among the levels"):
        multinom_response(["lo", "top"], levels)
    with pytest.raises(ValueError, match="codes must be in"):
        multinom_response(np.array([0, 3]), levels, codes=True)
    with pytest.raises(ValueError, match="integers"):
        multinom_response(np.array([0.5]), levels, codes=True)
    with pytest.raises(ValueError, match="missing"):
        multinom_response(np.array([0.0, np.nan]), levels, codes=True)
    with pytest.raises(ValueError, match="two levels"):
        multinom_response(["a"], ["a"])
    with pytest.raises(ValueError, match="unique"):
        multinom_response(["a"], ["a", "a"])


def test_coefficient_names_are_class_major() -> None:
    m = MultinomModel(("lo", "mid", "hi"), coef_names=("(Intercept)", "x"))
    assert m.names(("(Intercept)", "x")) == (
        "mid:(Intercept)",
        "mid:x",
        "hi:(Intercept)",
        "hi:x",
    )
