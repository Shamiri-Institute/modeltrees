"""The negative binomial model against the R fixtures.

In the order of CLAUDE.md: single fits (tests/fixtures/node_fits/negbin_*),
then split search and whole trees (tests/fixtures/extension_trees/), then
the thesis' quine tree (tests/fixtures/thesis_quine_crosscheck.json). The
oracle is r/extensions/negbin.R on MASS 7.3-66 (docs/extensions-spec.md,
section 3).

theta is compared at the usual tolerance, except where theta.ml stopped at
its iteration limit with a very large theta (near-Poisson nodes): there its
value depends on the exact Newton path and is compared loosely, as the spec
says. Coefficients, objectives and scores barely depend on such a theta and
are compared at the usual tolerance everywhere.
"""

import math
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures_io import FIXTURES, load_json, nan_array
from slow import mark_slow_trees
from tree_fixtures import cluster, inputs, variables

from modeltrees.instability import VariableKind
from modeltrees.models.families import GlmFitWarning, NegativeBinomial, Poisson
from modeltrees.models.glm_fit import glm_fit
from modeltrees.models.negbin import (
    NegBinModel,
    glm_nb,
    negbin_loglik,
    theta_ml,
)
from modeltrees.node_model import FitResult, FitStatus, extra_df
from modeltrees.split_search import find_split
from modeltrees.tree import GrownTree, MobControl, StopReason, mob

RTOL = 1e-8
ATOL = 1e-10
# theta after theta.ml's iteration limit (near-Poisson nodes).
THETA_LOOSE_RTOL = 1e-3
NEAR_POISSON = "iteration limit reached"

NODE_FITS = sorted(p.stem for p in (FIXTURES / "node_fits").glob("negbin_*.json"))
TREES = sorted(
    p.stem
    for p in (FIXTURES / "extension_trees").glob("*.json")
    if load_json(f"extension_trees/{p.name}")["fit_function"] == "negbinfit"
)


def test_fixture_lists() -> None:
    assert len(NODE_FITS) == 5
    assert TREES == [
        "quine_negbin_intercept",
        "quine_negbin_intercept_deviance",
        "sim_negbin_cluster",
        "sim_negbin_nosplit",
        "sim_negbin_onesplit",
        "sim_negbin_theta_only",
        "sim_negbin_weights",
    ]


def _close(got: float | None, expected: float | None, rtol: float = RTOL) -> None:
    if expected is None:
        assert got is None or math.isnan(got)
        return
    assert got is not None
    assert got == pytest.approx(expected, rel=rtol, abs=ATOL)


def _assert_model_info(extras: Any, info: dict[str, Any]) -> None:
    loose = info["theta_warning"] == NEAR_POISSON
    rtol = THETA_LOOSE_RTOL if loose else RTOL
    _close(extras["theta"], info["theta"], rtol)
    _close(extras["theta_irls"], info["theta_irls"], rtol)
    _close(extras["theta_se"], info["theta_se"], rtol)
    assert extras["theta_warning"] == info["theta_warning"]
    assert extras["converged"] == info["converged"]
    assert extras["iter"] == info["iter"]


# 1. Single fits.


@cache
def _node_fixture(name: str) -> dict[str, Any]:
    fixture: dict[str, Any] = load_json(f"node_fits/{name}.json")
    return fixture


def _node_fit(name: str) -> FitResult:
    f = _node_fixture(name)
    model = NegBinModel(coef_names=f["x_names"], **f["fit_args"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # none of these fits warns in R
        return model.fit(
            np.array(f["y"], dtype=np.float64),
            np.array(f["x"], dtype=np.float64),
            start=None if f["start"] is None else np.array(f["start"], np.float64),
            weights=np.array(f["weights"], dtype=np.float64),
            offset=None if f["offset"] is None else np.array(f["offset"], np.float64),
            estfun=True,
        )


@pytest.mark.parametrize("name", NODE_FITS)
def test_node_fit(name: str) -> None:
    expected = _node_fixture(name)["result"]
    got = _node_fit(name)
    assert got.status is FitStatus.OK
    assert got.coef_names == tuple(expected["coefficient_names"])
    np.testing.assert_allclose(
        got.coefficients, nan_array(expected["coefficients"]), rtol=RTOL, atol=ATOL
    )
    assert got.objfun == pytest.approx(expected["objfun"], rel=RTOL, abs=ATOL)
    assert got.estfun is not None
    np.testing.assert_allclose(
        got.estfun, np.array(expected["estfun"]), rtol=RTOL, atol=ATOL
    )
    _assert_model_info(got.extras, expected["model_info"])


def test_node_fit_objectives_differ() -> None:
    # The same fit, objective loglik and deviance.
    loglik = _node_fit("negbin_quine_eth_a_intercept")
    deviance = _node_fit("negbin_quine_deviance")
    np.testing.assert_array_equal(loglik.coefficients, deviance.coefficients)
    assert loglik.extras["theta"] == deviance.extras["theta"]
    assert loglik.objfun != pytest.approx(deviance.objfun)


def test_node_fit_theta_score_vanishes() -> None:
    # theta_score_sum in the fixtures: the theta score at glm.nb's theta.
    for name in NODE_FITS:
        f = _node_fixture(name)
        r = _node_fit(name)
        y = np.array(f["y"], np.float64)
        w = np.array(f["weights"], np.float64)
        eta = np.array(f["x"], np.float64) @ r.coefficients
        if f["offset"] is not None:
            eta = eta + np.array(f["offset"], np.float64)
        th = theta_ml(y, np.exp(eta), float(np.sum(w)), w, limit=25)
        assert th.theta == pytest.approx(r.extras["theta"], rel=1e-6)
        score = f["checks"].get("theta_score_sum")
        assert score is None or abs(score) < 1e-6


def test_objective_is_minus_half_twologlik() -> None:
    f = _node_fixture("negbin_sim_offset_weights")
    y = np.array(f["y"], np.float64)
    x = np.array(f["x"], np.float64)
    w = np.array(f["weights"], np.float64)
    off = np.array(f["offset"], np.float64)
    z = glm_nb(x, y, weights=w, offset=off)
    mu = np.exp(x @ z.coefficients + off)
    lm = negbin_loglik(z.theta, mu, y, w)
    assert -z.twologlik / 2 == pytest.approx(-lm, rel=1e-12)
    assert _node_fit("negbin_sim_offset_weights").objfun == -z.twologlik / 2


# 2. Split search and trees.


@cache
def _tree_fixture(name: str) -> dict[str, Any]:
    fixture: dict[str, Any] = load_json(f"extension_trees/{name}.json")
    return fixture


def _model(fixture: dict[str, Any]) -> NegBinModel:
    return NegBinModel(coef_names=fixture["coefficient_names"], **fixture["fit_args"])


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
    y, x, w, off = inputs(fixture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlmFitWarning)
        return mob(
            _model(fixture),
            y,
            x,
            variables(fixture),
            weights=w,
            offset=off,
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


@pytest.mark.parametrize(("name", "node_id"), mark_slow_trees(SEARCH_NODES))
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


@pytest.mark.parametrize("name", mark_slow_trees(TREES))
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
        np.testing.assert_allclose(
            g.fit.coefficients, nan_array(e["coefficients"]), rtol=RTOL, atol=ATOL
        )
        assert g.fit.objfun == pytest.approx(e["objfun"], rel=RTOL, abs=ATOL)
        assert g.fit.estfun is not None
        np.testing.assert_allclose(
            g.fit.estfun, np.array(e["estfun"]), rtol=RTOL, atol=ATOL
        )
        _assert_model_info(g.fit.extras, e["model_info"])


@pytest.mark.parametrize("name", mark_slow_trees(TREES))
def test_tree_tests_and_searches(name: str) -> None:
    for g, e in zip(_grown(name).nodes, _tree_fixture(name)["nodes"], strict=True):
        status = e["test"]["status"] if e["test"] else "not_run"
        if status == "not_run":
            assert g.test is None and not g.test_failed and g.p_value is None
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


@pytest.mark.parametrize("name", mark_slow_trees(TREES))
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
    # logLik.modelparty's df: logLik.negbin's rank + 1 per leaf, plus one per
    # split. That is the pruning df with extra_df = 1 (docs/decisions.md).
    k = len(fixture["coefficient_names"]) + extra_df(_model(fixture))
    assert len(leaves) * k + summary["n_inner"] == summary["loglik_df"]


@pytest.mark.slow
def test_near_poisson_leaves_are_compared_loosely() -> None:
    # The fixture the spec names: two leaves where theta.ml stopped at its
    # iteration limit, with theta about 1.7e4 and 8.4e4.
    nodes = _tree_fixture("sim_negbin_weights")["nodes"]
    near = [n for n in nodes if n["model_info"]["theta_warning"] == NEAR_POISSON]
    assert [n["id"] for n in near] == [10, 15]
    got = _grown("sim_negbin_weights")
    for e in near:
        g = got.node(e["id"])
        assert g.fit.extras["theta"] > 1e4
        assert g.fit.extras["theta_warning"] == NEAR_POISSON


def test_theta_only_change_does_not_split() -> None:
    got = _grown("sim_negbin_theta_only")
    assert len(got.nodes) == 1
    assert got.root.stop_reason is StopReason.NOT_SIGNIFICANT


# 3. The thesis' quine tree.


def test_thesis_quine_tree() -> None:
    cross = load_json("thesis_quine_crosscheck.json")
    assert cross["agree"] is True
    name = Path(cross["tree_fixture"]).stem
    assert name == "quine_negbin_intercept_deviance"
    got = _grown(name)
    root, node2 = got.node(1), got.node(2)
    variables_ = _tree_fixture(name)["partitioning"]
    for node, key in ((root, "root"), (node2, "node2")):
        r, t = cross["partykit"][key], cross["thesis"][key]
        assert node.test is not None and node.test.best is not None
        assert variables_[node.test.best] == r["variable"] == t["variable"]
        stat = float(node.test.statistic[node.test.best])
        assert stat == pytest.approx(r["statistic"][0], rel=RTOL)
        assert node.p_value is not None
        assert 1 - node.p_value == pytest.approx(r["criterion"], rel=RTOL)
        # The thesis prints 3 decimals of the criterion, and the statistic
        # to the digits shown.
        assert round(1 - node.p_value, 3) == t["criterion"]
        digits = len(str(t["statistic"]).split(".")[1])
        assert round(stat, digits) == t["statistic"]
    leaves = [n for n in got.nodes if n.terminal]
    for g, r, t in zip(
        leaves, cross["partykit"]["terminal"], cross["thesis"]["terminal"], strict=True
    ):
        assert g.rows.shape[0] == r["n"] == t["n"]
        intercept = float(g.fit.coefficients[0])
        assert intercept == pytest.approx(r["intercept"], rel=RTOL)
        assert g.fit.extras["theta"] == pytest.approx(r["theta"], rel=RTOL)
        assert round(intercept, 2) == t["intercept"]
        assert float(f"{g.fit.extras['theta']:.4g}") == t["theta"]


def test_quine_objectives_choose_different_splits() -> None:
    # docs/extensions-spec.md, section 6: node 2 splits Age differently.
    by_loglik = _grown("quine_negbin_intercept").node(2).split
    by_deviance = _grown("quine_negbin_intercept_deviance").node(2).split
    assert by_loglik is not None and by_deviance is not None
    assert by_loglik.index != by_deviance.index


# 4. Unit tests without an R fixture.


def _sim(seed: int, n: int = 200, theta: float = 2.0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    mu = np.exp(x @ np.array([0.5, 0.4]))
    y = rng.poisson(rng.gamma(theta, mu / theta)).astype(np.float64)
    return x, y


def test_pruning_counts_theta() -> None:
    model = NegBinModel()
    assert extra_df(model) == 1
    fixture = _tree_fixture("quine_negbin_intercept")
    y, x, w, _ = inputs(fixture)
    seen: list[tuple[int, int]] = []

    def rule(
        objfun: tuple[float, float], df: tuple[int, int], nobs: tuple[float, float]
    ) -> bool:
        seen.append(df)
        return False

    import dataclasses

    control = dataclasses.replace(_control(fixture), prune=rule)
    mob(model, y, x, variables(fixture), weights=w, control=control)
    # Intercept only: k = 1 coefficient + theta.
    assert seen and all(df == (2, 5) for df in seen)


def test_zero_weights_equal_deleted_rows() -> None:
    x, y = _sim(3)
    n = int(y.shape[0])
    w = np.random.default_rng(4).integers(0, 3, size=(n,)).astype(np.float64)
    keep = w > 0
    m = NegBinModel()
    full = m.fit(y, x, weights=w, estfun=True)
    kept = m.fit(y[keep], x[keep], weights=w[keep], estfun=True)
    np.testing.assert_array_equal(full.coefficients, kept.coefficients)
    assert full.objfun == kept.objfun
    assert full.extras["theta"] == kept.extras["theta"]
    assert full.estfun is not None and kept.estfun is not None
    np.testing.assert_array_equal(full.estfun[keep], kept.estfun)
    assert np.all(full.estfun[~keep] == 0)


def test_all_zero_counts_fail_as_in_r() -> None:
    # R: glm.nb(y ~ 1) on 30 zeros stops in theta.ml with "missing value
    # where TRUE/FALSE needed" (checked by hand in R 4.6.1, MASS 7.3-66).
    r = NegBinModel().fit(np.zeros(30), None, weights=np.ones(30), estfun=True)
    assert r.status is FitStatus.FAILED and math.isinf(r.objfun)
    assert r.reason is not None and "missing value" in r.reason
    assert r.coef_names == ("(Intercept)",)


def test_near_poisson_node_matches_r_by_hand() -> None:
    # 29 zeros and a one, intercept only. R 4.6.1, MASS 7.3-66, by hand:
    # -twologlik / 2 = 4.40122018151033, theta = 730.987499089747, with
    # "iteration limit reached" (twice).
    y = np.r_[np.zeros(29), 1.0]
    with pytest.warns(GlmFitWarning, match="iteration limit reached"):
        r = NegBinModel().fit(y, None, weights=np.ones(30), estfun=True)
    assert r.objfun == pytest.approx(4.40122018151033, rel=1e-10)
    assert r.extras["theta"] == pytest.approx(730.987499089747, rel=THETA_LOOSE_RTOL)
    assert r.extras["theta_warning"] == NEAR_POISSON
    assert r.extras["warnings"] == (NEAR_POISSON, NEAR_POISSON)


def test_negative_counts_raise() -> None:
    with pytest.raises(ValueError, match="negative values"):
        NegBinModel().fit(np.array([1.0, -1.0, 2.0]), None, weights=np.ones(3))


def test_alternation_limit() -> None:
    x, y = _sim(5)
    z = glm_nb(x, y, maxit=1)
    assert z.alternations == 2
    assert z.th_warn == "alternation limit reached"
    assert z.warnings[-1] == "alternation limit reached"
    full = glm_nb(x, y)
    assert full.th_warn is None and full.alternations <= 25


def test_theta_ml_iteration_limit_and_attributes() -> None:
    rng = np.random.default_rng(6)
    mu = np.full(500, 3.0)
    y = rng.poisson(mu).astype(np.float64)
    w = np.ones(500)
    t = theta_ml(y, mu, 500.0, w, limit=25)
    assert t.iterations <= 25
    short = theta_ml(y, mu, 500.0, w, limit=2)
    assert short.iterations == 2 and short.warning == "iteration limit reached"
    assert short.warnings == ("iteration limit reached",)
    # SE is sqrt(1 / info) at the last Newton step's start.
    assert short.se > 0


def test_split_search_fits_are_quiet_and_node_fits_warn() -> None:
    y = np.r_[np.zeros(29), 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = NegBinModel().fit(y, None, weights=np.ones(30), estfun=False)
    assert r.extras["warnings"] == (NEAR_POISSON, NEAR_POISSON)


def test_objective_option() -> None:
    with pytest.raises(ValueError, match="objective"):
        NegBinModel(objective="aic")  # type: ignore[arg-type]
    x, y = _sim(7)
    w = np.ones(y.shape[0])
    dev = NegBinModel(objective="deviance").fit(y, x, weights=w)
    z = glm_nb(x, y)
    assert dev.objfun == z.deviance


def test_negative_binomial_family_against_statsmodels() -> None:
    sm = pytest.importorskip("statsmodels.api")
    x, y = _sim(8)
    for theta in (0.7, 2.0, 50.0):
        # Both converged far past glm.fit's default, whose deviance
        # criterion leaves the coefficients about sqrt(epsilon) from the MLE.
        ours = glm_fit(
            x, y, NegativeBinomial(theta), epsilon=1e-15, maxit=100, intercept=False
        )
        theirs = sm.GLM(y, x, family=sm.families.NegativeBinomial(alpha=1 / theta)).fit(
            tol=1e-14, tol_criterion="params", maxiter=200
        )
        np.testing.assert_allclose(ours.coefficients, theirs.params, rtol=RTOL)
        assert ours.deviance == pytest.approx(theirs.deviance, rel=RTOL)
        mu = ours.fitted_values
        assert -ours.aic / 2 + ours.rank == pytest.approx(
            negbin_loglik(theta, mu, y, np.ones(y.shape[0])), rel=1e-12
        )


def test_etastart_takes_precedence_over_start() -> None:
    x, y = _sim(9)
    fam = Poisson()
    ref = glm_fit(x, y, fam)
    again = glm_fit(x, y, fam, etastart=np.log(ref.fitted_values))
    np.testing.assert_allclose(again.coefficients, ref.coefficients, rtol=1e-8)
    assert again.iter <= 2
    # R evaluates etastart first and never checks start's length then.
    both = glm_fit(x, y, fam, etastart=np.log(ref.fitted_values), start=[0.0, 0.0])
    np.testing.assert_array_equal(both.coefficients, again.coefficients)
