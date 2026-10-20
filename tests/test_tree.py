"""Tree growing and pruning against the partykit tree fixtures.

The tests go component by component, in the order of CLAUDE.md:

1. split search: every candidate's objective, on the fixture node's rows;
2. the chosen split variable and break;
3. node sizes, rows and coefficients of the grown tree;
4. whole trees: node ids, structure, stop reasons, tests, routing of rows
   to leaves and pruning.

The zero-weight fixture is grown with its zero weights and compared with
the fixture grown on the data with those rows deleted (docs/decisions.md).
"""

import dataclasses
import math
import warnings
from functools import cache
from typing import Any

import numpy as np
import pytest
from fixtures_io import nan_array
from tree_fixtures import (
    TREE_FILES,
    cluster,
    data,
    inputs,
    model,
    tree,
    variables,
)

from modeltrees.instability import TooFewClustersWarning, VariableKind
from modeltrees.models.glm import GlmModel
from modeltrees.models.lm import LmModel
from modeltrees.node_model import FitResult
from modeltrees.pruning import lm_prune_rule, mob_prune_rule
from modeltrees.split_search import Split, SplitSearch, find_split, getlevels
from modeltrees.tree import GrownTree, MobControl, Node, StopReason, mob

RTOL = 1e-8
ATOL = 1e-10

# partykit fails the root test on zero weights; the port treats them as
# deleted rows, so this fixture is not an oracle (docs/instability-tests.md).
R_ZERO_WEIGHTS = "sim_gaussian_lmtree_zero_weights"
DELETED = "sim_gaussian_lmtree_zero_weights_deleted"
ORACLE_TREES = [name for name in TREE_FILES if name != R_ZERO_WEIGHTS]
FEW_CLUSTERS = "sim_cluster_few_lmtree_cluster"


def _control(fixture: dict[str, Any]) -> MobControl:
    c = fixture["control"]
    maxdepth = math.inf if c["maxdepth"] == "Inf" else float(c["maxdepth"])
    return MobControl(
        alpha=c["alpha"],
        bonferroni=c["bonferroni"],
        minsize=c["minsize"],
        maxdepth=maxdepth,
        trim=c["trim"],
        dfsplit=c["dfsplit"],
        prune=c["prune"],
        restart=c["restart"],
        caseweights=c["caseweights"],
        numsplit=c["numsplit"],
        catsplit=c["catsplit"],
    )


def _grow(name: str, *, quiet: bool = True) -> GrownTree:
    fixture = tree(name)
    y, x, w, off = inputs(fixture)
    with warnings.catch_warnings():
        # Only the few-clusters fixture warns; it is checked on its own.
        if quiet:
            warnings.simplefilter("ignore", TooFewClustersWarning)
        return mob(
            model(fixture),
            y,
            x,
            variables(fixture),
            weights=w,
            offset=off,
            cluster=cluster(fixture),
            control=_control(fixture),
        )


@cache
def _grown(name: str) -> GrownTree:
    return _grow(name)


def _search_nodes() -> list[Any]:
    return [
        pytest.param(name, node["id"], id=f"{name}-{node['id']}")
        for name in ORACLE_TREES
        for node in tree(name)["nodes"]
        if node["split_search"] is not None
    ]


SEARCH_NODES = _search_nodes()


def _node(name: str, node_id: int) -> dict[str, Any]:
    return next(n for n in tree(name)["nodes"] if n["id"] == node_id)


@cache
def _search(name: str, node_id: int) -> SplitSearch:
    """Split search on the fixture node's rows in R's selected variable."""
    fixture = tree(name)
    node = _node(name, node_id)
    y, x, w, off = inputs(fixture)
    var = variables(fixture)[node["split_search"]["variable"]]
    rows = np.array(node["rows"], dtype=np.intp)
    m = model(fixture)
    cl = cluster(fixture)

    def fit(mask: np.ndarray, start: np.ndarray | None) -> FitResult:
        r = rows[mask]
        return m.fit(
            y[r],
            None if x is None else x[r],
            start=start,
            weights=w[r],
            offset=None if off is None else off[r],
            cluster=None if cl is None else cl[r],
            caseweights=fixture["control"]["caseweights"],
            estfun=False,
            vcov=False,
        )

    return find_split(
        fixture["partitioning"].index(var.name),
        var.kind,
        var.values[rows],
        var.levels,
        w[rows],
        fit,
        minsize=node["minsize"],
        caseweights=fixture["control"]["caseweights"],
        restart=fixture["control"]["restart"],
        numsplit=fixture["control"]["numsplit"],
    )


def _objective(values: list[float | None]) -> np.ndarray:
    """R's Inf objective is null in the fixture."""
    return np.array([np.inf if v is None else v for v in values], dtype=np.float64)


def _assert_split(got: Split | None, expected: dict[str, Any] | None) -> None:
    if expected is None:
        assert got is None
        return
    assert got is not None
    assert got.variable == expected["variable_index"]
    assert got.kind.value == expected["kind"]
    if expected["kind"] == "nominal":
        assert got.breaks is None
        assert got.index == tuple(expected["index"])
    else:
        assert got.index is None
        assert got.breaks == expected["breaks"]


# Level 1: split candidate objectives.


def test_search_nodes_cover_every_fixture_split() -> None:
    assert len(SEARCH_NODES) == 134
    for name in ORACLE_TREES:
        for node in tree(name)["nodes"]:
            if node["split"] is not None:
                assert node["split_search"] is not None


@pytest.mark.parametrize(("name", "node_id"), SEARCH_NODES)
def test_split_candidates(name: str, node_id: int) -> None:
    expected = _node(name, node_id)["split_search"]
    got = _search(name, node_id)
    assert got.minsize == expected["minsize"]
    if got.kind is VariableKind.NUMERIC:
        assert got.values is not None
        np.testing.assert_array_equal(got.values, np.array(expected["candidates"]))
    else:
        assert got.labels == tuple(expected["candidates"])


@pytest.mark.parametrize(("name", "node_id"), SEARCH_NODES)
def test_split_objectives(name: str, node_id: int) -> None:
    expected = _objective(_node(name, node_id)["split_search"]["objective"])
    got = _search(name, node_id).objective
    np.testing.assert_array_equal(np.isinf(got), np.isinf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(got[finite], expected[finite], rtol=RTOL, atol=ATOL)


# Level 2: the chosen split variable and break.


@pytest.mark.parametrize(("name", "node_id"), SEARCH_NODES)
def test_chosen_split(name: str, node_id: int) -> None:
    node = _node(name, node_id)
    got = _search(name, node_id)
    if node["stop_reason"] == "no_admissible_split":
        assert got.split is None and got.best is None
        return
    if node["stop_reason"] == "pruned":
        # partykit drops the split of a pruned node; it was found, though.
        assert got.split is not None
        return
    _assert_split(got.split, node["split"])


# Level 3: node sizes, rows and coefficients of the grown tree.


TREES = [pytest.param(name, id=name) for name in ORACLE_TREES]


@pytest.mark.parametrize("name", TREES)
def test_node_sizes(name: str) -> None:
    got = _grown(name).nodes
    expected = tree(name)["nodes"]
    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        assert g.rows.shape[0] == e["n_rows"]
        np.testing.assert_array_equal(g.rows, e["rows"])
        assert g.nobs == e["nobs"]


@pytest.mark.parametrize("name", TREES)
def test_node_coefficients(name: str) -> None:
    for g, e in zip(_grown(name).nodes, tree(name)["nodes"], strict=True):
        assert g.fit.coef_names == tuple(e["coefficient_names"])
        np.testing.assert_allclose(
            g.fit.coefficients, nan_array(e["coefficients"]), rtol=RTOL, atol=ATOL
        )
        assert g.fit.objfun == pytest.approx(e["objfun"], rel=RTOL, abs=ATOL)


# Level 4: whole trees.


def _parents(root: Node) -> dict[int, int | None]:
    out: dict[int, int | None] = {root.id: None}
    for node in root.walk():
        for kid in node.kids:
            out[kid.id] = node.id
    return out


@pytest.mark.parametrize("name", TREES)
def test_tree_structure(name: str) -> None:
    fixture = tree(name)
    got = _grown(name)
    parents = _parents(got.root)
    assert got.minsize == fixture["root_minsize"]
    for g, e in zip(got.nodes, fixture["nodes"], strict=True):
        assert g.id == e["id"]
        assert g.depth == e["depth"]
        assert parents[g.id] == e["parent"]
        assert [k.id for k in g.kids] == e["kids"]
        assert g.terminal == e["terminal"]
        assert g.stop_reason.value == e["stop_reason"]
        _assert_split(g.split, e["split"])
        assert got.minsize == e["minsize"]


@pytest.mark.parametrize("name", TREES)
def test_fitted_node(name: str) -> None:
    np.testing.assert_array_equal(_grown(name).fitted_node, tree(name)["fitted_node"])


@pytest.mark.parametrize("name", TREES)
def test_node_tests(name: str) -> None:
    for g, e in zip(_grown(name).nodes, tree(name)["nodes"], strict=True):
        status = e["test"]["status"] if e["test"] else "not_run"
        if status == "not_run":
            assert g.test is None and not g.test_failed
            assert g.p_value is None
            continue
        if status == "failed":
            assert g.test is None and g.test_failed
            assert g.p_value is None
            continue
        t = e["test"]
        assert g.test is not None
        np.testing.assert_allclose(g.test.statistic, t["statistic"], rtol=RTOL)
        np.testing.assert_allclose(
            g.test.p_value_adjusted, nan_array(t["p_value_adjusted"]), rtol=RTOL
        )
        assert g.test.best == t["best"]
        assert g.p_value == pytest.approx(e["p_value"], rel=RTOL)


@pytest.mark.parametrize("name", TREES)
def test_split_searches_in_grown_tree(name: str) -> None:
    for g, e in zip(_grown(name).nodes, tree(name)["nodes"], strict=True):
        if e["split_search"] is None:
            assert g.split_search is None
            continue
        assert g.split_search is not None
        assert g.split_search.variable == e["test"]["best"]
        expected = _objective(e["split_search"]["objective"])
        got = g.split_search.objective
        np.testing.assert_array_equal(np.isinf(got), np.isinf(expected))
        finite = np.isfinite(expected)
        np.testing.assert_allclose(got[finite], expected[finite], rtol=RTOL, atol=ATOL)


def test_summary_counts() -> None:
    for name in ORACLE_TREES:
        got = _grown(name)
        summary = tree(name)["summary"]
        assert len(got.nodes) == summary["n_nodes"]
        assert sum(n.terminal for n in got.nodes) == summary["n_terminal"]
        assert max(n.depth for n in got.nodes) == summary["depth"]
        terminal_objfun = sum(n.fit.objfun for n in got.nodes if n.terminal)
        assert terminal_objfun == pytest.approx(summary["objfun"], rel=RTOL)


# Oracle checks for the options that used to have only Python consistency
# tests. The fixtures are in ORACLE_TREES, so the level 1 to 4 tests above grow
# them with the fixture's own control settings. The tests below check that the
# setting is really in force and matters, against the fixture of the default.

NOCASEWEIGHTS = [
    ("sim_gaussian_lmtree_weights_nocaseweights", "sim_gaussian_lmtree_weights"),
    (
        "sim_gaussian_glmtree_gaussian_weights_nocaseweights",
        "sim_gaussian_glmtree_gaussian_weights",
    ),
]


@pytest.mark.parametrize(("name", "default"), NOCASEWEIGHTS)
def test_caseweights_false_oracle(name: str, default: str) -> None:
    fixture = tree(name)
    assert fixture["control"]["caseweights"] is False
    assert fixture["weights"] == tree(default)["weights"] is not None
    got = _grown(name)
    # R counts rows, not the weight total, so nobs is the row count.
    assert got.root.nobs == fixture["n"] == fixture["nodes"][0]["nobs"]
    assert tree(default)["nodes"][0]["nobs"] > fixture["n"]
    # The setting matters: the same call with caseweights = TRUE counts the
    # weight total, as the default fixture does.
    y, x, w, off = inputs(fixture)
    grown_default = mob(
        model(fixture),
        y,
        x,
        variables(fixture),
        weights=w,
        offset=off,
        control=dataclasses.replace(_control(fixture), caseweights=True),
    )
    assert grown_default.root.nobs == tree(default)["nodes"][0]["nobs"]
    assert grown_default.root.nobs != got.root.nobs


def test_restart_false_oracle() -> None:
    name = "sim_binomial_glmtree_multisplit_norestart"
    default = "sim_binomial_glmtree_multisplit"
    fixture = tree(name)
    assert fixture["control"]["restart"] is False
    assert tree(default)["control"]["restart"] is True
    # R's restart = FALSE searches change the split objectives in the grown
    # tree, so the match in test_split_searches_in_grown_tree is not trivial.
    objective = [
        [n["split_search"]["objective"] for n in tree(f)["nodes"] if n["split_search"]]
        for f in (name, default)
    ]
    assert objective[0] != objective[1]
    got = _grown(name)
    expected = [n for n in fixture["nodes"] if n["split_search"]]
    for node in expected:
        search = got.node(node["id"]).split_search
        assert search is not None
        np.testing.assert_allclose(
            search.objective,
            _objective(node["split_search"]["objective"]),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
        )


def test_few_clusters_node_warns_and_is_terminal() -> None:
    with pytest.warns(TooFewClustersWarning, match="3 clusters for 4 parameters"):
        got = _grow(FEW_CLUSTERS, quiet=False)
    node = got.node(3)
    assert node.terminal and node.test_failed
    assert node.stop_reason is StopReason.TEST_FAILED


# Zero weights: grown with the zero weights, compared with the R tree grown on
# the data with those rows deleted.


def test_zero_weights_equal_deleted_rows() -> None:
    deleted = tree(DELETED)
    full = tree(R_ZERO_WEIGHTS)
    got = _grow(R_ZERO_WEIGHTS)
    kept = np.setdiff1d(np.arange(full["n"]), deleted["removed_rows"])
    w0 = data(full["data_csv"])[0]["w0"]
    assert np.array_equal(np.flatnonzero(w0 > 0), kept)
    nodes = got.nodes
    assert len(nodes) == len(deleted["nodes"])
    parents = _parents(got.root)
    for g, e in zip(nodes, deleted["nodes"], strict=True):
        assert g.id == e["id"]
        assert parents[g.id] == e["parent"]
        assert [k.id for k in g.kids] == e["kids"]
        assert g.stop_reason.value == e["stop_reason"]
        _assert_split(g.split, e["split"])
        positive = g.rows[w0[g.rows] > 0]
        np.testing.assert_array_equal(positive, kept[e["rows"]])
        assert g.nobs == e["nobs"]
        np.testing.assert_allclose(
            g.fit.coefficients, nan_array(e["coefficients"]), rtol=RTOL, atol=ATOL
        )
        assert g.fit.objfun == pytest.approx(e["objfun"], rel=RTOL, abs=ATOL)
        if e["split_search"] is not None:
            assert g.split_search is not None
            expected = _objective(e["split_search"]["objective"])
            finite = np.isfinite(expected)
            np.testing.assert_array_equal(np.isinf(g.split_search.objective), ~finite)
            np.testing.assert_allclose(
                g.split_search.objective[finite], expected[finite], rtol=RTOL
            )
    np.testing.assert_array_equal(got.fitted_node[kept], deleted["fitted_node"])
    # Every zero-weight row reaches a leaf.
    assert np.all(got.fitted_node > 0)


def test_zero_weights_r_fixture_is_a_documented_divergence() -> None:
    """partykit stops at the root; the port grows the deleted-rows tree."""
    r_nodes = tree(R_ZERO_WEIGHTS)["nodes"]
    assert len(r_nodes) == 1 and r_nodes[0]["stop_reason"] == "test_failed"
    assert len(_grow(R_ZERO_WEIGHTS).nodes) == len(tree(DELETED)["nodes"])


def test_zero_weight_values_do_not_create_candidates() -> None:
    """A zero-weight row's value would tie with its neighbour; it must not
    become a candidate break (docs/decisions.md)."""
    rng = np.random.default_rng(1)
    n = 200
    z = np.sort(rng.uniform(0, 10, n))
    y = np.where(z <= 5, 0.0, 3.0) + rng.normal(0, 0.1, n)
    w = np.ones(n)
    # A zero-weight row between the last left value and the first right one.
    i = int(np.searchsorted(z, 5))
    z_extra = np.append(z, (z[i - 1] + z[i]) / 2)
    y_extra = np.append(y, 100.0)
    w_extra = np.append(w, 0.0)
    m = LmModel()
    a = mob(m, y, None, {"z": ("numeric", z)}, control=MobControl(minsize=20))
    b = mob(
        m,
        y_extra,
        None,
        {"z": ("numeric", z_extra)},
        weights=w_extra,
        control=MobControl(minsize=20),
    )
    assert a.root.split == b.root.split
    assert a.root.split_search is not None and b.root.split_search is not None
    assert b.root.split_search.values is not None
    assert z_extra[-1] not in b.root.split_search.values
    # lm.fit (unit weights) and lm.wfit (a zero weight) differ in rounding.
    np.testing.assert_allclose(
        a.root.split_search.objective, b.root.split_search.objective, rtol=1e-12
    )


def test_zero_weight_rows_with_an_absent_level_go_to_the_larger_kid() -> None:
    rng = np.random.default_rng(2)
    n = 300
    g = rng.integers(0, 3, n)  # levels a, b, c; d only in zero-weight rows
    y = np.where(g == 0, 0.0, 4.0) + rng.normal(0, 0.1, n)
    g_extra = np.append(g, [3, 3])
    y_extra = np.append(y, [0.0, 0.0])
    w = np.append(np.ones(n), [0.0, 0.0])
    got = mob(
        LmModel(),
        y_extra,
        None,
        {"g": ("nominal", g_extra, ["a", "b", "c", "d"])},
        weights=w,
        control=MobControl(minsize=20, maxdepth=2),
    )
    split = got.root.split
    assert split is not None and split.index is not None
    assert split.index[3] is None
    sizes = [kid.nobs for kid in got.root.kids]
    larger = got.root.kids[int(np.argmax(sizes))]
    assert np.all(got.fitted_node[-2:] == larger.id)


# Unit tests of the pieces, and options without fixtures.


def test_getlevels_nominal_matches_r() -> None:
    # mob_grow_getlevels(factor(c("a","b","c"))) in R:
    #   a: T F F; b: F T F; a+b: T T F
    np.testing.assert_array_equal(
        getlevels(3, ordered=False),
        [[True, False, False], [False, True, False], [True, True, False]],
    )
    al = getlevels(4, ordered=False)
    assert al.shape == (7, 4)
    assert not al[:, -1].any()
    assert len({tuple(r) for r in al}) == 7


def test_getlevels_ordered_matches_r() -> None:
    np.testing.assert_array_equal(
        getlevels(4, ordered=True),
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
        ],
    )


def _toy(seed: int, n: int = 300) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    z1 = rng.uniform(-1, 1, n)
    z2 = rng.integers(0, 4, n)
    y = np.where(z1 > 0, 1.0, -1.0) * x[:, 1] + (z2 == 2) + rng.normal(0, 0.5, n)
    z = {
        "z1": ("numeric", z1),
        "z2": ("nominal", z2, ["a", "b", "c", "d"]),
        "z3": ("ordered", rng.integers(0, 3, n), ["lo", "mid", "hi"]),
    }
    return y, x, z


def test_default_minsize_is_set_once_at_the_root() -> None:
    y, x, z = _toy(3)
    got = mob(LmModel(), y, x, z)
    assert got.minsize == 20  # ceiling(10 * 2 / 1)
    for node in got.nodes:
        if node.split_search is not None:
            assert node.split_search.minsize == 20


def test_minsize_is_truncated_as_r_as_integer() -> None:
    y, x, z = _toy(3)
    assert mob(LmModel(), y, x, z, control=MobControl(minsize=25.9)).minsize == 25


def test_restart_false_reuses_starting_values() -> None:
    y, x, z = _toy(4)
    yb = (y > 0).astype(float)
    a = mob(GlmModel("binomial"), yb, x, z)
    b = mob(GlmModel("binomial"), yb, x, z, control=MobControl(restart=False))
    assert [n.split for n in a.nodes] == [n.split for n in b.nodes]
    for na, nb in zip(a.nodes, b.nodes, strict=True):
        if na.split_search is not None:
            assert nb.split_search is not None
            np.testing.assert_allclose(
                na.split_search.objective, nb.split_search.objective, rtol=1e-6
            )


def test_restart_false_passes_previous_coefficients() -> None:
    y, x, z = _toy(5)
    starts: list[np.ndarray | None] = []

    class Recording(LmModel):
        def fit(self, *args: Any, **kwargs: Any) -> FitResult:
            starts.append(kwargs["start"])
            return super().fit(*args, **kwargs)

    mob(Recording(), y, x, z, control=MobControl(restart=False, maxdepth=2))
    # Node fit, then left/right pairs: the first pair starts from None,
    # later pairs from the previous admissible pair's coefficients.
    assert starts[0] is None and starts[1] is None and starts[2] is None
    assert starts[3] is not None and starts[4] is not None


def test_caseweights_false_with_unit_weights_matches_default() -> None:
    y, x, z = _toy(6)
    a = mob(LmModel(), y, x, z)
    b = mob(LmModel(), y, x, z, control=MobControl(caseweights=False))
    assert [n.split for n in a.nodes] == [n.split for n in b.nodes]
    for na, nb in zip(a.nodes, b.nodes, strict=True):
        assert na.nobs == nb.nobs
        if na.test is not None:
            assert nb.test is not None
            np.testing.assert_allclose(na.test.statistic, nb.test.statistic)


def test_caseweights_false_counts_positive_weights() -> None:
    y, x, z = _toy(7)
    w = np.full(y.shape[0], 2.0)
    got = mob(LmModel(), y, x, z, weights=w, control=MobControl(caseweights=False))
    assert got.root.nobs == y.shape[0]
    got_cw = mob(LmModel(), y, x, z, weights=w)
    assert got_cw.root.nobs == 2 * y.shape[0]


def test_maxdepth_and_alpha() -> None:
    y, x, z = _toy(8)
    assert len(mob(LmModel(), y, x, z, control=MobControl(maxdepth=1)).nodes) == 1
    root = mob(LmModel(), y, x, z, control=MobControl(alpha=0.0)).root
    assert root.terminal and root.stop_reason is StopReason.NOT_SIGNIFICANT


def test_bonferroni_off_uses_raw_p_values() -> None:
    y, x, z = _toy(9)
    root = mob(LmModel(), y, x, z, control=MobControl(bonferroni=False)).root
    assert root.test is not None
    np.testing.assert_array_equal(root.test.p_value_adjusted, root.test.p_value)


def test_numsplit_center_is_the_midpoint() -> None:
    y, x, z = _toy(10)
    a = mob(LmModel(), y, x, z, control=MobControl(maxdepth=2)).root
    b = mob(LmModel(), y, x, z, control=MobControl(maxdepth=2, numsplit="center")).root
    assert a.split_search is not None and a.split_search.values is not None
    assert a.split is not None and b.split is not None
    assert a.split.variable == 0 and a.split_search.best is not None
    values = a.split_search.values
    i = a.split_search.best
    assert b.split.breaks == (values[i] + values[i + 1]) / 2


def test_one_candidate_factor_objective_is_one() -> None:
    rng = np.random.default_rng(11)
    n = 200
    g = rng.integers(0, 2, n)
    y = 3.0 * g + rng.normal(size=n)
    root = mob(LmModel(), y, None, {"g": ("nominal", g, ["a", "b"])}).root
    assert root.split_search is not None
    np.testing.assert_array_equal(root.split_search.objective, [1.0])
    assert root.split is not None and root.split.index == (1, 2)


def test_labels_and_codes_give_the_same_tree() -> None:
    y, x, z = _toy(12)
    codes = z["z2"][1]
    labels = np.array(["a", "b", "c", "d"])[codes]
    a = mob(LmModel(), y, x, z)
    z_labels = dict(z, z2=("nominal", labels))
    b = mob(LmModel(), y, x, z_labels)
    assert [n.split for n in a.nodes] == [n.split for n in b.nodes]


def test_unsupported_options_raise() -> None:
    for kwargs in (
        {"mtry": 2},
        {"breakties": True},
        {"ordinal": "L2"},
        {"ordinal": "max"},
        {"vcov": "sandwich"},
        {"vcov": "info"},
        {"parm": [0]},
        {"catsplit": "multiway"},
    ):
        with pytest.raises(NotImplementedError):
            MobControl(**kwargs)


def test_invalid_inputs_raise() -> None:
    y, x, z = _toy(13)
    with pytest.raises(ValueError, match="kind"):
        mob(LmModel(), y, x, {"z": ("binary", z["z1"][1])})
    with pytest.raises(ValueError, match="missing"):
        mob(LmModel(), y, x, {"z": ("numeric", np.full(y.shape[0], np.nan))})
    with pytest.raises(ValueError, match="weights"):
        mob(LmModel(), y, x, z, weights=-np.ones(y.shape[0]))
    with pytest.raises(ValueError, match="prune"):
        mob(LmModel(), y, x, z, control=MobControl(prune="cv"))


def test_prune_rules() -> None:
    aic = mob_prune_rule("AIC")
    bic = mob_prune_rule("bic")
    assert aic is not None and bic is not None
    assert mob_prune_rule("none") is None
    # Parent -loglik 100 with 2 df vs kids 97 with 5 df: AIC 204 vs 204 keeps
    # the split (strict <), BIC with n = 100 prunes.
    assert not aic((100.0, 97.0), (2, 5), (50.0, 100.0))
    assert bic((100.0, 97.0), (2, 5), (50.0, 100.0))
    # The mob BIC uses log(n) of the full sample, nobs[1], not the node's.
    assert bic((100.0, 97.0), (2, 5), (8.0, 100.0)) == bic(
        (100.0, 97.0), (2, 5), (50.0, 100.0)
    )
    lm_aic = lm_prune_rule("aic")
    assert lm_aic is not None
    rss_parent, rss_kids, nobs = 50.0, 45.0, 100.0
    expected = nobs * math.log(rss_parent) + 2 * 2 < nobs * math.log(rss_kids) + 2 * 5
    assert lm_aic((rss_parent, rss_kids), (2, 5), (nobs, 1000.0)) == expected


def test_lmtree_uses_its_own_prune_rules() -> None:
    y, x, z = _toy(14)
    control = MobControl(alpha=0.9, prune="aic", maxdepth=3)
    got = mob(LmModel(), y, x, z, control=control)
    for node in got.nodes:
        if node.stop_reason is StopReason.PRUNED:
            assert node.terminal and node.split is None
    # The ids are preorder after pruning.
    assert [n.id for n in got.nodes] == list(range(1, len(got.nodes) + 1))
    for node in got.nodes:
        if node.terminal:
            assert np.all(got.fitted_node[node.rows] == node.id)


def test_prune_none_keeps_the_tree() -> None:
    y, x, z = _toy(15)
    a = mob(LmModel(), y, x, z, control=MobControl(alpha=0.9, maxdepth=3))
    b = mob(LmModel(), y, x, z, control=MobControl(alpha=0.9, maxdepth=3, prune="none"))
    assert len(a.nodes) == len(b.nodes)


def test_callable_prune_rule() -> None:
    y, x, z = _toy(16)
    calls: list[tuple[Any, Any, Any]] = []

    def always(objfun: Any, df: Any, nobs: Any) -> bool:
        calls.append((objfun, df, nobs))
        return True

    got = mob(LmModel(), y, x, z, control=MobControl(alpha=0.9, prune=always))
    assert len(got.nodes) == 1
    assert got.root.stop_reason is StopReason.PRUNED
    assert all(df == (2, 5) for _, df, _ in calls)  # 2 kids * 2 + dfsplit
    assert all(nobs[1] == y.shape[0] for _, _, nobs in calls)
    no_dfsplit = MobControl(alpha=0.9, prune=always, dfsplit=False)
    calls.clear()
    mob(LmModel(), y, x, z, control=no_dfsplit)
    assert all(df == (2, 4) for _, df, _ in calls)
