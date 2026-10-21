"""Every tree fixture grown through the public entry points
(docs/api-design.md, section 9, items 1, 3, 4 and 5).

Each tree and extension tree fixture is read into a polars frame and grown
by its entry point with the fixture's formula, weights, offset, cluster
and control. The grown tree must equal the array-path tree of
tests/grown_trees.py exactly (same arrays, same code), which the model
tests compare with R. On top: the fitted nodes against R's, the tree
log-likelihood against R's ``logLik(tree)``, predictions against leaf fits
computed by hand, and the coefficient table against the fixture.
"""

import math
import warnings
from dataclasses import fields
from functools import cache
from typing import Any

import numpy as np
import polars as pl
import pytest
from fixture_frames import offset_column, ordered_kinds, polars_frame
from fixtures_io import nan_array
from grown_trees import ALL_TREES, control, fixture, grow, model, tree_inputs
from slow import mark_slow_trees
from test_multinom import _coef_tol, _node_data, _r_offset
from test_safelogit import SEPARATED_RTOL
from tree_fixtures import cluster

import modeltrees as mt
from modeltrees.models import GlmModel, LmModel, MultinomModel, SafeLogitModel
from modeltrees.tree import Node

# partykit's tree on zero weights is not an oracle (docs/decisions.md).
NOT_ORACLE = "sim_gaussian_lmtree_zero_weights"
RTOL = 1e-8
ATOL = 1e-10
PREDICT_RTOL = 1e-12
DBL_EPSILON = float(np.finfo(np.float64).eps)

ORACLE_TREES = mark_slow_trees(
    [(d, n) for d, n in ALL_TREES if n != NOT_ORACLE], name_at=1
)
TREES = mark_slow_trees(ALL_TREES, name_at=1)


@cache
def public(directory: str, name: str) -> mt.ModelTree:
    f = fixture(directory, name)
    df = polars_frame(f)
    common: dict[str, Any] = {
        "weights": f["weights"],
        "offset": offset_column(f)[0],
        "cluster": f.get("cluster"),
        "kinds": ordered_kinds(f),
        "control": control(f),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mt.ModelTreesWarning)
        if f["function_name"] == "lmtree":
            return mt.lmtree(f["formula"], df, **common)
        if f["function_name"] == "glmtree":
            return mt.glmtree(
                f["formula"],
                df,
                family=f["family"]["family"],
                **f["glm_control"],
                **common,
            )
        fit_function = f["fit_function"]
        if fit_function == "negbinfit":
            return mt.negbintree(f["formula"], df, **f["fit_args"], **common)
        if fit_function == "safelogitfit":
            return mt.safelogittree(f["formula"], df, **f["fit_args"], **common)
        assert fit_function == "multinomfit"
        # The fit arguments are nnet's; the Newton fit has its own.
        common.pop("offset")
        return mt.multinomtree(f["formula"], df, **common)


# 1. The public tree is the array-path tree.


def _same(a: Any, b: Any) -> bool:
    """Exact equality, NaN equal to NaN, arrays elementwise."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
            return False
        if a.dtype.kind == "f":
            return bool(np.array_equal(a, b, equal_nan=True))
        return bool(np.array_equal(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if hasattr(a, "__dataclass_fields__") and type(a) is type(b):
        return all(_same(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    return bool(a == b) and type(a) is type(b)


def _assert_same_node(got: Node, expected: Node) -> None:
    where = f"node {expected.id}"
    for attr in ("id", "depth", "nobs", "stop_reason", "test_failed", "p_value"):
        assert _same(getattr(got, attr), getattr(expected, attr)), (where, attr)
    assert _same(got.rows, expected.rows), (where, "rows")
    for f in fields(expected.fit):
        assert _same(getattr(got.fit, f.name), getattr(expected.fit, f.name)), (
            where,
            f"fit.{f.name}",
        )
    for attr in ("test", "split_search", "split"):
        assert _same(getattr(got, attr), getattr(expected, attr)), (where, attr)
    assert [k.id for k in got.kids] == [k.id for k in expected.kids], where


@pytest.mark.parametrize(("directory", "name"), TREES)
def test_public_tree_equals_the_array_path_tree(directory: str, name: str) -> None:
    got = public(directory, name)._grown
    expected = grow(directory, name)
    assert got.minsize == expected.minsize
    assert got.control == expected.control
    assert _same(got.fitted_node, expected.fitted_node)
    assert len(got.variables) == len(expected.variables)
    for a, b in zip(got.variables, expected.variables, strict=True):
        assert (a.name, a.kind, a.levels) == (b.name, b.kind, b.levels)
        assert np.array_equal(a.values, b.values)
    assert len(got.nodes) == len(expected.nodes)
    for a_node, b_node in zip(got.nodes, expected.nodes, strict=True):
        _assert_same_node(a_node, b_node)


@pytest.mark.parametrize(("directory", "name"), TREES)
def test_tree_fields(directory: str, name: str) -> None:
    f = fixture(directory, name)
    tree = public(directory, name)
    n = len(f["fitted_node"])
    assert tree.n_rows == tree.n_used == n and tree.used_rows.all()
    assert tree.formula is not None
    levels = f["response_levels"]
    assert tree.response_levels == (None if levels is None else tuple(levels))
    assert tree.depth == f["summary"]["depth"] or name == NOT_ORACLE
    assert len(tree.leaves) == f["summary"]["n_terminal"] or name == NOT_ORACLE


# 3. fitted_node and predict(type="node") against R.


@pytest.mark.parametrize(("directory", "name"), ORACLE_TREES)
def test_fitted_nodes_match_r(directory: str, name: str) -> None:
    f = fixture(directory, name)
    tree = public(directory, name)
    expected = np.asarray(f["fitted_node"], dtype=np.int64)
    np.testing.assert_array_equal(tree.fitted_node, expected)
    np.testing.assert_array_equal(tree.predict(type="node"), expected)
    np.testing.assert_array_equal(tree.predict(polars_frame(f), type="node"), expected)


def test_zero_weights_fixture_routes_as_the_array_path() -> None:
    # R's tree is not an oracle here; the routing of the new data equals the
    # array path's fitted nodes, zero-weight rows included.
    f = fixture("trees", NOT_ORACLE)
    tree = public("trees", NOT_ORACLE)
    expected = grow("trees", NOT_ORACLE).fitted_node
    np.testing.assert_array_equal(tree.predict(polars_frame(f), type="node"), expected)
    assert not np.array_equal(expected, f["fitted_node"])


# 5. logLik against R.


@pytest.mark.parametrize(("directory", "name"), ORACLE_TREES)
def test_loglik_matches_r(directory: str, name: str) -> None:
    f = fixture(directory, name)
    tree = public(directory, name)
    ll = tree.loglik()
    assert ll.value == pytest.approx(f["summary"]["loglik"], rel=RTOL, abs=0)
    assert ll.df == f["summary"]["loglik_df"]
    assert ll.nobs == f["nobs"]
    assert tree.aic() == -2 * ll.value + 2 * ll.df
    assert tree.bic() == -2 * ll.value + math.log(ll.nobs) * ll.df


# 4. Predictions against leaf fits computed by hand.


def _inverse_link(m: Any, eta: np.ndarray) -> np.ndarray:
    """The inverse links written out, with R's bounds (family.c and
    make.link("log"))."""
    if isinstance(m, LmModel):
        return eta
    family = "binomial" if isinstance(m, SafeLogitModel) else None
    if isinstance(m, GlmModel):
        family = m.family.family
    if family == "gaussian":
        return eta
    if family == "binomial":
        e = np.exp(np.clip(eta, -30, 30))
        e = np.where(eta < -30, DBL_EPSILON, np.where(eta > 30, 1 / DBL_EPSILON, e))
        return e / (1 + e)
    return np.maximum(np.exp(eta), DBL_EPSILON)


def _leaf_refit(f: dict[str, Any], leaf: Node) -> Any:
    """The leaf's model fitted directly on its rows."""
    m = model(f)
    y, x, w, off = tree_inputs(f)
    cl = cluster(f)
    rows = leaf.rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mt.GlmFitWarning)
        return m.fit(
            y[rows],
            None if x is None else x[rows],
            start=None,
            weights=w[rows],
            offset=None if off is None else off[rows],
            cluster=None if cl is None else cl[rows],
            caseweights=control(f).caseweights,
            estfun=True,
            vcov=False,
        )


@pytest.mark.parametrize(("directory", "name"), TREES)
def test_predict_equals_hand_computed_leaf_fits(directory: str, name: str) -> None:
    f = fixture(directory, name)
    tree = public(directory, name)
    df = polars_frame(f)
    m = model(f)
    y, x, _, off = tree_inputs(f)
    xm = np.ones((y.shape[0], 1)) if x is None else x
    multinom = isinstance(m, MultinomModel)
    kinds = ("link", "prob", "class") if multinom else ("link", "response")
    got = {kind: tree.predict(df, type=kind) for kind in kinds}  # type: ignore[arg-type]
    for kind in kinds:
        same = tree.predict(type=kind)  # type: ignore[arg-type]
        assert _same(np.asarray(got[kind]), np.asarray(same)), kind
    nodes = tree.predict(df, type="node")
    for leaf in (node for node in grow(directory, name).nodes if node.terminal):
        rows = leaf.rows
        np.testing.assert_array_equal(np.flatnonzero(nodes == leaf.id), rows)
        fit = _leaf_refit(f, leaf)
        coef = fit.coefficients
        xr = xm[rows]
        if multinom and fit.extras["status"] == "reference_absent":
            # No coefficients without the reference class: no prediction
            # (docs/api-port.md 3.3, item 5).
            assert np.all(np.isnan(got["link"][rows]))
            assert np.all(np.isnan(got["prob"][rows]))
            assert all(label is None for label in got["class"][rows])
            continue
        if multinom:
            assert isinstance(m, MultinomModel)
            k = len(m.levels)
            beta = coef.reshape(k - 1, xr.shape[1])
            absent = np.all(np.isnan(beta), axis=1)
            eta = np.zeros((rows.shape[0], k))
            eta[:, 1:] = xr @ np.where(np.isnan(beta), 0, beta).T
            expo = np.exp(eta)
            expo[:, 1:][:, absent] = 0.0
            prob = expo / expo.sum(axis=1, keepdims=True)
            link = eta[:, 1:].copy()
            link[:, absent] = np.nan
            np.testing.assert_allclose(
                got["link"][rows], link, rtol=PREDICT_RTOL, atol=0
            )
            np.testing.assert_allclose(
                got["prob"][rows], prob, rtol=PREDICT_RTOL, atol=0
            )
            labels = [m.levels[i] for i in np.argmax(prob, axis=1)]
            assert got["class"][rows].tolist() == labels
            continue
        keep = ~np.isnan(coef)
        eta = xr[:, keep] @ coef[keep] + (0 if off is None else off[rows])
        np.testing.assert_allclose(got["link"][rows], eta, rtol=PREDICT_RTOL, atol=0)
        np.testing.assert_allclose(
            got["response"][rows], _inverse_link(m, eta), rtol=PREDICT_RTOL, atol=0
        )


# The coefficient table against the fixture.


@pytest.mark.parametrize(("directory", "name"), TREES)
def test_coef_table_equals_the_leaf_coefficients(directory: str, name: str) -> None:
    tree = public(directory, name)
    table = tree.coef_table()
    leaves = [n for n in grow(directory, name).nodes if n.terminal]
    assert table["node"].to_list() == [leaf.id for leaf in leaves]
    names = leaves[0].fit.coef_names
    assert table.columns[5 : 5 + len(names)] == list(names)
    for j, coef_name in enumerate(names):
        column = table[coef_name].to_numpy()
        expected = np.array([leaf.fit.coefficients[j] for leaf in leaves])
        np.testing.assert_array_equal(column, expected)
    long = tree.coef_table(layout="long")
    np.testing.assert_array_equal(
        long["estimate"].to_numpy(),
        np.concatenate([leaf.fit.coefficients for leaf in leaves]),
    )


@pytest.mark.parametrize(("directory", "name"), ORACLE_TREES)
def test_coef_table_matches_the_fixture(directory: str, name: str) -> None:
    """Within the tolerances the model tests use for the same coefficients:
    RTOL, loose for completely separated safe logit leaves, and from R's
    convergence offset for multinomial leaves."""
    f = fixture(directory, name)
    tree = public(directory, name)
    table = tree.coef_table(backend="numpy")
    expected_nodes = {n["id"]: n for n in f["nodes"]}
    m = model(f)
    for i, node_id in enumerate(table["node"].tolist()):
        e = expected_nodes[node_id]
        got = np.array([table[c][i] for c in e["coefficient_names"]], dtype=np.float64)
        expected = nan_array(e["coefficients"])
        if isinstance(m, MultinomModel) and e["estfun"] is None:
            # A leaf without the reference class has no coefficients.
            assert np.all(np.isnan(expected)) and np.all(np.isnan(got))
            continue
        if isinstance(m, MultinomModel):
            leaf = grow(directory, name).node(node_id)
            y, x, w, r_estfun = _node_data(name, e)
            tol = _coef_tol(_r_offset(leaf.fit, x, y, w, r_estfun))
            np.testing.assert_allclose(got, expected, rtol=0, atol=tol)
            continue
        separated = table.get("separation") is not None and (
            table["separation"][i] == "complete"
        )
        rtol = SEPARATED_RTOL if separated else RTOL
        np.testing.assert_allclose(got, expected, rtol=rtol, atol=ATOL)


# Printed output, spot checks.


def test_print_glmtree_binomial() -> None:
    text = str(public("trees", "sim_diabetes_like_glmtree_binomial"))
    lines = text.splitlines()
    assert lines[:4] == [
        "Generalized linear model tree (family: binomial)",
        "",
        "Model formula:",
        (
            "diabetes ~ glucose + mass | pregnant + pressure + triceps + insulin + "
            "pedigree + age"
        ),
    ]
    assert (
        "Partitioning variables: pregnant (numeric), pressure (numeric), "
        "triceps (numeric), insulin (numeric), pedigree (numeric), age (numeric)"
    ) in lines
    assert "n = 768" in lines
    assert "|   [2] age <= 35: n = 514" in lines
    assert "|   [3] age > 35" in lines
    assert "|   |   [4] pedigree <= 0.499: n = 166" in lines
    assert lines[-4:] == [
        "Number of inner nodes:    2",
        "Number of terminal nodes: 3",
        "Number of parameters per node: 3",
        "Objective function (negative log-likelihood): 380.2112",
    ]


def test_print_lmtree_with_every_kind() -> None:
    tree = public("trees", "sim_gaussian_lmtree_multisplit")
    text = str(tree)
    assert text.splitlines()[0] == "Linear model tree"
    assert (
        "Partitioning variables: z1 (numeric), z2 (nominal), z3 (ordered), "
        "z4 (numeric), z5 (nominal)"
    ) in text
    assert "|   |   [3] z3 <= L2: n = 93" in text
    assert "|   |   [6] z2 in {a, c}: n = 184" in text
    assert "Objective function (residual sum of squares): " in text
    # Leaf coefficients are printed under the leaf, four significant digits.
    c = tree.node(3).coefficients
    assert f"{c['x']:.4g}" in text


def test_print_safelogit_marks_leaves_without_scores() -> None:
    tree = public("extension_trees", "sim_separation_safelogit")
    text = str(tree)
    assert text.splitlines()[0] == "Safe logistic regression tree"
    no_scores = [line for line in text.splitlines() if "[no scores:" in line]
    assert len(no_scores) == sum(i.status == "no_estfun" for i in tree.nodes) == 2
    assert all("separation: complete" in line for line in no_scores)


def test_print_negbin_shows_theta() -> None:
    tree = public("extension_trees", "quine_negbin_intercept")
    text = str(tree)
    for leaf in tree.leaves:
        theta = tree.node(leaf).extras["theta"]
        assert f"theta = {theta:.4g}" in text
    assert "Eth (nominal)" in text


# Missing data against hand-deleted rows (section 9, item 2).


def test_missing_rows_give_the_tree_of_the_deleted_rows() -> None:
    f = fixture("trees", "sim_gaussian_lmtree_multisplit")
    df = polars_frame(f)
    n = df.height
    holes = {"y_multi": [3, 50], "x": [10, 50], "z2": [77], "z5": [200, 201]}
    nan_rows = [120, 121]  # NaN in a float column counts as missing too
    drop = np.zeros(n, dtype=bool)
    for rows in holes.values():
        drop[rows] = True
    drop[nan_rows] = True
    index = pl.int_range(pl.len())
    with_na = df.with_columns(
        *(
            pl.when(index.is_in(rows)).then(None).otherwise(pl.col(c)).alias(c)
            for c, rows in holes.items()
        ),
        pl.when(index.is_in(nan_rows))
        .then(float("nan"))
        .otherwise(pl.col("z1"))
        .alias("z1"),
    )
    deleted = df.filter(pl.Series(~drop))
    kinds = ordered_kinds(f)
    with pytest.warns(mt.MissingDataWarning) as record:
        a = mt.lmtree(f["formula"], with_na, kinds=kinds, control=control(f))
    assert len(record) == 1
    assert str(record[0].message) == (
        "8 of 600 rows dropped for missing values "
        "(y_multi: 2, x: 2, z1: 2, z2: 1, z5: 2)"
    )
    b = mt.lmtree(f["formula"], deleted, kinds=kinds, control=control(f))
    assert a.n_rows == n and a.n_used == n - 8
    np.testing.assert_array_equal(a.used_rows, ~drop)
    assert len(a._grown.nodes) == len(b._grown.nodes)
    for x_node, y_node in zip(a._grown.nodes, b._grown.nodes, strict=True):
        _assert_same_node(x_node, y_node)
    np.testing.assert_array_equal(a.fitted_node[~drop], b.fitted_node)
    assert np.all(a.fitted_node[drop] == 0)
    pa, pb = a.predict(), b.predict()
    assert np.isnan(pa[drop]).all()
    np.testing.assert_array_equal(pa[~drop], pb)
    # The rows with missing values routed as new data: z2 (row 77) is not on
    # the path of a z1 <= 3.99 row; y alone does not matter for routing.
    nodes = a.predict(with_na, type="node")
    np.testing.assert_array_equal(nodes[~drop], b.fitted_node)
    assert nodes[121] == 0  # z1 missing at the root
    assert a.coef_table().equals(b.coef_table())
    assert str(a).splitlines()[6] == (
        "n = 592 (8 rows dropped for missing values: "
        "y_multi: 2, x: 2, z1: 2, z2: 1, z5: 2)"
    )


def test_too_few_clusters_warning_is_recorded() -> None:
    tree = public("trees", "sim_cluster_few_lmtree_cluster")
    assert mt.TooFewClustersWarning in {w.category for w in tree.warnings}


@pytest.mark.slow
def test_negbin_warnings_are_merged() -> None:
    tree = public("extension_trees", "sim_negbin_weights")
    glm = [w for w in tree.warnings if w.category is mt.GlmFitWarning]
    assert glm
    assert len({w.message for w in glm}) == len(glm)
    total = sum(len(i.extras["warnings"]) for i in tree.nodes)
    assert sum(w.count for w in glm) == total
