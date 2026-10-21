"""ModelTree.vcov and coef_table(se="robust") (docs/api-design.md, section 5).

1. The public API: refits, cluster semantics, the refit check, statuses,
   warnings and the standard-error columns.
"""

import warnings
from functools import cache
from typing import Any

import numpy as np
import polars as pl
import pytest
from fixture_frames import polars_frame
from fixtures_io import load_json
from grown_trees import ALL_TREES, fixture, grow, model, tree_inputs
from slow import mark_slow_trees
from test_api_fixtures import public
from tree_fixtures import cluster, variables

import modeltrees as mt
from modeltrees._api import ModelTree
from modeltrees._vcov import sandwich
from modeltrees.models import LmModel, SafeLogitModel
from modeltrees.node_model import FitResult

LM = ("trees", "sim_cluster_lmtree_cluster")
BINOMIAL = ("trees", "sim_cluster_glmtree_binomial_cluster")
UNCLUSTERED = ("trees", "sim_gaussian_lmtree_multisplit")


def _refit(directory: str, name: str, node_id: int) -> tuple[FitResult, np.ndarray]:
    """The node refitted by hand on the fixture arrays, and its weights."""
    f = fixture(directory, name)
    y, x, w, off = tree_inputs(f)
    rows = grow(directory, name).node(node_id).rows
    fit = model(f).fit(
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
    return fit, w[rows]


def _tree(directory: str, name: str) -> ModelTree:
    return public(directory, name)


def test_every_leaf_by_default() -> None:
    tree = _tree(*LM)
    got = tree.vcov()
    assert list(got) == list(tree.leaves)
    for node_id, v in got.items():
        assert isinstance(v, mt.LeafVcov)
        assert v.node == node_id and v.status == "ok" and v.reason is None
        assert v.names == tuple(tree.node(node_id).coefficients)
        assert v.matrix.shape == (len(v.names),) * 2
        assert not v.matrix.flags.writeable
        np.testing.assert_array_equal(v.matrix, v.matrix.T)
        assert v.n_clusters == fixture(*LM)["nodes"][node_id - 1]["n_clusters"]
        assert v.condition >= 1
        assert v.std_errors == {
            n: float(np.sqrt(v.matrix[j, j])) for j, n in enumerate(v.names)
        }


@pytest.mark.parametrize("cadjust", [True, False])
@pytest.mark.parametrize("type", ["HC0", "HC1"])
def test_equals_the_sandwich_of_the_refit(type: str, cadjust: bool) -> None:
    tree = _tree(*LM)
    cl = cluster(fixture(*LM))
    assert cl is not None
    for node_id in tree.leaves:
        fit, w = _refit(*LM, node_id)
        rows = grow(*LM).node(node_id).rows
        for clusters in (cl[rows], None):
            expected = sandwich(
                fit.hessian,
                fit.estfun,
                weights=w,
                cluster=clusters,
                caseweights=True,
                type=type,  # type: ignore[arg-type]
                cadjust=cadjust,
            )
            got = tree.vcov(
                node_id,
                cluster=None if clusters is not None else False,
                type=type,  # type: ignore[arg-type]
                cadjust=cadjust,
            )[node_id]
            np.testing.assert_array_equal(got.matrix, expected.matrix)
            assert got.condition == expected.condition
            assert got.n_clusters == (None if clusters is None else expected.n_clusters)


def test_inner_nodes_and_node_ids() -> None:
    tree = _tree(*BINOMIAL)
    inner = [i.id for i in tree.nodes if not i.is_leaf]
    assert inner
    got = tree.vcov(inner[0])
    assert list(got) == [inner[0]] and got[inner[0]].status == "ok"
    with pytest.raises(KeyError, match="no node"):
        tree.vcov(len(tree.nodes) + 1)
    with pytest.raises(ValueError, match="type"):
        tree.vcov(type="HC3")  # type: ignore[arg-type]


def test_cluster_arguments() -> None:
    tree = _tree(*LM)
    f = fixture(*LM)
    df = polars_frame(f)
    default = tree.vcov()
    for spec in ("school", df["school"].to_numpy(), True):
        same = tree.vcov(cluster=spec)
        for i in tree.leaves:
            np.testing.assert_array_equal(same[i].matrix, default[i].matrix)
    none = tree.vcov(cluster=False)
    assert all(v.n_clusters is None for v in none.values())
    assert not np.array_equal(none[2].matrix, default[2].matrix)
    # A coarser level: pairs of schools.
    coarse = tree.vcov(cluster=df["school"].to_numpy() // 2)
    for i in tree.leaves:
        g_coarse, g = coarse[i].n_clusters, default[i].n_clusters
        assert g_coarse is not None and g is not None and g_coarse < g


def test_cluster_argument_errors() -> None:
    tree = _tree(*LM)
    df = polars_frame(fixture(*LM))
    school = df["school"].to_numpy().astype(np.float64)
    with pytest.raises(mt.DataError, match="not in the data"):
        tree.vcov(cluster="district")
    with pytest.raises(mt.DataError, match="rows"):
        tree.vcov(cluster=school[:-1])
    missing = school.copy()
    missing[3] = np.nan
    with pytest.raises(mt.DataError, match="missing on 1 of the rows"):
        tree.vcov(cluster=missing)
    unclustered = _tree(*UNCLUSTERED)
    with pytest.raises(ValueError, match="without clusters"):
        unclustered.vcov(cluster=True)
    assert all(v.n_clusters is None for v in unclustered.vcov().values())


def test_missing_cluster_on_dropped_rows_is_allowed() -> None:
    # The rule of section 4.1: only the rows the tree used must have one.
    df = polars_frame(fixture(*LM))
    x = df["x"].to_numpy().copy()
    x[:5] = np.nan
    df = df.with_columns(pl.Series("x", x))
    school = df["school"].to_numpy().astype(np.float64)
    school[:5] = np.nan
    f = fixture(*LM)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mt.MissingDataWarning)
        tree = mt.lmtree(f["formula"], df, cluster="school")
    got = tree.vcov(cluster=school)
    ref = tree.vcov()
    for i in tree.leaves:
        np.testing.assert_array_equal(got[i].matrix, ref[i].matrix)


def test_mob_trees() -> None:
    f = fixture(*LM)
    y, x, w, _ = tree_inputs(f)
    cl = cluster(f)
    tree = mt.mob(
        LmModel(coef_names=f["coefficient_names"]),
        y,
        x,
        variables(f),
        weights=w,
        cluster=cl,
    )
    ref = _tree(*LM).vcov()
    got = tree.vcov()
    assert list(got) == list(ref)
    for i in got:
        np.testing.assert_array_equal(got[i].matrix, ref[i].matrix)
    with pytest.raises(TypeError, match="array"):
        tree.vcov(cluster="school")
    assert cl is not None
    np.testing.assert_array_equal(tree.vcov(cluster=cl)[2].matrix, ref[2].matrix)


class _DriftingLm(LmModel):
    """An lm whose refits for vcov move the coefficients slightly."""

    def __init__(self, drift: float) -> None:
        super().__init__()
        self.drift = drift

    def fit(self, *args: Any, **kwargs: Any) -> FitResult:
        r = super().fit(*args, **kwargs)
        if not kwargs.get("vcov"):
            return r
        return FitResult(
            coefficients=r.coefficients * (1 + self.drift),
            coef_names=r.coef_names,
            objfun=r.objfun,
            estfun=r.estfun,
            extras=r.extras,
            hessian=r.hessian,
        )


@pytest.mark.parametrize(("drift", "warns"), [(1e-8, True), (1e-12, False)])
def test_refit_check(drift: float, warns: bool) -> None:
    f = fixture(*UNCLUSTERED)
    y, x, w, _ = tree_inputs(f)
    tree = mt.mob(_DriftingLm(drift), y, x, variables(f), weights=w)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = tree.vcov()
    messages = [str(c.message) for c in caught if c.category is mt.ModelTreesWarning]
    if warns:
        assert len(messages) == len(tree.leaves)
        assert all("refit of node" in m and "relative" in m for m in messages)
        assert any(f"node {tree.leaves[0]} " in m for m in messages)
    else:
        assert messages == []
    assert all(v.status == "ok" for v in got.values())


@pytest.mark.parametrize(("directory", "name"), mark_slow_trees(ALL_TREES, name_at=1))
def test_refits_equal_the_stored_fits(directory: str, name: str) -> None:
    """The refit check of section 5.2 on every node of every fixture tree:
    no ModelTreesWarning other than TooFewClustersWarning."""
    tree = _tree(directory, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for node in tree.nodes:
            got = tree.vcov(node.id)[node.id]
            expected = "ok" if node.status == "ok" else "unavailable"
            assert got.status == expected or got.status == "singular"
    other = [
        c
        for c in caught
        if issubclass(c.category, mt.ModelTreesWarning)
        and c.category is not mt.TooFewClustersWarning
    ]
    assert other == []


def test_unavailable_leaves() -> None:
    # A completely separated safe logit leaf and a multinomial leaf without
    # the reference class have no scores, so no covariance.
    for directory, name, status in (
        ("extension_trees", "sim_separation_safelogit", "no_estfun"),
        ("extension_trees", "sim_multinom_noref_onesplit", "no_estfun"),
    ):
        tree = _tree(directory, name)
        leaves = [i for i in tree.leaves if tree.node(i).status == status]
        assert leaves
        got = tree.vcov()
        for i in leaves:
            v = got[i]
            assert v.status == "unavailable" and np.all(np.isnan(v.matrix))
            assert v.reason is not None and "no_estfun" in v.reason
            assert np.isnan(v.condition)


@pytest.mark.parametrize("layout", ["wide", "long"])
def test_coef_table_robust_standard_errors(layout: str) -> None:
    tree = _tree(*BINOMIAL)
    v = tree.vcov()
    table = tree.coef_table(layout=layout, se="robust", backend="numpy")  # type: ignore[arg-type]
    plain = tree.coef_table(layout=layout, backend="numpy")  # type: ignore[arg-type]
    names = list(tree.node(tree.leaves[0]).coefficients)
    if layout == "long":
        assert list(table) == ["node", "term", "estimate", "std_error"]
        for node_id, term, se in zip(
            table["node"], table["term"], table["std_error"], strict=True
        ):
            assert se == v[int(node_id)].std_errors[term]
    else:
        cols = list(table)
        first = cols.index(names[0])
        assert cols[first : first + 2 * len(names)] == names + [
            f"std_error_{n}" for n in names
        ]
        for i, node_id in enumerate(table["node"]):
            for n in names:
                assert table[f"std_error_{n}"][i] == v[int(node_id)].std_errors[n]
        assert [c for c in cols if not c.startswith("std_error_")] == list(plain)
    for c in plain:
        np.testing.assert_array_equal(np.asarray(table[c]), np.asarray(plain[c]))


def test_coef_table_robust_nan_where_unavailable() -> None:
    tree = _tree("extension_trees", "sim_separation_safelogit")
    table = tree.coef_table(se="robust", backend="numpy")
    names = list(tree.node(1).coefficients)
    for i, node_id in enumerate(table["node"]):
        se = np.array([table[f"std_error_{n}"][i] for n in names])
        ok = tree.node(int(node_id)).status == "ok"
        assert np.all(np.isfinite(se)) if ok else np.all(np.isnan(se))


@cache
def _few() -> ModelTree:
    return _tree("trees", "sim_cluster_few_lmtree_cluster")


def test_too_few_clusters_warns_and_names_the_node() -> None:
    tree = _few()
    with pytest.warns(mt.TooFewClustersWarning, match=r"node 3 has 3 clusters for 4"):
        got = tree.vcov(3)
    assert got[3].status == "ok" and got[3].n_clusters == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error", mt.TooFewClustersWarning)
        tree.vcov(2)
        tree.vcov(3, cluster=False)


@pytest.mark.parametrize("weights", ["unit", "equal", "unequal"])
@pytest.mark.parametrize(("n_clusters", "warns"), [(2, True), (3, True), (4, False)])
def test_too_few_clusters_warns_under_any_weights(
    weights: str, n_clusters: int, warns: bool
) -> None:
    """vcov warns at G <= k under unit, equal and unequal case weights: its
    meat sums the scores, whose cluster sums add up to zero at the estimate
    whatever the weights (the instability tests' guard differs,
    docs/decisions.md)."""
    rng = np.random.default_rng(30 + n_clusters)
    n = 90
    w = {
        "unit": np.ones(n),
        "equal": np.full(n, 2.0),
        "unequal": rng.integers(1, 5, n).astype(np.float64),
    }[weights]
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "z": rng.uniform(size=n),
            "w": w,
            "school": np.arange(n) % n_clusters,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mt.ModelTreesWarning)
        tree = mt.lmtree(
            "y ~ x1 + x2 | z", df, weights="w", cluster="school", maxdepth=1
        )
    if warns:
        with pytest.warns(
            mt.TooFewClustersWarning, match=f"node 1 has {n_clusters} clusters for 3"
        ):
            got = tree.vcov(1)[1]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", mt.TooFewClustersWarning)
            got = tree.vcov(1)[1]
    assert got.status == "ok" and got.n_clusters == n_clusters


def test_quasi_separation_warns_and_names_the_coefficient() -> None:
    # node_fits/safelogit_quasi as a one-node tree: every d4 = 1 row has
    # y = 0, so the d4 coefficient diverges. The scaled Hessian is well
    # conditioned, so the status stays "ok" (docs/api-port.md, section 15).
    f = load_json("node_fits/safelogit_quasi.json")
    y = np.array(f["y"], np.float64)
    tree = mt.mob(
        SafeLogitModel(coef_names=f["x_names"], **f["fit_args"]),
        y,
        np.array(f["x"], np.float64),
        {"z": ("numeric", np.zeros_like(y))},
        weights=np.array(f["weights"], np.float64),
    )
    assert tree.leaves == (1,) and tree.node(1).extras["separation"] == "quasi"
    expected = "node 1 is quasi-separated: coefficient 'd4' diverges"
    with pytest.warns(mt.ModelTreesWarning, match=expected) as caught:
        got = tree.vcov()[1]
    assert len(caught) == 1
    assert got.status == "ok" and got.reason == str(caught[0].message)
    np.testing.assert_allclose(tree.node(1).coefficients["d4"], -18.567, rtol=1e-4)


def test_no_quasi_warning_without_separation() -> None:
    f = load_json("node_fits/safelogit_none.json")
    y = np.array(f["y"], np.float64)
    tree = mt.mob(
        SafeLogitModel(coef_names=f["x_names"], **f["fit_args"]),
        y,
        np.array(f["x"], np.float64),
        {"z": ("numeric", np.zeros_like(y))},
        weights=np.array(f["weights"], np.float64),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", mt.ModelTreesWarning)
        got = tree.vcov()[1]
    assert got.status == "ok" and got.reason is None
