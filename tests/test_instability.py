"""Instability test statistics against the per-node values in the tree fixtures.

Each node's scores (``estfun``) come from R, so these tests isolate the
statistic computation from model fitting. The statistic, the p-value
conversion and the full test are checked separately.
"""

import math
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures_io import FIXTURES, load_dataset, load_json, nan_array

from modeltrees.instability import (
    InstabilityTestError,
    PartitioningVariable,
    TooFewClustersError,
    TooFewClustersWarning,
    VariableKind,
    best_variable,
    fluctuation_statistics,
    fluctuation_tests,
    root_matrix,
    statistics_to_log_pvalues,
)
from modeltrees.pvalues import logp_chisq, logp_suplm

RTOL = 1e-9
ATOL = 1e-12

TREE_FILES = sorted(p.name for p in (FIXTURES / "trees").glob("*.json"))
KINDS = {
    "double": VariableKind.NUMERIC,
    "integer": VariableKind.NUMERIC,
    "factor": VariableKind.NOMINAL,
    "ordered": VariableKind.ORDERED,
}


@cache
def _tree(name: str) -> dict[str, Any]:
    tree: dict[str, Any] = load_json(f"trees/{name}")
    return tree


@cache
def _dataset(csv_relative: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_dataset(csv_relative)


def _node_inputs(
    tree: dict[str, Any], node: dict[str, Any]
) -> tuple[np.ndarray, list[PartitioningVariable], np.ndarray]:
    data, columns = _dataset(tree["data_csv"])
    rows = np.asarray(node["rows"], dtype=np.intp)
    z = [
        PartitioningVariable(name, KINDS[columns[name]["type"]], data[name][rows])
        for name in tree["partitioning"]
    ]
    weights = data[tree["weights"]][rows] if tree["weights"] else np.ones(rows.shape[0])
    estfun = np.asarray(node["estfun"], dtype=np.float64)
    return estfun, z, weights


def _node_cluster(tree: dict[str, Any], node: dict[str, Any]) -> np.ndarray | None:
    """The node's cluster ids, from the cluster column of the data CSV."""
    if "cluster" not in tree:
        return None
    data, _ = _dataset(tree["data_csv"])
    rows = np.asarray(node["rows"], dtype=np.intp)
    return data[tree["cluster"]][rows].astype(np.intp)


def _tested_nodes(status: str) -> list[Any]:
    cases = []
    for name in TREE_FILES:
        tree = _tree(name)
        assert tree["control"]["caseweights"] is True
        assert tree["control"]["vcov"] == "opg"
        assert tree["control"]["ordinal"] == "chisq"
        for node in tree["nodes"]:
            if node["test"]["status"] == status:
                cases.append(
                    pytest.param(
                        name, node["id"], id=f"{Path(name).stem}-node{node['id']}"
                    )
                )
    return cases


OK_NODES = _tested_nodes("ok")
FAILED_NODES = _tested_nodes("failed")
# Failed nodes split by cause: zero case weights, or fewer clusters than
# parameters.
ZERO_WEIGHT_FAILED = [c for c in FAILED_NODES if "cluster" not in _tree(c.values[0])]
FEW_CLUSTER_FAILED = [c for c in FAILED_NODES if "cluster" in _tree(c.values[0])]
CLUSTER_TREES = sorted(n for n in TREE_FILES if "cluster" in _tree(n))


def _get(name: str, node_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    tree = _tree(name)
    node = next(n for n in tree["nodes"] if n["id"] == node_id)
    return tree, node


def test_fixtures_cover_every_variable_kind() -> None:
    kinds = set()
    for name in TREE_FILES:
        tree = _tree(name)
        _, columns = _dataset(tree["data_csv"])
        kinds |= {columns[v]["type"] for v in tree["partitioning"]}
    assert {"double", "factor", "ordered"} <= kinds
    assert len(OK_NODES) > 50
    assert len(ZERO_WEIGHT_FAILED) == 1
    assert len(FEW_CLUSTER_FAILED) == 1


@pytest.mark.parametrize(("name", "node_id"), OK_NODES)
def test_statistic(name: str, node_id: int) -> None:
    tree, node = _get(name, node_id)
    test = node["test"]
    estfun, z, weights = _node_inputs(tree, node)
    stats = fluctuation_statistics(
        estfun,
        z,
        weights,
        minsize=tree["root_minsize"],
        trim=tree["control"]["trim"],
        cluster=_node_cluster(tree, node),
    )
    assert stats.nobs == test["nobs"]
    assert stats.k == test["k"]
    assert stats.trimming.start == test["trim_from"]
    assert stats.trimming.end == test["trim_to"]
    assert stats.trimming.lam == pytest.approx(test["lambda"], rel=1e-15)
    # The fixture's `tested` is mtest membership (all variables when mtry = Inf).
    assert all(test["tested"])
    expected_computed = ~np.isnan(nan_array(test["log_p_value_raw"]))
    np.testing.assert_array_equal(stats.computed, expected_computed)
    np.testing.assert_allclose(
        stats.statistic, np.asarray(test["statistic"]), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(("name", "node_id"), OK_NODES)
def test_pvalue_conversion_from_r_statistic(name: str, node_id: int) -> None:
    """R's statistic in, R's raw log p-value out, without our statistic."""
    tree, node = _get(name, node_id)
    test = node["test"]
    _, columns = _dataset(tree["data_csv"])
    data, _ = _dataset(tree["data_csv"])
    rows = np.asarray(node["rows"], dtype=np.intp)
    expected = nan_array(test["log_p_value_raw"])
    for i, var in enumerate(tree["partitioning"]):
        if math.isnan(expected[i]):
            assert test["statistic"][i] == 0
            continue
        stat = test["statistic"][i]
        if KINDS[columns[var]["type"]] is VariableKind.NUMERIC:
            got = logp_suplm(stat, test["k"], test["lambda"])
        else:
            values = data[var][rows]
            w = (
                data[tree["weights"]][rows]
                if tree["weights"]
                else np.ones(rows.shape[0])
            )
            n_levels = np.unique(values[w > 0]).shape[0]
            got = logp_chisq(stat, test["k"] * (n_levels - 1))
        assert got == pytest.approx(expected[i], rel=RTOL, abs=ATOL), var


@pytest.mark.parametrize(("name", "node_id"), OK_NODES)
def test_full_tests(name: str, node_id: int) -> None:
    tree, node = _get(name, node_id)
    test = node["test"]
    control = tree["control"]
    estfun, z, weights = _node_inputs(tree, node)
    result = fluctuation_tests(
        estfun,
        z,
        weights,
        minsize=tree["root_minsize"],
        trim=control["trim"],
        bonferroni=control["bonferroni"],
        cluster=_node_cluster(tree, node),
    )
    assert result.variables == tuple(test["variables"])
    np.testing.assert_allclose(
        result.log_p_value,
        nan_array(test["log_p_value_raw"]),
        rtol=RTOL,
        atol=ATOL,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.p_value,
        nan_array(test["p_value_raw"]),
        rtol=1e-8,
        atol=0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.p_value_adjusted,
        nan_array(test["p_value_adjusted"]),
        rtol=1e-8,
        atol=0,
        equal_nan=True,
    )
    assert result.best == test["best"]
    if node["p_value"] is not None:
        assert result.best is not None
        assert result.p_value_adjusted[result.best] == pytest.approx(
            node["p_value"], rel=1e-8
        )


def _partykit_zero_weight_meat(estfun: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """partykit 1.3-0's meat with zero weights (modelparty.R:220-246).

    The process drops zero-weight rows but is divided by the unfiltered
    weights. R recycles the shorter vector over the column-major matrix, so
    every weight is used and a zero weight puts Inf or NaN into the meat.
    """
    process = estfun[weights > 0] / math.sqrt(weights.sum())
    flat = process.reshape(-1, order="F")
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = (flat / np.resize(np.sqrt(weights), flat.shape[0])).reshape(
            process.shape, order="F"
        )
        return scaled.T @ scaled


@pytest.mark.parametrize(("name", "node_id"), ZERO_WEIGHT_FAILED)
def test_partykit_zero_weight_failure_is_a_documented_divergence(
    name: str, node_id: int
) -> None:
    """Fixture 5(a): partykit 1.3-0 fails the root test with zero weights and
    the node becomes terminal. The port does not replicate that (see
    docs/decisions.md and docs/instability-tests.md, section 4, item 6): the
    zero-weight rows are dropped consistently and the tests run."""
    tree, node = _get(name, node_id)
    assert tree["summary"]["n_terminal"] == 1
    assert node["stop_reason"] == "test_failed"
    estfun, z, weights = _node_inputs(tree, node)
    assert np.any(weights == 0)

    # R's arithmetic: the meat is not finite, so chol() fails.
    meat = _partykit_zero_weight_meat(estfun, weights)
    assert not np.all(np.isfinite(meat))

    # The port: no error, finite statistics.
    result = fluctuation_tests(
        estfun,
        z,
        weights,
        minsize=tree["root_minsize"],
        trim=tree["control"]["trim"],
    )
    assert np.all(np.isfinite(result.statistic))
    assert result.best is not None


def test_zero_weights_test_like_deleted_rows() -> None:
    """Fixture 5(a)'s root node, run with zero weights, against the R oracle
    5(b): the same data with the zero-weight rows removed before fitting."""
    full = _tree("sim_gaussian_lmtree_zero_weights.json")
    oracle = _tree("sim_gaussian_lmtree_zero_weights_deleted.json")
    assert full["control"] == oracle["control"]
    assert full["partitioning"] == oracle["partitioning"]
    assert full["weights"] == oracle["weights"] == "w0"
    removed = np.asarray(oracle["removed_rows"], dtype=np.intp)
    assert removed.shape[0] > 0

    data, _ = _dataset(full["data_csv"])
    np.testing.assert_array_equal(np.flatnonzero(data["w0"] == 0), removed)
    assert oracle["n"] == full["n"] - removed.shape[0]

    root_full = full["nodes"][0]
    root_oracle = oracle["nodes"][0]
    assert root_oracle["test"]["status"] == "ok"
    estfun, z, weights = _node_inputs(full, root_full)
    result = fluctuation_tests(
        estfun,
        z,
        weights,
        minsize=full["root_minsize"],
        trim=full["control"]["trim"],
        bonferroni=full["control"]["bonferroni"],
    )
    test = root_oracle["test"]
    assert result.nobs == test["nobs"]
    assert result.k == test["k"]
    assert result.trimming.start == test["trim_from"]
    assert result.trimming.end == test["trim_to"]
    np.testing.assert_allclose(
        result.statistic, np.asarray(test["statistic"]), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        result.log_p_value,
        nan_array(test["log_p_value_raw"]),
        rtol=RTOL,
        atol=ATOL,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.p_value_adjusted,
        nan_array(test["p_value_adjusted"]),
        rtol=1e-8,
        atol=0,
        equal_nan=True,
    )
    assert result.best == test["best"]


def test_zero_weight_rows_do_not_enter_the_tests() -> None:
    """Values of z, and of the scores, in zero-weight rows change nothing."""
    tree = _tree("sim_gaussian_lmtree_zero_weights.json")
    estfun, z, weights = _node_inputs(tree, tree["nodes"][0])
    zero = weights == 0
    assert zero.any()
    expected = fluctuation_tests(estfun, z, weights, minsize=tree["root_minsize"])

    rng = np.random.default_rng(0)
    estfun2 = estfun.copy()
    estfun2[zero] = rng.normal(size=(int(zero.sum()), estfun.shape[1]))
    z2 = []
    for v in z:
        values = np.array(v.values, copy=True)
        values[zero] = rng.permutation(values)[: int(zero.sum())]
        z2.append(PartitioningVariable(v.name, v.kind, values))
    got = fluctuation_tests(estfun2, z2, weights, minsize=tree["root_minsize"])
    np.testing.assert_array_equal(got.statistic, expected.statistic)
    np.testing.assert_array_equal(got.log_p_value, expected.log_p_value)
    assert got.best == expected.best


def _random_scores(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    x = rng.standard_normal((n, k))
    return x - x.mean(axis=0)


def test_one_parameter_uses_scalar_root() -> None:
    assert root_matrix(np.array([[4.0]])) == pytest.approx(np.array([[2.0]]))
    rng = np.random.default_rng(1)
    n = 200
    estfun = _random_scores(rng, n, 1)
    z = [PartitioningVariable("z", VariableKind.NUMERIC, rng.uniform(size=n))]
    stats = fluctuation_statistics(estfun, z, np.ones(n), minsize=10)
    assert stats.k == 1
    # With one parameter the decorrelated process is e / sqrt(sum(e^2)), so
    # the statistic is a ratio of partial sums of squares.
    e = estfun[np.argsort(z[0].values, kind="stable"), 0]
    cum = np.cumsum(e) / math.sqrt(np.sum(e**2))
    t = np.arange(1, n + 1) / n
    sel = slice(int(stats.trimming.start) - 1, int(stats.trimming.end))
    expected = np.max(cum[sel] ** 2 / (t[sel] * (1 - t[sel])))
    assert stats.statistic[0] == pytest.approx(expected, rel=1e-12)


def test_one_parameter_factor() -> None:
    rng = np.random.default_rng(2)
    n = 300
    estfun = _random_scores(rng, n, 1)
    codes = rng.integers(0, 3, size=n)
    z = [PartitioningVariable("f", VariableKind.NOMINAL, codes)]
    result = fluctuation_tests(estfun, z, np.ones(n), minsize=10)
    e = estfun[:, 0] / math.sqrt(np.sum(estfun[:, 0] ** 2))
    expected = sum(np.sum(e[codes == j]) ** 2 / np.mean(codes == j) for j in range(3))
    assert result.statistic[0] == pytest.approx(expected, rel=1e-12)
    assert result.log_p_value[0] == pytest.approx(logp_chisq(expected, 2), rel=1e-12)


@pytest.mark.parametrize("k", [39, 40, 41, 45])
def test_many_parameters(k: int) -> None:
    rng = np.random.default_rng(k)
    n = 2000
    estfun = _random_scores(rng, n, k)
    z = [
        PartitioningVariable("x", VariableKind.NUMERIC, rng.uniform(size=n)),
        PartitioningVariable("f", VariableKind.NOMINAL, rng.integers(0, 4, size=n)),
    ]
    result = fluctuation_tests(estfun, z, np.ones(n), minsize=10)
    assert result.k == k
    assert np.all(np.isfinite(result.log_p_value))
    assert result.log_p_value[0] == logp_suplm(
        result.statistic[0], k, result.trimming.lam
    )
    assert result.log_p_value[1] == logp_chisq(result.statistic[1], 3 * k)
    # The decorrelated process has identity cross product, so the factor
    # statistic sums k independent chi-squared(3) contributions.
    assert 0.3 * 3 * k < result.statistic[1] < 3 * 3 * k


def test_constant_variable_is_not_tested() -> None:
    rng = np.random.default_rng(3)
    n = 100
    estfun = _random_scores(rng, n, 2)
    z = [
        PartitioningVariable("c", VariableKind.NUMERIC, np.full(n, 5.0)),
        PartitioningVariable("f", VariableKind.NOMINAL, np.zeros(n, dtype=np.intp)),
        PartitioningVariable("x", VariableKind.NUMERIC, rng.uniform(size=n)),
    ]
    result = fluctuation_tests(estfun, z, np.ones(n), minsize=5)
    assert list(result.statistic[:2]) == [0.0, 0.0]
    assert np.all(np.isnan(result.log_p_value[:2]))
    assert np.isnan(result.p_value_adjusted[0])
    assert result.best == 2
    # Only one non-NaN p-value, so the adjustment is the identity up to rounding.
    assert result.p_value_adjusted[2] == pytest.approx(result.p_value[2], rel=1e-12)


def test_no_admissible_trimming_window() -> None:
    # minsize above nobs / 2 leaves from > to: no supLM statistic, as in R.
    rng = np.random.default_rng(4)
    n = 50
    estfun = _random_scores(rng, n, 2)
    z = [PartitioningVariable("x", VariableKind.NUMERIC, rng.uniform(size=n))]
    result = fluctuation_tests(estfun, z, np.ones(n), minsize=30)
    assert result.statistic[0] == 0.0
    assert np.isnan(result.log_p_value[0])
    assert result.best is None


def test_singular_scores_raise() -> None:
    rng = np.random.default_rng(5)
    n = 50
    col = rng.standard_normal(n)
    estfun = np.column_stack([col, 2 * col])
    z = [PartitioningVariable("x", VariableKind.NUMERIC, rng.uniform(size=n))]
    with pytest.raises(InstabilityTestError):
        fluctuation_tests(estfun, z, np.ones(n), minsize=5)


def test_best_variable_ties_go_first() -> None:
    assert best_variable(np.array([np.nan, -2.0, -2.0])) == 1
    assert best_variable(np.array([np.nan, np.nan])) is None


def test_statistics_to_log_pvalues_matches_full() -> None:
    tree, node = _get(*OK_NODES[0].values)
    estfun, z, weights = _node_inputs(tree, node)
    stats = fluctuation_statistics(estfun, z, weights, minsize=tree["root_minsize"])
    logp = statistics_to_log_pvalues(stats, [v.kind for v in z])
    full = fluctuation_tests(estfun, z, weights, minsize=tree["root_minsize"])
    np.testing.assert_array_equal(logp, full.log_p_value)


# ---------------------------------------------------------------------------
# Clusters (modelparty.R:240-245)


def _cluster_root(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    tree = _tree(name)
    return tree, tree["nodes"][0]


def test_cluster_fixtures_record_cluster_counts() -> None:
    assert [n.removesuffix(".json") for n in CLUSTER_TREES] == [
        "sim_cluster_few_lmtree_cluster",
        "sim_cluster_glmtree_binomial_cluster",
        "sim_cluster_lmtree_cluster",
    ]
    for name in CLUSTER_TREES:
        tree = _tree(name)
        assert tree["weights"] is None
        for node in tree["nodes"]:
            ids = _node_cluster(tree, node)
            assert ids is not None
            assert np.unique(ids).shape[0] == node["n_clusters"]


@pytest.mark.parametrize(
    ("clustered", "plain"),
    [
        ("sim_cluster_lmtree_cluster.json", "sim_cluster_lmtree_nocluster.json"),
        (
            "sim_cluster_glmtree_binomial_cluster.json",
            "sim_cluster_glmtree_binomial_nocluster.json",
        ),
    ],
)
def test_cluster_changes_the_p_values(clustered: str, plain: str) -> None:
    """Same data and model, with and without cluster. R's root p-values for the
    school-level variables zs and zt are far apart, and so are ours, computed
    from the same scores both ways."""
    tree_c, root_c = _cluster_root(clustered)
    tree_p, root_p = _cluster_root(plain)
    assert tree_c["data_csv"] == tree_p["data_csv"]
    school_level = [tree_c["partitioning"].index(v) for v in ("zs", "zt")]
    r_cluster = nan_array(root_c["test"]["p_value_raw"])[school_level]
    r_plain = nan_array(root_p["test"]["p_value_raw"])[school_level]
    assert np.all(r_cluster > 0.05)
    assert np.all(r_plain < 1e-3)

    estfun, z, weights = _node_inputs(tree_c, root_c)
    kw = {"minsize": tree_c["root_minsize"]}
    with_cluster = fluctuation_tests(
        estfun, z, weights, cluster=_node_cluster(tree_c, root_c), **kw
    )
    without = fluctuation_tests(estfun, z, weights, **kw)
    # Each is R's answer for its own call (the scores are the same in both
    # fixtures only when the fits are, which they are: cluster does not change
    # the fit).
    assert np.all(with_cluster.p_value[school_level] > 0.05)
    assert np.all(without.p_value[school_level] < 1e-3)
    np.testing.assert_allclose(
        without.p_value, nan_array(root_p["test"]["p_value_raw"]), rtol=1e-8
    )


def test_cluster_does_not_change_the_fit() -> None:
    """The scores at the root are the same with and without cluster, which is
    why the same estfun can be tested both ways above."""
    _, root_c = _cluster_root("sim_cluster_lmtree_cluster.json")
    _, root_p = _cluster_root("sim_cluster_lmtree_nocluster.json")
    np.testing.assert_array_equal(root_c["estfun"], root_p["estfun"])
    assert root_c["coefficients"] == root_p["coefficients"]


def test_singleton_clusters_equal_no_clusters() -> None:
    """One row per cluster: the within-cluster sums are the rows themselves."""
    tree, root = _cluster_root("sim_cluster_lmtree_nocluster.json")
    estfun, z, weights = _node_inputs(tree, root)
    kw = {"minsize": tree["root_minsize"]}
    plain = fluctuation_tests(estfun, z, weights, **kw)
    single = fluctuation_tests(
        estfun, z, weights, cluster=np.arange(weights.shape[0]), **kw
    )
    np.testing.assert_allclose(single.statistic, plain.statistic, rtol=1e-12)
    np.testing.assert_allclose(
        single.log_p_value, plain.log_p_value, rtol=1e-12, equal_nan=True
    )


def test_cluster_labels_and_row_order_do_not_matter() -> None:
    tree, root = _cluster_root("sim_cluster_lmtree_cluster.json")
    estfun, z, weights = _node_inputs(tree, root)
    ids = _node_cluster(tree, root)
    assert ids is not None
    kw = {"minsize": tree["root_minsize"]}
    expected = fluctuation_tests(estfun, z, weights, cluster=ids, **kw)
    labels = np.array([f"school-{int(i) * 7 + 3}" for i in ids])
    got = fluctuation_tests(estfun, z, weights, cluster=labels, **kw)
    np.testing.assert_allclose(got.statistic, expected.statistic, rtol=1e-12)
    np.testing.assert_allclose(
        got.log_p_value, expected.log_p_value, rtol=1e-12, equal_nan=True
    )


@pytest.mark.parametrize(("name", "node_id"), FEW_CLUSTER_FAILED)
def test_fewer_clusters_than_parameters_stops_the_node(name: str, node_id: int) -> None:
    """R's chol() fails on the cluster-summed meat and try() makes the node
    terminal. The port warns and raises, which the tree code turns into a
    terminal node (docs/decisions.md)."""
    tree, node = _get(name, node_id)
    assert node["stop_reason"] == "test_failed"
    assert node["n_clusters"] < len(node["coefficient_names"])
    estfun, z, weights = _node_inputs(tree, node)
    kw = {"minsize": tree["root_minsize"], "cluster": _node_cluster(tree, node)}
    message = f"{node['n_clusters']} clusters for {estfun.shape[1]} parameters"
    with (
        pytest.warns(TooFewClustersWarning, match=message),
        pytest.raises(TooFewClustersError),
    ):
        fluctuation_tests(estfun, z, weights, **kw)
    # It is an InstabilityTestError, the class tree code already catches.
    with pytest.warns(TooFewClustersWarning), pytest.raises(InstabilityTestError):
        fluctuation_statistics(estfun, z, weights, **kw)
    # Without the cluster vector the same scores test fine: the failure comes
    # from the cluster sums, not from the scores.
    assert (
        fluctuation_tests(estfun, z, weights, minsize=tree["root_minsize"]).best
        is not None
    )


def test_sibling_with_enough_clusters_is_tested() -> None:
    """In the same fixture the left child has 21 clusters for 4 parameters."""
    name = FEW_CLUSTER_FAILED[0].values[0]
    tree = _tree(name)
    sibling = next(n for n in tree["nodes"] if n["id"] == 2)
    assert sibling["test"]["status"] == "ok"
    assert sibling["n_clusters"] == 21


@pytest.mark.parametrize("n_clusters", [1, 2, 3])
def test_fewer_clusters_than_parameters_raises(n_clusters: int) -> None:
    rng = np.random.default_rng(10 + n_clusters)
    n, k = 120, 4
    estfun = _random_scores(rng, n, k)
    z = [PartitioningVariable("z", VariableKind.NUMERIC, rng.uniform(size=n))]
    cluster = np.arange(n) % n_clusters
    with (
        pytest.warns(TooFewClustersWarning, match=f"{n_clusters} clusters for 4"),
        pytest.raises(TooFewClustersError),
    ):
        fluctuation_tests(estfun, z, np.ones(n), minsize=10, cluster=cluster)


def test_cluster_count_at_and_above_the_number_of_parameters() -> None:
    """Scores that sum to zero (as fitted scores do) make the cluster sums sum
    to zero, so with exactly k clusters the meat has rank k - 1. Whether chol
    fails there depends on rounding, in R as in the port (see
    docs/instability-tests.md, section 4, item 14), so only the absence of the
    few-clusters rule is asserted. With more clusters the tests run."""
    rng = np.random.default_rng(14)
    n, k = 120, 4
    estfun = _random_scores(rng, n, k)
    z = [PartitioningVariable("z", VariableKind.NUMERIC, rng.uniform(size=n))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            fluctuation_tests(
                estfun, z, np.ones(n), minsize=10, cluster=np.arange(n) % k
            )
        except TooFewClustersError:  # pragma: no cover
            pytest.fail("k clusters for k parameters is not the few-clusters case")
        except InstabilityTestError:
            pass
        result = fluctuation_tests(
            estfun, z, np.ones(n), minsize=10, cluster=np.arange(n) % (k + 2)
        )
    assert np.all(np.isfinite(result.statistic))


def test_one_cluster_one_parameter_runs() -> None:
    """R: one cluster and one parameter gives a 1 x 1 meat, which is fine."""
    rng = np.random.default_rng(11)
    n = 60
    estfun = _random_scores(rng, n, 1)
    z = [PartitioningVariable("z", VariableKind.NUMERIC, rng.uniform(size=n))]
    result = fluctuation_tests(estfun, z, np.ones(n), minsize=5, cluster=np.zeros(n))
    assert np.all(np.isfinite(result.statistic))


def _cluster_fixture_root_with_zero_weight_rows() -> tuple[
    dict[str, Any],
    dict[str, Any],
    np.ndarray,
    list[PartitioningVariable],
    np.ndarray,
    np.ndarray,
]:
    tree, root = _cluster_root("sim_cluster_lmtree_cluster.json")
    estfun, z, weights = _node_inputs(tree, root)
    cluster = _node_cluster(tree, root)
    assert cluster is not None
    return tree, root, estfun, z, weights, cluster


def test_zero_weight_rows_are_dropped_from_the_cluster_vector_too() -> None:
    """Extra rows with weight 0, one in an existing school and some in a school
    that has no other rows, change nothing: the result still equals R's on the
    original data, so the cluster vector is filtered with the scores."""
    tree, root, estfun, z, weights, cluster = (
        _cluster_fixture_root_with_zero_weight_rows()
    )
    rng = np.random.default_rng(12)
    extra = 25
    new_ids = np.concatenate(
        [rng.choice(np.unique(cluster), size=extra - 5), np.full(5, 999)]
    )
    pos = np.sort(rng.integers(0, weights.shape[0] + 1, size=extra))
    estfun2 = np.insert(
        estfun, pos, rng.normal(size=(extra, estfun.shape[1])) * 50, axis=0
    )
    cluster2 = np.insert(cluster, pos, new_ids)
    weights2 = np.insert(weights, pos, 0.0)
    z2 = [
        PartitioningVariable(v.name, v.kind, np.insert(v.values, pos, v.values[:extra]))
        for v in z
    ]
    kw = {"minsize": tree["root_minsize"], "trim": tree["control"]["trim"]}
    got = fluctuation_tests(estfun2, z2, weights2, cluster=cluster2, **kw)
    test = root["test"]
    assert got.nobs == test["nobs"]
    np.testing.assert_allclose(
        got.statistic, np.asarray(test["statistic"]), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        got.log_p_value,
        nan_array(test["log_p_value_raw"]),
        rtol=RTOL,
        atol=ATOL,
        equal_nan=True,
    )


def test_zero_weight_rows_equal_deleted_rows_with_clusters() -> None:
    """Zero weights on every row of three schools and on some rows of the rest
    test like the data with those rows deleted, including the cluster count."""
    tree, _, estfun, z, _, cluster = _cluster_fixture_root_with_zero_weight_rows()
    rng = np.random.default_rng(13)
    n = estfun.shape[0]
    zero = np.isin(cluster, np.unique(cluster)[:3]) | (rng.uniform(size=n) < 0.1)
    weights = np.where(zero, 0.0, 1.0)
    kw = {"minsize": tree["root_minsize"]}
    got = fluctuation_tests(estfun, z, weights, cluster=cluster, **kw)
    keep = ~zero
    z_kept = [PartitioningVariable(v.name, v.kind, v.values[keep]) for v in z]
    expected = fluctuation_tests(
        estfun[keep], z_kept, np.ones(int(keep.sum())), cluster=cluster[keep], **kw
    )
    assert got.nobs == expected.nobs == keep.sum()
    np.testing.assert_array_equal(got.statistic, expected.statistic)
    np.testing.assert_array_equal(got.log_p_value, expected.log_p_value)


def test_clusters_with_only_zero_weights_do_not_count() -> None:
    rng = np.random.default_rng(14)
    n, k = 100, 4
    estfun = _random_scores(rng, n, k)
    z = [PartitioningVariable("z", VariableKind.NUMERIC, rng.uniform(size=n))]
    cluster = np.arange(n) % 5  # five clusters, enough for four parameters
    ok = fluctuation_tests(estfun, z, np.ones(n), minsize=5, cluster=cluster)
    assert ok.best is not None
    weights = np.where(cluster < 2, 0.0, 1.0)  # two clusters left with weight 0
    with (
        pytest.warns(TooFewClustersWarning, match="3 clusters for 4 parameters"),
        pytest.raises(TooFewClustersError),
    ):
        fluctuation_tests(estfun, z, weights, minsize=5, cluster=cluster)


def test_cluster_input_validation() -> None:
    rng = np.random.default_rng(15)
    n = 40
    estfun = _random_scores(rng, n, 2)
    z = [PartitioningVariable("z", VariableKind.NUMERIC, rng.uniform(size=n))]
    with pytest.raises(ValueError, match="one id per row"):
        fluctuation_tests(estfun, z, np.ones(n), minsize=5, cluster=np.arange(n - 1))
    bad = np.arange(n, dtype=np.float64)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="missing"):
        fluctuation_tests(estfun, z, np.ones(n), minsize=5, cluster=bad)
