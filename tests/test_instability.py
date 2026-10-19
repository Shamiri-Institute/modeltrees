"""Instability test statistics against the per-node values in the tree fixtures.

Each node's scores (``estfun``) come from R, so these tests isolate the
statistic computation from model fitting. The statistic, the p-value
conversion and the full test are checked separately.
"""

import math
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures_io import FIXTURES, load_dataset, load_json, nan_array

from modeltrees.instability import (
    InstabilityTestError,
    PartitioningVariable,
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
    assert len(FAILED_NODES) == 1


@pytest.mark.parametrize(("name", "node_id"), OK_NODES)
def test_statistic(name: str, node_id: int) -> None:
    tree, node = _get(name, node_id)
    test = node["test"]
    estfun, z, weights = _node_inputs(tree, node)
    stats = fluctuation_statistics(
        estfun, z, weights, minsize=tree["root_minsize"], trim=tree["control"]["trim"]
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


@pytest.mark.parametrize(("name", "node_id"), FAILED_NODES)
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
