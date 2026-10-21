"""The public ModelTree's node views (docs/api-design.md, sections 3.1 and
3.2), built around the core trees of the fixtures."""

import math
from typing import Any

import numpy as np
import pytest
from grown_trees import ALL_TREES, fixture, grow, model
from slow import mark_slow_trees

from modeltrees._api import ModelTree, NodeInfo, SplitRule


def _wrap(directory: str, name: str) -> ModelTree:
    f = fixture(directory, name)
    g = grow(directory, name)
    return ModelTree(
        g,
        model(f),
        model_name="test",
        formula=None,
        design=None,
        response_levels=None,
        warnings=(),
        used_rows=np.ones(g.fitted_node.shape[0], dtype=bool),
        x=None,
        offset=None,
    )


def _expected_nodes(f: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(f["nodes"], key=lambda n: n["id"])


@pytest.mark.parametrize(("directory", "name"), mark_slow_trees(ALL_TREES, name_at=1))
def test_structure_and_splits_match_the_fixture(directory: str, name: str) -> None:
    if name == "sim_gaussian_lmtree_zero_weights":
        pytest.skip("not an R oracle (docs/decisions.md)")
    f = fixture(directory, name)
    tree = _wrap(directory, name)
    expected = _expected_nodes(f)
    assert [n.id for n in tree.nodes] == [n["id"] for n in expected]
    for info, e in zip(tree.nodes, expected, strict=True):
        assert info.parent == e["parent"]
        assert list(info.kids) == list(e["kids"] or [])
        assert info.depth == e["depth"]
        assert info.is_leaf == (not e["kids"])
        assert info.n_rows == len(e["rows"])
        assert info.stop_reason == e["stop_reason"]
        split = e.get("split")
        if split is None:
            assert info.split is None
            continue
        rule = info.split
        assert rule is not None and rule.variable == split["variable"]
        assert rule.kind == split["kind"]
        if split["kind"] == "numeric":
            assert rule.threshold == split["breaks"]
            assert rule.left_levels is None
        else:
            assert rule.left_levels == tuple(split["left_levels"])
            assert rule.right_levels == tuple(split["right_levels"])
            if split["kind"] == "ordered":
                assert rule.threshold == split["break_level"]
            else:
                assert rule.threshold is None
                assert rule.absent_levels == tuple(split["absent_levels"])
    assert tree.leaves == tuple(n["id"] for n in expected if not n["kids"])
    assert tree.depth == max(n["depth"] for n in expected)


@pytest.mark.parametrize(("directory", "name"), mark_slow_trees(ALL_TREES, name_at=1))
def test_node_views_carry_the_core_values(directory: str, name: str) -> None:
    tree = _wrap(directory, name)
    for info, node in zip(tree.nodes, grow(directory, name).nodes, strict=True):
        assert info.core is node
        assert info.nobs == node.nobs
        assert info.objective == node.fit.objfun
        assert list(info.coefficients) == list(node.fit.coef_names)
        assert np.array_equal(
            list(info.coefficients.values()), node.fit.coefficients, equal_nan=True
        )
        assert info.status == node.fit.status.value
        assert info.p_value == node.p_value
        if node.test is None:
            assert info.tests is None
        else:
            assert info.tests is not None
            assert list(info.tests) == list(node.test.variables)
            for (stat, p), r in zip(
                zip(node.test.statistic, node.test.p_value_adjusted, strict=True),
                info.tests.values(),
                strict=True,
            ):
                assert r.statistic == stat
                assert (r.p_value is None) == math.isnan(p)
                assert r.p_value is None or r.p_value == p


def test_rules_are_readable_paths() -> None:
    tree = _wrap("trees", "sim_gaussian_lmtree_multisplit")
    # R prints the break as round(3.9957765955477953, 5).
    assert tree.node(3).rule == ("z1 <= 3.99578", "z3 <= L2")
    assert tree.node(4).rule == ("z1 <= 3.99578", "z3 > L2")
    assert tree.node(6).rule == ("z1 > 3.99578", "z2 in {a, c}")
    assert tree.node(7).rule == ("z1 > 3.99578", "z2 in {b, d}")
    assert tree.node(1).rule == ()


def test_split_rule_conditions() -> None:
    boolean = SplitRule("b", "numeric", 0.0, None, None, (), True)
    assert boolean.conditions() == ("b = false", "b = true")
    numeric = SplitRule("z", "numeric", 1e-7, None, None)
    assert numeric.conditions() == ("z <= 0", "z > 0")
    nominal = SplitRule("f", "nominal", None, ("a",), ("c", "d"), ("b",))
    assert nominal.conditions() == ("f in {a}", "f in {c, d}")


def test_tree_is_immutable() -> None:
    tree = _wrap("trees", "sim_gaussian_lmtree_onesplit")
    with pytest.raises(AttributeError):
        tree.foo = 1  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        tree.fitted_node[0] = 5
    with pytest.raises(ValueError):
        tree.used_rows[0] = False
    info: NodeInfo = tree.node(1)
    with pytest.raises(AttributeError):
        info.id = 3  # type: ignore[misc]
    with pytest.raises(KeyError):
        tree.node(0)
    with pytest.raises(KeyError):
        tree.node(len(tree.nodes) + 1)


def test_fitted_node_is_zero_for_dropped_rows() -> None:
    f = fixture("trees", "sim_gaussian_lmtree_onesplit")
    g = grow("trees", "sim_gaussian_lmtree_onesplit")
    n = g.fitted_node.shape[0]
    used = np.ones(n + 3, dtype=bool)
    used[[0, 10, n + 2]] = False
    tree = ModelTree(
        g,
        model(f),
        model_name="lmtree",
        formula=None,
        design=None,
        response_levels=None,
        warnings=(),
        used_rows=used,
        x=None,
        offset=None,
    )
    assert tree.n_rows == n + 3 and tree.n_used == n
    assert tree.fitted_node.dtype == np.int64
    assert np.array_equal(tree.fitted_node[used], g.fitted_node)
    assert np.all(tree.fitted_node[~used] == 0)


@pytest.mark.slow
def test_extras_hold_the_report_and_warnings() -> None:
    tree = _wrap("extension_trees", "sim_negbin_weights")
    for info in tree.nodes:
        assert set(info.extras) == {
            "theta",
            "theta_se",
            "theta_warning",
            "converged",
            "n_warnings",
            "warnings",
        }
        assert info.extras["n_warnings"] == len(info.extras["warnings"])
    lm = _wrap("trees", "sim_gaussian_lmtree_multisplit")
    assert all(info.extras == {} for info in lm.nodes)
