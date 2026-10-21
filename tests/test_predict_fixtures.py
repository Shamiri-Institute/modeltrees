"""predict on new data against R (tests/fixtures/predict/,
docs/sandwich-fixtures.md, section 2).

On the rows R routes deterministically, ``predict`` equals R's
``predict(type = "node" / "response" / "link")``. The rows R routes at
random (a missing split variable on the path, -Inf at a numeric split)
have no R value: ours gives no prediction for a missing split variable and
sends -Inf to kid 1 (docs/api-port.md, section 11, item 4). lmtree's
"link" has no R counterpart (predict.lm has no such type); ours equals
"response".
"""

import re
import warnings
from functools import cache
from typing import Any

import numpy as np
import polars as pl
import pytest
from fixture_frames import offset_column, schema
from fixtures_io import FIXTURES, load_json
from test_api_fixtures import public

import modeltrees as mt

RTOL = 1e-8

PREDICT_FILES = sorted(p.stem for p in (FIXTURES / "predict").glob("*.json"))


@cache
def _fixture(name: str) -> dict[str, Any]:
    f: dict[str, Any] = load_json(f"predict/{name}.json")
    return f


def _tree_fixture(name: str) -> dict[str, Any]:
    f: dict[str, Any] = load_json(_fixture(name)["tree_fixture"])
    return f


def _tree(name: str) -> mt.ModelTree:
    directory, file = _fixture(name)["tree_fixture"].split("/")
    return public(directory, file.removesuffix(".json"))


def _number(v: Any) -> float | None:
    """JSON null is NA; "Inf" and "-Inf" are strings."""
    return None if v is None else float(v)


@cache
def _newdata(name: str) -> pl.DataFrame:
    """The fixture's new rows as a polars frame: factor labels as strings,
    numbers with null for NA, and the offset column the tree was grown with
    (log(exposure), as a precomputed column)."""
    f = _fixture(name)
    tf = _tree_fixture(name)
    types = {c["name"]: c["type"] for c in schema(tf["data_csv"])}
    columns = []
    for col, values in f["newdata"].items():
        if types[col] in ("factor", "ordered"):
            columns.append(pl.Series(col, values, dtype=pl.String))
        else:
            numbers = [_number(v) for v in values]
            columns.append(pl.Series(col, numbers, dtype=pl.Float64))
    df = pl.DataFrame(columns)
    off, _ = offset_column(tf)
    if off is not None:
        df = df.with_columns(pl.col("exposure").log().alias(off))
    return df


def _deterministic(name: str) -> np.ndarray:
    return np.array([c["routing"] == "deterministic" for c in _fixture(name)["cases"]])


def _expected(name: str, type: str) -> np.ndarray:
    return np.array(
        [np.nan if v is None else v for v in _fixture(name)["expected"][type]],
        dtype=np.float64,
    )


def test_fixture_lists() -> None:
    assert PREDICT_FILES == [
        "sim_binomial_glmtree_multisplit",
        "sim_gaussian_lmtree_weights",
        "sim_poisson_glmtree_multisplit",
    ]


@pytest.mark.parametrize("name", PREDICT_FILES)
def test_node(name: str) -> None:
    det = _deterministic(name)
    assert det.any() and not det.all()
    got = _tree(name).predict(_newdata(name), type="node")
    expected = _expected(name, "node")
    assert not np.any(np.isnan(expected[det]))
    np.testing.assert_array_equal(got[det], expected[det].astype(np.int64))
    # R's random rows are left out of its call.
    assert np.all(np.isnan(expected[~det]))


@pytest.mark.parametrize(
    ("name", "type"),
    [
        (n, t)
        for n in PREDICT_FILES
        for t in ("response", "link")
        if not (t == "link" and _tree_fixture(n)["function_name"] == "lmtree")
    ],
)
def test_response_and_link(name: str, type: str) -> None:
    det = _deterministic(name)
    got = _tree(name).predict(_newdata(name), type=type)  # type: ignore[arg-type]
    expected = _expected(name, type)[det]
    # R's NA (a missing regressor or offset) is our NaN.
    np.testing.assert_array_equal(np.isnan(got[det]), np.isnan(expected))
    keep = ~np.isnan(expected)
    assert keep.any()
    np.testing.assert_allclose(got[det][keep], expected[keep], rtol=RTOL, atol=0)


def test_lmtree_link_is_our_extension() -> None:
    name = "sim_gaussian_lmtree_weights"
    f = _fixture(name)
    assert f["expected"]["link"] is None
    (call,) = f["failing_calls"]
    assert 'type = "link"' in call["r_call"] and "should be one of" in call["error"]
    tree = _tree(name)
    df = _newdata(name)
    np.testing.assert_array_equal(
        tree.predict(df, type="link"), tree.predict(df, type="response")
    )


@pytest.mark.parametrize("name", PREDICT_FILES)
def test_rows_r_routes_at_random(name: str) -> None:
    """A missing split variable on the path: no prediction (node 0). -Inf
    at a numeric split: kid 1, as the most negative finite value."""
    f = _fixture(name)
    tree = _tree(name)
    df = _newdata(name)
    nodes = tree.predict(df, type="node")
    response = tree.predict(df, type="response")
    reasons = set()
    for case in f["cases"]:
        if case["routing"] != "random":
            continue
        row = case["row"]
        reasons.add(case["case"])
        if case["case"] == "missing_on_path":
            assert nodes[row] == 0 and np.isnan(response[row])
            continue
        assert case["case"] == "minus_inf"
        (col,) = case["changed"]
        assert df[col][row] == -np.inf
        finite = df[row].with_columns(pl.lit(-np.finfo(np.float64).max).alias(col))
        assert nodes[row] == tree.predict(finite, type="node")[0] != 0
        # The path R walked ends at the split where -Inf is drawn at random;
        # ours continues to that split's first kid.
        split_node = case["path"][-1]
        info = tree.node(split_node)
        assert info.split is not None and info.split.variable == col
        assert _on_path(tree, nodes[row], info.kids[0])
    assert reasons == {"missing_on_path", "minus_inf"}


def _on_path(tree: mt.ModelTree, leaf: int, node: int) -> bool:
    """Whether ``leaf`` is in the subtree of ``node``."""
    while leaf is not None:
        if leaf == node:
            return True
        parent = tree.node(int(leaf)).parent
        if parent is None:
            return False
        leaf = parent
    return False


@pytest.mark.parametrize("name", PREDICT_FILES)
def test_unseen_levels(name: str) -> None:
    """R stops on a new level in a split variable and ignores one in a
    partitioning variable no split uses; ours raises for both (stricter,
    docs/api-port.md, section 11, item 3), and with unseen="null" gives the
    row no prediction."""
    tree = _tree(name)
    df = _newdata(name)
    for call in _fixture(name)["unseen_level_calls"]:
        match = re.search(r"newdata\[(\d+), \]", call["r_call"])
        assert match is not None
        row = int(match.group(1)) - 1
        new = df[row].with_columns(pl.lit("zz_unseen").alias(call["variable"]))
        assert tree.predict(df[row], type="node")[0] == call["node_without_change"]
        with pytest.raises(mt.UnseenLevelError, match="zz_unseen"):
            tree.predict(new, type="node")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert tree.predict(new, type="node", unseen="null")[0] == 0
        if call["split_variable"]:
            assert call["error"] is not None and "new level" in call["error"]
        else:
            assert call["error"] is None and call["node"] == call["node_without_change"]
