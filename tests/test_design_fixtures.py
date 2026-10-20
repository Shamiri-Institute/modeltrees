"""The design built from each tree fixture's formula equals the arrays
tests/tree_fixtures.py rebuilds (docs/api-port.md, section 7).

Every tree and extension tree fixture is read into a polars frame with its
schema: factor columns as pl.Enum with the schema's levels, ordered columns
declared ordered through ``kinds``. A subset is repeated with a pandas frame
(ordered columns as ordered categoricals) and another with a dict of numpy
arrays (factor columns as strings, with ``levels`` where the schema's order
is not the code-point sort). An offset ``log(exposure)`` cannot be written
in the formula subset or as a column name, so it is a precomputed column.
"""

import csv
import warnings
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest
from fixtures_io import FIXTURES, load_json
from grown_trees import ALL_TREES, fixture, tree_inputs
from tree_fixtures import cluster, data, variables

from modeltrees._design import Design, build_design

PANDAS_TREES = [
    ("trees", "sim_gaussian_lmtree_weights"),
    ("trees", "sim_poisson_glmtree_weights"),
    ("trees", "synth_diabetes_glmtree_binomial"),
    ("trees", "sim_cluster_glmtree_binomial_cluster_weights"),
    ("extension_trees", "quine_negbin_intercept"),
    ("extension_trees", "sim_multinom_cluster"),
]
DICT_TREES = [
    ("trees", "sim_gaussian_lmtree_multisplit"),
    ("trees", "sim_gaussian_lmtree_intercept_only"),
    ("trees", "sim_binomial_glmtree_multisplit"),
    ("trees", "sim_cluster_lmtree_cluster"),
    ("extension_trees", "sim_negbin_cluster"),
    ("extension_trees", "sim_separation_safelogit"),
]


def _schema(csv_relative: str) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = load_json(
        csv_relative.replace(".csv", ".schema.json")
    )["columns"]
    return columns


def _raw(csv_relative: str) -> dict[str, list[str]]:
    with open(FIXTURES / csv_relative, newline="") as f:
        rows = list(csv.DictReader(f))
    return {name: [r[name] for r in rows] for name in rows[0]}


def _offset_column(f: dict[str, Any]) -> tuple[str | None, str | None]:
    """The offset as a column name, and the column to precompute it from."""
    expr = f["offset"]
    if expr is None:
        return None, None
    if expr.startswith("log(") and expr.endswith(")"):
        return f"log_{expr[4:-1]}", expr[4:-1]
    return expr, None


def _polars(f: dict[str, Any]) -> pl.DataFrame:
    raw = _raw(f["data_csv"])
    series = []
    for col in _schema(f["data_csv"]):
        name, kind = col["name"], col["type"]
        if kind in ("factor", "ordered"):
            series.append(pl.Series(name, raw[name], dtype=pl.Enum(col["levels"])))
        elif kind == "integer":
            series.append(pl.Series(name, [int(v) for v in raw[name]], dtype=pl.Int64))
        else:
            assert kind == "double"
            series.append(
                pl.Series(name, [float(v) for v in raw[name]], dtype=pl.Float64)
            )
    df = pl.DataFrame(series)
    name, source = _offset_column(f)
    if name is not None and source is not None:
        df = df.with_columns(pl.Series(name, np.log(df[source].to_numpy())))
    return df


def _pandas(f: dict[str, Any]) -> pd.DataFrame:
    raw = _raw(f["data_csv"])
    out: dict[str, Any] = {}
    for col in _schema(f["data_csv"]):
        name, kind = col["name"], col["type"]
        if kind in ("factor", "ordered"):
            out[name] = pd.Categorical(
                raw[name], categories=col["levels"], ordered=kind == "ordered"
            )
        elif kind == "integer":
            out[name] = np.array([int(v) for v in raw[name]], dtype=np.int64)
        else:
            out[name] = np.array([float(v) for v in raw[name]])
    name, source = _offset_column(f)
    if name is not None and source is not None:
        out[name] = np.log(out[source])
    return pd.DataFrame(out)


def _dict(f: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Numpy arrays, and the levels= a factor needs when its schema order is
    not the code-point sort."""
    raw = _raw(f["data_csv"])
    out: dict[str, np.ndarray] = {}
    levels: dict[str, list[str]] = {}
    for col in _schema(f["data_csv"]):
        name, kind = col["name"], col["type"]
        if kind in ("factor", "ordered"):
            out[name] = np.array(raw[name])
            if kind == "ordered" or col["levels"] != sorted(col["levels"]):
                levels[name] = col["levels"]
        elif kind == "integer":
            out[name] = np.array([int(v) for v in raw[name]], dtype=np.int64)
        else:
            out[name] = np.array([float(v) for v in raw[name]])
    name, source = _offset_column(f)
    if name is not None and source is not None:
        out[name] = np.log(out[source])
    used = {f["response"], *f["regressors"], *f["partitioning"]}
    return out, {k: v for k, v in levels.items() if k in used}


def _ordered(f: dict[str, Any]) -> dict[str, Any]:
    used = {f["response"], *f["regressors"], *f["partitioning"]}
    return {
        c["name"]: "ordered"
        for c in _schema(f["data_csv"])
        if c["type"] == "ordered" and c["name"] in used
    }


def _build(f: dict[str, Any], frame: Any, **kwargs: Any) -> Design:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no missing data in the fixtures
        return build_design(
            f["formula"],
            frame,
            weights=f["weights"],
            offset=_offset_column(f)[0],
            cluster=f.get("cluster"),
            **kwargs,
        )


def _assert_matches(f: dict[str, Any], design: Design) -> None:
    y, x, w, off = tree_inputs(f)
    n = y.shape[0]
    assert design.n_used == n and design.used_rows.all()
    assert design.coef_names == tuple(
        f["coefficient_names"]
        if f.get("fit_function") != "multinomfit"
        else ["(Intercept)", *f["regressors"]]
    )
    # tree_fixtures passes None for an intercept-only model.
    assert np.array_equal(design.x, np.ones((n, 1)) if x is None else x)
    assert np.array_equal(design.weights, w)
    if off is None:
        assert design.offset is None
    else:
        assert design.offset is not None and np.array_equal(design.offset, off)
    cl = cluster(f)
    if cl is None:
        assert design.cluster is None
    else:
        assert design.cluster is not None and np.array_equal(design.cluster, cl)
    expected = variables(f)
    assert list(design.partitioning) == list(expected)
    for name, v in expected.items():
        got = design.partitioning[name]
        assert got.kind is v.kind
        assert got.levels == v.levels
        assert np.array_equal(got.values, v.values)
    # The raw response: values, or codes in the schema's level order.
    d, columns = data(f["data_csv"])
    assert np.array_equal(design.response.values, d[f["response"]])
    assert design.response.term.levels == (
        tuple(columns[f["response"]]["levels"])
        if "levels" in columns[f["response"]]
        else None
    )


@pytest.mark.parametrize(("directory", "name"), ALL_TREES)
def test_polars_frame(directory: str, name: str) -> None:
    f = fixture(directory, name)
    _assert_matches(f, _build(f, _polars(f), kinds=_ordered(f)))


@pytest.mark.parametrize(("directory", "name"), PANDAS_TREES)
def test_pandas_frame(directory: str, name: str) -> None:
    f = fixture(directory, name)
    _assert_matches(f, _build(f, _pandas(f)))


@pytest.mark.parametrize(("directory", "name"), DICT_TREES)
def test_dict_of_arrays(directory: str, name: str) -> None:
    f = fixture(directory, name)
    arrays, levels = _dict(f)
    _assert_matches(f, _build(f, arrays, kinds=_ordered(f), levels=levels))


def test_dict_subset_exercises_the_code_point_sort() -> None:
    # At least one factor in the dict subset gets its levels from the sort.
    sorted_factors = [
        c["name"]
        for directory, name in DICT_TREES
        for c in _schema(fixture(directory, name)["data_csv"])
        if c["type"] == "factor"
        and c["name"] in fixture(directory, name)["partitioning"]
        and c["levels"] == sorted(c["levels"])
    ]
    assert sorted_factors
