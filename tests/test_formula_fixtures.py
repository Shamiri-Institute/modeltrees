"""The formula features of the public layer against R
(tests/fixtures/formula_trees/, docs/formula-fixtures.md).

Each fixture's CSV is read with its schema into a polars frame (factors as
``pl.Enum`` with the schema's levels, ordered factors declared through
``kinds``, logicals as Booleans, ``NA`` as null) and grown by the fixture's
entry point with its formula, weights and control. Then, against R: the
rows used (R's ``na.action``), the coefficient names, the structure and
splits, coefficients, objectives, scores, instability tests and split
searches at the tolerances of tests/test_tree.py, the kept and dropped
levels, the log-likelihood, the fitted nodes, and ``predict`` on the
fixture's new rows. A subset is repeated from a pandas frame and from a
dict of numpy arrays, which must give the polars tree bit for bit.

R stops on a split on a logical partitioning variable; the port splits it as
the numeric 0/1 (docs/api-design.md, section 1.4), so that fixture is
compared with R's tree on the same data with the variable stored as 0/1
(docs/formula-fixtures.md, section 3, and docs/api-port.md, section 18).
"""

import csv
import re
import warnings
from functools import cache
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest
from fixtures_io import FIXTURES, load_json, nan_array
from grown_trees import control
from test_api_fixtures import _assert_same_node, _same
from test_tree import ATOL, RTOL, _assert_split, _objective, _parents

import modeltrees as mt

PREDICT_RTOL = 1e-8  # as tests/test_predict_fixtures.py

FORMULA_TREES = sorted(p.stem for p in (FIXTURES / "formula_trees").glob("*.json"))
LOGICAL_SPLIT = "sim_formula_logical_lmtree_logical_split"


@cache
def fixture(name: str) -> dict[str, Any]:
    f: dict[str, Any] = load_json(f"formula_trees/{name}.json")
    return f


OK_TREES = [n for n in FORMULA_TREES if fixture(n)["r_outcome"] == "ok"]
# Repeated from pandas and from a dict: every feature but the binomial
# response and the second one-part formula.
SUBSET = [
    "sim_formula_factor_lmtree_factor",
    "sim_formula_factor_lmtree_noint",
    "sim_formula_logical_lmtree_logical",
    "sim_formula_logical_lmtree_logical_noint",
    "sim_formula_ordered_lmtree",
    "sim_formula_na_lmtree_weights",
    "sim_formula_factor_lmtree_onepart",
]
BACKENDS = ["pandas", "dict"]


# Reading the data.


def _schema(csv_relative: str) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = load_json(
        csv_relative.replace(".csv", ".schema.json")
    )["columns"]
    return columns


@cache
def _typed_columns(csv_relative: str) -> dict[str, list[Any]]:
    """The CSV's columns as Python values, None for NA."""
    with open(FIXTURES / csv_relative, newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, list[Any]] = {}
    for col in _schema(csv_relative):
        name, kind = col["name"], col["type"]
        raw = [None if r[name] == "NA" else r[name] for r in rows]
        if kind == "logical":
            out[name] = [None if v is None else v == "TRUE" for v in raw]
        elif kind == "integer":
            out[name] = [None if v is None else int(v) for v in raw]
        elif kind == "double":
            out[name] = [None if v is None else float(v) for v in raw]
        else:
            assert kind in ("factor", "ordered")
            out[name] = raw
    return out


def _polars(schema: list[dict[str, Any]], values: dict[str, list[Any]]) -> pl.DataFrame:
    dtypes = {
        "logical": pl.Boolean,
        "integer": pl.Int64,
        "double": pl.Float64,
    }
    series = []
    for col in schema:
        name, kind = col["name"], col["type"]
        if name not in values:
            continue
        dtype = pl.Enum(col["levels"]) if "levels" in col else dtypes[kind]
        series.append(pl.Series(name, values[name], dtype=dtype))
    return pl.DataFrame(series)


def _pandas(schema: list[dict[str, Any]], values: dict[str, list[Any]]) -> pd.DataFrame:
    out: dict[str, Any] = {}
    for col in schema:
        name, kind = col["name"], col["type"]
        v = values[name]
        if kind in ("factor", "ordered"):
            out[name] = pd.Categorical(
                v, categories=col["levels"], ordered=kind == "ordered"
            )
        elif kind == "logical":
            assert None not in v
            out[name] = np.array(v, dtype=np.bool_)
        elif kind == "integer" and None not in v:
            out[name] = np.array(v, dtype=np.int64)
        else:
            # pandas has no missing value in an int64 column: NaN in floats.
            out[name] = np.array([np.nan if x is None else x for x in v], np.float64)
    return pd.DataFrame(out)


def _dict(
    schema: list[dict[str, Any]], values: dict[str, list[Any]]
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Numpy arrays (labels as objects with None), and the levels= of the
    factors whose declared order is not the code-point sort."""
    out: dict[str, np.ndarray] = {}
    levels: dict[str, list[str]] = {}
    for col in schema:
        name, kind = col["name"], col["type"]
        v = values[name]
        if kind in ("factor", "ordered"):
            out[name] = np.array(v, dtype=object)
            if kind == "ordered" or col["levels"] != sorted(col["levels"]):
                levels[name] = col["levels"]
        elif kind == "logical":
            assert None not in v
            out[name] = np.array(v, dtype=np.bool_)
        elif kind == "integer" and None not in v:
            out[name] = np.array(v, dtype=np.int64)
        else:
            out[name] = np.array([np.nan if x is None else x for x in v], np.float64)
    return out, levels


def _used_names(f: dict[str, Any]) -> set[str]:
    return {f["response"], *f["regressors"], *f["partitioning"]}


def _grow_data(f: dict[str, Any], backend: str) -> tuple[Any, dict[str, Any]]:
    """The fixture's data in a backend, and the levels= and kinds= it needs."""
    schema = _schema(f["data_csv"])
    values = _typed_columns(f["data_csv"])
    used = _used_names(f) if "response" in f else set(values)
    kinds = {
        c["name"]: "ordered"
        for c in schema
        if c["type"] == "ordered" and c["name"] in used
    }
    if backend == "polars":
        return _polars(schema, values), {"kinds": kinds}
    if backend == "pandas":
        return _pandas(schema, values), {"kinds": kinds}
    assert backend == "dict"
    data, levels = _dict(schema, values)
    return data, {
        "kinds": kinds,
        "levels": {k: v for k, v in levels.items() if k in used},
    }


def _entry(f: dict[str, Any], data: Any, **kw: Any) -> mt.ModelTree:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mt.MissingDataWarning)
        if f["function_name"] == "lmtree":
            return mt.lmtree(
                f["formula"], data, weights=f["weights"], control=control(f), **kw
            )
        assert f["function_name"] == "glmtree"
        return mt.glmtree(
            f["formula"],
            data,
            family=f["family"]["family"],
            weights=f["weights"],
            control=control(f),
            **f["glm_control"],
            **kw,
        )


@cache
def public(name: str, backend: str = "polars") -> mt.ModelTree:
    f = fixture(name)
    data, kw = _grow_data(f, backend)
    return _entry(f, data, **kw)


def _used_mask(f: dict[str, Any]) -> np.ndarray:
    used = np.ones(f["na_action"]["n_input"], dtype=np.bool_)
    used[f["na_action"]["removed_rows"]] = False
    return used


def _newdata(f: dict[str, Any], backend: str = "polars") -> Any:
    """The fixture's new rows in a backend, with the data's declared levels."""
    p = f["predict"]
    schema = [c for c in _schema(f["data_csv"]) if c["name"] in p["newdata"]]
    values = {name: list(v) for name, v in p["newdata"].items()}
    if backend == "polars":
        return _polars(schema, values)
    if backend == "pandas":
        return _pandas(schema, values)
    return _dict(schema, values)[0]


def test_fixture_list() -> None:
    assert len(FORMULA_TREES) == 11
    assert {fixture(n)["case"] for n in FORMULA_TREES} == set(range(1, 8))
    assert [n for n in FORMULA_TREES if n not in OK_TREES] == [LOGICAL_SPLIT]


# The design: rows, coefficient names and levels.


@pytest.mark.parametrize("name", OK_TREES)
def test_used_rows_are_the_rows_r_kept(name: str) -> None:
    f = fixture(name)
    tree = public(name)
    used = _used_mask(f)
    np.testing.assert_array_equal(tree.used_rows, used)
    assert tree.n_rows == f["na_action"]["n_input"]
    assert tree.n_used == f["n"] == f["na_action"]["n_used"]


@pytest.mark.parametrize("name", OK_TREES)
def test_coefficient_names_are_r_names(name: str) -> None:
    f = fixture(name)
    tree = public(name)
    assert tree._design is not None
    assert tree._design.coef_names == tuple(f["coefficient_names"])
    for info, e in zip(tree.nodes, f["nodes"], strict=True):
        assert tuple(info.coefficients) == tuple(e["coefficient_names"])
    assert tree.coef_table().columns[5 : 5 + len(f["coefficient_names"])] == list(
        f["coefficient_names"]
    )


@pytest.mark.parametrize("name", OK_TREES)
def test_levels_are_the_model_frame_levels(name: str) -> None:
    """Kept and dropped levels of every factor in R's model frame."""
    f = fixture(name)
    design = public(name)._design
    assert design is not None
    terms = {
        t.name: t
        for t in (design.response.term, *design.regressors, *design.partitioning_terms)
    }
    mf = f["model_frame"]
    for column, levels in mf["levels"].items():
        assert terms[column].levels == tuple(levels), column
        assert terms[column].dropped_levels == tuple(mf["dropped_levels"][column])
    assert design.dropped_levels == {
        k: tuple(v) for k, v in mf["dropped_levels"].items() if v
    }
    # A logical is a factor of FALSE and TRUE for the model matrix, numeric
    # as a partitioning variable; R's model frame keeps it logical.
    for column, cls in mf["classes"].items():
        if cls == "logical":
            assert terms[column].storage == "boolean"
            assert column not in mf["levels"]


def test_one_part_formulas_expand_to_an_intercept_only_model() -> None:
    for name in (
        "sim_formula_factor_lmtree_onepart",
        "sim_formula_factor_glmtree_poisson_onepart",
    ):
        f = fixture(name)
        tree = public(name)
        assert tree.formula == f["expanded_formula"]
        assert f["regressors"] == []
        assert tree._design is not None and tree._design.x.shape[1] == 1


# The tree against R, component by component.


def _assert_tree_matches_r(tree: mt.ModelTree, f: dict[str, Any]) -> None:
    got = tree._grown
    parents = _parents(got.root)
    assert got.minsize == f["root_minsize"]
    assert len(got.nodes) == len(f["nodes"])
    for g, e in zip(got.nodes, f["nodes"], strict=True):
        where = f"node {e['id']}"
        assert g.id == e["id"] and g.depth == e["depth"], where
        assert parents[g.id] == e["parent"], where
        assert [k.id for k in g.kids] == e["kids"], where
        assert g.stop_reason.value == e["stop_reason"], where
        np.testing.assert_array_equal(g.rows, e["rows"])
        assert g.nobs == e["nobs"], where
        _assert_split(g.split, e["split"])
        assert g.fit.coef_names == tuple(e["coefficient_names"]), where
        np.testing.assert_allclose(
            g.fit.coefficients, nan_array(e["coefficients"]), rtol=RTOL, atol=ATOL
        )
        assert g.fit.objfun == pytest.approx(e["objfun"], rel=RTOL, abs=ATOL), where
        if e["estfun"] is not None:
            assert g.fit.estfun is not None
            np.testing.assert_allclose(
                g.fit.estfun, np.array(e["estfun"]), rtol=RTOL, atol=ATOL
            )
        t = e["test"]
        if t["status"] == "not_run":
            assert g.test is None and g.p_value is None, where
        else:
            assert t["status"] == "ok" and g.test is not None, where
            np.testing.assert_allclose(g.test.statistic, t["statistic"], rtol=RTOL)
            np.testing.assert_allclose(
                g.test.p_value, nan_array(t["p_value_raw"]), rtol=RTOL
            )
            np.testing.assert_allclose(
                g.test.p_value_adjusted, nan_array(t["p_value_adjusted"]), rtol=RTOL
            )
            assert g.test.best == t["best"], where
            assert g.p_value == pytest.approx(e["p_value"], rel=RTOL), where
        ss = e["split_search"]
        if ss is None:
            assert g.split_search is None, where
        else:
            assert g.split_search is not None, where
            expected = _objective(ss["objective"])
            obj = g.split_search.objective
            np.testing.assert_array_equal(np.isinf(obj), np.isinf(expected))
            finite = np.isfinite(expected)
            np.testing.assert_allclose(
                obj[finite], expected[finite], rtol=RTOL, atol=ATOL
            )
    used = tree.used_rows
    np.testing.assert_array_equal(tree.fitted_node[used], f["fitted_node"])
    assert np.all(tree.fitted_node[~used] == 0)
    ll = tree.loglik()
    assert ll.value == pytest.approx(f["summary"]["loglik"], rel=RTOL, abs=0)
    assert ll.df == f["summary"]["loglik_df"]
    assert ll.nobs == f["nobs"]


@pytest.mark.parametrize("name", OK_TREES)
def test_tree_matches_r(name: str) -> None:
    _assert_tree_matches_r(public(name), fixture(name))


@pytest.mark.parametrize("name", OK_TREES)
def test_training_nodes_match_r(name: str) -> None:
    f = fixture(name)
    tree = public(name)
    used = tree.used_rows
    nodes = tree.predict(type="node")
    np.testing.assert_array_equal(nodes[used], f["fitted_node"])
    assert np.all(nodes[~used] == 0)
    # The training frame as new data routes the used rows the same way. Its
    # dropped rows may hold levels the model frame dropped, which are
    # unseen (R stops on them too, docs/formula-fixtures.md, section 6).
    frame, _ = _grow_data(f, "polars")
    np.testing.assert_array_equal(
        tree.predict(frame, type="node", unseen="null")[used], f["fitted_node"]
    )


def test_levels_dropped_with_missing_rows_are_unseen_at_predict() -> None:
    """f = e and zf = q occur only in rows R drops; predicting a row that has
    one raises, as R's model.frame(xlev = ...) stops with "factor f has new
    level e" (and "factor zf has new level q")."""
    name = "sim_formula_na_lmtree_weights"
    tree = public(name)
    frame, _ = _grow_data(fixture(name), "polars")
    for column, level in (("f", "e"), ("zf", "q")):
        rows = frame.filter(pl.col(column) == level).with_columns(
            pl.lit(0.0).alias("x"), pl.lit(1.0).alias("z1"), pl.lit(1.0).alias("y")
        )
        assert rows.height == 3
        kind = "response" if column == "f" else "node"
        with pytest.raises(mt.UnseenLevelError, match=f"{column}: '{level}'"):
            tree.predict(rows.with_columns(pl.col("zf").fill_null("k")), type=kind)  # type: ignore[arg-type]
        got = tree.predict(rows, type=kind, unseen="null")  # type: ignore[arg-type]
        assert np.all(got == 0) if kind == "node" else np.all(np.isnan(got))


@pytest.mark.parametrize("name", OK_TREES)
def test_predict_new_rows_matches_r(name: str) -> None:
    f = fixture(name)
    tree = public(name)
    newdata = _newdata(f)
    expected = f["predict"]["expected"]
    np.testing.assert_array_equal(tree.predict(newdata, type="node"), expected["node"])
    np.testing.assert_allclose(
        tree.predict(newdata, type="response"),
        np.array(expected["response"]),
        rtol=PREDICT_RTOL,
        atol=0,
    )


def test_new_rows_have_a_subset_of_the_levels() -> None:
    """What the new rows test: a factor regressor lacks its first level, a
    factor partitioning variable its last, a logical regressor is TRUE."""
    seen = 0
    for name in OK_TREES:
        p = fixture(name)["predict"]
        for column, level in p["excluded_levels"].items():
            labels = [
                ("TRUE" if v else "FALSE") if isinstance(v, bool) else v
                for v in p["newdata"][column]
            ]
            assert level not in labels, (name, column)
            seen += 1
    assert seen == 9


# Printed split conditions against partykit's print.


_PRINT_RULE = re.compile(r"^[| ]*\[(\d+)\] (.+?)(?:: n = .*)?$")


def _r_rules(f: dict[str, Any]) -> dict[int, str]:
    out = {}
    for line in f["print"]:
        m = _PRINT_RULE.match(line)
        if m and m.group(2) != "root":
            out[int(m.group(1))] = m.group(2)
    return out


@pytest.mark.parametrize("name", OK_TREES)
def test_split_conditions_match_the_r_print(name: str) -> None:
    """Equal to R's, except the documented nominal braces (R prints
    ``zf in k, j``, docs/api-port.md, section 9, item 2)."""
    f = fixture(name)
    tree = public(name)
    r_rules = _r_rules(f)
    for info in tree.nodes:
        if info.split is None:
            continue
        left, right = info.split.conditions()
        if info.split.kind == "nominal":
            left, right = (
                re.sub(r" in \{(.*)\}$", r" in \1", c) for c in (left, right)
            )
        assert (left, right) == (r_rules[info.kids[0]], r_rules[info.kids[1]])


# Other backends.


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", SUBSET)
def test_other_backends_give_the_polars_tree(name: str, backend: str) -> None:
    f = fixture(name)
    got = public(name, backend)
    expected = public(name)
    np.testing.assert_array_equal(got.used_rows, expected.used_rows)
    assert got._design is not None and expected._design is not None
    assert got._design.coef_names == expected._design.coef_names
    assert np.array_equal(got._design.x, expected._design.x)
    assert len(got._grown.nodes) == len(expected._grown.nodes)
    for a, b in zip(got._grown.nodes, expected._grown.nodes, strict=True):
        _assert_same_node(a, b)
    assert _same(got.fitted_node, expected.fitted_node)
    _assert_tree_matches_r(got, f)
    newdata = _newdata(f, backend)
    for kind in ("node", "response"):
        assert _same(
            got.predict(newdata, type=kind),  # type: ignore[arg-type]
            expected.predict(_newdata(f), type=kind),  # type: ignore[arg-type]
        ), kind


# The logical partitioning variable R cannot split on.


def test_r_stops_on_a_split_on_a_logical_partitioning_variable() -> None:
    f = fixture(LOGICAL_SPLIT)
    assert f["error"] == {
        "message": "variable 2 is not integer",
        "call": "kidids_split(sp, data = z)",
    }
    assert f["oracle_fixture"] == (
        "formula_trees/sim_formula_logical_numeric_lmtree_split.json"
    )


def test_a_logical_partitioning_variable_splits_as_numeric_zero_one() -> None:
    """A documented difference: R stops, the port grows R's tree of the
    variable stored as 0/1 (docs/api-design.md, section 1.4)."""
    f = fixture(LOGICAL_SPLIT)
    oracle_name = (
        f["oracle_fixture"].removeprefix("formula_trees/").removesuffix(".json")
    )
    oracle = fixture(oracle_name)
    assert (f["formula"], f["control"]) == (oracle["formula"], oracle["control"])
    data, kw = _grow_data(f, "polars")
    assert data["zb"].dtype == pl.Boolean
    tree = _entry(f, data, **kw)
    _assert_tree_matches_r(tree, oracle)
    # Bit for bit the tree of the 0/1 column, printed as false and true
    # where R prints zb <= 0 and zb > 0.
    numeric = public(oracle_name)
    for a, b in zip(tree._grown.nodes, numeric._grown.nodes, strict=True):
        _assert_same_node(a, b)
    split = tree.node(1).split
    assert split is not None and split.boolean and split.threshold == 0.0
    assert split.conditions() == ("zb = false", "zb = true")
    numeric_split = numeric.node(1).split
    assert numeric_split is not None
    assert numeric_split.conditions() == ("zb <= 0", "zb > 0")
    assert (_r_rules(oracle)[2], _r_rules(oracle)[3]) == ("zb <= 0", "zb > 0")
    np.testing.assert_array_equal(
        tree.predict(
            _newdata(oracle).with_columns(pl.col("zb").cast(pl.Boolean)), type="node"
        ),
        oracle["predict"]["expected"]["node"],
    )


def test_a_tested_logical_partitioning_variable_matches_r() -> None:
    """zb is tested as numeric at every tested node, and R's statistics are
    the port's (compared in test_tree_matches_r); no split selects it."""
    f = fixture("sim_formula_logical_lmtree_logical")
    tested = [n for n in f["nodes"] if n["test"]["status"] == "ok"]
    assert tested
    for node in tested:
        assert node["test"]["variables"] == ["z1", "zb"]
        assert node["test"]["statistic"][1] > 0
    assert all(n["split"] is None or n["split"]["variable"] == "z1" for n in f["nodes"])
