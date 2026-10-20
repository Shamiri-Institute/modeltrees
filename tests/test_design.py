"""The design builder (docs/api-design.md, sections 1.3, 1.4, 4.1 and 4.3).

Expected column names and level labels were checked against R 4.6.1's
model.matrix, factor and as.character.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest

from modeltrees._design import Design, build_design, r_double_label, r_name
from modeltrees.errors import DataError, FormulaError, MissingDataWarning
from modeltrees.instability import VariableKind


def _build(formula: str, data: Any, **kwargs: Any) -> Design:
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingDataWarning)
        return build_design(formula, data, **kwargs)


@pytest.fixture
def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            "f": ["b", "a", "c", "a", "b", "b"],
            "g": ["u", "v", "u", "v", "u", "v"],
            "b": [True, False, True, True, False, True],
            "z": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


# Treatment coding and R's names


def test_treatment_coding_with_intercept(frame: pl.DataFrame) -> None:
    d = _build("y ~ x + f | z", frame)
    assert d.coef_names == ("(Intercept)", "x", "fb", "fc")
    f = np.array(["b", "a", "c", "a", "b", "b"])
    expected = np.column_stack(
        [np.ones(6), frame["x"].to_numpy(), f == "b", f == "c"]
    ).astype(np.float64)
    assert d.x.dtype == np.float64
    assert np.array_equal(d.x, expected)
    term = d.regressors[1]
    assert term.levels == ("a", "b", "c")
    assert term.coding == "treatment" and term.columns == ("fb", "fc")


def test_without_intercept_the_first_factor_gets_every_level(
    frame: pl.DataFrame,
) -> None:
    # R: colnames(model.matrix(y ~ 0 + x + f + g)) is x fa fb fc gv.
    d = _build("y ~ 0 + x + f + g | z", frame)
    assert d.coef_names == ("x", "fa", "fb", "fc", "gv")
    f = np.array(["b", "a", "c", "a", "b", "b"])
    assert np.array_equal(d.x[:, 1:4], np.column_stack([f == "a", f == "b", f == "c"]))
    assert [t.coding for t in d.regressors] == ["numeric", "full", "treatment"]
    d = _build("y ~ x - 1 | z", frame)
    assert d.coef_names == ("x",)
    assert np.array_equal(d.x, frame["x"].to_numpy()[:, None])


def test_intercept_only(frame: pl.DataFrame) -> None:
    d = _build("y ~ z", frame)
    assert d.formula.expanded == "y ~ 1 | z"
    assert d.coef_names == ("(Intercept)",)
    assert np.array_equal(d.x, np.ones((6, 1)))
    assert d.regressors == ()


def test_column_order_follows_the_formula(frame: pl.DataFrame) -> None:
    d = _build("y ~ g + x + f | z", frame)
    assert d.coef_names == ("(Intercept)", "gv", "x", "fb", "fc")


def test_r_names_backtick_non_syntactic_names() -> None:
    # R: (Intercept) `my x` `my f`b `2z` `b b`TRUE .x
    data = {
        "y": np.arange(4.0),
        "my x": np.array([1.0, 2.0, 3.0, 5.0]),
        "my f": np.array(["a", "b", "a", "b"]),
        "2z": np.array([1.0, 3.0, 2.0, 4.0]),
        "b b": np.array([True, False, True, True]),
        ".x": np.array([1.0, 2.0, 4.0, 3.0]),
        "z": np.arange(4.0),
    }
    d = _build("y ~ `my x` + `my f` + `2z` + `b b` + .x | z", data)
    assert d.coef_names == (
        "(Intercept)",
        "`my x`",
        "`my f`b",
        "`2z`",
        "`b b`TRUE",
        ".x",
    )
    assert [r_name(n) for n in ("x", "x.1", "_x", ".1x", "if", "TRUE", "é")] == [
        "x",
        "x.1",
        "`_x`",
        "`.1x`",
        "`if`",
        "`TRUE`",
        "é",
    ]


def test_r_double_labels() -> None:
    # R: as.character(c(1e5, 1e15, 0.1, 1/3, 123456789012, 2.5, 10))
    values = [1e5, 1e15, 0.1, 1 / 3, 123456789012.0, 2.5, 10.0]
    assert [r_double_label(v) for v in values] == [
        "1e+05",
        "1e+15",
        "0.1",
        "0.333333333333333",
        "123456789012",
        "2.5",
        "10",
    ]


# Boolean regressors


def test_boolean_regressor_is_coded_as_a_logical(frame: pl.DataFrame) -> None:
    d = _build("y ~ b | z", frame)
    assert d.coef_names == ("(Intercept)", "bTRUE")
    assert np.array_equal(d.x[:, 1], frame["b"].to_numpy().astype(np.float64))
    assert d.regressors[0].coding == "logical"
    assert d.regressors[0].levels == ("FALSE", "TRUE")


def test_boolean_regressor_without_intercept(frame: pl.DataFrame) -> None:
    # R: model.matrix(y ~ 0 + b + f) is bFALSE bTRUE fb fc, and
    # model.matrix(y ~ 0 + f + b) is fa fb fc bTRUE.
    assert _build("y ~ 0 + b + f | z", frame).coef_names == (
        "bFALSE",
        "bTRUE",
        "fb",
        "fc",
    )
    assert _build("y ~ 0 + f + b | z", frame).coef_names == ("fa", "fb", "fc", "bTRUE")


def test_constant_boolean_keeps_both_levels() -> None:
    # R keeps a logical's FALSE level even when unused (btTRUE is all 1).
    data = {"y": np.arange(3.0), "b": np.ones(3, dtype=bool), "z": np.arange(3.0)}
    d = _build("y ~ b | z", data)
    assert d.coef_names == ("(Intercept)", "bTRUE")
    assert np.array_equal(d.x[:, 1], np.ones(3))
    assert d.dropped_levels == {}


def test_boolean_regressor_declared_numeric(frame: pl.DataFrame) -> None:
    d = _build("y ~ b | z", frame, kinds={"b": "numeric"})
    assert d.coef_names == ("(Intercept)", "b")
    assert np.array_equal(d.x[:, 1], frame["b"].to_numpy().astype(np.float64))


# Dropped levels


def test_unused_levels_are_dropped_and_recorded() -> None:
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "f": pl.Series(
                ["lo", "hi", "lo", "hi"], dtype=pl.Enum(["lo", "mid", "hi"])
            ),
            "z": pl.Series(["a", "a", "c", "c"], dtype=pl.Enum(["a", "b", "c"])),
        }
    )
    d = _build("y ~ f | z", df)
    assert d.coef_names == ("(Intercept)", "fhi")
    assert d.regressors[0].levels == ("lo", "hi")
    assert d.regressors[0].dropped_levels == ("mid",)
    assert d.partitioning["z"].levels == ("a", "c")
    assert np.array_equal(d.partitioning["z"].values, [0, 0, 1, 1])
    assert d.dropped_levels == {"f": ("mid",), "z": ("b",)}


def test_levels_are_computed_after_missing_data_removal() -> None:
    data = {
        "y": np.array([1.0, np.nan, 3.0, 4.0]),
        "f": np.array(["a", "b", "c", "a"]),
        "z": np.arange(4.0),
    }
    with pytest.warns(MissingDataWarning):
        d = build_design("y ~ f | z", data)
    assert d.coef_names == ("(Intercept)", "fc")
    assert d.dropped_levels == {"f": ("b",)}


# Level order (section 4.3)


def test_rule_1_levels_argument_wins() -> None:
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "f": pl.Series(["a", "b", "c"], dtype=pl.Enum(["a", "b", "c"])),
            "z": [1.0, 2.0, 3.0],
        }
    )
    d = _build("y ~ f | z", df, levels={"f": ["c", "a", "b", "extra"]})
    assert d.coef_names == ("(Intercept)", "fa", "fb")
    assert d.regressors[0].dropped_levels == ("extra",)


def test_rule_1_values_outside_levels_raise() -> None:
    data = {"y": np.arange(3.0), "f": np.array(["a", "b", "c"]), "z": np.arange(3.0)}
    with pytest.raises(DataError, match="not in its levels: 'c'"):
        build_design("y ~ f | z", data, levels={"f": ["a", "b"]})
    with pytest.raises(DataError, match="duplicated"):
        build_design("y ~ f | z", data, levels={"f": ["a", "b", "c", "a"]})


def test_rule_2_declared_dtype_order() -> None:
    enum = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "f": pl.Series(["lo", "hi", "mid"], dtype=pl.Enum(["lo", "mid", "hi"])),
            "z": [1.0, 2.0, 3.0],
        }
    )
    assert _build("y ~ f | z", enum).coef_names == ("(Intercept)", "fmid", "fhi")
    pandas = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "f": pd.Categorical(["z", "y", "x"], categories=["z", "y", "x"]),
            "z": [1.0, 2.0, 3.0],
        }
    )
    assert _build("y ~ f | z", pandas).coef_names == ("(Intercept)", "fy", "fx")


def test_rule_3_integer_made_categorical_uses_numeric_order() -> None:
    data = {
        "y": np.arange(4.0),
        "school": np.array([10, 2, 10, 2]),
        "z": np.arange(4.0),
    }
    d = _build(
        "y ~ x | school", {**data, "x": np.arange(4.0)}, kinds={"school": "nominal"}
    )
    assert d.partitioning["school"].kind is VariableKind.NOMINAL
    assert d.partitioning["school"].levels == ("2", "10")  # a string sort gives 10, 2
    assert np.array_equal(d.partitioning["school"].values, [1, 0, 1, 0])
    d = _build("y ~ school | z", data, kinds={"school": "nominal"})
    assert d.coef_names == ("(Intercept)", "school10")
    d = _build(
        "y ~ x | school", {**data, "x": np.arange(4.0)}, kinds={"school": "ordered"}
    )
    assert d.partitioning["school"].kind is VariableKind.ORDERED
    assert d.partitioning["school"].levels == ("2", "10")


def test_rule_4_code_point_sort_not_locale() -> None:
    # A locale collation (R's sort in en_US) gives a A b B; code points give
    # A B a b.
    for data in (
        pl.DataFrame(
            {"y": [1.0, 2.0, 3.0, 4.0], "f": ["b", "B", "a", "A"], "z": [1.0, 2, 3, 4]}
        ),
        pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "f": pl.Series(["b", "B", "a", "A"], dtype=pl.Categorical),
                "z": [1.0, 2, 3, 4],
            }
        ),
        {
            "y": np.arange(4.0),
            "f": np.array(["b", "B", "a", "A"]),
            "z": np.arange(4.0),
        },
    ):
        d = _build("y ~ f | z", data)
        assert d.regressors[0].levels == ("A", "B", "a", "b")
        assert d.coef_names == ("(Intercept)", "fB", "fa", "fb")


def test_ordered_variable_needs_a_declared_order() -> None:
    data = {
        "y": np.arange(3.0),
        "x": np.arange(3.0),
        "z": np.array(["low", "mid", "high"]),
    }
    with pytest.raises(DataError, match="no declared level order"):
        build_design("y ~ x | z", data, kinds={"z": "ordered"})
    d = _build(
        "y ~ x | z", data, kinds={"z": "ordered"}, levels={"z": ["low", "mid", "high"]}
    )
    assert d.partitioning["z"].kind is VariableKind.ORDERED
    assert d.partitioning["z"].levels == ("low", "mid", "high")
    assert np.array_equal(d.partitioning["z"].values, [0, 1, 2])


def test_pandas_ordered_categorical_is_ordered_and_polars_enum_nominal() -> None:
    cats = ["lo", "mid", "hi"]
    pandas = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [1.0, 2.0, 3.0],
            "z": pd.Categorical(["hi", "lo", "mid"], categories=cats, ordered=True),
        }
    )
    z = _build("y ~ x | z", pandas).partitioning["z"]
    assert z.kind is VariableKind.ORDERED and z.levels == tuple(cats)
    assert np.array_equal(z.values, [2, 0, 1])
    polars = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [1.0, 2.0, 3.0],
            "z": pl.Series(["hi", "lo", "mid"], dtype=pl.Enum(cats)),
        }
    )
    assert _build("y ~ x | z", polars).partitioning["z"].kind is VariableKind.NOMINAL
    z = _build("y ~ x | z", polars, kinds={"z": "ordered"}).partitioning["z"]
    assert z.kind is VariableKind.ORDERED and z.levels == tuple(cats)


def test_ordered_regressor_uses_treatment_coding() -> None:
    pandas = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "o": pd.Categorical(
                ["hi", "lo", "mid"], categories=["lo", "mid", "hi"], ordered=True
            ),
            "z": [1.0, 2.0, 3.0],
        }
    )
    d = _build("y ~ o | z", pandas)
    assert d.coef_names == ("(Intercept)", "omid", "ohi")


# kinds


def test_kinds_overrides(frame: pl.DataFrame) -> None:
    with pytest.raises(DataError, match="cannot be 'numeric'"):
        build_design("y ~ x | f", frame, kinds={"f": "numeric"})
    with pytest.raises(ValueError, match="must be one of"):
        build_design("y ~ x | f", frame, kinds={"f": "factor"})  # type: ignore[dict-item]
    with pytest.raises(DataError, match="does not use: 'w'"):
        build_design("y ~ x | f", frame, kinds={"w": "nominal"})
    with pytest.raises(DataError, match="does not use: 'w'"):
        build_design("y ~ x | f", frame, levels={"w": ["a"]})
    with pytest.raises(DataError, match="declares numeric"):
        build_design("y ~ x | z", frame, kinds={"z": "numeric"}, levels={"z": [1]})


def test_float_made_nominal_uses_r_labels() -> None:
    data = {"y": np.arange(3.0), "x": np.array([10.0, 2.5, 1e5]), "z": np.arange(3.0)}
    d = _build("y ~ x | z", data, kinds={"x": "nominal"})
    assert d.regressors[0].levels == ("2.5", "10", "1e+05")
    assert d.coef_names == ("(Intercept)", "x10", "x1e+05")


# Partitioning variables


def test_boolean_partitioning_variable_is_numeric(frame: pl.DataFrame) -> None:
    v = _build("y ~ x | b", frame).partitioning["b"]
    assert v.kind is VariableKind.NUMERIC and v.levels is None
    assert v.values.dtype == np.float64
    assert np.array_equal(v.values, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0])


def test_partitioning_variables_in_formula_order(frame: pl.DataFrame) -> None:
    d = _build("y ~ x | z + f + x", frame)
    assert list(d.partitioning) == ["z", "f", "x"]
    assert d.partitioning["f"].kind is VariableKind.NOMINAL
    assert d.partitioning["f"].levels == ("a", "b", "c")
    assert np.array_equal(d.partitioning["f"].values, [1, 0, 2, 0, 1, 1])
    assert np.array_equal(d.partitioning["x"].values, frame["x"].to_numpy())
    assert [t.kind for t in d.partitioning_terms] == ["numeric", "nominal", "numeric"]


# Response, weights, offset, cluster


def test_response(frame: pl.DataFrame) -> None:
    r = _build("f ~ x | z", frame).response
    assert r.term.kind == "nominal" and r.term.levels == ("a", "b", "c")
    assert np.array_equal(r.values, [1, 0, 2, 0, 1, 1])
    r = _build("b ~ x | z", frame).response
    assert r.term.storage == "boolean" and r.values.dtype == np.float64
    assert np.array_equal(r.values, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    # Levels make an integer response categorical.
    data = {"y": np.array([2, 0, 1]), "x": np.arange(3.0), "z": np.arange(3.0)}
    r = _build("y ~ x | z", data, levels={"y": [0, 1, 2]}).response
    assert r.term.kind == "nominal" and r.term.levels == ("0", "1", "2")
    assert np.array_equal(r.values, [2, 0, 1])


def test_weights_offset_cluster_as_columns_or_arrays(frame: pl.DataFrame) -> None:
    df = frame.with_columns(
        w=pl.Series([1, 2, 3, 1, 2, 3]),
        o=pl.Series([0.1] * 6),
        s=pl.Series(list("aabbcc")),
    )
    d = _build("y ~ x | z", df, weights="w", offset="o", cluster="s")
    assert d.weights.dtype == np.float64
    assert np.array_equal(d.weights, [1, 2, 3, 1, 2, 3])
    assert d.offset is not None and np.array_equal(d.offset, [0.1] * 6)
    assert d.cluster is not None and d.cluster.tolist() == list("aabbcc")
    d = _build(
        "y ~ x | z",
        frame,
        weights=np.array([1.0, 2, 3, 1, 2, 3]),
        offset=[0.1] * 6,
        cluster=np.array([1, 1, 2, 2, 3, 3]),
    )
    assert np.array_equal(d.weights, [1, 2, 3, 1, 2, 3])
    assert d.cluster is not None and d.cluster.tolist() == [1, 1, 2, 2, 3, 3]
    no = _build("y ~ x | z", frame)
    assert np.array_equal(no.weights, np.ones(6)) and no.offset is None
    assert no.cluster is None


def test_case_argument_errors(frame: pl.DataFrame) -> None:
    with pytest.raises(DataError, match="negative"):
        build_design("y ~ x | z", frame, weights=-np.ones(6))
    with pytest.raises(DataError, match="has 3 values but the data has 6"):
        build_design("y ~ x | z", frame, weights=np.ones(3))
    with pytest.raises(DataError, match="must be numeric"):
        build_design("y ~ x | z", frame, weights="f")
    with pytest.raises(DataError, match="'nope' is not in the data"):
        build_design("y ~ x | z", frame, offset="nope")
    with pytest.raises(DataError, match="'q' is not in the data"):
        build_design("y ~ q | z", frame)
    with pytest.raises(FormulaError):
        build_design("y ~ log(x) | z", frame)
    with pytest.raises(ValueError, match="na_action"):
        build_design("y ~ x | z", frame, na_action="drop")  # type: ignore[arg-type]


# Missing data (section 4.1)


def _with_missing() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "y": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x": [0.5, 1.5, float("nan"), 3.5, 4.5, 5.5, 6.5, 7.5],
            "f": ["b", "a", "c", None, "b", "b", "a", "c"],
            "z": [0.1, 0.2, 0.3, 0.4, None, 0.6, 0.7, 0.8],
            "w": [1, 2, 1, 2, 1, None, 2, 1],
            "o": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, float("nan"), 0.7],
            "s": [1, 1, 2, 2, 3, 3, 4, None],
            "unused": [None] * 8,
        },
        schema_overrides={"unused": pl.Float64},
    )


def test_missing_rows_are_removed_as_complete_cases() -> None:
    df = _with_missing()
    kwargs: dict[str, Any] = {"weights": "w", "offset": "o", "cluster": "s"}
    with pytest.warns(MissingDataWarning) as record:
        d = build_design("y ~ x + f | z", df, **kwargs)
    assert len(record) == 1
    assert str(record[0].message) == (
        "7 of 8 rows dropped for missing values "
        "(y: 1, x: 1, f: 1, z: 1, w: 1, o: 1, s: 1)"
    )
    # The same frame with the rows deleted by hand (only row 0 is complete),
    # extended so that every level is still present.
    df = pl.concat([df, df.slice(0, 1).with_columns(f=pl.lit("a"))])
    df = pl.concat([df, df.slice(0, 1).with_columns(f=pl.lit("c"))])
    with pytest.warns(MissingDataWarning, match="7 of 10 rows dropped"):
        d = build_design("y ~ x + f | z", df, **kwargs)
    by_hand = df[[0, 8, 9]]
    e = _build("y ~ x + f | z", by_hand, **kwargs)
    assert np.array_equal(d.used_rows, [True] + [False] * 7 + [True, True])
    assert d.n_used == 3 and d.n_rows == 10
    assert d.missing_counts == {"y": 1, "x": 1, "f": 1, "z": 1, "w": 1, "o": 1, "s": 1}
    assert np.array_equal(d.x, e.x) and d.coef_names == e.coef_names
    assert np.array_equal(d.response.values, e.response.values)
    for name in d.partitioning:
        assert np.array_equal(d.partitioning[name].values, e.partitioning[name].values)
    assert np.array_equal(d.weights, e.weights)
    assert d.offset is not None and e.offset is not None
    assert np.array_equal(d.offset, e.offset)
    assert d.cluster is not None and e.cluster is not None
    assert np.array_equal(d.cluster, e.cluster)


def test_missing_rows_in_pandas_and_dict_input() -> None:
    values: dict[str, Any] = {
        "y": [1.0, np.nan, 3.0, 4.0],
        "x": [1.0, 2.0, 3.0, 4.0],
        "z": [0.1, 0.2, np.nan, 0.4],
    }
    for data in (pd.DataFrame(values), {k: np.array(v) for k, v in values.items()}):
        with pytest.warns(MissingDataWarning, match=r"2 of 4 rows .* \(y: 1, z: 1\)"):
            d = build_design("y ~ x | z", data)
        assert np.array_equal(d.used_rows, [True, False, False, True])
        assert np.array_equal(d.response.values, [1.0, 4.0])


def test_columns_the_formula_does_not_use_are_ignored() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = build_design("y ~ x | f", _with_missing().slice(0, 1))
    assert d.n_used == 1


def test_na_action_fail() -> None:
    with pytest.raises(
        DataError, match=r"1 of 8 rows .* \(y: 1\) and na_action='fail'"
    ):
        build_design(
            "y ~ 1 | f", _with_missing().with_columns(f=pl.lit("a")), na_action="fail"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_design("y ~ x | f", _with_missing().slice(0, 1), na_action="fail")


def test_no_complete_rows() -> None:
    with (
        pytest.raises(DataError, match="no rows are left"),
        pytest.warns(MissingDataWarning),
    ):
        build_design(
            "y ~ x | z", {"y": [np.nan, 1.0], "x": [1.0, np.nan], "z": [1.0, 2.0]}
        )


def test_infinite_values() -> None:
    base = {"y": np.arange(3.0), "x": np.arange(3.0), "z": np.arange(3.0)}
    inf = np.array([0.0, np.inf, 1.0])
    d = _build("y ~ x | z", {**base, "z": inf})
    assert np.array_equal(d.partitioning["z"].values, inf)
    for column in ("y", "x"):
        with pytest.raises(DataError, match="infinite"):
            build_design("y ~ x | z", {**base, column: inf})
    with pytest.raises(DataError, match="weights .* infinite"):
        build_design("y ~ x | z", base, weights=inf)
    with pytest.raises(DataError, match="offset .* infinite"):
        build_design("y ~ x | z", base, offset=-inf)
