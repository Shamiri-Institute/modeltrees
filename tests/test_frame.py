"""The dataframe boundary (docs/api-design.md, sections 1.4, 2 and 4.3)."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest

from modeltrees._frame import Column, Frame, array_column, read_columns
from modeltrees.errors import DataError


def _column(data: Any, name: str) -> Column:
    return Frame(data).column(name)


def _assert_strings(column: Column, expected: list[str | None]) -> None:
    assert column.values.dtype == object
    assert column.values.tolist() == expected
    assert column.null.tolist() == [v is None for v in expected]


# polars


def test_polars_numeric_columns() -> None:
    df = pl.DataFrame(
        {
            "f": [1.5, None, float("nan"), 2.0],
            "i": [1, None, 3, 4],
            "u8": pl.Series([1, 2, 3, 4], dtype=pl.UInt8),
            "f32": pl.Series([0.5, 1.5, None, 2.5], dtype=pl.Float32),
        }
    )
    f = _column(df, "f")
    assert f.storage == "float" and f.kind == "numeric"
    assert f.values.dtype == np.float64
    assert f.null.tolist() == [False, True, False, False]
    assert f.nan.tolist() == [False, False, True, False]
    assert f.missing.tolist() == [False, True, True, False]
    assert f.values[0] == 1.5 and f.values[3] == 2.0
    i = _column(df, "i")
    assert i.storage == "integer" and i.kind == "numeric"
    assert i.values.dtype.kind == "i"
    assert i.values.tolist() == [1, 0, 3, 4]  # 0 is the placeholder under null
    assert i.null.tolist() == [False, True, False, False]
    assert not i.nan.any()
    assert _column(df, "u8").storage == "integer"
    f32 = _column(df, "f32")
    assert f32.values.dtype == np.float64
    assert f32.values[1] == 1.5 and f32.null.tolist() == [False, False, True, False]
    assert i.kind_candidates == ("numeric", "nominal", "ordered")


def test_polars_boolean() -> None:
    b = _column(pl.DataFrame({"b": [True, None, False]}), "b")
    assert b.storage == "boolean" and b.kind == "numeric"
    assert b.values.dtype == np.bool_
    assert b.values.tolist() == [True, False, False]
    assert b.null.tolist() == [False, True, False]


def test_polars_string() -> None:
    s = _column(pl.DataFrame({"s": ["b", None, "a", "B"]}), "s")
    assert s.storage == "string" and s.kind == "nominal"
    assert s.levels is None and not s.ordered
    _assert_strings(s, ["b", None, "a", "B"])
    assert s.kind_candidates == ("nominal", "ordered")


def test_polars_enum_is_nominal_with_declared_levels() -> None:
    e = _column(
        pl.DataFrame(
            {
                "e": pl.Series(
                    ["lo", "hi", None, "lo"], dtype=pl.Enum(["lo", "mid", "hi"])
                )
            }
        ),
        "e",
    )
    assert e.storage == "categorical"
    assert e.kind == "nominal" and not e.ordered
    assert e.levels == ("lo", "mid", "hi")
    _assert_strings(e, ["lo", "hi", None, "lo"])


def test_polars_categorical_has_no_declared_order() -> None:
    # polars' physical category order is never a level order (section 4.3).
    c = _column(
        pl.DataFrame({"c": pl.Series(["z", "y", None], dtype=pl.Categorical)}), "c"
    )
    assert c.storage == "categorical" and c.kind == "nominal"
    assert c.levels is None
    _assert_strings(c, ["z", "y", None])


def test_polars_lazy_frame_raises() -> None:
    with pytest.raises(DataError, match=r"collect\(\)"):
        Frame(pl.DataFrame({"a": [1]}).lazy())  # type: ignore[arg-type]


def test_polars_unsupported_dtype() -> None:
    df = pl.DataFrame({"d": pl.Series([1, 2]).cast(pl.Date)})
    frame = Frame(df)  # only used columns are converted
    with pytest.raises(DataError, match="'d' has unsupported dtype"):
        frame.column("d")


# pandas (only in the tests: the library never imports it)


def test_pandas_columns() -> None:
    df = pd.DataFrame(
        {
            "f": [1.0, np.nan, 2.0],
            "i": pd.array([1, None, 3], dtype="Int64"),
            "ii": [1, 2, 3],
            "b": [True, False, True],
            "bn": pd.array([True, None, False], dtype="boolean"),
            "s": ["b", None, "a"],
            "st": pd.array(["a", None, "b"], dtype="string"),
        }
    )
    f = _column(df, "f")
    assert f.storage == "float"
    assert f.missing.tolist() == [False, True, False]
    i = _column(df, "i")
    assert i.storage == "integer" and i.values.tolist() == [1, 0, 3]
    assert i.null.tolist() == [False, True, False]
    assert _column(df, "ii").values.tolist() == [1, 2, 3]
    b = _column(df, "b")
    assert b.storage == "boolean" and b.values.tolist() == [True, False, True]
    bn = _column(df, "bn")
    assert bn.storage == "boolean" and bn.null.tolist() == [False, True, False]
    _assert_strings(_column(df, "s"), ["b", None, "a"])
    _assert_strings(_column(df, "st"), ["a", None, "b"])


def test_pandas_ordered_categorical_is_ordered() -> None:
    df = pd.DataFrame(
        {
            "o": pd.Categorical(
                ["mid", "lo", None, "hi"], categories=["lo", "mid", "hi"], ordered=True
            )
        }
    )
    o = _column(df, "o")
    assert o.storage == "categorical"
    assert o.ordered and o.kind == "ordered"
    assert o.levels == ("lo", "mid", "hi")
    _assert_strings(o, ["mid", "lo", None, "hi"])


def test_pandas_unordered_categorical_keeps_declared_order() -> None:
    df = pd.DataFrame(
        {"u": pd.Categorical(["z", "y", None], categories=["z", "y", "x"])}
    )
    u = _column(df, "u")
    assert u.kind == "nominal" and not u.ordered
    assert u.levels == ("z", "y", "x")  # section 13, decision 3
    _assert_strings(u, ["z", "y", None])


def test_pandas_integer_categories_become_strings() -> None:
    df = pd.DataFrame({"c": pd.Categorical([3, 1, None], categories=[3, 1])})
    c = _column(df, "c")
    assert c.levels == ("3", "1")
    _assert_strings(c, ["3", "1", None])


def test_pandas_object_column() -> None:
    df = pd.DataFrame({"o": pd.Series(["a", None, "b"], dtype=object)})
    _assert_strings(_column(df, "o"), ["a", None, "b"])
    mixed = pd.DataFrame({"o": pd.Series([1, "a"], dtype=object)})
    with pytest.raises(DataError, match="not strings"):
        _column(mixed, "o")


# Mapping of arrays (no narwhals)


def test_mapping_columns() -> None:
    data = {
        "f": np.array([1.0, np.nan, 3.0]),
        "i": np.array([1, 2, 3]),
        "b": np.array([True, False, True]),
        "s": np.array(["b", "a", "c"]),
        "o": np.array(["b", None, "a"], dtype=object),
        "lst": [1.5, 2.5, 3.5],
    }
    f = _column(data, "f")
    assert f.storage == "float"
    assert f.nan.tolist() == [False, True, False] and not f.null.any()
    assert _column(data, "i").storage == "integer"
    b = _column(data, "b")
    assert b.storage == "boolean" and b.kind == "numeric"
    s = _column(data, "s")
    assert s.storage == "string" and s.levels is None
    _assert_strings(s, ["b", "a", "c"])
    _assert_strings(_column(data, "o"), ["b", None, "a"])
    assert _column(data, "lst").values.tolist() == [1.5, 2.5, 3.5]


def test_mapping_errors() -> None:
    with pytest.raises(DataError, match="different lengths"):
        Frame({"a": np.zeros(3), "b": np.zeros(4)})
    with pytest.raises(DataError, match="1-D"):
        Frame({"a": np.zeros((3, 2))})
    with pytest.raises(DataError, match="only strings"):
        _column({"a": np.array([1, "x"], dtype=object)}, "a")
    with pytest.raises(DataError, match="unsupported dtype"):
        _column({"a": np.array(["2020-01-01"], dtype="datetime64[D]")}, "a")


def test_array_column() -> None:
    # array_column is the Mapping path, also used for arrays passed as
    # weights, offset or cluster.
    col = array_column("w", np.array([1, 2]))
    assert col.storage == "integer" and col.name == "w"


def test_missing_column_names_the_columns() -> None:
    with pytest.raises(DataError, match="'zz' is not in the data"):
        Frame(pl.DataFrame({"a": [1]})).column("zz")
    with pytest.raises(DataError, match="'zz' is not in the data"):
        Frame({"a": [1]}).column("zz")


def test_unsupported_input_type() -> None:
    with pytest.raises(TypeError, match="eager dataframe"):
        Frame([1, 2, 3])  # type: ignore[arg-type]


def test_read_columns_in_every_backend() -> None:
    values = {"x": [1.0, 2.0], "g": ["a", "b"]}
    inputs: list[Any] = [pl.DataFrame(values), pd.DataFrame(values), values]
    for data in inputs:
        columns, n = read_columns(data, ["x", "g", "x"])
        assert n == 2
        assert list(columns) == ["x", "g"]
        assert columns["x"].values.tolist() == [1.0, 2.0]
        assert columns["g"].values.tolist() == ["a", "b"]
