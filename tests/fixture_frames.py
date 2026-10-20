"""Fixture datasets as dataframes, as the formula entry points take them.

A dataset CSV is read with its ``.schema.json``: factor columns as
``pl.Enum`` (pandas categoricals, numpy string arrays) with the schema's
levels, integer and double columns as such. An offset ``log(exposure)``
cannot be written in the formula subset, so it becomes a precomputed
column ``log_exposure``.
"""

import csv
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from fixtures_io import FIXTURES, load_json


def schema(csv_relative: str) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = load_json(
        csv_relative.replace(".csv", ".schema.json")
    )["columns"]
    return columns


def raw_columns(csv_relative: str) -> dict[str, list[str]]:
    with open(FIXTURES / csv_relative, newline="") as f:
        rows = list(csv.DictReader(f))
    return {name: [r[name] for r in rows] for name in rows[0]}


def offset_column(f: dict[str, Any]) -> tuple[str | None, str | None]:
    """The offset as a column name, and the column to precompute it from."""
    expr = f["offset"]
    if expr is None:
        return None, None
    if expr.startswith("log(") and expr.endswith(")"):
        return f"log_{expr[4:-1]}", expr[4:-1]
    return expr, None


def polars_frame(f: dict[str, Any]) -> pl.DataFrame:
    raw = raw_columns(f["data_csv"])
    series = []
    for col in schema(f["data_csv"]):
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
    name, source = offset_column(f)
    if name is not None and source is not None:
        df = df.with_columns(pl.Series(name, np.log(df[source].to_numpy())))
    return df


def pandas_frame(f: dict[str, Any]) -> pd.DataFrame:
    raw = raw_columns(f["data_csv"])
    out: dict[str, Any] = {}
    for col in schema(f["data_csv"]):
        name, kind = col["name"], col["type"]
        if kind in ("factor", "ordered"):
            out[name] = pd.Categorical(
                raw[name], categories=col["levels"], ordered=kind == "ordered"
            )
        elif kind == "integer":
            out[name] = np.array([int(v) for v in raw[name]], dtype=np.int64)
        else:
            out[name] = np.array([float(v) for v in raw[name]])
    name, source = offset_column(f)
    if name is not None and source is not None:
        out[name] = np.log(out[source])
    return pd.DataFrame(out)


def dict_arrays(
    f: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Numpy arrays, and the levels= a factor needs when its schema order is
    not the code-point sort."""
    raw = raw_columns(f["data_csv"])
    out: dict[str, np.ndarray] = {}
    levels: dict[str, list[str]] = {}
    for col in schema(f["data_csv"]):
        name, kind = col["name"], col["type"]
        if kind in ("factor", "ordered"):
            out[name] = np.array(raw[name])
            if kind == "ordered" or col["levels"] != sorted(col["levels"]):
                levels[name] = col["levels"]
        elif kind == "integer":
            out[name] = np.array([int(v) for v in raw[name]], dtype=np.int64)
        else:
            out[name] = np.array([float(v) for v in raw[name]])
    name, source = offset_column(f)
    if name is not None and source is not None:
        out[name] = np.log(out[source])
    used = {f["response"], *f["regressors"], *f["partitioning"]}
    return out, {k: v for k, v in levels.items() if k in used}


def ordered_kinds(f: dict[str, Any]) -> dict[str, Any]:
    used = {f["response"], *f["regressors"], *f["partitioning"]}
    return {
        c["name"]: "ordered"
        for c in schema(f["data_csv"])
        if c["type"] == "ordered" and c["name"] in used
    }
