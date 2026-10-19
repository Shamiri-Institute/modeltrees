"""Readers for the R-generated fixtures in tests/fixtures/."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FIXTURES = Path(__file__).parent / "fixtures"


def load_json(relative: str) -> Any:
    with open(FIXTURES / relative) as f:
        return json.load(f)


def nan_array(values: list[float | None]) -> NDArray[np.float64]:
    """JSON null (R's NA) becomes NaN."""
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


def load_dataset(
    csv_relative: str,
) -> tuple[dict[str, NDArray[Any]], dict[str, dict[str, Any]]]:
    """Read a dataset CSV and its schema.

    Factor columns are returned as 0-based integer codes in R's level order,
    every other column as float64.
    """
    schema = load_json(csv_relative.replace(".csv", ".schema.json"))
    columns = {c["name"]: c for c in schema["columns"]}
    with open(FIXTURES / csv_relative, newline="") as f:
        rows = list(csv.DictReader(f))
    data: dict[str, NDArray[Any]] = {}
    for name, col in columns.items():
        raw = [r[name] for r in rows]
        if col["type"] in ("factor", "ordered"):
            index = {level: i for i, level in enumerate(col["levels"])}
            data[name] = np.array([index[v] for v in raw], dtype=np.intp)
        else:
            data[name] = np.array([float(v) for v in raw], dtype=np.float64)
    return data, columns
