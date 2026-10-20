"""The dataframe boundary (docs/api-design.md, sections 1.4, 2 and 4.3).

The only module that imports narwhals. A :data:`DataInput` (an eager
dataframe narwhals supports, or a mapping of equal-length 1-D arrays)
becomes per-column numpy arrays with the metadata the design builder needs:
the storage type, the inferred kind, the declared level order, and the null
and NaN masks. A mapping does not go through narwhals. Lazy frames raise
:class:`DataError`.

Level order from a dtype (section 4.3, rule 2): an Enum's categories and a
pandas-like categorical's categories are declared orders. A polars
Categorical's physical order is not, so its levels are left to the design
builder's code-point sort, as for strings.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoDataFrame
from numpy.typing import ArrayLike, NDArray

from modeltrees.errors import DataError

DataInput = IntoDataFrame | Mapping[str, ArrayLike]

Storage = Literal["float", "integer", "boolean", "string", "categorical"]
Kind = Literal["numeric", "nominal", "ordered"]

_NUMERIC_STORAGE = ("float", "integer", "boolean")


@dataclass(frozen=True, eq=False)
class Column:
    """One input column as numpy arrays.

    ``values`` is float64 for ``"float"``, an integer array for
    ``"integer"``, bool for ``"boolean"`` and an object array of ``str``
    for ``"string"`` and ``"categorical"``. Entries under ``null`` hold a
    placeholder (0, False, None, or NaN for floats). ``nan`` marks NaN in a
    float column that is not null; pandas reports NaN as null, so a pandas
    NaN is under ``null``. ``levels`` is the declared level order of a categorical
    dtype, None otherwise. ``ordered`` is true for a pandas-like ordered
    categorical only (section 1.4).
    """

    name: str
    storage: Storage
    values: NDArray[Any]
    null: NDArray[np.bool_]
    nan: NDArray[np.bool_]
    levels: tuple[str, ...] | None = None
    ordered: bool = False

    @property
    def missing(self) -> NDArray[np.bool_]:
        """R's ``is.na``: null, or NaN in a float column."""
        return self.null | self.nan

    @property
    def kind(self) -> Kind:
        """The kind inferred from the dtype (section 1.4)."""
        if self.storage in _NUMERIC_STORAGE:
            return "numeric"
        return "ordered" if self.ordered else "nominal"

    @property
    def kind_candidates(self) -> tuple[Kind, ...]:
        """The kinds ``kinds=`` may give this column: a numeric column may be
        made categorical, a string or categorical column cannot be numeric."""
        if self.storage in _NUMERIC_STORAGE:
            return ("numeric", "nominal", "ordered")
        return ("nominal", "ordered")


class Frame:
    """Columns of a :data:`DataInput`, converted on first use."""

    def __init__(self, data: DataInput) -> None:
        self._columns: dict[str, Column] = {}
        if isinstance(data, Mapping):
            self._native: nw.DataFrame[Any] | None = None
            self._arrays = {str(k): v for k, v in data.items()}
            lengths = {k: _length(k, v) for k, v in self._arrays.items()}
            if len(set(lengths.values())) > 1:
                detail = ", ".join(f"{k}: {n}" for k, n in lengths.items())
                raise DataError(f"columns have different lengths ({detail})")
            self.names: tuple[str, ...] = tuple(self._arrays)
            self.n_rows: int = next(iter(lengths.values()), 0)
            return
        native = nw.from_native(data, pass_through=True)
        if isinstance(native, nw.LazyFrame):
            raise DataError(
                "lazy frames are not supported; call .collect() on the frame first"
            )
        if not isinstance(native, nw.DataFrame):
            raise TypeError(
                "data must be an eager dataframe (polars, pandas, pyarrow, ...) "
                f"or a mapping of column names to arrays, got {type(data).__name__}"
            )
        self._native = native
        self._arrays = {}
        self.names = tuple(str(c) for c in native.columns)
        self.n_rows = int(native.shape[0])

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def column(self, name: str) -> Column:
        """The column ``name``; :class:`DataError` if there is none."""
        if name not in self._columns:
            if name not in self.names:
                raise DataError(
                    f"column {name!r} is not in the data (columns: "
                    f"{', '.join(map(repr, self.names))})"
                )
            if self._native is None:
                self._columns[name] = array_column(name, self._arrays[name])
            else:
                self._columns[name] = _series_column(name, self._native[name])
        return self._columns[name]


def _length(name: str, values: ArrayLike) -> int:
    array = np.asarray(values)
    if array.ndim != 1:
        raise DataError(
            f"column {name!r} must be 1-D, got an array of shape {array.shape}"
        )
    return int(array.shape[0])


def array_column(name: str, values: ArrayLike) -> Column:
    """A column from a 1-D array, without narwhals (section 2).

    Float, integer and Boolean arrays are numeric; string arrays, and
    object arrays of ``str`` with None for missing, are strings.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise DataError(
            f"column {name!r} must be 1-D, got an array of shape {array.shape}"
        )
    n = array.shape[0]
    no = np.zeros(n, dtype=np.bool_)
    if array.dtype.kind == "b":
        return Column(name, "boolean", array.astype(np.bool_), no, no.copy())
    if array.dtype.kind in "iu":
        return Column(name, "integer", array, no, no.copy())
    if array.dtype.kind == "f":
        values64 = array.astype(np.float64)
        return Column(name, "float", values64, no, np.isnan(values64))
    if array.dtype.kind in "US":
        strings = np.array([str(v) for v in array.tolist()], dtype=object)
        return Column(name, "string", strings, no, no.copy())
    if array.dtype.kind == "O":
        items = array.tolist()
        null = np.array([v is None for v in items], dtype=np.bool_)
        if all(isinstance(v, str) for v, m in zip(items, null, strict=True) if not m):
            return Column(name, "string", np.array(items, dtype=object), null, no)
        raise DataError(
            f"column {name!r} is an object array that does not hold only strings "
            "and None; pass a typed numeric, Boolean or string array"
        )
    raise DataError(f"column {name!r} has unsupported dtype {array.dtype}")


def _series_column(name: str, series: nw.Series[Any]) -> Column:
    dtype = series.dtype
    null = np.asarray(series.is_null().to_numpy(), dtype=np.bool_)
    n = null.shape[0]
    no = np.zeros(n, dtype=np.bool_)
    if dtype == nw.Boolean:
        values = np.asarray(series.fill_null(False).to_numpy(), dtype=np.bool_)
        return Column(name, "boolean", values, null, no)
    if dtype.is_integer():
        values = np.asarray(series.fill_null(0).to_numpy())
        return Column(name, "integer", values, null, no)
    if dtype.is_numeric():  # floats and decimals
        values = np.asarray(series.cast(nw.Float64).to_numpy(), dtype=np.float64)
        nan = np.isnan(values) & ~null
        values = values.copy()
        values[null] = np.nan
        return Column(name, "float", values, null, nan)
    if dtype == nw.String:
        return Column(name, "string", _strings(series, null), null, no)
    if isinstance(dtype, nw.Enum):
        levels = tuple(str(c) for c in dtype.categories)
        # narwhals maps a pandas-like ordered categorical to Enum; a polars
        # Enum declares categories, not an order (section 1.4).
        ordered = series.implementation.is_pandas_like()
        return Column(
            name, "categorical", _strings(series, null), null, no, levels, ordered
        )
    if dtype == nw.Categorical:
        levels_or_none = None
        if series.implementation.is_pandas_like():
            categories = series.cat.get_categories().to_list()
            levels_or_none = tuple(str(c) for c in categories)
        return Column(
            name, "categorical", _strings(series, null), null, no, levels_or_none
        )
    if dtype == nw.Object:
        items = series.to_list()
        if all(isinstance(v, str) for v, m in zip(items, null, strict=True) if not m):
            return Column(name, "string", _strings(series, null), null, no)
        raise DataError(
            f"column {name!r} has object dtype with values that are not strings"
        )
    raise DataError(f"column {name!r} has unsupported dtype {dtype}")


def _strings(series: nw.Series[Any], null: NDArray[np.bool_]) -> NDArray[np.object_]:
    """The values as ``str``, None where null."""
    items = series.to_list()
    out = np.empty(len(items), dtype=object)
    for i, (v, m) in enumerate(zip(items, null, strict=True)):
        out[i] = None if m else str(v)
    return out


def read_columns(
    data: DataInput, names: Sequence[str]
) -> tuple[dict[str, Column], int]:
    """The columns ``names`` of ``data`` and the number of rows."""
    frame = Frame(data)
    return {name: frame.column(name) for name in dict.fromkeys(names)}, frame.n_rows
