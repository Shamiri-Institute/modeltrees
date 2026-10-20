"""Prediction from a grown tree (docs/api-design.md, sections 3.1 and 4.2).

New data is coded with the training design's term descriptions (R's
``xlevels``): numeric columns as they are, categorical ones through the
levels kept in training. Each row is routed from the root and stops at the
first split whose variable it lacks (no node, id 0). A training level
absent from the node at a nominal split goes to the kid with the larger
``nobs`` (kid 1 on a tie), the rule the core uses for zero-weight rows. A
level never seen in training raises :class:`UnseenLevelError`, or with
``unseen="null"`` gives the row no prediction.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from modeltrees._design import ColumnSpec, Design, Term, _label
from modeltrees._frame import Column, DataInput, Frame, array_column
from modeltrees.errors import DataError, UnseenLevelError
from modeltrees.instability import VariableKind
from modeltrees.node_model import NodeModel
from modeltrees.tree import Node, Variable

PredictType = Literal["node", "response", "link", "prob", "class"]
Unseen = Literal["error", "null"]
_TYPES: tuple[str, ...] = ("node", "response", "link", "prob", "class")

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Coded:
    """New data coded for the tree.

    ``z`` holds one array per partitioning variable: floats (NaN for
    missing) for a numeric variable, level codes (-1 for missing) for a
    factor. ``x`` is the regressor matrix, None when it was not needed;
    ``x_missing`` marks rows whose regressors or offset are missing.
    ``unseen`` marks rows with a level never seen in training, and
    ``unseen_levels`` lists those levels per column.
    """

    n: int
    z: list[NDArray[Any]]
    x: FloatArray | None
    offset: FloatArray | None
    x_missing: NDArray[np.bool_]
    unseen: NDArray[np.bool_]
    unseen_levels: dict[str, list[str]]


def _labels(column: Column) -> list[str | None]:
    """Each value's level label, None where missing."""
    missing = column.missing
    return [
        None if m else _label(v)
        for v, m in zip(column.values.tolist(), missing.tolist(), strict=True)
    ]


def _codes(
    column: Column, levels: Sequence[str], unseen_levels: dict[str, list[str]]
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Codes into the training ``levels`` (-1 for missing or unseen) and
    the mask of unseen values."""
    position = {level: i for i, level in enumerate(levels)}
    labels = _labels(column)
    codes = np.full(len(labels), -1, dtype=np.intp)
    unseen = np.zeros(len(labels), dtype=np.bool_)
    new: dict[str, None] = {}
    for i, label in enumerate(labels):
        if label is None:
            continue
        code = position.get(label)
        if code is None:
            unseen[i] = True
            new[label] = None
        else:
            codes[i] = code
    if new:
        unseen_levels.setdefault(column.name, []).extend(new)
    return codes, unseen


def _numeric(column: Column, role: str) -> FloatArray:
    if column.storage not in ("float", "integer", "boolean"):
        raise DataError(
            f"{role} {column.name!r} was numeric in training but has "
            f"{column.storage} values in the new data"
        )
    values = np.asarray(column.values, dtype=np.float64).copy()
    values[column.missing] = np.nan
    return values


def code_newdata(
    design: Design,
    data: DataInput,
    *,
    offset: ColumnSpec,
    needs_x: bool,
) -> Coded:
    """Code ``data`` with the training design (section 4.2)."""
    frame = Frame(data)
    n = frame.n_rows
    unseen = np.zeros(n, dtype=np.bool_)
    unseen_levels: dict[str, list[str]] = {}

    z: list[NDArray[Any]] = []
    for term in design.partitioning_terms:
        column = frame.column(term.name)
        if term.levels is None:
            z.append(_numeric(column, "partitioning variable"))
        else:
            codes, new = _codes(column, term.levels, unseen_levels)
            z.append(codes)
            unseen |= new

    x = None
    off = None
    x_missing = np.zeros(n, dtype=np.bool_)
    if needs_x:
        blocks: list[FloatArray] = []
        if design.formula.intercept:
            blocks.append(np.ones((n, 1)))
        for term in design.regressors:
            block, missing, new = _regressor(
                frame.column(term.name), term, unseen_levels
            )
            blocks.append(block)
            x_missing |= missing
            unseen |= new
        x = np.hstack(blocks) if blocks else np.zeros((n, 0))
        off = _offset(design, frame, offset, n)
        if off is not None:
            x_missing |= np.isnan(off)
    return Coded(n, z, x, off, x_missing, unseen, unseen_levels)


def _regressor(
    column: Column, term: Term, unseen_levels: dict[str, list[str]]
) -> tuple[FloatArray, NDArray[np.bool_], NDArray[np.bool_]]:
    """The term's matrix columns, the rows with a missing value, and the
    rows with an unseen level."""
    n = column.values.shape[0]
    if term.coding == "numeric":
        values = _numeric(column, "regressor")
        return values[:, None], np.isnan(values), np.zeros(n, dtype=np.bool_)
    assert term.levels is not None
    if term.coding == "logical":
        if column.storage != "boolean":
            raise DataError(
                f"regressor {term.name!r} was Boolean in training but has "
                f"{column.storage} values in the new data"
            )
        codes = np.where(column.missing, -1, column.values.astype(np.intp))
        unseen = np.zeros(n, dtype=np.bool_)
    else:
        codes, unseen = _codes(column, term.levels, unseen_levels)
    # The matrix columns are the indicators of the last len(columns) levels:
    # every level but the first (treatment, logical with an intercept) or
    # all of them (the first factor of a model without an intercept).
    first = len(term.levels) - len(term.columns)
    indicators = (codes[:, None] == np.arange(first, len(term.levels))[None, :]).astype(
        np.float64
    )
    bad = codes < 0
    indicators[bad] = np.nan
    return indicators, bad & ~unseen, unseen


def _offset(
    design: Design, frame: Frame, offset: ColumnSpec, n: int
) -> FloatArray | None:
    """The offset of new data: ``offset=`` if given, else the training
    offset's column, read from the new data."""
    if offset is None:
        if design.offset is None:
            return None
        if design.offset_column is None:
            raise DataError(
                "the tree was grown with an offset array; pass offset= (a column "
                "name or an array) for the new data"
            )
        offset = design.offset_column
    if isinstance(offset, str):
        column = frame.column(offset)
    else:
        column = array_column("(offset)", offset)
        if column.values.shape[0] != n:
            raise DataError(
                f"offset has {column.values.shape[0]} values but the new data has "
                f"{n} rows"
            )
    return _numeric(column, "offset")


def code_arrays(
    variables: Sequence[Variable],
    z: Mapping[str, ArrayLike],
    x: ArrayLike | None,
    offset: ArrayLike | None,
    *,
    has_x: bool,
    has_offset: bool,
    needs_x: bool,
) -> Coded:
    """Code the arrays of a tree grown by ``mob()``.

    ``z`` maps every partitioning variable to its values: floats (NaN for
    missing) for a numeric variable; for a factor, integer codes into the
    training levels or labels (None or NaN for missing).
    """
    missing_names = [v.name for v in variables if v.name not in z]
    if missing_names:
        raise DataError(
            f"z lacks the partitioning variable(s) {', '.join(map(repr, missing_names))}"
        )
    unseen_levels: dict[str, list[str]] = {}
    lengths = {np.asarray(z[v.name]).shape[0] for v in variables}
    if len(lengths) != 1:
        raise DataError("the arrays in z have different lengths")
    n = lengths.pop()
    unseen = np.zeros(n, dtype=np.bool_)
    coded_z: list[NDArray[Any]] = []
    for v in variables:
        values = np.asarray(z[v.name])
        if v.kind is VariableKind.NUMERIC:
            coded_z.append(np.asarray(values, dtype=np.float64))
            continue
        assert v.levels is not None
        if values.dtype.kind in "iu":
            codes = values.astype(np.intp)
            new = (codes < -1) | (codes >= len(v.levels))
            if new.any():
                unseen_levels[v.name] = [str(c) for c in np.unique(codes[new])]
            codes = np.where(new, -1, codes)
        else:
            codes, new = _codes(array_column(v.name, values), v.levels, unseen_levels)
        coded_z.append(codes)
        unseen |= new

    xm = None
    off = None
    x_missing = np.zeros(n, dtype=np.bool_)
    if needs_x:
        if x is None:
            if has_x:
                raise ValueError(
                    "the tree's model has regressors; pass x for predictions "
                    "other than type='node'"
                )
            xm = np.ones((n, 1))
        else:
            xm = np.asarray(x, dtype=np.float64)
            if xm.ndim == 1:
                xm = xm[:, None]
            if xm.shape[0] != n:
                raise DataError(f"x has {xm.shape[0]} rows but z has {n}")
        x_missing |= np.any(np.isnan(xm), axis=1)
        if offset is None and has_offset:
            raise DataError("the tree was grown with an offset; pass offset=")
        if offset is not None:
            off = np.asarray(offset, dtype=np.float64)
            if off.shape != (n,):
                raise DataError(f"offset must have {n} values")
            x_missing |= np.isnan(off)
    return Coded(n, coded_z, xm, off, x_missing, unseen, unseen_levels)


def route(root: Node, z: Sequence[NDArray[Any]], n: int) -> NDArray[np.int64]:
    """The leaf id of each row, 0 where a split variable on the row's path
    is missing (section 4.2)."""
    out = np.zeros(n, dtype=np.int64)

    def visit(node: Node, rows: NDArray[np.intp]) -> None:
        if node.terminal:
            out[rows] = node.id
            return
        split = node.split
        assert split is not None
        values = z[split.variable][rows]
        if split.kind is VariableKind.NUMERIC:
            assert split.breaks is not None
            missing = np.isnan(values)
            # .bincode(right = TRUE): z <= break goes to kid 1, and so does
            # -Inf; Inf goes to kid 2.
            kid = np.where(values <= split.breaks, 1, 2)
        elif split.kind is VariableKind.ORDERED:
            assert split.breaks is not None
            missing = values < 0
            kid = np.where(values + 1 <= split.breaks, 1, 2)
        else:
            assert split.index is not None
            missing = values < 0
            lookup = np.array([0 if k is None else k for k in split.index], np.intp)
            kid = lookup[np.where(missing, 0, values)]
            # A training level absent from the node: the larger kid.
            larger = 1 if node.kids[0].nobs >= node.kids[1].nobs else 2
            kid = np.where(kid == 0, larger, kid)
        for k, child in enumerate(node.kids, start=1):
            visit(child, rows[~missing & (kid == k)])

    visit(root, np.arange(n, dtype=np.intp))
    return out


def predict_leaves(
    model: NodeModel,
    leaves: Mapping[int, Node],
    nodes: NDArray[np.int64],
    x: FloatArray,
    offset: FloatArray | None,
    no_prediction: NDArray[np.bool_],
    kind: str,
    template: Node,
) -> FloatArray:
    """The model's predictions of ``kind`` per row, from each row's leaf.

    Rows in ``no_prediction`` and rows without a node are NaN. ``template``
    is a node whose fit gives the shape of an empty prediction (and checks
    that the model defines ``kind`` before anything is computed).
    """
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise TypeError(
            f"{type(model).__name__} has no predict method; only type='node' "
            "is available"
        )
    off_empty = None if offset is None else offset[:0]
    empty = np.asarray(predict(template.fit, x[:0], off_empty, kind))
    out = np.full((x.shape[0], *empty.shape[1:]), np.nan)
    for leaf_id, leaf in leaves.items():
        rows = np.flatnonzero((nodes == leaf_id) & ~no_prediction)
        if rows.shape[0] == 0:
            continue
        out[rows] = predict(
            leaf.fit, x[rows], None if offset is None else offset[rows], kind
        )
    return out


def check_type(type_: str, unseen: str) -> None:
    if type_ not in _TYPES:
        raise ValueError(f"type must be one of {', '.join(_TYPES)}, got {type_!r}")
    if unseen not in ("error", "null"):
        raise ValueError(f"unseen must be 'error' or 'null', got {unseen!r}")


def raise_unseen(unseen_levels: Mapping[str, list[str]]) -> None:
    detail = "; ".join(
        f"{name}: {', '.join(map(repr, levels))}"
        for name, levels in unseen_levels.items()
    )
    raise UnseenLevelError(
        f"the new data has levels not seen in training ({detail}); pass "
        "unseen='null' to leave those rows without a prediction"
    )
