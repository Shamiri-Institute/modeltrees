"""The design: model matrix, partitioning variables and case data
(docs/api-design.md, sections 1.3, 1.4, 4.1 and 4.3).

From a parsed formula, the data and the keyword arguments (``weights``,
``offset``, ``cluster``, ``levels``, ``kinds``, ``na_action``) this builds
what the core's ``mob()`` takes: the regressor matrix with R's column names,
the partitioning variables as :class:`~modeltrees.tree.Variable`, and the
weights, offset and cluster arrays, all restricted to the complete cases.
The response is returned raw, with its kind and levels; the model-specific
coding (binomial, multinomial) is done by the entry points. Each regressor
and partitioning variable keeps a :class:`Term` description from which
``predict`` can rebuild the matrix for new data, as R's ``xlevels``.
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from modeltrees._formula import ParsedFormula, parse_formula
from modeltrees._frame import Column, DataInput, Frame, Kind, Storage, array_column
from modeltrees.errors import DataError, MissingDataWarning
from modeltrees.instability import VariableKind
from modeltrees.tree import Variable

NaAction = Literal["omit", "fail"]
Coding = Literal["numeric", "logical", "treatment", "full"]
ColumnSpec = str | ArrayLike | None

_KINDS: tuple[Kind, ...] = get_args(Kind)
_VARIABLE_KIND = {
    "numeric": VariableKind.NUMERIC,
    "nominal": VariableKind.NOMINAL,
    "ordered": VariableKind.ORDERED,
}
# R's logical as a factor: both levels, always (model.matrix keeps an unused one).
_LOGICAL_LEVELS = ("FALSE", "TRUE")

# Words R does not accept as syntactic names (?Reserved).
_R_RESERVED = frozenset(
    [
        "if",
        "else",
        "repeat",
        "while",
        "function",
        "for",
        "next",
        "break",
        "TRUE",
        "FALSE",
        "NULL",
        "Inf",
        "NaN",
        "NA",
        "NA_integer_",
        "NA_real_",
        "NA_character_",
        "NA_complex_",
        "in",
    ]
)


@dataclass(frozen=True)
class Term:
    """A response, regressor or partitioning variable as used in the fit.

    ``levels`` are the levels kept after missing-data removal, in level
    order, for a categorical term (None for a numeric one);
    ``dropped_levels`` the declared or observed levels that no used row
    has. For a regressor, ``coding`` says how it enters the matrix and
    ``columns`` names its matrix columns: ``"numeric"`` one column,
    ``"logical"`` R's coding of a Boolean (levels FALSE and TRUE, never
    dropped), ``"treatment"`` an indicator per level but the first, and
    ``"full"`` an indicator per level (the first categorical term of a
    model without an intercept).
    """

    name: str
    kind: Kind
    storage: Storage
    levels: tuple[str, ...] | None = None
    dropped_levels: tuple[str, ...] = ()
    coding: Coding | None = None
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Response:
    """The response before model-specific coding.

    ``values`` is float64 for a numeric response (a Boolean as 0/1, with
    ``term.storage == "boolean"``) and 0-based codes into ``term.levels``
    for a categorical one.
    """

    term: Term
    values: NDArray[Any]


@dataclass(frozen=True, eq=False)
class Design:
    """Everything ``mob()`` needs, restricted to the used rows.

    ``used_rows`` is a mask over the input rows. ``missing_counts`` gives,
    per used column with missing values, how many rows it has missing,
    over all input rows (a row can count for several columns).
    """

    formula: ParsedFormula
    response: Response
    x: NDArray[np.float64]
    coef_names: tuple[str, ...]
    regressors: tuple[Term, ...]
    partitioning: dict[str, Variable]
    partitioning_terms: tuple[Term, ...]
    weights: NDArray[np.float64]
    offset: NDArray[np.float64] | None
    cluster: NDArray[Any] | None
    used_rows: NDArray[np.bool_]
    missing_counts: dict[str, int] = field(default_factory=dict)
    # The offset's column name when it was given as one (predict reads it
    # from new data), and the input's dataframe backend (None for a mapping).
    offset_column: str | None = None
    backend: str | None = None

    @property
    def n_rows(self) -> int:
        return int(self.used_rows.shape[0])

    @property
    def n_used(self) -> int:
        return int(self.used_rows.sum())

    @property
    def dropped_levels(self) -> dict[str, tuple[str, ...]]:
        """Levels that no used row has, per categorical column."""
        terms = [self.response.term, *self.regressors, *self.partitioning_terms]
        return {t.name: t.dropped_levels for t in terms if t.dropped_levels}


def r_name(name: str) -> str:
    """The name as R writes it in coefficient names: backticked unless it is
    syntactic (``make.names``-valid and not reserved)."""
    first = name[:1]
    syntactic = (
        bool(name)
        and (first.isalpha() or (first == "." and not name[1:2].isdigit()))
        and all(c.isalnum() or c in "._" for c in name)
        and name not in _R_RESERVED
    )
    return name if syntactic else f"`{name}`"


def r_double_label(value: float) -> str:
    """``as.character`` of a double: 15 significant digits, the shorter of
    fixed and scientific notation, fixed on a tie."""
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if value == 0:
        return "0"
    target = float(f"{value:.14e}")
    digits = next(d for d in range(1, 16) if float(f"{value:.{d - 1}e}") == target)
    sci = f"{value:.{digits - 1}e}"
    exponent = int(sci.split("e")[1])
    fixed = f"{value:.{max(0, digits - 1 - exponent)}f}"
    return fixed if len(fixed) <= len(sci) else sci


def _label(value: Any) -> str:
    """A value's level label, as R's factor() would write it."""
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return r_double_label(float(value))
    return str(value)


def _resolve_kind(
    column: Column, kinds: Mapping[str, Kind], levels: Mapping[str, Any]
) -> Kind:
    if column.name in kinds:
        kind = kinds[column.name]
        if kind not in column.kind_candidates:
            raise DataError(
                f"column {column.name!r} has {column.storage} values and cannot be "
                f"{kind!r} (possible: {', '.join(column.kind_candidates)})"
            )
        if kind == "numeric" and column.name in levels:
            raise DataError(
                f"levels are given for {column.name!r}, which kinds declares numeric"
            )
        return kind
    if column.name in levels and column.kind == "numeric":
        # Levels make a numeric column categorical (for example an integer
        # response of class codes).
        return "nominal"
    return column.kind


def _categorical(
    column: Column,
    kind: Kind,
    declared: Sequence[Any] | None,
    used: NDArray[np.bool_],
) -> tuple[NDArray[np.intp], tuple[str, ...], tuple[str, ...]]:
    """Codes of the used rows, kept levels and dropped levels (section 4.3)."""
    present = ~column.missing
    numeric = column.storage in ("float", "integer", "boolean")
    observed = np.unique(column.values[present])
    observed_labels = [_label(v) for v in observed.tolist()]
    if declared is not None:
        # Rule 1: levels=, which must cover every observed value.
        order = [_label(v) for v in declared]
        if len(set(order)) != len(order):
            raise DataError(f"levels for {column.name!r} has duplicated levels")
        outside = sorted(set(observed_labels) - set(order))
        if outside:
            raise DataError(
                f"column {column.name!r} has values not in its levels: "
                f"{', '.join(map(repr, outside))}"
            )
    elif column.levels is not None:
        # Rule 2: the dtype's declared order.
        order = list(column.levels)
    elif numeric:
        # Rule 3: numeric order (np.unique sorts numerically). Values with
        # the same label (15 significant digits) are one level, as in R.
        order = list(dict.fromkeys(observed_labels))
    else:
        # Rule 4: code-point sort; never an order for an ordered variable.
        if kind == "ordered":
            raise DataError(
                f"column {column.name!r} is ordered but has no declared level order; "
                "pass levels= or use an Enum or ordered categorical dtype (a sort "
                "of the strings would not be a meaningful order)"
            )
        order = sorted(observed_labels)
    values = column.values[used]
    unique, inverse = np.unique(values, return_inverse=True)
    used_labels = [_label(v) for v in unique.tolist()]
    present_set = set(used_labels)
    kept = tuple(level for level in order if level in present_set)
    dropped = tuple(level for level in order if level not in present_set)
    position = {level: i for i, level in enumerate(kept)}
    lookup = np.array([position[label] for label in used_labels], dtype=np.intp)
    codes = lookup[inverse.reshape(-1)] if values.shape[0] else np.zeros(0, np.intp)
    return codes, kept, dropped


def _numeric(column: Column, used: NDArray[np.bool_]) -> NDArray[np.float64]:
    return np.asarray(column.values[used], dtype=np.float64)


def _check_finite(name: str, values: NDArray[np.float64], role: str) -> None:
    bad = ~np.isfinite(values)
    if bad.any():
        raise DataError(
            f"{role} {name!r} has {int(bad.sum())} infinite values; "
            "infinite values are allowed only in partitioning variables"
        )


def _case_column(
    spec: ColumnSpec, role: str, frame: Frame
) -> tuple[str, Column] | None:
    """A weights, offset or cluster argument: a column name or an array."""
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec, frame.column(spec)
    column = array_column(f"({role})", spec)
    if column.values.shape[0] != frame.n_rows:
        raise DataError(
            f"{role} has {column.values.shape[0]} values but the data has "
            f"{frame.n_rows} rows"
        )
    return column.name, column


def build_design(
    formula: str | ParsedFormula,
    data: DataInput,
    *,
    weights: ColumnSpec = None,
    offset: ColumnSpec = None,
    cluster: ColumnSpec = None,
    levels: Mapping[str, Sequence[Any]] | None = None,
    kinds: Mapping[str, Kind] | None = None,
    na_action: NaAction = "omit",
) -> Design:
    """Build the design of a formula entry point.

    Raises :class:`~modeltrees.errors.DataError` for unusable data and
    emits one :class:`~modeltrees.errors.MissingDataWarning` when rows are
    dropped for missing values.
    """
    parsed = parse_formula(formula) if isinstance(formula, str) else formula
    levels = dict(levels or {})
    kinds = dict(kinds or {})
    if na_action not in ("omit", "fail"):
        raise ValueError(f"na_action must be 'omit' or 'fail', got {na_action!r}")
    for name, kind in kinds.items():
        if kind not in _KINDS:
            raise ValueError(
                f"kinds[{name!r}] must be one of {', '.join(_KINDS)}, got {kind!r}"
            )

    frame = Frame(data)
    formula_names = list(
        dict.fromkeys([parsed.response, *parsed.regressors, *parsed.partitioning])
    )
    unknown = sorted((set(kinds) | set(levels)) - set(formula_names))
    if unknown:
        raise DataError(
            f"kinds or levels name columns the formula does not use: "
            f"{', '.join(map(repr, unknown))}"
        )
    columns = {name: frame.column(name) for name in formula_names}
    case: dict[str, tuple[str, Column] | None] = {
        role: _case_column(spec, role, frame)
        for role, spec in (
            ("weights", weights),
            ("offset", offset),
            ("cluster", cluster),
        )
    }
    for role in ("weights", "offset"):
        entry = case[role]
        if entry is not None and entry[1].storage not in ("float", "integer"):
            raise DataError(
                f"{role} column {entry[0]!r} must be numeric, got {entry[1].storage}"
            )

    # Complete cases over the used columns (section 4.1).
    used_columns: dict[str, Column] = dict(columns)
    for entry in case.values():
        if entry is not None:
            used_columns.setdefault(entry[0], entry[1])
    missing_counts = {
        name: int(col.missing.sum())
        for name, col in used_columns.items()
        if col.missing.any()
    }
    complete = np.ones(frame.n_rows, dtype=np.bool_)
    for col in used_columns.values():
        complete &= ~col.missing
    n_dropped = int(frame.n_rows - complete.sum())
    if n_dropped:
        detail = ", ".join(f"{name}: {count}" for name, count in missing_counts.items())
        if na_action == "fail":
            raise DataError(
                f"{n_dropped} of {frame.n_rows} rows have missing values ({detail}) "
                "and na_action='fail'"
            )
        warnings.warn(
            f"{n_dropped} of {frame.n_rows} rows dropped for missing values ({detail})",
            MissingDataWarning,
            stacklevel=2,
        )
    if not complete.any():
        raise DataError("no rows are left after removing missing values")

    response = _response(columns[parsed.response], kinds, levels, complete)
    x, coef_names, regressor_terms = _regressors(
        parsed, columns, kinds, levels, complete
    )
    variables, partitioning_terms = _partitioning(
        parsed, columns, kinds, levels, complete
    )

    n_used = int(complete.sum())
    w = np.ones(n_used)
    if case["weights"] is not None:
        name, col = case["weights"]
        w = _numeric(col, complete)
        _check_finite(name, w, "weights")
        if np.any(w < 0):
            raise DataError(f"weights {name!r} has negative values")
    off = None
    if case["offset"] is not None:
        name, col = case["offset"]
        off = _numeric(col, complete)
        _check_finite(name, off, "offset")
    cl = None
    if case["cluster"] is not None:
        cl = case["cluster"][1].values[complete]

    return Design(
        formula=parsed,
        response=response,
        x=x,
        coef_names=coef_names,
        regressors=regressor_terms,
        partitioning=variables,
        partitioning_terms=partitioning_terms,
        weights=w,
        offset=off,
        cluster=cl,
        used_rows=complete,
        missing_counts=missing_counts,
        offset_column=offset if isinstance(offset, str) else None,
        backend=frame.backend,
    )


def _response(
    column: Column,
    kinds: Mapping[str, Kind],
    levels: Mapping[str, Sequence[Any]],
    used: NDArray[np.bool_],
) -> Response:
    kind = _resolve_kind(column, kinds, levels)
    if kind == "numeric":
        values = _numeric(column, used)
        _check_finite(column.name, values, "response")
        return Response(Term(column.name, kind, column.storage), values)
    codes, kept, dropped = _categorical(column, kind, levels.get(column.name), used)
    return Response(Term(column.name, kind, column.storage, kept, dropped), codes)


def _regressors(
    parsed: ParsedFormula,
    columns: Mapping[str, Column],
    kinds: Mapping[str, Kind],
    levels: Mapping[str, Sequence[Any]],
    used: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], tuple[str, ...], tuple[Term, ...]]:
    """The model matrix with R's column names (section 1.3)."""
    n = int(used.sum())
    blocks: list[NDArray[np.float64]] = []
    names: list[str] = []
    if parsed.intercept:
        blocks.append(np.ones((n, 1)))
        names.append("(Intercept)")
    terms: list[Term] = []
    first_factor = True
    for name in parsed.regressors:
        column = columns[name]
        kind = _resolve_kind(column, kinds, levels)
        label = r_name(name)
        logical = column.storage == "boolean" and name not in kinds
        if kind == "numeric" and not logical:
            values = _numeric(column, used)
            _check_finite(name, values, "regressor")
            blocks.append(values[:, None])
            names.append(label)
            terms.append(
                Term(name, kind, column.storage, coding="numeric", columns=(label,))
            )
            continue
        if logical:
            # R turns a logical into a factor with levels FALSE and TRUE.
            kept: tuple[str, ...] = _LOGICAL_LEVELS
            dropped: tuple[str, ...] = ()
            codes = column.values[used].astype(np.intp)
        else:
            codes, kept, dropped = _categorical(column, kind, levels.get(name), used)
        # model.matrix: without an intercept the first factor gets every level.
        full = not parsed.intercept and first_factor
        first_factor = False
        coded = kept if full else kept[1:]
        indicators = (codes[:, None] == np.arange(len(kept))[None, :]).astype(
            np.float64
        )
        blocks.append(indicators if full else indicators[:, 1:])
        cols = tuple(f"{label}{level}" for level in coded)
        names.extend(cols)
        coding: Coding = "logical" if logical else ("full" if full else "treatment")
        terms.append(Term(name, kind, column.storage, kept, dropped, coding, cols))
    x = np.hstack(blocks) if blocks else np.zeros((n, 0))
    return np.ascontiguousarray(x, dtype=np.float64), tuple(names), tuple(terms)


def _partitioning(
    parsed: ParsedFormula,
    columns: Mapping[str, Column],
    kinds: Mapping[str, Kind],
    levels: Mapping[str, Sequence[Any]],
    used: NDArray[np.bool_],
) -> tuple[dict[str, Variable], tuple[Term, ...]]:
    """The core's partitioning variables (section 1.4). A Boolean is numeric
    0/1, as partykit treats a logical; infinite values are allowed."""
    variables: dict[str, Variable] = {}
    terms: list[Term] = []
    for name in parsed.partitioning:
        column = columns[name]
        kind = _resolve_kind(column, kinds, levels)
        if kind == "numeric":
            variables[name] = Variable(
                name, VariableKind.NUMERIC, _numeric(column, used)
            )
            terms.append(Term(name, kind, column.storage))
            continue
        codes, kept, dropped = _categorical(column, kind, levels.get(name), used)
        variables[name] = Variable(name, _VARIABLE_KIND[kind], codes, kept)
        terms.append(Term(name, kind, column.storage, kept, dropped))
    return variables, tuple(terms)
