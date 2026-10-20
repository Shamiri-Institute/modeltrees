"""Tables and printed output of a fitted tree (docs/api-design.md, sections
3.3 and 3.4).

The tables are built as columns of numpy arrays and converted to the
requested backend at the dataframe boundary (:func:`_frame.build_frame`).
The printed tree follows ``print.modelparty`` (partykit 1.3-0,
modelparty.R:958-1016, and print.partynode in print.R): a header, the tree
with split rules, node sizes and leaf coefficients, and a footer. Printed
output is for humans and not stable (section 8.4).
"""

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from modeltrees._frame import build_frame, is_importable

if TYPE_CHECKING:
    from modeltrees._api import ModelTree, NodeInfo

Backend = Literal["polars", "pandas", "pyarrow", "numpy"] | None
_BACKENDS = ("polars", "pandas", "pyarrow", "modin", "cudf", "numpy")

# R's print of an objective uses getOption("digits") = 7.
_R_DIGITS = 7


def resolve_backend(backend: str | None, data_backend: str | None) -> str:
    """The backend of a result table (section 3.3): the requested one, else
    the caller's dataframe backend, else polars if importable, else numpy."""
    if backend is not None:
        if backend not in _BACKENDS:
            raise ValueError(
                f"backend must be one of {', '.join(_BACKENDS)} or None, got {backend!r}"
            )
        return backend
    if data_backend is not None:
        return data_backend
    return "polars" if is_importable("polars") else "numpy"


def make_table(
    columns: Mapping[str, Sequence[Any] | np.ndarray] | Mapping[str, list[Any]],
    backend: str,
) -> Any:
    """A table in ``backend``: a native frame, or for "numpy" a dict of
    numpy arrays (object arrays where values are not numbers)."""
    arrays: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        if isinstance(values, np.ndarray):
            arrays[name] = values
            continue
        items = list(values)
        if items and all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_))
            for v in items
        ):
            arrays[name] = np.asarray(items)
        elif items and all(isinstance(v, (bool, np.bool_)) for v in items):
            arrays[name] = np.asarray(items, dtype=np.bool_)
        else:
            out = np.empty(len(items), dtype=object)
            out[:] = items
            arrays[name] = out
    if backend == "numpy":
        return arrays
    return build_frame(arrays, backend)


def coef_columns(
    tree: "ModelTree", layout: str
) -> dict[str, Sequence[Any] | np.ndarray]:
    """The leaf coefficient table's columns (section 3.3)."""
    leaves = [tree.node(i) for i in tree.leaves]
    if layout == "long":
        node: list[int] = []
        term: list[str] = []
        estimate: list[float] = []
        for info in leaves:
            for name, value in info.coefficients.items():
                node.append(info.id)
                term.append(name)
                estimate.append(value)
        return {
            "node": np.asarray(node, dtype=np.int64),
            "term": term,
            "estimate": np.asarray(estimate, dtype=np.float64),
        }
    columns: dict[str, Sequence[Any] | np.ndarray] = {
        "node": np.asarray([i.id for i in leaves], dtype=np.int64),
        "n_rows": np.asarray([i.n_rows for i in leaves], dtype=np.int64),
        "nobs": np.asarray([i.nobs for i in leaves], dtype=np.float64),
        "status": [i.status for i in leaves],
        "stop_reason": [i.stop_reason for i in leaves],
    }
    names = list(leaves[0].coefficients)
    for name in names:
        columns[name] = np.asarray(
            [i.coefficients[name] for i in leaves], dtype=np.float64
        )
    for key in report_keys(tree):
        columns[key] = [i.extras.get(key) for i in leaves]
    return columns


def report_keys(tree: "ModelTree") -> list[str]:
    """The model's report columns (section 3.5), without the warning texts."""
    return [k for k in tree.node(1).extras if k != "warnings"]


# Printing


def title(tree: "ModelTree") -> str:
    name = tree.model_name
    if name == "lmtree":
        return "Linear model tree"
    if name.startswith("glmtree("):
        return f"Generalized linear model tree (family: {name[8:-1]})"
    if name == "negbintree":
        return "Negative binomial regression tree"
    if name == "safelogittree":
        return "Safe logistic regression tree"
    if name == "multinomtree":
        return "Multinomial logit tree"
    return f"Model-based recursive partitioning ({name[4:-1] or name})"


def objective_label(tree: "ModelTree") -> str:
    name = tree.model_name
    if name == "lmtree":
        return "residual sum of squares"
    if name == "negbintree" and getattr(tree.model, "objective", "") == "deviance":
        return "deviance"
    if name.startswith("mob("):
        return ""
    return "negative log-likelihood"


def _number(value: float, digits: int) -> str:
    """``value`` with ``digits`` significant digits (R's ``format``)."""
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    return f"{value:.{digits}g}"


def _count(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else _number(value, _R_DIGITS)


def _named_values(
    names: Sequence[str], values: Sequence[str], width: int = 72
) -> list[str]:
    """Names over values, right-aligned in columns, as R prints a named
    vector, wrapped at ``width`` characters."""
    cells = [(n, v, max(len(n), len(v))) for n, v in zip(names, values, strict=True)]
    lines: list[str] = []
    row: list[tuple[str, str, int]] = []
    used = 0
    for cell in cells:
        if row and used + cell[2] + 1 > width:
            lines += _cell_lines(row)
            row, used = [], 0
        row.append(cell)
        used += cell[2] + 1
    if row:
        lines += _cell_lines(row)
    return lines


def _cell_lines(row: Sequence[tuple[str, str, int]]) -> list[str]:
    return [
        " ".join(n.rjust(w) for n, _, w in row),
        " ".join(v.rjust(w) for _, v, w in row),
    ]


def _leaf_lines(
    tree: "ModelTree", info: "NodeInfo", digits: int
) -> tuple[str, list[str]]:
    """The text after a leaf's label and the lines below it."""
    head = f": n = {_count(info.nobs)}"
    if info.status == "failed":
        return f"{head} [fit failed: {info.reason}]", []
    lines = _named_values(
        list(info.coefficients),
        [_number(v, digits) for v in info.coefficients.values()],
    )
    extras = info.extras
    notes: list[str] = []
    if "theta" in extras and extras["theta"] is not None:
        notes.append(f"theta = {_number(float(extras['theta']), digits)}")
    separation = extras.get("separation")
    if separation not in (None, "none"):
        notes.append(f"separation: {separation}")
    if info.status == "no_estfun":
        notes.append(f"[no scores: {info.reason}]")
    if notes:
        lines.append(", ".join(notes))
    return head, lines


def _variable_kinds(tree: "ModelTree") -> str:
    parts = []
    for v in tree._variables:
        kind = "numeric, Boolean" if v.boolean else v.kind.value
        parts.append(f"{v.name} ({kind})")
    return ", ".join(parts)


def _size_line(tree: "ModelTree") -> str:
    line = f"n = {tree.n_used}"
    dropped = tree.n_rows - tree.n_used
    if dropped:
        counts = tree._design.missing_counts if tree._design is not None else {}
        detail = ", ".join(f"{name}: {count}" for name, count in counts.items())
        line += f" ({dropped} rows dropped for missing values: {detail})"
    return line


def format_tree(tree: "ModelTree", digits: int = 4) -> str:
    """The tree print of section 3.4."""
    lines = [title(tree), ""]
    if tree.formula is not None:
        lines += ["Model formula:", tree.formula, ""]
    lines.append(f"Partitioning variables: {_variable_kinds(tree)}")
    lines += [_size_line(tree), "", "Fitted tree:"]

    def leaf(info: "NodeInfo", prefix: str, label: str) -> None:
        head, below = _leaf_lines(tree, info, digits)
        lines.append(f"{prefix}{label}{head}")
        # print.partynode indents a terminal panel by four spaces.
        lines.extend(f"{prefix}    {line}" for line in below)

    def visit(info: "NodeInfo", prefix: str) -> None:
        assert info.split is not None
        inner = prefix + "|   "
        for kid_id, condition in zip(info.kids, info.split.conditions(), strict=True):
            kid = tree.node(kid_id)
            label = f"[{kid_id}] {condition}"
            if kid.is_leaf:
                leaf(kid, inner, label)
            else:
                lines.append(f"{inner}{label}")
                visit(kid, inner)

    root = tree.node(1)
    if root.is_leaf:
        leaf(root, "", "[1] root")
    else:
        lines.append("[1] root")
        visit(root, "")
    leaves = [tree.node(i) for i in tree.leaves]
    k = sum(len(i.coefficients) for i in leaves) / len(leaves)
    objective = sum(i.objective for i in leaves)
    label = objective_label(tree)
    label = f" ({label})" if label else ""
    lines += [
        "",
        f"Number of inner nodes:    {len(tree.nodes) - len(leaves)}",
        f"Number of terminal nodes: {len(leaves)}",
        f"Number of parameters per node: {_number(k, _R_DIGITS)}",
        f"Objective function{label}: {_number(objective, _R_DIGITS)}",
    ]
    return "\n".join(lines)


# Summary


def summary_tables(
    tree: "ModelTree",
) -> tuple[dict[str, list[Any]], dict[str, list[Any]], dict[str, list[Any]]]:
    """The variables, nodes and tests tables of ``summary()``."""
    dropped = tree._design.dropped_levels if tree._design is not None else {}
    variables: dict[str, list[Any]] = {
        "variable": [],
        "kind": [],
        "levels": [],
        "dropped_levels": [],
    }
    for v in tree._variables:
        variables["variable"].append(v.name)
        variables["kind"].append(v.kind.value)
        variables["levels"].append(None if v.levels is None else ", ".join(v.levels))
        lost = dropped.get(v.name)
        variables["dropped_levels"].append(None if not lost else ", ".join(lost))

    nodes: dict[str, list[Any]] = {
        key: []
        for key in (
            "node",
            "parent",
            "depth",
            "n_rows",
            "nobs",
            "stop_reason",
            "status",
            "variable",
            "statistic",
            "p_value",
            "split",
        )
    }
    tests: dict[str, list[Any]] = {
        "node": [],
        "variable": [],
        "statistic": [],
        "p_value": [],
    }
    for info in tree.nodes:
        core_test = info.core.test
        best = None if core_test is None else core_test.best
        variable = (
            None if best is None or core_test is None else core_test.variables[best]
        )
        nodes["node"].append(info.id)
        nodes["parent"].append(info.parent)
        nodes["depth"].append(info.depth)
        nodes["n_rows"].append(info.n_rows)
        nodes["nobs"].append(info.nobs)
        nodes["stop_reason"].append(info.stop_reason)
        nodes["status"].append(info.status)
        nodes["variable"].append(variable)
        nodes["statistic"].append(
            None
            if best is None or core_test is None
            else float(core_test.statistic[best])
        )
        nodes["p_value"].append(info.p_value)
        nodes["split"].append(
            None if info.split is None else info.split.conditions()[0]
        )
        if info.tests is not None:
            for name, result in info.tests.items():
                tests["node"].append(info.id)
                tests["variable"].append(name)
                tests["statistic"].append(result.statistic)
                tests["p_value"].append(
                    math.nan if result.p_value is None else result.p_value
                )
    return variables, nodes, tests


def format_summary(tree: "ModelTree", digits: int = 4) -> str:
    """The printed summary: the tree, then per tested node its instability
    tests, the dropped levels and the recorded warnings."""
    lines = [format_tree(tree, digits), "", "Parameter instability tests:"]
    for info in tree.nodes:
        if info.tests is None:
            continue
        lines.append(f"-- Node {info.id} ({info.stop_reason}) --")
        names = list(info.tests)
        width = max(len(n) for n in [*names, "variable"])
        lines.append(f"{'variable'.ljust(width)}  {'statistic':>10}  {'p_value':>10}")
        for name, result in info.tests.items():
            p = "NA" if result.p_value is None else _number(result.p_value, digits)
            lines.append(
                f"{name.ljust(width)}  {_number(result.statistic, digits):>10}  {p:>10}"
            )
    dropped = tree._design.dropped_levels if tree._design is not None else {}
    if dropped:
        lines += ["", "Levels dropped (no used row):"]
        lines += [f"  {name}: {', '.join(levels)}" for name, levels in dropped.items()]
    if tree.warnings:
        lines += ["", "Warnings:"]
        for w in tree.warnings:
            times = "" if w.count == 1 else f" ({w.count} times)"
            lines.append(f"  {w.category.__name__}: {w.message}{times}")
    return "\n".join(lines)
