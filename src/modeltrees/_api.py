"""The public tree class and the entry points (docs/api-design.md, sections 1
and 3).

:class:`ModelTree` wraps the core's :class:`~modeltrees.tree.GrownTree`
and, for the formula entry points, the :class:`~modeltrees._design.Design`
it was grown from. It is immutable: every view (:class:`NodeInfo`,
:class:`SplitRule`, :class:`TestResult`) is built from the stored nodes.
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from modeltrees._design import Design, r_double_label
from modeltrees.instability import VariableKind
from modeltrees.node_model import NodeModel
from modeltrees.split_search import Split
from modeltrees.tree import GrownTree, MobControl, Node

# R prints numeric split points rounded to getOption("digits") - 2 = 5
# decimals (print.partynode, character_split).
_BREAK_DECIMALS = 5


@dataclass(frozen=True)
class RecordedWarning:
    """A warning emitted while the tree was grown, identical messages merged
    (docs/api-design.md, section 7.3)."""

    category: type[Warning]
    message: str
    count: int


@dataclass(frozen=True)
class TestResult:
    """One partitioning variable's instability test at a node: the statistic
    and the adjusted p-value (None where the variable was not tested)."""

    __test__ = False  # not a pytest test class

    statistic: float
    p_value: float | None


@dataclass(frozen=True)
class SplitRule:
    """A node's binary split in labels.

    ``kind`` is the partitioning variable's kind. Numeric: ``threshold`` is
    the break, ``z <= threshold`` goes left. Ordered: ``threshold`` is the
    last level that goes left. Nominal: ``threshold`` is None. For factors
    ``left_levels`` and ``right_levels`` list the levels of each kid, and
    ``absent_levels`` the training levels absent from the node, which
    ``predict`` sends to the larger kid.
    """

    variable: str
    kind: str
    threshold: float | str | None
    left_levels: tuple[str, ...] | None
    right_levels: tuple[str, ...] | None
    absent_levels: tuple[str, ...] = ()
    boolean: bool = False  # a Boolean variable, split into false and true

    def conditions(self) -> tuple[str, str]:
        """The readable conditions of the left and the right kid."""
        if self.boolean:
            return f"{self.variable} = false", f"{self.variable} = true"
        if self.kind == "numeric":
            assert isinstance(self.threshold, float)
            label = r_double_label(round(self.threshold, _BREAK_DECIMALS))
            return f"{self.variable} <= {label}", f"{self.variable} > {label}"
        if self.kind == "ordered":
            return (
                f"{self.variable} <= {self.threshold}",
                f"{self.variable} > {self.threshold}",
            )
        assert self.left_levels is not None and self.right_levels is not None
        return (
            f"{self.variable} in {{{', '.join(self.left_levels)}}}",
            f"{self.variable} in {{{', '.join(self.right_levels)}}}",
        )


@dataclass(frozen=True)
class NodeInfo:
    """A frozen view of one node (docs/api-design.md, section 3.2).

    ``core`` is the core :class:`~modeltrees.tree.Node`, for advanced use;
    it is not part of the stable API.
    """

    id: int
    parent: int | None
    kids: tuple[int, ...]
    depth: int
    is_leaf: bool
    n_rows: int
    nobs: float
    rule: tuple[str, ...]
    split: SplitRule | None
    coefficients: dict[str, float]
    objective: float
    status: str
    stop_reason: str
    reason: str | None
    p_value: float | None
    tests: dict[str, TestResult] | None
    extras: dict[str, Any]
    core: Node


@dataclass(frozen=True)
class TreeSummary:
    """The tables behind ``summary()`` (docs/api-design.md, section 3.4),
    as frames in the tree's backend, and their printed form.

    ``variables``: each partitioning variable with its kind and levels.
    ``nodes``: per node its id, parent, depth, n, nobs, stop reason,
    status, selected variable, statistic, adjusted p-value and split.
    ``tests``: the full instability-test table of every tested node.
    ``coefficients``: the leaf coefficients and model reports, as
    ``coef_table()``. ``dropped_levels`` and ``missing_counts`` describe
    the missing-data removal, ``dropped_rows`` counts the rows it dropped.
    """

    variables: Any
    nodes: Any
    tests: Any
    coefficients: Any
    dropped_levels: dict[str, tuple[str, ...]]
    dropped_rows: int
    missing_counts: dict[str, int]
    warnings: tuple[RecordedWarning, ...]
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class _VariableInfo:
    """A partitioning variable's metadata: its kind and levels as the tree
    used them, and whether it was a Boolean column (numeric 0/1)."""

    name: str
    kind: VariableKind
    levels: tuple[str, ...] | None
    boolean: bool = False


def _readonly(array: NDArray[Any]) -> NDArray[Any]:
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


class ModelTree:
    """A fitted model-based tree (docs/api-design.md, section 3.1).

    Returned by every entry point (:func:`lmtree`, :func:`glmtree`,
    :func:`negbintree`, :func:`safelogittree`, :func:`multinomtree`) and by
    :func:`mob`. Immutable after construction.
    """

    def __init__(
        self,
        grown: GrownTree,
        model: NodeModel,
        *,
        model_name: str,
        formula: str | None,
        design: Design | None,
        response_levels: tuple[str, ...] | None,
        warnings: tuple[RecordedWarning, ...],
        used_rows: NDArray[np.bool_],
        x: NDArray[np.float64] | None,
        offset: NDArray[np.float64] | None,
        boolean: frozenset[str] = frozenset(),
        backend: str | None = None,
    ) -> None:
        self._grown = grown
        self._model = model
        self._model_name = model_name
        self._formula = formula
        self._design = design
        self._response_levels = response_levels
        self._warnings = warnings
        self._used_rows = _readonly(np.asarray(used_rows, dtype=np.bool_))
        fitted = np.zeros(self._used_rows.shape[0], dtype=np.int64)
        fitted[self._used_rows] = grown.fitted_node
        self._fitted_node = _readonly(fitted)
        self._x = x
        self._offset = offset
        self._variables = tuple(
            _VariableInfo(v.name, v.kind, v.levels, v.name in boolean)
            for v in grown.variables
        )
        self._backend = backend
        self._frozen = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("ModelTree is immutable")
        object.__setattr__(self, name, value)

    # Settings and data

    @property
    def formula(self) -> str | None:
        """The expanded formula, e.g. ``"y ~ 1 | z1 + z2"``; None from mob()."""
        return self._formula

    @property
    def model_name(self) -> str:
        """``"lmtree"``, ``"glmtree(binomial)"``, ``"negbintree"``, ..."""
        return self._model_name

    @property
    def control(self) -> MobControl:
        return self._grown.control

    @property
    def model(self) -> NodeModel:
        """The node model the tree was grown with."""
        return self._model

    @property
    def n_rows(self) -> int:
        """Rows in the input data."""
        return int(self._used_rows.shape[0])

    @property
    def n_used(self) -> int:
        """Rows left after missing-data removal."""
        return int(self._used_rows.sum())

    @property
    def used_rows(self) -> NDArray[np.bool_]:
        """Mask over the input rows: True for the rows the tree used."""
        return self._used_rows

    @property
    def fitted_node(self) -> NDArray[np.int64]:
        """The leaf id of each input row, 0 for rows dropped for missing values."""
        return self._fitted_node

    @property
    def response_levels(self) -> tuple[str, ...] | None:
        """The level order of a categorical response (binomial: the first
        level is failure; multinomial: the first is the reference), else None."""
        return self._response_levels

    @property
    def warnings(self) -> tuple[RecordedWarning, ...]:
        """The warnings emitted while the tree was grown, merged."""
        return self._warnings

    # Nodes

    @cached_property
    def _node_infos(self) -> tuple[NodeInfo, ...]:
        infos: list[NodeInfo] = []
        self._collect(self._grown.root, None, (), infos)
        infos.sort(key=lambda info: info.id)
        return tuple(infos)

    def _collect(
        self,
        node: Node,
        parent: int | None,
        rule: tuple[str, ...],
        out: list[NodeInfo],
    ) -> None:
        split = None if node.split is None else self._split_rule(node)
        out.append(self._node_info(node, parent, rule, split))
        if split is not None:
            for kid, condition in zip(node.kids, split.conditions(), strict=True):
                self._collect(kid, node.id, (*rule, condition), out)

    def _split_rule(self, node: Node) -> SplitRule:
        split: Split | None = node.split
        assert split is not None
        var = self._variables[split.variable]
        if split.kind is VariableKind.NUMERIC:
            assert split.breaks is not None
            return SplitRule(
                var.name, "numeric", float(split.breaks), None, None, (), var.boolean
            )
        assert var.levels is not None
        if split.kind is VariableKind.ORDERED:
            assert split.breaks is not None
            b = int(split.breaks)
            return SplitRule(
                var.name, "ordered", var.levels[b - 1], var.levels[:b], var.levels[b:]
            )
        assert split.index is not None
        sides: dict[int | None, list[str]] = {1: [], 2: [], None: []}
        for level, kid in zip(var.levels, split.index, strict=True):
            sides[kid].append(level)
        return SplitRule(
            var.name,
            "nominal",
            None,
            tuple(sides[1]),
            tuple(sides[2]),
            tuple(sides[None]),
        )

    def _node_info(
        self,
        node: Node,
        parent: int | None,
        rule: tuple[str, ...],
        split: SplitRule | None,
    ) -> NodeInfo:
        fit = node.fit
        tests = None
        if node.test is not None:
            tests = {
                name: TestResult(float(stat), None if math.isnan(p) else float(p))
                for name, stat, p in zip(
                    node.test.variables,
                    node.test.statistic,
                    node.test.p_value_adjusted,
                    strict=True,
                )
            }
        return NodeInfo(
            id=node.id,
            parent=parent,
            kids=tuple(kid.id for kid in node.kids),
            depth=node.depth,
            is_leaf=node.terminal,
            n_rows=int(node.rows.shape[0]),
            nobs=float(node.nobs),
            rule=rule,
            split=split,
            coefficients={
                name: float(c)
                for name, c in zip(fit.coef_names, fit.coefficients, strict=True)
            },
            objective=float(fit.objfun),
            status=fit.status.value,
            stop_reason=node.stop_reason.value,
            reason=fit.reason,
            p_value=node.p_value,
            tests=tests,
            extras=self._extras(node),
            core=node,
        )

    def _extras(self, node: Node) -> dict[str, Any]:
        """The model report (section 3.5), plus the fit's warning texts."""
        report = getattr(self._model, "report", None)
        extras: dict[str, Any] = dict(report(node.fit)) if callable(report) else {}
        messages = node.fit.extras.get("warnings")
        if messages is not None:
            extras["warnings"] = tuple(messages)
        return extras

    def node(self, node_id: int) -> NodeInfo:
        """The node with id ``node_id`` (1-based, preorder, as partykit)."""
        infos = self._node_infos
        if not isinstance(node_id, (int, np.integer)) or not (
            1 <= node_id <= len(infos)
        ):
            raise KeyError(f"no node {node_id!r}; ids are 1 to {len(infos)}")
        return infos[int(node_id) - 1]

    @property
    def nodes(self) -> tuple[NodeInfo, ...]:
        """Every node, in id (preorder) order."""
        return self._node_infos

    @property
    def leaves(self) -> tuple[int, ...]:
        """The ids of the terminal nodes."""
        return tuple(info.id for info in self._node_infos if info.is_leaf)

    @property
    def depth(self) -> int:
        """The largest node depth, the root at depth 1 (so ``maxdepth``
        bounds it; partykit's ``depth()`` is one less)."""
        return max(info.depth for info in self._node_infos)

    def __repr__(self) -> str:
        n_inner = len(self._node_infos) - len(self.leaves)
        what = self._formula if self._formula is not None else "mob()"
        return (
            f"<ModelTree {self._model_name}: {what}, {n_inner} inner and "
            f"{len(self.leaves)} terminal nodes>"
        )


def _boolean_variables(design: Design) -> frozenset[str]:
    """Partitioning variables that come from Boolean columns."""
    return frozenset(
        t.name
        for t in design.partitioning_terms
        if t.storage == "boolean" and t.kind == "numeric"
    )
