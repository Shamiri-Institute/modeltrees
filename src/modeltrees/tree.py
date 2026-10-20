"""Model-based recursive partitioning: growing and pruning the tree.

Port of ``mob_grow`` and ``mob_prune`` from partykit 1.3-0
(reference/partykit/R/modelparty.R:516-676), with the ``mob_control()``
options listed in docs/tree-growing.md. Line numbers below refer to
modelparty.R. The input is numeric arrays: a response, an optional regressor
matrix and a dict of partitioning variables with declared kinds. There is no
formula or dataframe layer here.
"""

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from modeltrees._reference import reference_active
from modeltrees.instability import (
    FluctuationTests,
    InstabilityTestError,
    PartitioningVariable,
    VariableKind,
    fluctuation_tests,
)
from modeltrees.node_model import FitResult, NodeModel
from modeltrees.pruning import PruneRule, mob_prune_rule
from modeltrees.split_search import (
    BatchObjectives,
    SideFit,
    Split,
    SplitObjectives,
    SplitSearch,
    find_split,
    find_split_reference,
    kid_ids,
    w2n,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


class StopReason(Enum):
    SPLIT = "split"
    MINSIZE = "minsize"  # fewer than 2 * minsize observations
    MAXDEPTH = "maxdepth"
    TEST_FAILED = "test_failed"  # the instability tests raised an error
    NOT_SIGNIFICANT = "not_significant"  # adjusted p-value above alpha
    NO_ADMISSIBLE_SPLIT = "no_admissible_split"
    PRUNED = "pruned"  # split found, then removed by post-pruning


@dataclass(frozen=True)
class MobControl:
    """The ``mob_control()`` settings that are ported (modelparty.R:726-778).

    ``minsize = None`` gives ``ceiling(10 * k / NCOL(y))`` with ``k`` the
    root's coefficient count, set once at the root. ``prune`` is "aic",
    "bic", "none", a :data:`~modeltrees.pruning.PruneRule` or None. Strings
    take the model's rules if it has a ``prune_rule(name)`` method (lmtree's
    RSS rules for LmModel), otherwise mob()'s log-likelihood rules.

    The remaining fields exist so that non-default values fail loudly:
    they need R's random numbers or are out of the v0.1 scope
    (docs/decisions.md) and raise NotImplementedError.
    """

    alpha: float = 0.05
    bonferroni: bool = True
    minsize: float | None = None
    maxdepth: float = math.inf
    trim: float = 0.1
    dfsplit: bool = True
    prune: str | PruneRule | None = None
    restart: bool = True
    caseweights: bool = True
    numsplit: Literal["left", "center"] = "left"
    catsplit: Literal["binary", "multiway"] = "binary"
    mtry: float = math.inf
    breakties: bool = False
    ordinal: str = "chisq"
    vcov: str = "opg"
    parm: Sequence[int] | Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.mtry < math.inf and self.mtry >= 1:
            raise NotImplementedError("mtry < Inf needs R's random numbers")
        if self.breakties:
            raise NotImplementedError("breakties = TRUE needs R's random numbers")
        if self.ordinal.lower() != "chisq":
            raise NotImplementedError(
                f'ordinal = "{self.ordinal}" needs strucchange or mvtnorm '
                "simulation; only the chi-squared test is ported"
            )
        if self.vcov.lower() != "opg":
            raise NotImplementedError(
                f'vcov = "{self.vcov}" is not in the v0.1 scope; only "opg"'
            )
        if self.parm is not None:
            raise NotImplementedError("parm is not ported")
        if self.catsplit == "multiway":
            raise NotImplementedError("catsplit = 'multiway' is not ported")
        if self.catsplit != "binary":
            raise ValueError(f"unknown catsplit {self.catsplit!r}")
        if self.numsplit not in ("left", "center"):
            raise ValueError(f"unknown numsplit {self.numsplit!r}")


@dataclass(frozen=True)
class Variable:
    """A partitioning variable over the full data.

    Numeric values are floats. Factor values are 0-based codes into
    ``levels``, in R's level order (the order of an ordered factor).
    """

    name: str
    kind: VariableKind
    values: NDArray[np.generic]
    levels: tuple[str, ...] | None = None


@dataclass
class Node:
    id: int  # 1-based, preorder, as in partykit
    depth: int  # root is 1
    rows: IntArray  # row indices into the full data
    nobs: float  # w2n(weights)
    fit: FitResult
    stop_reason: StopReason
    test: FluctuationTests | None = None  # None if not run or failed
    test_failed: bool = False
    p_value: float | None = None  # adjusted p-value of the selected variable
    split_search: SplitSearch | None = None
    split: Split | None = None
    kids: list["Node"] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return not self.kids

    def walk(self) -> Iterator["Node"]:
        """The subtree in preorder, which is node id order."""
        yield self
        for kid in self.kids:
            yield from kid.walk()


@dataclass
class ModelTree:
    root: Node
    variables: tuple[Variable, ...]
    control: MobControl
    minsize: int  # as used by the tree, after the root default
    fitted_node: IntArray  # terminal node id per row

    @property
    def nodes(self) -> list[Node]:
        return list(self.root.walk())

    def node(self, node_id: int) -> Node:
        return self.nodes[node_id - 1]


_KINDS = {k.value: k for k in VariableKind}


def _as_variable(name: str, spec: Any, n: int) -> Variable:
    """A Variable from a Variable, a dict or a (kind, values[, levels]) tuple."""
    if isinstance(spec, Variable):
        kind, values, levels = spec.kind, spec.values, spec.levels
    elif isinstance(spec, Mapping):
        kind, values, levels = spec["kind"], spec["values"], spec.get("levels")
    else:
        kind, values, *rest = spec
        levels = rest[0] if rest else None
    if isinstance(kind, str):
        if kind not in _KINDS:
            raise ValueError(f"{name}: kind must be one of {sorted(_KINDS)}")
        kind = _KINDS[kind]
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{name}: expected {n} values, got shape {arr.shape}")
    if kind is VariableKind.NUMERIC:
        num = arr.astype(np.float64)
        if np.any(np.isnan(num)):
            raise ValueError(f"{name}: missing values are not supported")
        return Variable(name, kind, num, None)
    if levels is not None and arr.dtype.kind in "iu":
        codes = arr.astype(np.intp)
        level_tuple = tuple(str(v) for v in levels)
        if codes.size and (codes.min() < 0 or codes.max() >= len(level_tuple)):
            raise ValueError(f"{name}: codes must index into the levels")
        return Variable(name, kind, codes, level_tuple)
    # Labels: map through the given levels, or sorted unique values (R sorts
    # by locale, so pass levels when the order matters).
    if arr.dtype.kind == "f" and np.any(np.isnan(arr)):
        raise ValueError(f"{name}: missing values are not supported")
    order = list(levels) if levels is not None else list(np.unique(arr))
    index = {v: i for i, v in enumerate(order)}
    try:
        codes = np.array([index[v] for v in arr.tolist()], dtype=np.intp)
    except KeyError as exc:
        raise ValueError(f"{name}: value {exc.args[0]!r} is not a level") from exc
    return Variable(name, kind, codes, tuple(str(v) for v in order))


def mob(
    model: NodeModel,
    y: ArrayLike,
    x: ArrayLike | None,
    z: Mapping[str, Any],
    *,
    weights: ArrayLike | None = None,
    offset: ArrayLike | None = None,
    cluster: ArrayLike | None = None,
    control: MobControl | None = None,
) -> ModelTree:
    """Grow (and optionally prune) a model-based tree, as ``partykit::mob``.

    ``z`` maps each partitioning variable's name to its kind ("numeric",
    "ordered" or "nominal") and values: a :class:`Variable`, a dict with
    ``kind``, ``values`` and optionally ``levels``, or a tuple
    ``(kind, values)`` or ``(kind, values, levels)``. Factors given as integer
    codes need ``levels``. ``x`` is the regressor matrix, intercept column
    included, or None for an intercept-only model.
    """
    control = control or MobControl()
    yv = np.asarray(y)
    n = yv.shape[0]
    xv = None if x is None else np.asarray(x, dtype=np.float64)
    if xv is not None and xv.shape[0] != n:
        raise ValueError(f"x has {xv.shape[0]} rows but y has {n}")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights must have {n} values")
    if np.any(w < 0) or np.any(np.isnan(w)):
        raise ValueError("weights must be non-negative")
    off = None if offset is None else np.asarray(offset, dtype=np.float64)
    if off is not None and off.shape != (n,):
        raise ValueError(f"offset must have {n} values")
    cl = None if cluster is None else np.asarray(cluster)
    if cl is not None and cl.shape != (n,):
        raise ValueError(f"cluster must have {n} values")
    if not z:
        raise ValueError("at least one partitioning variable is needed")
    variables = tuple(_as_variable(name, spec, n) for name, spec in z.items())

    grower = _Grower(model, yv, xv, variables, w, off, cl, control)
    root = grower.grow(np.arange(n, dtype=np.intp), node_id=1, depth=1)
    assert grower.minsize is not None

    rule = _prune_rule(control.prune, model)
    if rule is not None:
        k = len(root.fit.coefficients)
        # R's n is nrow(Z). Rows with zero weight are not counted, so that
        # zero weights act as deleted rows (docs/decisions.md); without zero
        # weights this is R's n.
        n_full = float(np.sum(w > 0))
        _prune(root, rule, k, control.dfsplit, n_full)
        _renumber(root)

    fitted = np.zeros(n, dtype=np.intp)
    for node in root.walk():
        if node.terminal:
            fitted[node.rows] = node.id
    return ModelTree(root, variables, control, grower.minsize, fitted)


def _prune_rule(prune: str | PruneRule | None, model: NodeModel) -> PruneRule | None:
    if prune is None:
        return None
    if isinstance(prune, str):
        own = getattr(model, "prune_rule", None)
        rule: PruneRule | None = own(prune) if callable(own) else mob_prune_rule(prune)
        return rule
    return prune


class _Grower:
    """State shared by the recursion, as the closures in mob_partynode."""

    def __init__(
        self,
        model: NodeModel,
        y: NDArray[Any],
        x: FloatArray | None,
        variables: tuple[Variable, ...],
        weights: FloatArray,
        offset: FloatArray | None,
        cluster: NDArray[Any] | None,
        control: MobControl,
    ) -> None:
        self.model = model
        self.y = y
        self.x = x
        self.variables = variables
        self.weights = weights
        self.offset = offset
        self.cluster = cluster
        self.control = control
        # mob_partynode coerces minsize with as.integer (:167-168).
        self.minsize: int | None = (
            None if control.minsize is None else int(control.minsize)
        )

    def fit(
        self, rows: IntArray, *, start: FloatArray | None = None, estfun: bool = False
    ) -> FitResult:
        return self.model.fit(
            self.y[rows],
            None if self.x is None else self.x[rows],
            start=start,
            weights=self.weights[rows],
            offset=None if self.offset is None else self.offset[rows],
            cluster=None if self.cluster is None else self.cluster[rows],
            caseweights=self.control.caseweights,
            estfun=estfun,
            vcov=False,
        )

    def _node_fitters(self, rows: IntArray) -> tuple[SideFit, BatchObjectives | None]:
        """Side fits on the node's rows, extracted once per node.

        Masking the node's contiguous arrays selects the same rows in the
        same order as indexing the full data with ``rows[mask]``, so the
        fits are those of the reference path, without a fancy-indexed copy
        of the full data per candidate. The batch scorer is the model's
        ``split_objectives``, if it has one (docs/model-interface.md).
        """
        y = self.y[rows]
        x = None if self.x is None else self.x[rows]
        w = self.weights[rows]
        off = None if self.offset is None else self.offset[rows]
        cl = None if self.cluster is None else self.cluster[rows]
        caseweights = self.control.caseweights
        model = self.model

        def side_fit(mask: NDArray[np.bool_], start: FloatArray | None) -> FitResult:
            return model.fit(
                y[mask],
                None if x is None else x[mask],
                start=start,
                weights=w[mask],
                offset=None if off is None else off[mask],
                cluster=None if cl is None else cl[mask],
                caseweights=caseweights,
                estfun=False,
                vcov=False,
            )

        scorer = getattr(model, "split_objectives", None)
        if not callable(scorer):
            return side_fit, None

        def batch(order: IntArray, cuts: IntArray) -> SplitObjectives | None:
            assert callable(scorer)
            result: SplitObjectives | None = scorer(
                y,
                x,
                weights=w,
                offset=off,
                cluster=cl,
                caseweights=caseweights,
                order=order,
                cuts=cuts,
            )
            return result

        return side_fit, batch

    def grow(self, rows: IntArray, node_id: int, depth: int) -> Node:
        control = self.control
        w = self.weights[rows]
        mod = self.fit(rows, estfun=True)
        nobs = w2n(w, control.caseweights)

        # Default minsize, assigned once at the root with <<- (:533).
        if self.minsize is None:
            ncol = self.model.n_response_columns(self.y)
            self.minsize = math.ceil(10 * len(mod.coefficients) / ncol)
        minsize = self.minsize

        def terminal(reason: StopReason, **kw: Any) -> Node:
            return Node(node_id, depth, rows, nobs, mod, reason, **kw)

        # Stopping rules (:536-547). minsize is checked first, so a node that
        # meets both is recorded as stopped by minsize.
        if nobs < 2 * minsize:
            return terminal(StopReason.MINSIZE)
        if depth >= control.maxdepth:
            return terminal(StopReason.MAXDEPTH)

        # Instability tests (:550). partykit wraps them in try(); a failure
        # makes the node terminal (:569-573).
        if mod.estfun is None:
            return terminal(StopReason.TEST_FAILED, test_failed=True)
        z_node = [
            PartitioningVariable(v.name, v.kind, v.values[rows]) for v in self.variables
        ]
        try:
            test = fluctuation_tests(
                mod.estfun,
                z_node,
                w,
                minsize=minsize,
                trim=control.trim,
                bonferroni=control.bonferroni,
                cluster=None if self.cluster is None else self.cluster[rows],
                caseweights=control.caseweights,
            )
        except InstabilityTestError:
            return terminal(StopReason.TEST_FAILED, test_failed=True)

        best = test.best
        if best is None:
            return terminal(StopReason.NOT_SIGNIFICANT, test=test)
        p_value = float(test.p_value_adjusted[best])
        if p_value > control.alpha:
            return terminal(StopReason.NOT_SIGNIFICANT, test=test, p_value=p_value)

        var = self.variables[best]
        if reference_active():

            def side_fit(
                mask: NDArray[np.bool_], start: FloatArray | None
            ) -> FitResult:
                return self.fit(rows[mask], start=start)

            search = find_split_reference(
                best,
                var.kind,
                var.values[rows],
                var.levels,
                w,
                side_fit,
                minsize=minsize,
                caseweights=control.caseweights,
                restart=control.restart,
                numsplit=control.numsplit,
            )
        else:
            node_fit, batch = self._node_fitters(rows)
            search = find_split(
                best,
                var.kind,
                var.values[rows],
                var.levels,
                w,
                node_fit,
                minsize=minsize,
                caseweights=control.caseweights,
                restart=control.restart,
                numsplit=control.numsplit,
                batch=batch,
            )
        if search.split is None:
            return terminal(
                StopReason.NO_ADMISSIBLE_SPLIT,
                test=test,
                p_value=p_value,
                split_search=search,
            )

        kids_of_rows = _route(search.split, var.values[rows], w, control.caseweights)
        node = Node(
            node_id,
            depth,
            rows,
            nobs,
            mod,
            StopReason.SPLIT,
            test=test,
            p_value=p_value,
            split_search=search,
            split=search.split,
        )
        # Preorder ids: kid 1 gets id + 1, each later kid one more than the
        # largest id in the previous kid's subtree (:600-615).
        next_id = node_id + 1
        for kid in (1, 2):
            child = self.grow(rows[kids_of_rows == kid], next_id, depth + 1)
            node.kids.append(child)
            next_id = max(n.id for n in child.walk()) + 1
        return node


def _route(
    split: Split, z: NDArray[np.generic], weights: FloatArray, caseweights: bool
) -> IntArray:
    """Kid of each row of a node.

    A nominal level without a kid can only occur in rows with zero weight,
    since the split's levels come from the positive-weight rows. R would
    give NA and route such a row at random (node.R:189-194); here it goes
    to the kid with more observations, kid 1 on a tie.
    """
    kids = kid_ids(split, z)
    missing = kids == 0
    if np.any(missing):
        n1 = w2n(weights[kids == 1], caseweights)
        n2 = w2n(weights[kids == 2], caseweights)
        kids[missing] = 1 if n1 >= n2 else 2
    return kids


def _prune(node: Node, rule: PruneRule, k: int, dfsplit: bool, n: float) -> None:
    """Post-pruning, as ``mob_prune`` (:625-676).

    partykit repeatedly checks every inner node whose kids are all
    terminal and prunes those the rule accepts. Each check depends only on
    the node and its kids, so pruning the kids first and then the node gives
    the same tree.
    """
    for kid in node.kids:
        _prune(kid, rule, k, dfsplit, n)
    if not node.kids or not all(kid.terminal for kid in node.kids):
        return
    objfun = (node.fit.objfun, sum(kid.fit.objfun for kid in node.kids))
    # k is the root's coefficient count, aliased ones included (:647).
    df = (k, len(node.kids) * k + int(dfsplit))
    if rule(objfun, df, (node.nobs, n)):
        node.kids = []
        node.split = None
        node.stop_reason = StopReason.PRUNED


def _renumber(root: Node) -> None:
    """Preorder ids after pruning, as partykit relabels them."""
    for i, node in enumerate(root.walk(), start=1):
        node.id = i
