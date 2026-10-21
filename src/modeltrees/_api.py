"""The public tree class and the entry points (docs/api-design.md, sections 1
and 3).

:class:`ModelTree` wraps the core's :class:`~modeltrees.tree.GrownTree`
and, for the formula entry points, the :class:`~modeltrees._design.Design`
it was grown from. It is immutable: every view (:class:`NodeInfo`,
:class:`SplitRule`, :class:`TestResult`) is built from the stored nodes.
"""

import math
import warnings as _warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, TypedDict, Unpack

import numpy as np
from numpy.typing import ArrayLike, NDArray

from modeltrees._design import (
    ColumnSpec,
    Design,
    NaAction,
    build_design,
    r_double_label,
)
from modeltrees._frame import DataInput, Frame, Kind, array_column
from modeltrees._loglik import LogLik, tree_loglik
from modeltrees._predict import (
    PredictType,
    Unseen,
    check_type,
    code_arrays,
    code_newdata,
    predict_leaves,
    raise_unseen,
    route,
)
from modeltrees._report import (
    Backend,
    coef_columns,
    format_summary,
    format_tree,
    make_table,
    resolve_backend,
    summary_tables,
)
from modeltrees._response import CodedResponse, ResponseRule, code_response
from modeltrees._vcov import LeafVcov, VcovType, expand, sandwich
from modeltrees.errors import (
    DataError,
    GlmFitWarning,
    ModelTreesWarning,
    NodeFitFailedWarning,
    TooFewClustersWarning,
)
from modeltrees.instability import VariableKind
from modeltrees.models.glm import GlmModel
from modeltrees.models.lm import LmModel
from modeltrees.models.multinom import MultinomModel
from modeltrees.models.negbin import NegBinModel
from modeltrees.models.safelogit import SafeLogitModel, diverging_coefficient
from modeltrees.node_model import FitStatus, NodeModel
from modeltrees.pruning import PruneRule
from modeltrees.split_search import Split
from modeltrees.tree import GrownTree, MobControl, Node
from modeltrees.tree import mob as _grow

# R prints numeric split points rounded to getOption("digits") - 2 = 5
# decimals (print.partynode, character_split).
_BREAK_DECIMALS = 5
# vcov() warns when a node's refit coefficients differ from the stored ones
# by more than this, relative (docs/api-design.md, section 5.2).
_REFIT_RTOL = 1e-10


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
        y: NDArray[Any] | None = None,
        weights: NDArray[np.float64] | None = None,
        cluster: NDArray[Any] | None = None,
        frame: Frame | None = None,
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
        # What vcov() refits from, over the used rows: the coded response,
        # the weights and the tree's clusters; and the training data, for a
        # cluster column given by name.
        self._y = y
        self._weights = weights
        self._cluster = cluster
        self._frame = frame
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

    # Prediction

    def predict(
        self,
        newdata: DataInput | None = None,
        *,
        type: PredictType = "response",
        offset: ColumnSpec = None,
        unseen: Unseen = "error",
        x: ArrayLike | None = None,
        z: Mapping[str, ArrayLike] | None = None,
    ) -> NDArray[Any]:
        """Predictions as a numpy array (docs/api-design.md, section 4.2).

        ``type``: "node" (the leaf id per row, int64, 0 for no node),
        "link" (the linear predictor), "response" (its inverse link),
        "prob" and "class" (multinomial: the probability matrix with
        columns in ``response_levels`` order, and the most probable label
        as an object array). For a multinomial tree "link" is the
        ``(n, K - 1)`` matrix of the non-reference classes and "response"
        is undefined. Rows without a prediction are NaN (None for
        "class").

        ``newdata`` is coded with the training levels; None predicts the
        training rows (rows dropped for missing values get none). A row
        stops at the first split whose variable it lacks (node 0). A level
        absent from the node at a split goes to the larger kid. A level
        never seen in training raises :class:`UnseenLevelError`, or with
        ``unseen="null"`` gives the row no prediction. ``offset`` (a column
        name or an array) overrides the offset of new data, which is
        otherwise read from the training offset's column. Trees grown by
        :func:`mob` take ``z`` (and ``x``, ``offset``) instead of
        ``newdata``.
        """
        check_type(type, unseen)
        needs_x = type != "node"
        if self._design is None:
            if newdata is not None:
                raise TypeError(
                    "a tree grown by mob() predicts from x and z, not newdata"
                )
            if z is None and (x is not None or offset is not None):
                raise TypeError("pass z, the partitioning variables, with x or offset")
            coded = None
            if z is not None:
                if offset is not None and isinstance(offset, str):
                    raise TypeError("offset must be an array for a tree grown by mob()")
                coded = code_arrays(
                    self._grown.variables,
                    z,
                    x,
                    offset,
                    has_x=self._x is not None,
                    has_offset=self._offset is not None,
                    needs_x=needs_x,
                )
        else:
            if x is not None or z is not None:
                raise TypeError("x and z are for trees grown by mob(); pass newdata")
            if newdata is None and offset is not None:
                raise TypeError("offset applies to newdata; pass the data as newdata")
            coded = (
                None
                if newdata is None
                else code_newdata(self._design, newdata, offset=offset, needs_x=needs_x)
            )

        if coded is None:
            # The training rows.
            nodes = np.array(self._fitted_node)
            no_prediction = ~self._used_rows
            xm, off = self._training_x() if needs_x else (None, None)
        else:
            if coded.unseen.any() and unseen == "error":
                raise_unseen(coded.unseen_levels)
            nodes = route(self._grown.root, coded.z, coded.n)
            nodes[coded.unseen] = 0
            no_prediction = coded.x_missing | coded.unseen
            xm, off = coded.x, coded.offset
        if type == "node":
            return nodes
        assert xm is not None
        leaves = {node.id: node for node in self._grown.root.walk() if node.terminal}
        kind = type
        values = predict_leaves(
            self._model,
            leaves,
            nodes,
            xm,
            off,
            no_prediction | (nodes == 0),
            kind,
            self._grown.root,
        )
        if type != "class":
            return values
        levels = self._response_levels or tuple(
            str(v) for v in getattr(self._model, "levels", ())
        )
        labels = np.full(values.shape[0], None, dtype=object)
        defined = ~np.isnan(values)
        labels[defined] = [levels[int(c)] for c in values[defined]]
        return labels

    def _training_x(self) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """The training regressors and offset over the input rows, NaN for
        dropped rows (an intercept column when the model had none)."""
        used = self._used_rows
        x = self._x if self._x is not None else np.ones((int(used.sum()), 1))
        xm = np.full((used.shape[0], x.shape[1]), np.nan)
        xm[used] = x
        off = None
        if self._offset is not None:
            off = np.full(used.shape[0], np.nan)
            off[used] = self._offset
        return xm, off

    # Tables, likelihood and printing

    def coef_table(
        self,
        *,
        layout: Literal["wide", "long"] = "wide",
        se: Literal["robust"] | None = None,
        backend: Backend = None,
    ) -> Any:
        """The leaf coefficients, one row per leaf (section 3.3).

        ``layout="wide"``: ``node``, ``n_rows``, ``nobs``, ``status``,
        ``stop_reason``, one column per coefficient (NaN for aliased or
        unidentified ones), with ``se="robust"`` one ``std_error_<term>``
        column per coefficient, then the model's report columns.
        ``layout="long"``: ``node``, ``term``, ``estimate``, and with
        ``se="robust"`` ``std_error``. The standard errors are those of
        :meth:`vcov` with its defaults (the tree's clusters, HC1 with
        cadjust), NaN where the covariance is not "ok". The frame is in the
        backend of the data the tree was grown from, polars for arrays (a
        dict of numpy arrays without polars), or ``backend``: "polars",
        "pandas", "pyarrow" or "numpy".

        No p-values or confidence intervals are given: the leaves were
        selected by tests on the same data, so standard errors are
        conditional on the tree and too small for inference about the
        subgroups (section 5.4). Grow the tree on one part of the data and
        estimate on the other for honest inference.
        """
        if se not in (None, "robust"):
            raise ValueError(f"se must be None or 'robust', got {se!r}")
        if layout not in ("wide", "long"):
            raise ValueError(f"layout must be 'wide' or 'long', got {layout!r}")
        std_errors = None
        if se == "robust":
            std_errors = {
                i: np.sqrt(np.diag(v.matrix))
                for i, v in self._vcov(None, None, "HC1", True, stacklevel=4).items()
            }
        return make_table(
            coef_columns(self, layout, std_errors),
            resolve_backend(backend, self._backend),
        )

    def vcov(
        self,
        node_id: int | None = None,
        *,
        cluster: str | ArrayLike | bool | None = None,
        type: VcovType = "HC1",
        cadjust: bool = True,
    ) -> dict[int, LeafVcov]:
        """Cluster-robust covariances of node coefficients (section 5).

        ``V = H^-1 (c sum_g s_g s_g') H^-1``, with ``H`` the node model's
        Hessian and ``s_g`` its scores summed within cluster ``g``, from a
        refit of each requested node on its rows. ``c`` is ``G / (G - 1)``
        with ``cadjust`` and also ``(n - 1) / (n - k)`` with ``type="HC1"``,
        G the clusters with positive weight in the node, n its ``nobs`` and
        k its estimated coefficients. Without clusters each row is its own
        cluster; under case weights that is the sandwich of the data with
        each row repeated ``w_i`` times.

        ``node_id`` None gives every leaf; an inner node is allowed too.
        ``cluster`` None uses the tree's clusters if it was grown with them,
        else none; False forces no clusters; True requires the tree's; a
        column name (of the training data) or an array over the input rows
        gives other clusters, for example a coarser level than the tests
        used. They must not be missing on the rows the tree used.

        The default, HC1 with ``cadjust``, is the same for every model and
        equals statsmodels' ``cov_type="cluster"``. R's
        ``sandwich::vcovCL`` defaults to HC1 only for ``lm`` fits and to HC0
        for every other model, glm included: to compare a glmtree,
        negbintree, safelogittree or multinomtree leaf with R, pass
        ``type="HC0"``.

        Each :class:`LeafVcov` has status "ok", "singular" (the Hessian
        scaled by its diagonal is not positive definite, or has a condition
        number above 1 / eps) or "unavailable" (the fit failed or has no
        scores, as a completely separated safe logit or a multinomial leaf
        without the reference class, or a single cluster). With ``G <= k``
        clusters the meat has rank below k and a
        :class:`TooFewClustersWarning` names the node. Negative binomial
        covariances are conditional on theta, as ``vcov(glm.nb)``.

        The leaves were selected by tests on the same data, so these
        covariances are conditional on the tree and too small for
        inference about the subgroups (section 5.4); grow on one part of
        the data and estimate on the other for honest inference.
        """
        if type not in ("HC0", "HC1"):
            raise ValueError(f"type must be 'HC0' or 'HC1', got {type!r}")
        return self._vcov(node_id, cluster, type, bool(cadjust), stacklevel=3)

    def _vcov(
        self,
        node_id: int | None,
        cluster: str | ArrayLike | bool | None,
        type: VcovType,
        cadjust: bool,
        *,
        stacklevel: int,
    ) -> dict[int, LeafVcov]:
        if self._y is None or self._weights is None:
            raise TypeError("this tree keeps no training data to refit its nodes")
        ids = self.leaves if node_id is None else (self.node(node_id).id,)
        clusters = self._vcov_cluster(cluster)
        return {
            i: self._node_vcov(i, clusters, type, cadjust, stacklevel + 1) for i in ids
        }

    def _vcov_cluster(
        self, cluster: str | ArrayLike | bool | None
    ) -> NDArray[Any] | None:
        """The clusters over the used rows, or None (section 5.1)."""
        if cluster is None:
            return self._cluster
        if cluster is False:
            return None
        if cluster is True:
            if self._cluster is None:
                raise ValueError(
                    "cluster=True, but the tree was grown without clusters"
                )
            return self._cluster
        if isinstance(cluster, str):
            if self._frame is None:
                raise TypeError(
                    "a tree grown by mob() takes other clusters as an array, "
                    "not a column name"
                )
            column = self._frame.column(cluster)
            name = repr(cluster)
        else:
            column = array_column("(cluster)", cluster)
            name = "(cluster)"
            if column.values.shape[0] != self.n_rows:
                raise DataError(
                    f"cluster has {column.values.shape[0]} values but the data has "
                    f"{self.n_rows} rows"
                )
        missing = column.missing & self._used_rows
        if missing.any():
            raise DataError(
                f"cluster {name} is missing on {int(missing.sum())} of the rows "
                "the tree used"
            )
        return column.values[self._used_rows]

    def _node_vcov(
        self,
        node_id: int,
        clusters: NDArray[Any] | None,
        type: VcovType,
        cadjust: bool,
        stacklevel: int,
    ) -> LeafVcov:
        node = self._grown.node(node_id)
        assert self._y is not None and self._weights is not None
        rows = node.rows
        weights = self._weights[rows]
        with _warnings.catch_warnings():
            # The node's fit warned while the tree was grown, and the
            # warnings were recorded then.
            _warnings.simplefilter("ignore", GlmFitWarning)
            fit = self._model.fit(
                self._y[rows],
                None if self._x is None else self._x[rows],
                start=None,
                weights=weights,
                offset=None if self._offset is None else self._offset[rows],
                cluster=None if self._cluster is None else self._cluster[rows],
                caseweights=self.control.caseweights,
                estfun=True,
                vcov=True,
            )
        stored = node.fit.coefficients
        got = fit.coefficients
        same_nan = np.array_equal(np.isnan(stored), np.isnan(got))
        finite = ~np.isnan(stored) & ~np.isnan(got)
        with np.errstate(invalid="ignore"):
            diff = np.abs(got[finite] - stored[finite])
            scale = np.maximum(np.abs(got[finite]), np.abs(stored[finite]))
        if not same_nan or np.any(diff > _REFIT_RTOL * scale):
            rel = float(np.max(diff / np.where(scale > 0, scale, 1.0), initial=0.0))
            detail = "NaN in other places" if not same_nan else f"{rel:.1e} relative"
            _warnings.warn(
                f"the refit of node {node_id} for vcov() differs from its stored "
                f"coefficients ({detail}); the covariance is the refit's",
                ModelTreesWarning,
                stacklevel=stacklevel + 1,
            )
        result = sandwich(
            fit.hessian,
            fit.estfun,
            weights=weights,
            cluster=None if clusters is None else clusters[rows],
            caseweights=self.control.caseweights,
            type=type,
            cadjust=cadjust,
        )
        reason = result.reason
        if fit.status is not FitStatus.OK:
            reason = f"the fit has status {fit.status.value}: {fit.reason}"
        elif result.too_few_clusters and clusters is not None:
            k = int(np.sum(~np.isnan(got)))
            _warnings.warn(
                f"node {node_id} has {result.n_clusters:g} clusters for {k} "
                "coefficients, so its cluster-robust covariance has rank below "
                f"{k} and is unreliable",
                TooFewClustersWarning,
                stacklevel=stacklevel + 1,
            )
        if (
            fit.status is FitStatus.OK
            and fit.extras.get("separation") == "quasi"
            and fit.hessian is not None
        ):
            estimated = ~np.isnan(got)
            j = diverging_coefficient(
                fit.hessian,
                None if self._x is None else self._x[rows][:, estimated],
                weights,
            )
            term = [n for n, e in zip(fit.coef_names, estimated, strict=True) if e][j]
            # Quasi-separation need not make the scaled Hessian ill
            # conditioned, so the status can stay "ok" (docs/api-port.md,
            # section 15.1, item 6); the reason says which term diverges.
            quasi = (
                f"node {node_id} is quasi-separated: coefficient {term!r} "
                "diverges, so its variance is not reliable"
            )
            _warnings.warn(quasi, ModelTreesWarning, stacklevel=stacklevel + 1)
            reason = quasi if reason is None else f"{reason}; {quasi}"
        return LeafVcov(
            node=node_id,
            names=fit.coef_names,
            matrix=expand(result, ~np.isnan(got)),
            n_clusters=None
            if clusters is None
            else int(np.unique(clusters[rows][weights > 0]).shape[0]),
            status=result.status,
            reason=reason,
            condition=result.condition,
        )

    def loglik(self) -> LogLik:
        """The tree's log-likelihood, as ``logLik.modelparty``: the sum over
        the leaves, df the leaf df plus ``dfsplit`` per inner node, nobs
        the summed leaf ``nobs`` (section 3.6). NaN, with a
        :class:`NodeFitFailedWarning`, if a leaf fit failed."""
        if not callable(getattr(self._model, "loglik", None)):
            raise TypeError(
                f"{type(self._model).__name__} has no loglik method "
                "(the LogLikModel protocol)"
            )
        model: Any = self._model
        failed = [i.id for i in self.nodes if i.is_leaf and i.status == "failed"]
        if failed:
            _warnings.warn(
                f"the fit of leaf {', '.join(map(str, failed))} failed, so the "
                "tree's log-likelihood is undefined (NaN)",
                NodeFitFailedWarning,
                stacklevel=3,
            )
        return tree_loglik(self._grown, model)

    def aic(self) -> float:
        """``-2 loglik + 2 df``."""
        ll = self.loglik()
        return -2 * ll.value + 2 * ll.df

    def bic(self) -> float:
        """``-2 loglik + log(nobs) df``."""
        ll = self.loglik()
        return -2 * ll.value + math.log(ll.nobs) * ll.df

    def print(self, *, digits: int = 4) -> None:
        """Print the tree with leaf coefficients to ``digits`` significant
        digits (``str(tree)`` uses 4, as R's ``getOption("digits") - 3``)."""
        print(format_tree(self, digits))

    def __str__(self) -> str:
        return format_tree(self)

    def summary(self, *, backend: Backend = None) -> TreeSummary:
        """The tree's tables and their printed form (section 3.4). The
        tables follow ``coef_table``'s backend rules."""
        name = resolve_backend(backend, self._backend)
        variables, nodes, tests = summary_tables(self)
        design = self._design
        return TreeSummary(
            variables=make_table(variables, name),
            nodes=make_table(nodes, name),
            tests=make_table(tests, name),
            coefficients=make_table(coef_columns(self, "wide"), name),
            dropped_levels={} if design is None else dict(design.dropped_levels),
            dropped_rows=self.n_rows - self.n_used,
            missing_counts={} if design is None else dict(design.missing_counts),
            warnings=self._warnings,
            text=format_summary(self),
        )

    def __repr__(self) -> str:
        n_inner = len(self._node_infos) - len(self.leaves)
        what = self._formula if self._formula is not None else "mob()"
        return (
            f"<ModelTree {self._model_name}: {what}, {n_inner} inner and "
            f"{len(self.leaves)} terminal nodes>"
        )


# Entry points


class ControlOptions(TypedDict, total=False):
    """The :class:`~modeltrees.tree.MobControl` fields, as keywords of the
    entry points (``lmtree(f, df, alpha=0.01)``). The last five raise
    ``NotImplementedError`` for anything but their defaults."""

    alpha: float
    bonferroni: bool
    minsize: float | None
    maxdepth: float
    trim: float
    dfsplit: bool
    prune: str | PruneRule | None
    restart: bool
    caseweights: bool
    numsplit: Literal["left", "center"]
    catsplit: Literal["binary", "multiway"]
    mtry: float
    breakties: bool
    ordinal: str
    vcov: str
    parm: Sequence[int] | Sequence[str] | None


def _control(control: MobControl | None, options: Mapping[str, Any]) -> MobControl:
    unknown = sorted(set(options) - set(ControlOptions.__annotations__))
    if unknown:
        raise TypeError(
            f"unexpected keyword argument(s): {', '.join(unknown)} (control options "
            f"are {', '.join(ControlOptions.__annotations__)})"
        )
    if control is not None:
        if options:
            raise TypeError(
                "pass either control=MobControl(...) or control options such as "
                f"{next(iter(options))}=..., not both"
            )
        if not isinstance(control, MobControl):
            raise TypeError(
                f"control must be a MobControl, got {type(control).__name__}"
            )
        return control
    return MobControl(**options)


def _check_n_jobs(n_jobs: int | None) -> None:
    if n_jobs is None or (
        isinstance(n_jobs, (int, np.integer))
        and not isinstance(n_jobs, bool)
        and n_jobs == 1
    ):
        return
    if not isinstance(n_jobs, (int, np.integer)) or isinstance(n_jobs, bool):
        raise TypeError(f"n_jobs must be an int or None, got {n_jobs!r}")
    raise NotImplementedError(
        "parallel split search (n_jobs other than None or 1) is not available yet"
    )


def _merge(caught: Sequence[_warnings.WarningMessage]) -> tuple[RecordedWarning, ...]:
    """Identical (category, message) pairs merged, in order of first
    occurrence (section 7.3)."""
    counts: dict[tuple[type[Warning], str], int] = {}
    for w in caught:
        key = (w.category, str(w.message))
        counts[key] = counts.get(key, 0) + 1
    return tuple(RecordedWarning(c, m, n) for (c, m), n in counts.items())


def _recorded[T](
    run: Callable[[], T], stacklevel: int
) -> tuple[T, tuple[RecordedWarning, ...]]:
    """Run ``run`` while recording its warnings, then re-emit each distinct
    warning once, ``stacklevel`` frames up from here (the user's call).

    Inside, every warning is recorded ("always"), so user filters such as
    "error" apply only to the re-emitted warnings. Warnings recorded before
    an exception are re-emitted before it propagates.
    """
    caught: list[_warnings.WarningMessage] = []
    try:
        with _warnings.catch_warnings(record=True) as recording:
            _warnings.simplefilter("always")
            caught = recording
            result = run()
    finally:
        recorded = _merge(caught)
        for r in recorded:
            text = r.message if r.count == 1 else f"{r.message} ({r.count} times)"
            _warnings.warn(text, r.category, stacklevel=stacklevel + 1)
    return result, recorded


def _formula_tree(
    formula: str,
    data: DataInput,
    *,
    rule: ResponseRule,
    make_model: Callable[[tuple[str, ...], CodedResponse], NodeModel],
    model_name: str,
    weights: ColumnSpec,
    offset: ColumnSpec,
    cluster: ColumnSpec,
    levels: Mapping[str, Sequence[Any]] | None,
    kinds: Mapping[str, Kind] | None,
    na_action: NaAction,
    n_jobs: int | None,
    control: MobControl | None,
    options: Mapping[str, Any],
) -> ModelTree:
    """The shared body of the formula entry points."""
    ctrl = _control(control, options)
    _check_n_jobs(n_jobs)

    def run() -> tuple[Design, CodedResponse, NodeModel, GrownTree]:
        design = build_design(
            formula,
            data,
            weights=weights,
            offset=offset,
            cluster=cluster,
            levels=levels,
            kinds=kinds,
            na_action=na_action,
        )
        coded = code_response(design.response, rule, weighted=weights is not None)
        model = make_model(design.coef_names, coded)
        grown = _grow(
            model,
            coded.values,
            design.x,
            design.partitioning,
            weights=design.weights,
            offset=design.offset,
            cluster=design.cluster,
            control=ctrl,
        )
        return design, coded, model, grown

    # Frames: _formula_tree, the entry point, the user's call.
    (design, coded, model, grown), recorded = _recorded(run, 3)
    return ModelTree(
        grown,
        model,
        model_name=model_name,
        formula=design.formula.expanded,
        design=design,
        response_levels=coded.levels,
        warnings=recorded,
        used_rows=design.used_rows,
        x=design.x,
        offset=design.offset,
        boolean=_boolean_variables(design),
        backend=design.backend,
        y=coded.values,
        weights=design.weights,
        cluster=design.cluster,
        frame=Frame(data),
    )


def lmtree(
    formula: str,
    data: DataInput,
    *,
    weights: ColumnSpec = None,
    offset: ColumnSpec = None,
    cluster: ColumnSpec = None,
    levels: Mapping[str, Sequence[Any]] | None = None,
    kinds: Mapping[str, Kind] | None = None,
    na_action: NaAction = "omit",
    n_jobs: int | None = None,
    control: MobControl | None = None,
    **control_options: Unpack[ControlOptions],
) -> ModelTree:
    """A linear model tree, as ``partykit::lmtree``.

    ``formula`` is ``"y ~ x1 + x2 | z1 + z2"``: the regressors of the
    leaf models, then the partitioning variables. A one-part formula
    ``"y ~ z1 + z2"`` means an intercept-only model partitioned by ``z1``
    and ``z2``, as in R (the tree's ``formula`` shows it expanded,
    ``"y ~ 1 | z1 + z2"``). ``data`` is an eager dataframe (polars,
    pandas, ...) or a mapping of column names to arrays.

    ``weights``, ``offset`` and ``cluster`` are column names or arrays;
    ``levels`` and ``kinds`` set level orders and variable kinds per column;
    ``na_action="omit"`` drops incomplete rows with one
    :class:`~modeltrees.errors.MissingDataWarning`. Control options
    (``alpha``, ``minsize``, ``maxdepth``, ``prune``, ...) are keywords, or
    a :class:`~modeltrees.tree.MobControl` as ``control``, not both.
    ``n_jobs`` accepts only None or 1 for now.
    """
    return _formula_tree(
        formula,
        data,
        rule="gaussian",
        make_model=lambda names, _: LmModel(coef_names=names),
        model_name="lmtree",
        weights=weights,
        offset=offset,
        cluster=cluster,
        levels=levels,
        kinds=kinds,
        na_action=na_action,
        n_jobs=n_jobs,
        control=control,
        options=control_options,
    )


_FAMILIES: tuple[str, ...] = ("gaussian", "binomial", "poisson")


def glmtree(
    formula: str,
    data: DataInput,
    *,
    family: Literal["gaussian", "binomial", "poisson"] = "gaussian",
    epsilon: float = 1e-8,
    maxit: int = 25,
    weights: ColumnSpec = None,
    offset: ColumnSpec = None,
    cluster: ColumnSpec = None,
    levels: Mapping[str, Sequence[Any]] | None = None,
    kinds: Mapping[str, Kind] | None = None,
    na_action: NaAction = "omit",
    n_jobs: int | None = None,
    control: MobControl | None = None,
    **control_options: Unpack[ControlOptions],
) -> ModelTree:
    """A generalized linear model tree, as ``partykit::glmtree``.

    ``family`` is "gaussian", "binomial" or "poisson", each with its
    canonical link; ``epsilon`` and ``maxit`` are glm.fit's controls. A
    binomial response may be 0/1, Boolean, categorical (the first level is
    failure, every other level success, as in R) or proportions with the
    number of trials as ``weights``; a poisson response non-negative
    counts. The other arguments are those of :func:`lmtree`.
    """
    if not isinstance(family, str) or family not in _FAMILIES:
        raise NotImplementedError(
            f"family {family!r} is not available; glmtree supports "
            f"{', '.join(_FAMILIES)} with their canonical links"
        )
    rule: ResponseRule = family
    return _formula_tree(
        formula,
        data,
        rule=rule,
        make_model=lambda names, _: GlmModel(
            family, epsilon=epsilon, maxit=maxit, coef_names=names
        ),
        model_name=f"glmtree({family})",
        weights=weights,
        offset=offset,
        cluster=cluster,
        levels=levels,
        kinds=kinds,
        na_action=na_action,
        n_jobs=n_jobs,
        control=control,
        options=control_options,
    )


def negbintree(
    formula: str,
    data: DataInput,
    *,
    objective: Literal["loglik", "deviance"] = "loglik",
    epsilon: float = 1e-8,
    maxit: int = 25,
    weights: ColumnSpec = None,
    offset: ColumnSpec = None,
    cluster: ColumnSpec = None,
    levels: Mapping[str, Sequence[Any]] | None = None,
    kinds: Mapping[str, Kind] | None = None,
    na_action: NaAction = "omit",
    n_jobs: int | None = None,
    control: MobControl | None = None,
    **control_options: Unpack[ControlOptions],
) -> ModelTree:
    """A negative binomial regression tree with theta estimated per leaf.

    The model follows Rusch's thesis: only the regression coefficients are
    tested for instability, so a change in overdispersion alone does not
    split. ``objective="deviance"`` compares candidate splits by deviance,
    as party 1.0-1 did in the thesis. The response must be non-negative
    counts. The other arguments are those of :func:`lmtree`.
    """
    return _formula_tree(
        formula,
        data,
        rule="negbin",
        make_model=lambda names, _: NegBinModel(
            objective=objective, epsilon=epsilon, maxit=maxit, coef_names=names
        ),
        model_name="negbintree",
        weights=weights,
        offset=offset,
        cluster=cluster,
        levels=levels,
        kinds=kinds,
        na_action=na_action,
        n_jobs=n_jobs,
        control=control,
        options=control_options,
    )


def safelogittree(
    formula: str,
    data: DataInput,
    *,
    separation_tol: float = 1e-6,
    epsilon: float = 1e-8,
    maxit: int = 25,
    weights: ColumnSpec = None,
    offset: ColumnSpec = None,
    cluster: ColumnSpec = None,
    levels: Mapping[str, Sequence[Any]] | None = None,
    kinds: Mapping[str, Kind] | None = None,
    na_action: NaAction = "omit",
    n_jobs: int | None = None,
    control: MobControl | None = None,
    **control_options: Unpack[ControlOptions],
) -> ModelTree:
    """A binary logistic regression tree that tolerates separation.

    This project's design, not Rusch's implementation (the thesis names a
    safe logit model but gives no code; docs/extensions-spec.md). A
    completely separated leaf (every row fitted within ``separation_tol``)
    keeps its fit but has no scores and is terminal. The response must be
    binary: 0/1, Boolean or categorical (first level 0, the rest 1);
    proportions raise. The other arguments are those of :func:`lmtree`.
    """
    return _formula_tree(
        formula,
        data,
        rule="safelogit",
        make_model=lambda names, _: SafeLogitModel(
            separation_tol=separation_tol,
            epsilon=epsilon,
            maxit=maxit,
            coef_names=names,
        ),
        model_name="safelogittree",
        weights=weights,
        offset=offset,
        cluster=cluster,
        levels=levels,
        kinds=kinds,
        na_action=na_action,
        n_jobs=n_jobs,
        control=control,
        options=control_options,
    )


def multinomtree(
    formula: str,
    data: DataInput,
    *,
    gtol: float = 1e-12,
    maxit: int = 100,
    weights: ColumnSpec = None,
    offset: ColumnSpec = None,
    cluster: ColumnSpec = None,
    levels: Mapping[str, Sequence[Any]] | None = None,
    kinds: Mapping[str, Kind] | None = None,
    na_action: NaAction = "omit",
    n_jobs: int | None = None,
    control: MobControl | None = None,
    **control_options: Unpack[ControlOptions],
) -> ModelTree:
    """A multinomial logit tree, fitted by Newton's method in each node.

    This project's design, not Rusch's implementation
    (docs/extensions-spec.md). The response must be categorical, or integer
    with ``levels``; its first level (after unused levels are dropped) is
    the reference class, and the tree's ``response_levels`` records the
    order. An offset is not supported. The other arguments are those of
    :func:`lmtree`.
    """
    if offset is not None:
        raise TypeError("multinomtree does not support an offset")
    return _formula_tree(
        formula,
        data,
        rule="multinom",
        make_model=lambda names, coded: MultinomModel(
            coded.levels or (), coef_names=names, gtol=gtol, maxit=maxit
        ),
        model_name="multinomtree",
        weights=weights,
        offset=offset,
        cluster=cluster,
        levels=levels,
        kinds=kinds,
        na_action=na_action,
        n_jobs=n_jobs,
        control=control,
        options=control_options,
    )


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
    n_jobs: int | None = None,
) -> ModelTree:
    """Grow a tree with any node model on arrays, as ``partykit::mob``.

    The array path for custom models: ``y`` the response as the model takes
    it, ``x`` the regressor matrix with its intercept column (or None for
    an intercept-only model), ``z`` the partitioning variables, each a
    :class:`~modeltrees.tree.Variable`, a dict with ``kind``, ``values``
    and optionally ``levels``, or a tuple ``(kind, values[, levels])``.
    There is no missing-data handling: every row is used.
    """
    _check_n_jobs(n_jobs)
    if control is not None and not isinstance(control, MobControl):
        raise TypeError(f"control must be a MobControl, got {type(control).__name__}")
    xv = None if x is None else np.asarray(x, dtype=np.float64)
    off = None if offset is None else np.asarray(offset, dtype=np.float64)

    def run() -> GrownTree:
        return _grow(
            model,
            y,
            xv,
            z,
            weights=weights,
            offset=off,
            cluster=cluster,
            control=control,
        )

    grown, recorded = _recorded(run, 2)  # mob, the user's call
    levels = getattr(model, "levels", None)
    return ModelTree(
        grown,
        model,
        model_name=f"mob({type(model).__name__})",
        formula=None,
        design=None,
        response_levels=None if levels is None else tuple(str(v) for v in levels),
        warnings=recorded,
        used_rows=np.ones(grown.fitted_node.shape[0], dtype=np.bool_),
        x=xv,
        offset=off,
        y=np.asarray(y),
        weights=np.ones(grown.fitted_node.shape[0])
        if weights is None
        else np.asarray(weights, dtype=np.float64),
        cluster=None if cluster is None else np.asarray(cluster),
    )


def _boolean_variables(design: Design) -> frozenset[str]:
    """Partitioning variables that come from Boolean columns."""
    return frozenset(
        t.name
        for t in design.partitioning_terms
        if t.storage == "boolean" and t.kind == "numeric"
    )
