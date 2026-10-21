"""The comparison of the blind differential run (task 6j) for the cases in
tests/fixtures/blind_cases/.

A case is grown through its public entry point from its CSV as a polars
frame (``n_jobs=1``) and compared with the R tree in the order of
CLAUDE.md, with the helpers and tolerances of the standing tests:

1. split candidates and objectives, by ``find_split`` on R's node rows;
2. the chosen splits;
3. node sizes and coefficients (with objectives, scores and model info);
4. the whole tree: structure, stop reasons, splits, instability tests and
   the split searches of the grown tree;
5. ``fitted_node`` and ``predict(type="node")``;
6. ``loglik()`` and its df;
7. ``vcov`` on the leaves of unweighted clustered cases against
   ``sandwich::vcovCL`` (HC0 and HC1, ``cadjust = TRUE``).

Two comparisons here differ from the standing tests, for quantities the
data do not determine beyond rounding (docs/api-port.md, section 19), and
nowhere else:

- negative binomial theta under a weak-identification warning (theta.ml's
  iteration limit or glm.nb's alternation limit, from R or the port):
  theta, theta_irls and theta_se at ``WEAK_THETA_RTOL`` (theta.ml's
  iteration limit keeps the standing ``THETA_LOOSE_RTOL``); the
  alternation-limit flag need not agree, and when it differs the three
  are compared at ``ROUNDED_STOP_RTOL`` instead;
- completely separated safe logit leaves: the linear predictor on the
  leaf's rows at ``SEPARATED_RTOL`` and the fitted means at the usual
  tolerance, instead of each coefficient (the slopes are not identified),
  and the objective (of order exp(-eta)) relatively, without ``ATOL``;
- completely separated binomial and poisson glm nodes (every
  positive-weight fitted mean at R's coefficients within
  ``SEPARATION_TOL`` of the boundary, 0 or 1 for binomial, 0 for poisson):
  the same comparison on the positive-weight rows. Their coefficients are
  not compared: there is no finite MLE and where IRLS stops is set by
  rounding (docs/api-port.md, section 21).

:func:`compare` stops at the first divergence and raises
:class:`Divergence` with the component.
"""

import math
import warnings
from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np
from fixture_frames import offset_column, ordered_kinds, polars_frame
from fixtures_io import FIXTURES, load_json, nan_array
from grown_trees import control as fixture_control
from grown_trees import model as fixture_model
from grown_trees import tree_inputs
from test_multinom import OFFSET_FACTOR, _coef_tol, _r_offset
from test_negbin import NEAR_POISSON, THETA_LOOSE_RTOL
from test_safelogit import SEPARATED_RTOL
from tree_fixtures import cluster as fixture_cluster
from tree_fixtures import variables as fixture_variables

import modeltrees as mt
from modeltrees._vcov import sandwich
from modeltrees.instability import (
    PartitioningVariable,
    VariableKind,
    fluctuation_tests,
)
from modeltrees.models.families import get_family
from modeltrees.models.multinom import (
    multinom_evaluate,
    multinom_hessian,
    multinom_scores,
)
from modeltrees.node_model import FitResult, FitStatus
from modeltrees.split_search import SplitSearch, find_split
from modeltrees.tree import Node

RTOL = 1e-8
ATOL = 1e-10
# theta, theta_irls and theta_se when R or the port warns that theta is
# weakly identified.
WEAK_THETA_RTOL = 1e-6
ALTERNATION_LIMIT = "alternation limit reached"
WEAK_THETA = frozenset({NEAR_POISSON, ALTERNATION_LIMIT})
# theta, theta_irls and theta_se when only one side reached glm.nb's
# alternation limit, a stopping decision decided by rounding: twice the
# largest deviation from R's value over 2,300 end points of blind_36 node
# 112 refitted with its regressors perturbed by one ulp (9.3e-9, theta_se),
# where theta moves by up to 1.4e-9 per alternation (docs/api-port.md,
# section 20).
ROUNDED_STOP_RTOL = 2e-8
# A glm node is completely separated when every positive-weight fitted mean
# at R's coefficients is this close to the boundary of the family's mean.
SEPARATION_TOL = 1e-10
BLIND = FIXTURES / "blind_cases"
CASES = sorted(p.stem for p in (BLIND / "trees").glob("*.json"))

COMPONENTS = (
    "split candidates and objectives",
    "chosen splits",
    "node sizes and coefficients",
    "whole tree",
    "fitted_node",
    "loglik and df",
    "vcov",
)


class Divergence(AssertionError):
    """The first difference from R: the component and what differs."""

    def __init__(self, component: str, detail: str) -> None:
        super().__init__(f"{component}: {detail}")
        self.component = component
        self.detail = detail


class _Differs(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _check(ok: bool, detail: str) -> None:
    if not ok:
        raise _Differs(detail)


def _allclose(got: Any, expected: Any, rtol: float, atol: float, what: str) -> None:
    """np.testing.assert_allclose's rule, NaN equal to NaN."""
    g = np.asarray(got, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    _check(g.shape == e.shape, f"{what}: shape {g.shape} vs R {e.shape}")
    _check(bool(np.array_equal(np.isnan(g), np.isnan(e))), f"{what}: NaN pattern")
    inf_g, inf_e = np.isinf(g), np.isinf(e)
    _check(
        bool(np.array_equal(inf_g, inf_e)) and bool(np.array_equal(g[inf_g], e[inf_e])),
        f"{what}: Inf pattern",
    )
    fin = np.isfinite(e)
    err = np.abs(g[fin] - e[fin])
    allow = atol + rtol * np.abs(e[fin])
    if err.size and np.any(err > allow):
        i = int(np.argmax(err / allow))
        raise _Differs(f"{what}: {g[fin][i]!r} vs R {e[fin][i]!r} (abs {err[i]:.3g})")


def _approx(
    got: float | None, expected: float | None, rtol: float, atol: float, what: str
) -> None:
    if expected is None:
        _check(got is None or math.isnan(got), f"{what}: {got!r} vs R NA")
        return
    _check(got is not None, f"{what}: None vs R {expected!r}")
    _allclose([got], [expected], rtol, atol, what)


def _labels(value: str | list[str]) -> tuple[str, ...]:
    # jsonlite writes a length-one character vector as a string.
    return (value,) if isinstance(value, str) else tuple(value)


def _objective(values: list[float | None]) -> np.ndarray:
    return np.array([np.inf if v is None else v for v in values], dtype=np.float64)


def _assert_split(got: Any, expected: dict[str, Any] | None, where: str) -> None:
    if expected is None:
        _check(got is None, f"{where}: split {got} vs R none")
        return
    _check(got is not None, f"{where}: no split vs R {expected['variable']}")
    _check(
        got.variable == expected["variable_index"]
        and got.kind.value == expected["kind"],
        f"{where}: split variable {got.variable} vs R {expected['variable_index']}",
    )
    if expected["kind"] == "nominal":
        _check(got.index == tuple(expected["index"]), f"{where}: split index")
    else:
        _check(
            got.breaks == expected["breaks"],
            f"{where}: break {got.breaks!r} vs R {expected['breaks']!r}",
        )


@cache
def case_fixture(name: str) -> dict[str, Any]:
    f: dict[str, Any] = load_json(f"blind_cases/trees/{name}.json")
    # The fixture readers resolve data_csv against tests/fixtures.
    f["data_csv"] = f"blind_cases/{f['data_csv']}"
    return f


@cache
def case_settings(name: str) -> dict[str, Any]:
    s: dict[str, Any] = load_json(f"blind_cases/cases/{name}.json")
    return s


def case_vcov(name: str) -> dict[str, Any] | None:
    if not (BLIND / "vcov" / f"{name}.json").exists():
        return None
    v: dict[str, Any] = load_json(f"blind_cases/vcov/{name}.json")
    return v


class _Case:
    def __init__(self, name: str) -> None:
        self.f = case_fixture(name)
        self.model_name: str = case_settings(name)["model"]
        self.vcov = case_vcov(name)
        self.y, self.x, self.w, self.off = tree_inputs(self.f)
        self.cl = fixture_cluster(self.f)
        self.vars = fixture_variables(self.f)

    def node(self, node_id: int) -> dict[str, Any]:
        return next(n for n in self.f["nodes"] if n["id"] == node_id)

    def grow(self) -> mt.ModelTree:
        """As tests/test_api_fixtures.public, with n_jobs=1."""
        f = self.f
        df = polars_frame(f)
        common: dict[str, Any] = {
            "weights": f["weights"],
            "offset": offset_column(f)[0],
            "cluster": f.get("cluster"),
            "kinds": ordered_kinds(f),
            "control": fixture_control(f),
            "n_jobs": 1,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", mt.ModelTreesWarning)
            if f["function_name"] == "lmtree":
                return mt.lmtree(f["formula"], df, **common)
            if f["function_name"] == "glmtree":
                return mt.glmtree(
                    f["formula"],
                    df,
                    family=f["family"]["family"],
                    **f["glm_control"],
                    **common,
                )
            fit_function = f["fit_function"]
            if fit_function == "negbinfit":
                return mt.negbintree(f["formula"], df, **f["fit_args"], **common)
            if fit_function == "safelogitfit":
                return mt.safelogittree(f["formula"], df, **f["fit_args"], **common)
            assert fit_function == "multinomfit"
            common.pop("offset")
            return mt.multinomtree(f["formula"], df, **common)

    # 1 and 2.
    def search(self, node: dict[str, Any]) -> SplitSearch:
        """Split search on R's node rows, as tests/test_tree._search."""
        c = self.f["control"]
        var = self.vars[node["split_search"]["variable"]]
        rows = np.array(node["rows"], dtype=np.intp)
        m = fixture_model(self.f)
        y, x, w, off, cl = self.y, self.x, self.w, self.off, self.cl

        def fit(mask: np.ndarray, start: np.ndarray | None) -> FitResult:
            r = rows[mask]
            return m.fit(
                y[r],
                None if x is None else x[r],
                start=start,
                weights=w[r],
                offset=None if off is None else off[r],
                cluster=None if cl is None else cl[r],
                caseweights=c["caseweights"],
                estfun=False,
                vcov=False,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return find_split(
                self.f["partitioning"].index(var.name),
                var.kind,
                var.values[rows],
                var.levels,
                w[rows],
                fit,
                minsize=node["minsize"],
                caseweights=c["caseweights"],
                restart=c["restart"],
                numsplit=c["numsplit"],
            )

    def searches(self) -> dict[int, SplitSearch]:
        out = {}
        for node in self.f["nodes"]:
            if node["split_search"] is None:
                continue
            where = f"node {node['id']}"
            exp = node["split_search"]
            got = self.search(node)
            out[node["id"]] = got
            _check(got.minsize == exp["minsize"], f"{where}: minsize")
            if got.kind is VariableKind.NUMERIC:
                cand = np.asarray(exp["candidates"], dtype=np.float64)
                _check(
                    got.values is not None
                    and got.values.shape == cand.shape
                    and bool(np.array_equal(got.values, cand)),
                    f"{where}: numeric candidates",
                )
            else:
                _check(got.labels == tuple(exp["candidates"]), f"{where}: candidates")
            _allclose(
                got.objective,
                _objective(exp["objective"]),
                RTOL,
                ATOL,
                f"{where}: split objectives",
            )
        return out

    def chosen(self, searches: dict[int, SplitSearch]) -> None:
        for node in self.f["nodes"]:
            if node["id"] not in searches:
                continue
            got = searches[node["id"]]
            where = f"node {node['id']}"
            if node["stop_reason"] == "no_admissible_split":
                _check(got.split is None, f"{where}: split where R has none")
            elif node["stop_reason"] == "pruned":
                _check(got.split is not None, f"{where}: no split for a pruned node")
            else:
                _assert_split(got.split, node["split"], where)

    # 3.
    def node_data(self, e: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.array(e["rows"], dtype=np.intp)
        assert self.x is not None
        return self.y[rows], self.x[rows], self.w[rows]

    def nodes(self, tree: mt.ModelTree) -> None:
        got_nodes = tree._grown.nodes
        expected = self.f["nodes"]
        for i, e in enumerate(expected):
            where = f"node {e['id']}"
            _check(i < len(got_nodes), f"{where}: missing ({len(got_nodes)} nodes)")
            g = got_nodes[i]
            _check(
                g.rows.shape[0] == e["n_rows"]
                and bool(np.array_equal(g.rows, e["rows"])),
                f"{where}: rows differ ({g.rows.shape[0]} vs R {e['n_rows']})",
            )
            _check(g.nobs == e["nobs"], f"{where}: nobs {g.nobs} vs R {e['nobs']}")
            _check(
                g.fit.coef_names == tuple(e["coefficient_names"]),
                f"{where}: coefficient names",
            )
            self.fit(g, e, where)
        _check(
            len(got_nodes) == len(expected),
            f"{len(got_nodes)} nodes vs R {len(expected)}",
        )

    def fit(self, g: Node, e: dict[str, Any], where: str) -> None:
        fit = g.fit
        coefficients = nan_array(e["coefficients"])
        if self.model_name == "glm":
            separated = self.separated_glm(e, coefficients)
            if separated is not None:
                # The coefficients of a completely separated node are not
                # compared: it has no finite MLE, IRLS walks towards
                # infinity until the deviance change falls below epsilon,
                # and where it stops is set by rounding (blind_13 node 67:
                # 3.9e-10 from R here, 4.2e-7 on a CI runner, up to 4.2e-7
                # under one-ulp changes of x1; docs/api-port.md, section
                # 21). The objective, of order exp(-|eta|), is compared
                # relatively, without ATOL.
                compare_separated(
                    *self.glm_means(fit.coefficients, separated),
                    *self.glm_means(coefficients, separated),
                    where,
                )
                _approx(fit.objfun, e["objfun"], RTOL, 0.0, f"{where}: objective")
                return
        if self.model_name in ("lm", "glm"):
            _allclose(
                fit.coefficients, coefficients, RTOL, ATOL, f"{where}: coefficients"
            )
            _approx(fit.objfun, e["objfun"], RTOL, ATOL, f"{where}: objective")
            return
        info = e["model_info"]
        if self.model_name == "negbin":
            _allclose(
                fit.coefficients, coefficients, RTOL, ATOL, f"{where}: coefficients"
            )
            _approx(fit.objfun, e["objfun"], RTOL, ATOL, f"{where}: objective")
            _allclose(fit.estfun, np.array(e["estfun"]), RTOL, ATOL, f"{where}: scores")
            self.theta(fit, info, where)
            return
        if self.model_name == "safelogit":
            if fit.extras["separation"] == "complete":
                x = self.node_data(e)[1]
                compare_separated(
                    *_linear_predictor(x, fit.coefficients),
                    *_linear_predictor(x, coefficients),
                    where,
                )
                # The objective is of order exp(-eta), far below ATOL, so it
                # is compared relatively.
                atol = 0.0
            else:
                _allclose(
                    fit.coefficients, coefficients, RTOL, ATOL, f"{where}: coefficients"
                )
                atol = ATOL
            _approx(fit.objfun, e["objfun"], RTOL, atol, f"{where}: objective")
            for key in ("status", "separation", "n_boundary", "converged", "boundary"):
                _check(
                    fit.extras[key] == info[key],
                    f"{where}: {key} {fit.extras[key]!r} vs R {info[key]!r}",
                )
            _check(
                fit.extras["warnings"] == _labels(info["warnings"]),
                f"{where}: warnings",
            )
            if e["estfun"] is None:
                _check(
                    fit.status is FitStatus.NO_ESTFUN and fit.estfun is None,
                    f"{where}: scores where R has none",
                )
            else:
                _allclose(
                    fit.estfun, np.array(e["estfun"]), RTOL, ATOL, f"{where}: scores"
                )
            return
        assert self.model_name == "multinom"
        _approx(fit.objfun, e["objfun"], RTOL, 0.0, f"{where}: objective")
        if e["estfun"] is None:
            _check(
                fit.status is FitStatus.NO_ESTFUN
                and fit.estfun is None
                and bool(np.all(np.isnan(fit.coefficients))),
                f"{where}: coefficients or scores where R has none",
            )
        else:
            y, x, w = self.node_data(e)
            r_estfun = np.array(e["estfun"])
            tol = _coef_tol(_r_offset(fit, x, y, w, r_estfun))
            _allclose(
                fit.coefficients, coefficients, 0.0, tol, f"{where}: coefficients"
            )
            _check(
                fit.estfun is not None and fit.estfun.shape == r_estfun.shape,
                f"{where}: score shape",
            )
            assert fit.estfun is not None
            k = fit.estfun.shape[1] // x.shape[1]
            xi = np.tile(np.abs(x), k)
            bound = w[:, None] * xi * np.abs(x).sum(axis=1)[:, None] * tol / 2
            _check(
                bool(np.all(np.abs(fit.estfun - r_estfun) <= bound + ATOL)),
                f"{where}: scores beyond the convergence-offset bound",
            )
        _check(fit.extras["status"] == info["status"], f"{where}: status")
        _check(
            fit.extras["classes_present"] == _labels(info["classes_present"]),
            f"{where}: classes present",
        )

    def glm_means(
        self, coefficients: np.ndarray, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The glm node's linear predictor and fitted means on ``rows``."""
        assert self.x is not None
        eta = self.x[rows] @ np.where(np.isnan(coefficients), 0.0, coefficients)
        if self.off is not None:
            eta = eta + self.off[rows]
        return eta, get_family(self.f["family"]["family"]).linkinv(eta)

    def separated_glm(
        self, e: dict[str, Any], coefficients: np.ndarray
    ) -> np.ndarray | None:
        """The positive-weight rows of a completely separated binomial or
        poisson node, None for any other node."""
        family = self.f["family"]["family"]
        if family not in ("binomial", "poisson"):
            return None
        rows = np.array(e["rows"], dtype=np.intp)
        rows = rows[self.w[rows] > 0]
        mu = self.glm_means(coefficients, rows)[1]
        distance = np.minimum(mu, 1 - mu) if family == "binomial" else mu
        if rows.size == 0 or not bool(np.all(distance <= SEPARATION_TOL)):
            return None
        return rows

    def theta(self, fit: FitResult, info: dict[str, Any], where: str) -> None:
        got_warning, r_warning = fit.extras["theta_warning"], info["theta_warning"]
        # Only one side reached the alternation limit.
        stopped_by_rounding = got_warning != r_warning and ALTERNATION_LIMIT in (
            got_warning,
            r_warning,
        )
        if stopped_by_rounding:
            rtol = ROUNDED_STOP_RTOL
        elif r_warning == NEAR_POISSON:
            rtol = THETA_LOOSE_RTOL
        elif r_warning in WEAK_THETA or got_warning in WEAK_THETA:
            rtol = WEAK_THETA_RTOL
        else:
            rtol = RTOL
        try:
            for key in ("theta", "theta_irls", "theta_se"):
                _approx(fit.extras[key], info[key], rtol, ATOL, f"{where}: {key}")
            if not stopped_by_rounding:
                _check(
                    got_warning == r_warning,
                    f"{where}: theta_warning {got_warning!r} vs R {r_warning!r}",
                )
        except _Differs as d:
            raise _Differs(f"{d.detail}; {_theta_state(fit.extras, info)}") from None
        for key in ("converged", "iter"):
            _check(
                fit.extras[key] == info[key],
                f"{where}: {key} {fit.extras[key]!r} vs R {info[key]!r}",
            )

    # 4.
    def tree(self, tree: mt.ModelTree) -> None:
        grown = tree._grown
        f = self.f
        _check(grown.minsize == f["root_minsize"], "root minsize")
        parents = {k.id: n.id for n in grown.nodes for k in n.kids}
        for g, e in zip(grown.nodes, f["nodes"], strict=True):
            where = f"node {e['id']}"
            _check(g.id == e["id"], f"{where}: id {g.id}")
            _check(g.depth == e["depth"], f"{where}: depth")
            _check(parents.get(g.id) == e["parent"], f"{where}: parent")
            _check([k.id for k in g.kids] == e["kids"], f"{where}: kids")
            _check(g.terminal == e["terminal"], f"{where}: terminal")
            _check(
                g.stop_reason.value == e["stop_reason"],
                f"{where}: stop reason {g.stop_reason.value} vs R {e['stop_reason']}",
            )
            _assert_split(g.split, e["split"], where)
            self.tests(g, e, where)
            if e["split_search"] is None:
                _check(g.split_search is None, f"{where}: split search")
            else:
                _check(g.split_search is not None, f"{where}: no split search")
                assert g.split_search is not None
                _check(
                    g.split_search.variable == e["test"]["best"],
                    f"{where}: split search variable",
                )
                _allclose(
                    g.split_search.objective,
                    _objective(e["split_search"]["objective"]),
                    RTOL,
                    ATOL,
                    f"{where}: grown split objectives",
                )
        s = f["summary"]
        leaves = [n for n in grown.nodes if n.terminal]
        _check(len(leaves) == s["n_terminal"], "terminal node count")
        _check(max(n.depth for n in grown.nodes) == s["depth"], "depth")
        _approx(
            sum(n.fit.objfun for n in leaves), s["objfun"], RTOL, 0.0, "leaf objective"
        )

    def tests(self, g: Node, e: dict[str, Any], where: str) -> None:
        status = e["test"]["status"] if e["test"] else "not_run"
        if status == "not_run":
            _check(
                g.test is None and not g.test_failed and g.p_value is None,
                f"{where}: tests run where R ran none",
            )
            return
        if status == "failed":
            _check(
                g.test is None and g.test_failed and g.p_value is None,
                f"{where}: tests where R's failed",
            )
            return
        t = e["test"]
        _check(g.test is not None, f"{where}: no tests where R has them")
        assert g.test is not None
        if self.model_name == "multinom":
            self.multinom_tests(g, e, where)
        else:
            _allclose(
                g.test.statistic, t["statistic"], RTOL, 0.0, f"{where}: statistics"
            )
            _allclose(
                g.test.p_value_adjusted,
                nan_array(t["p_value_adjusted"]),
                RTOL,
                0.0,
                f"{where}: adjusted p-values",
            )
            _approx(g.p_value, e["p_value"], RTOL, 0.0, f"{where}: p-value")
        _check(g.test.best == t["best"], f"{where}: best variable")

    def moved(
        self, fit: FitResult, e: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Hessian, scores, weights and rows at our coefficients moved by
        R's offset -H^-1 g_R, over the present classes (tests/test_multinom.py,
        test_tree_tests; tests/test_vcov_fixtures.py, _moved_sandwich)."""
        y, x, w = self.node_data(e)
        offset = _r_offset(fit, x, y, w, np.array(e["estfun"]))
        p = x.shape[1]
        beta = fit.coefficients.reshape(-1, p)
        present = ~np.isnan(beta[:, 0])
        classes = np.r_[0, np.flatnonzero(present) + 1]
        recode = np.full(beta.shape[0] + 1, -1)
        recode[classes] = np.arange(classes.shape[0])
        keep = w > 0
        yk = np.where(keep, recode[y], 0)
        _, prob = multinom_evaluate(beta[present] + offset.reshape(-1, p), x, yk, w)
        rows = np.array(e["rows"], dtype=np.intp)
        hessian = multinom_hessian(prob[keep], x[keep], w[keep])
        return hessian, multinom_scores(prob, x, yk, w), w, rows

    def multinom_tests(self, g: Node, e: dict[str, Any], where: str) -> None:
        assert g.test is not None
        _, scores, w, rows = self.moved(g.fit, e)
        c = self.f["control"]
        z = [
            PartitioningVariable(v.name, v.kind, v.values[rows])
            for v in self.vars.values()
        ]
        shifted = fluctuation_tests(
            scores,
            z,
            w,
            minsize=e["minsize"],
            trim=c["trim"],
            bonferroni=c["bonferroni"],
            cluster=None if self.cl is None else self.cl[rows],
            caseweights=c["caseweights"],
        )
        t = e["test"]
        for what, got, moved, expected in (
            (
                "statistics",
                g.test.statistic,
                shifted.statistic,
                np.array(t["statistic"]),
            ),
            (
                "adjusted p-values",
                g.test.p_value_adjusted,
                shifted.p_value_adjusted,
                nan_array(t["p_value_adjusted"]),
            ),
        ):
            tol = OFFSET_FACTOR * np.abs(moved - got) + RTOL * np.abs(expected)
            ok = (np.abs(got - expected) <= tol) | (np.isnan(got) & np.isnan(expected))
            _check(bool(np.all(ok)), f"{where}: {what}")

    # 5 and 6.
    def fitted(self, tree: mt.ModelTree) -> None:
        expected = np.asarray(self.f["fitted_node"], dtype=np.int64)
        _check(bool(np.array_equal(tree.fitted_node, expected)), "fitted_node")
        _check(bool(np.array_equal(tree.predict(type="node"), expected)), "predict")

    def loglik(self, tree: mt.ModelTree) -> None:
        s = self.f["summary"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ll = tree.loglik()
        _approx(ll.value, s["loglik"], RTOL, 0.0, "loglik")
        _check(ll.df == s["loglik_df"], f"df {ll.df} vs R {s['loglik_df']}")
        _check(ll.nobs == self.f["nobs"], "nobs")

    # 7.
    def vcovs(self, tree: mt.ModelTree) -> None:
        """Unweighted cases only: sandwich is not an oracle under case
        weights (docs/sandwich-fixtures.md 1.5). Leaves without scores or
        with an absent class have no R value."""
        if self.vcov is None or self.vcov["weighted"]:
            return
        assert self.cl is not None
        for leaf in self.vcov["leaves"]:
            if leaf["status"] == "skipped":
                continue
            nid = leaf["id"]
            _check(leaf["status"] == "ok", f"leaf {nid}: R refit failed")
            for entry in leaf["vcovcl"]:
                where = f"leaf {nid} {entry['type']}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    got = tree.vcov(nid, type=entry["type"], cadjust=True)[nid]
                if "error" in entry:
                    _check(got.status == "unavailable", f"{where}: R stops")
                    continue
                _check(got.status == "ok", f"{where}: status {got.status}")
                _check(got.n_clusters == leaf["n_clusters"], f"{where}: clusters")
                expected = np.array(entry["matrix"], dtype=np.float64)
                if self.model_name != "multinom":
                    _allclose(got.matrix, expected, RTOL, 0.0, f"{where}: vcov")
                    continue
                hess, scores, w, rows = self.moved(
                    tree._grown.node(nid).fit, self.node(nid)
                )
                keep = w > 0
                moved = sandwich(
                    hess,
                    scores[keep],
                    weights=w[keep],
                    cluster=self.cl[rows][keep],
                    caseweights=True,
                    type=entry["type"],
                    cadjust=True,
                ).matrix
                tol = 2 * np.abs(moved - got.matrix) + RTOL * np.max(np.abs(expected))
                _check(
                    bool(np.all(np.abs(got.matrix - expected) <= tol)), f"{where}: vcov"
                )


def _theta_state(got: Mapping[str, Any], r: Mapping[str, Any]) -> str:
    """theta, theta_irls, iter and the warning on both sides, for a theta
    divergence."""

    def side(d: Mapping[str, Any]) -> str:
        return (
            f"theta {d['theta']!r}, theta_irls {d['theta_irls']!r}, "
            f"iter {d['iter']!r}, theta_warning {d['theta_warning']!r}"
        )

    return f"port {side(got)}; R {side(r)}"


def _linear_predictor(
    x: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The logit leaf's linear predictor and fitted means on its rows."""
    eta = x @ np.where(np.isnan(coefficients), 0.0, coefficients)
    return eta, 1 / (1 + np.exp(-eta))


def compare_separated(
    eta: np.ndarray, mu: np.ndarray, r_eta: np.ndarray, r_mu: np.ndarray, where: str
) -> None:
    """A completely separated leaf: its slopes are not identified, so the
    linear predictor (the intercept direction IRLS runs along) is compared
    at SEPARATED_RTOL and the fitted means at the usual tolerance."""
    _allclose(eta, r_eta, SEPARATED_RTOL, 0.0, f"{where}: linear predictor")
    _allclose(mu, r_mu, RTOL, ATOL, f"{where}: fitted values")


def compare(name: str) -> None:
    """Compare a blind case with R; raise Divergence at the first difference."""
    case = _Case(name)
    component = COMPONENTS[0]
    try:
        searches = case.searches()
        component = COMPONENTS[1]
        case.chosen(searches)
        component = COMPONENTS[2]
        tree = case.grow()
        case.nodes(tree)
        component = COMPONENTS[3]
        case.tree(tree)
        component = COMPONENTS[4]
        case.fitted(tree)
        component = COMPONENTS[5]
        case.loglik(tree)
        component = COMPONENTS[6]
        case.vcovs(tree)
    except _Differs as d:
        raise Divergence(component, d.detail) from None
