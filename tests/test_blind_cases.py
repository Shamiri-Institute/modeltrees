"""The cases of the blind differential runs (tasks 6j and 6l) where the port
and R differed, in tests/fixtures/blind_cases/ (README.md there has the
analysis).

Each test compares the case with R in the order of CLAUDE.md
(tests/blind_compare.py). The cases the port now matches (``MATCHING``) are
standing tests that must pass on every component. The others are marked
xfail with their first diverging component; such a test fails outright if
the case diverges anywhere else first, and xfail is strict, so a case that
starts to match R must move to ``MATCHING``.
"""

import copy
import warnings
from functools import cache
from typing import Any

import numpy as np
import pytest
from blind_compare import (
    ALTERNATION_LIMIT,
    CASES,
    Divergence,
    _Case,
    _Differs,
    _linear_predictor,
    compare,
    compare_separated,
)
from grown_trees import model as fixture_model
from slow import SLOW

import modeltrees as mt
from modeltrees.tree import Node, StopReason

# (case, slow): cases that match R on every component.
MATCHING = [
    # Nodes with as many clusters as coefficients under unequal case
    # weights, which the instability tests run as partykit does (task 6k).
    ("blind_03_lm", False),
    ("blind_13_glm", True),
    # Weakly identified theta (glm.nb's alternation limit): theta at
    # WEAK_THETA_RTOL, or at ROUNDED_STOP_RTOL where only one side reached
    # the limit.
    ("blind_29_negbin", True),
    # Pruning counts coefficients only (task 6k), then as blind_29.
    ("blind_36_negbin", True),
    # Completely separated leaves: linear predictor and fitted values
    # instead of the non-identified slopes.
    ("blind_43_safelogit", False),
    ("blind_47_safelogit", False),
]

# (case, first diverging component, start of the difference, reason, slow).
# The mismatching cases of the second blind run (task 6l), xfail strict.
EXPECTED: list[tuple[str, str, str, str, bool]] = [
    (
        "blind_22_glm",
        "whole tree",
        "node 9: stop reason test_failed vs R pruned",
        (
            "G = 2 < k = 3 under case weights: R's chol() of the singular meat "
            "succeeds by rounding, the port stops; R prunes the node"
        ),
        False,
    ),
    (
        "blind_52_multinom",
        "node sizes and coefficients",
        "node 6: rows differ (65 vs R 135)",
        (
            "G = 3 < k = 4 under case weights at node 6: R's chol() of the "
            "singular meat succeeds by rounding and R splits, the port stops "
            "and AIC prunes node 5"
        ),
        False,
    ),
    (
        "blind_60_multinom",
        "split candidates and objectives",
        "node 6: split objectives:",
        (
            "nnet stops at maxit in two small split-search fits, 1.8e-5 above "
            "the Newton optimum"
        ),
        True,
    ),
]


def test_every_blind_case_is_listed() -> None:
    assert sorted([m[0] for m in MATCHING] + [e[0] for e in EXPECTED]) == CASES


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, id=name, marks=[SLOW] if slow else [])
        for name, slow in MATCHING
    ],
)
def test_blind_case_matches(name: str) -> None:
    compare(name)


@pytest.mark.parametrize(
    ("name", "component", "detail"),
    [
        pytest.param(
            name,
            component,
            detail,
            id=name,
            marks=[
                pytest.mark.xfail(
                    raises=Divergence,
                    strict=True,
                    reason=f"first diverging component: {component} ({reason})",
                ),
                *([SLOW] if slow else []),
            ],
        )
        for name, component, detail, reason, slow in EXPECTED
    ],
)
def test_blind_case(name: str, component: str, detail: str) -> None:
    try:
        compare(name)
    except Divergence as d:
        if d.component != component or not d.detail.startswith(detail):
            # A different first divergence is a failure, not the known one.
            raise AssertionError(f"unexpected divergence: {d}") from None
        raise


# Sensitivity: the comparisons of blind_compare that allow for quantities
# the data do not determine still catch a 1e-6 relative change of an
# identified quantity.

PERTURBATION = 1e-6


@cache
def _grown(name: str) -> mt.ModelTree:
    return _Case(name).grow()


def _perturbed_case(name: str, node_id: int, change: Any) -> _Case:
    case = _Case(name)
    case.f = copy.deepcopy(case.f)
    change(next(n for n in case.f["nodes"] if n["id"] == node_id))
    return case


def test_unperturbed_separated_leaf_matches() -> None:
    case = _Case("blind_43_safelogit")
    case.nodes(_grown("blind_43_safelogit"))


def test_perturbed_objective_of_a_separated_leaf_is_caught() -> None:
    def change(node: dict[str, Any]) -> None:
        node["objfun"] *= 1 + PERTURBATION

    case = _perturbed_case("blind_43_safelogit", 3, change)
    with pytest.raises(_Differs, match="node 3: objective"):
        case.nodes(_grown("blind_43_safelogit"))


@pytest.mark.parametrize("name", ["blind_43_safelogit", "blind_47_safelogit"])
def test_perturbed_fitted_value_of_a_separated_leaf_is_caught(name: str) -> None:
    case = _Case(name)
    node = case.node(3)
    fit = _grown(name)._grown.node(3).fit
    assert fit.extras["separation"] == "complete"
    x = case.node_data(node)[1]
    eta, mu = _linear_predictor(x, fit.coefficients)
    r_eta, r_mu = _linear_predictor(x, np.array(node["coefficients"], dtype=float))
    compare_separated(eta, mu, r_eta, r_mu, "node 3")
    r_mu = r_mu.copy()
    r_mu[0] *= 1 - PERTURBATION
    with pytest.raises(_Differs, match="node 3: fitted values"):
        compare_separated(eta, mu, r_eta, r_mu, "node 3")


@SLOW
@pytest.mark.parametrize("key", ["theta", "theta_irls", "theta_se"])
def test_perturbed_theta_of_an_identified_node_is_caught(key: str) -> None:
    # blind_29 node 3: theta 4.83 with standard error 1.65 and no warning.
    def change(node: dict[str, Any]) -> None:
        assert node["model_info"]["theta_warning"] is None
        node["model_info"][key] *= 1 + PERTURBATION

    case = _perturbed_case("blind_29_negbin", 3, change)
    with pytest.raises(_Differs, match=f"node 3: {key}:"):
        case.nodes(_grown("blind_29_negbin"))


# blind_36 node 112: theta 176, SE 2513; the port reaches glm.nb's
# alternation limit and R does not (docs/api-port.md, section 20).


def _node_112() -> tuple[_Case, Node]:
    """R's node 112 and the port's fit on its rows (the grown tree's fit,
    bit for bit), with R's flag set to the opposite of the port's so that
    the flags differ on any platform."""
    case = _Case("blind_36_negbin")
    case.f = copy.deepcopy(case.f)
    e = case.node(112)
    rows = np.array(e["rows"], dtype=np.intp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fixture_model(case.f).fit(
            case.y[rows],
            case.x[rows] if case.x is not None else None,
            start=None,
            weights=case.w[rows],
            offset=None if case.off is None else case.off[rows],
            cluster=None,
            caseweights=case.f["control"]["caseweights"],
            estfun=True,
            vcov=False,
        )
    flag = fit.extras["theta_warning"]
    r_flag = None if flag == ALTERNATION_LIMIT else ALTERNATION_LIMIT
    e["model_info"]["theta_warning"] = r_flag
    node = Node(
        id=112,
        depth=e["depth"],
        rows=rows,
        nobs=e["nobs"],
        fit=fit,
        stop_reason=StopReason(e["stop_reason"]),
    )
    return case, node


def test_unperturbed_node_with_differing_flags_matches() -> None:
    case, node = _node_112()
    case.fit(node, case.node(112), "node 112")


@pytest.mark.parametrize("key", ["theta", "theta_irls", "theta_se"])
def test_perturbed_theta_with_differing_flags_is_caught(key: str) -> None:
    case, node = _node_112()
    case.node(112)["model_info"][key] *= 1 + 1e-5
    # The message carries both sides' theta, theta_irls, iter and flag.
    with pytest.raises(
        _Differs,
        match=(
            f"node 112: {key}:.*; port theta .*, theta_irls .*, iter 1, "
            "theta_warning .*; R theta .*, theta_irls .*, iter 1, theta_warning"
        ),
    ):
        case.fit(node, case.node(112), "node 112")


@pytest.mark.parametrize(
    ("what", "change"),
    [
        ("objective", lambda e: e.update(objfun=e["objfun"] * (1 + PERTURBATION))),
        (
            "coefficients",
            lambda e: e["coefficients"].__setitem__(
                1, e["coefficients"][1] * (1 + PERTURBATION)
            ),
        ),
    ],
)
def test_perturbed_fit_with_differing_flags_is_caught(what: str, change: Any) -> None:
    case, node = _node_112()
    change(case.node(112))
    with pytest.raises(_Differs, match=f"node 112: {what}:"):
        case.fit(node, case.node(112), "node 112")


# blind_13_glm: node 67 is completely separated by x1 and its coefficients
# are not compared; node 69, the nearest to the boundary of the nodes that
# are not separated, keeps the coefficient comparison (docs/api-port.md,
# section 21).


def _glm_node(node_id: int) -> tuple[_Case, Node]:
    """R's blind_13 node and the port's fit on its rows."""
    case = _Case("blind_13_glm")
    case.f = copy.deepcopy(case.f)
    e = case.node(node_id)
    rows = np.array(e["rows"], dtype=np.intp)
    assert case.x is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fixture_model(case.f).fit(
            case.y[rows],
            case.x[rows],
            start=None,
            weights=case.w[rows],
            offset=None,
            cluster=None,
            caseweights=case.f["control"]["caseweights"],
            estfun=False,
            vcov=False,
        )
    node = Node(
        id=node_id,
        depth=e["depth"],
        rows=rows,
        nobs=e["nobs"],
        fit=fit,
        stop_reason=StopReason(e["stop_reason"]),
    )
    return case, node


@pytest.mark.parametrize(("node_id", "separated"), [(67, True), (69, False)])
def test_unperturbed_glm_node_matches(node_id: int, separated: bool) -> None:
    case, node = _glm_node(node_id)
    e = case.node(node_id)
    coefficients = np.array(e["coefficients"], dtype=float)
    assert (case.separated_glm(e, coefficients) is not None) is separated
    case.fit(node, e, f"node {node_id}")


def test_perturbed_objective_of_a_separated_glm_node_is_caught() -> None:
    case, node = _glm_node(67)
    e = case.node(67)
    e["objfun"] *= 1 + PERTURBATION
    with pytest.raises(_Differs, match="node 67: objective"):
        case.fit(node, e, "node 67")


def test_perturbed_fitted_value_of_a_separated_glm_node_is_caught() -> None:
    case, node = _glm_node(67)
    e = case.node(67)
    coefficients = np.array(e["coefficients"], dtype=float)
    rows = case.separated_glm(e, coefficients)
    assert rows is not None
    eta, mu = case.glm_means(node.fit.coefficients, rows)
    r_eta, r_mu = case.glm_means(coefficients, rows)
    compare_separated(eta, mu, r_eta, r_mu, "node 67")
    r_mu = r_mu.copy()
    r_mu[np.argmax(r_mu)] *= 1 - PERTURBATION
    with pytest.raises(_Differs, match="node 67: fitted values"):
        compare_separated(eta, mu, r_eta, r_mu, "node 67")


@pytest.mark.parametrize("index", [0, 1])
def test_perturbed_coefficient_of_a_glm_node_not_separated_is_caught(
    index: int,
) -> None:
    case, node = _glm_node(69)
    e = case.node(69)
    e["coefficients"][index] *= 1 + PERTURBATION
    with pytest.raises(_Differs, match="node 69: coefficients"):
        case.fit(node, e, "node 69")
