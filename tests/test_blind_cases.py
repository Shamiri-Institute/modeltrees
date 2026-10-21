"""The cases of the blind differential run (task 6j) where the port and R
differ, in tests/fixtures/blind_cases/ (README.md there has the analysis).

Each test compares the case with R in the order of CLAUDE.md
(tests/blind_compare.py). The cases the port now matches (``MATCHING``) are
standing tests that must pass on every component. The others are marked
xfail with their first diverging component; such a test fails outright if
the case diverges anywhere else first, and xfail is strict, so a case that
starts to match R must move to ``MATCHING``.
"""

import copy
from functools import cache
from typing import Any

import numpy as np
import pytest
from blind_compare import (
    CASES,
    Divergence,
    _Case,
    _Differs,
    _linear_predictor,
    compare,
    compare_separated,
)
from slow import SLOW

import modeltrees as mt

# (case, slow): cases that match R on every component.
MATCHING = [
    # Nodes with as many clusters as coefficients under unequal case
    # weights, which the instability tests run as partykit does (task 6k).
    ("blind_03_lm", False),
    ("blind_13_glm", True),
    # Weakly identified theta (glm.nb's alternation limit): theta at
    # WEAK_THETA_RTOL, and the flag where the final thetas agree.
    ("blind_29_negbin", True),
    # Pruning counts coefficients only (task 6k), then as blind_29.
    ("blind_36_negbin", True),
    # Completely separated leaves: linear predictor and fitted values
    # instead of the non-identified slopes.
    ("blind_43_safelogit", False),
    ("blind_47_safelogit", False),
]

# (case, first diverging component, start of the difference, reason, slow).
# Empty since task 6k; a case that diverges from R goes here, xfail strict.
EXPECTED: list[tuple[str, str, str, str, bool]] = []


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
