"""The cases of the blind differential run (task 6j) where the port and R
differ, in tests/fixtures/blind_cases/ (README.md there has the analysis).

Each test compares the case with R in the order of CLAUDE.md
(tests/blind_compare.py). The cases the port now matches (``MATCHING``) are
standing tests that must pass on every component. The others are marked
xfail with their first diverging component; such a test fails outright if
the case diverges anywhere else first, and xfail is strict, so a case that
starts to match R must move to ``MATCHING``.
"""

import pytest
from blind_compare import CASES, Divergence, compare
from slow import SLOW

# (case, slow): cases that match R on every component.
MATCHING = [
    # Nodes with as many clusters as coefficients under unequal case
    # weights, which the instability tests run as partykit does (task 6k).
    ("blind_03_lm", False),
    ("blind_13_glm", True),
]

# (case, first diverging component, start of the difference, reason, slow)
EXPECTED = [
    (
        "blind_29_negbin",
        "node sizes and coefficients",
        "node 4: theta:",
        (
            "node 4's theta (4924, SE 81787, glm.nb alternation limit) differs by "
            "4.7e-8 relative: not identified beyond rounding"
        ),
        True,
    ),
    (
        "blind_36_negbin",
        "node sizes and coefficients",
        "node 86: theta_se:",
        (
            "node 86's theta_se (theta 272, SE 3826, glm.nb alternation limit) "
            "differs by 1.4e-8 relative: not identified beyond rounding"
        ),
        True,
    ),
    (
        "blind_43_safelogit",
        "node sizes and coefficients",
        "node 3: coefficients:",
        (
            "node 3 is completely separated (constant response): its slopes are "
            "rounding residue, 2e-16 against R's -7.7e-8"
        ),
        False,
    ),
    (
        "blind_47_safelogit",
        "node sizes and coefficients",
        "node 3: coefficients:",
        (
            "node 3 is completely separated (constant response): its slopes are "
            "rounding residue, -5.1e-11 against R's -1.2e-9"
        ),
        False,
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
