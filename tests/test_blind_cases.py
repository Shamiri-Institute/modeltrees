"""The cases of the blind differential run (task 6j) where the port and R
differ, in tests/fixtures/blind_cases/ (README.md there has the analysis).

Each test compares the case with R in the order of CLAUDE.md
(tests/blind_compare.py) and is marked xfail with its first diverging
component. The test fails outright if the case diverges anywhere else
first, and xfail is strict, so a case that starts to match R must have its
mark removed.
"""

import pytest
from blind_compare import CASES, Divergence, compare
from slow import SLOW

# (case, first diverging component, start of the difference, reason, slow)
EXPECTED = [
    (
        "blind_03_lm",
        "node sizes and coefficients",
        "node 21: rows differ",
        (
            "node 20 has as many clusters as coefficients (3) with case weights: "
            "partykit tests it, the port stops at G <= k"
        ),
        False,
    ),
    (
        "blind_13_glm",
        "node sizes and coefficients",
        "node 14: rows differ",
        (
            "node 13 has as many clusters as coefficients (2) with case weights: "
            "partykit tests it, the port stops at G <= k"
        ),
        True,
    ),
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
        "node 19: rows differ",
        (
            "node 18 is pruned: the port counts theta in the pruning df "
            "(docs/decisions.md), partykit does not"
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
    assert sorted(e[0] for e in EXPECTED) == CASES


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
