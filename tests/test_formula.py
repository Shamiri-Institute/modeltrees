"""The formula parser (docs/api-design.md, section 1.2)."""

import pytest

from modeltrees._formula import ParsedFormula, parse_formula
from modeltrees.errors import FormulaError


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        # Two parts, with an intercept.
        ("y ~ x | z", ParsedFormula("y", ("x",), True, ("z",), "y ~ x | z")),
        (
            "y ~ x1 + x2 | z1 + z2",
            ParsedFormula(
                "y", ("x1", "x2"), True, ("z1", "z2"), "y ~ x1 + x2 | z1 + z2"
            ),
        ),
        # One part: intercept-only, partitioned by every term.
        (
            "y ~ z1 + z2",
            ParsedFormula("y", (), True, ("z1", "z2"), "y ~ 1 | z1 + z2"),
        ),
        ("y ~ 1 | z", ParsedFormula("y", (), True, ("z",), "y ~ 1 | z")),
        ("y ~ 1 + x | z", ParsedFormula("y", ("x",), True, ("z",), "y ~ x | z")),
        # No intercept: 0, + 0, - 1, a leading - 1.
        ("y ~ 0 + x | z", ParsedFormula("y", ("x",), False, ("z",), "y ~ x - 1 | z")),
        ("y ~ x + 0 | z", ParsedFormula("y", ("x",), False, ("z",), "y ~ x - 1 | z")),
        ("y ~ x - 1 | z", ParsedFormula("y", ("x",), False, ("z",), "y ~ x - 1 | z")),
        ("y ~ -1 + x | z", ParsedFormula("y", ("x",), False, ("z",), "y ~ x - 1 | z")),
        # The last intercept token wins, as in R's terms().
        ("y ~ 0 + 1 + x | z", ParsedFormula("y", ("x",), True, ("z",), "y ~ x | z")),
        (
            "y ~ 1 + 0 + x | z",
            ParsedFormula("y", ("x",), False, ("z",), "y ~ x - 1 | z"),
        ),
        ("y ~ x - 1 + 1 | z", ParsedFormula("y", ("x",), True, ("z",), "y ~ x | z")),
        # A leading +, as R allows.
        ("y ~ +x | +z", ParsedFormula("y", ("x",), True, ("z",), "y ~ x | z")),
        # Duplicates are dropped; a column may be in both parts.
        (
            "y ~ x + x | x + z + x",
            ParsedFormula("y", ("x",), True, ("x", "z"), "y ~ x | x + z"),
        ),
        # Backticks, dots and underscores, case kept.
        (
            "`my y` ~ `x-1` + .x | `2z` + z_1 + Z.a",
            ParsedFormula(
                "my y",
                ("x-1", ".x"),
                True,
                ("2z", "z_1", "Z.a"),
                "`my y` ~ `x-1` + .x | `2z` + z_1 + Z.a",
            ),
        ),
        ("`x` ~ `y` | `z`", ParsedFormula("x", ("y",), True, ("z",), "x ~ y | z")),
        # Whitespace does not matter.
        ("y~x|z", ParsedFormula("y", ("x",), True, ("z",), "y ~ x | z")),
        ("  y ~\tx\n| z ", ParsedFormula("y", ("x",), True, ("z",), "y ~ x | z")),
        # Names that are not ASCII.
        ("ñ ~ é | ü", ParsedFormula("ñ", ("é",), True, ("ü",), "ñ ~ é | ü")),
        # Upper case is a different column.
        ("y ~ X | x", ParsedFormula("y", ("X",), True, ("x",), "y ~ X | x")),
    ],
)
def test_supported_forms(formula: str, expected: ParsedFormula) -> None:
    assert parse_formula(formula) == expected


@pytest.mark.parametrize(
    ("formula", "position", "fragment"),
    [
        # Unsupported constructs, each named.
        ("y ~ x:z | w", 5, "interaction ':'"),
        ("y ~ x*z | w", 5, "interaction '*'"),
        ("y ~ x/z | w", 5, "nesting '/'"),
        ("y ~ x^2 | w", 5, "power '^'"),
        ("y ~ x %in% z | w", 6, "operator '%...%'"),
        ("y ~ log(x) | z", 4, "function call log(...)"),
        ("y ~ I(x^2) | z", 4, "function call I(...)"),
        ("y ~ C(x) | z", 4, "function call C(...)"),
        ("y ~ poly(x, 2) | z", 4, "function call poly(...)"),
        ("log(y) ~ x | z", 0, "function call log(...)"),
        ("y ~ x + offset(e) | z", 8, "offset(...)"),
        ("cbind(s, f) ~ x | z", 0, "cbind(...) response"),
        ("y ~ . | z", 4, "'.' shorthand"),
        ("y ~ x | .", 8, "'.' shorthand"),
        ("y ~ .", 4, "'.' shorthand"),
        ("y ~ x - z | w", 8, "removing a term ('- z'"),
        ("y ~ x - 0 | w", 8, "'- 0'"),
        ("y ~ x | z | w", 10, "third '|' part"),
        ("y ~ (x) | z", 4, "parentheses"),
        # Intercept errors.
        ("y ~ 0 | z", 4, "no columns"),
        ("y ~ -1 | z", 5, "no columns"),
        ("y ~ 1 - 1 | z", 8, "no columns"),
        ("y ~ x | 1 + z", 8, "intercept terms are not allowed"),
        ("y ~ x | z - 1", 12, "intercept terms are not allowed"),
        ("y ~ x | 0 + z", 8, "intercept terms are not allowed"),
        ("y ~ 1", 4, "one-part formula"),
        # The response on the right.
        ("y ~ y | z", 4, "response 'y' appears"),
        ("y ~ x | z + y", 12, "response 'y' appears"),
        ("y ~ z + y", 8, "response 'y' appears"),
        # Malformed input.
        ("y ~ 2 | z", 4, "numeric constant '2'"),
        ("y ~ 1x | z", 4, "numeric constant '1x'"),
        ("y x | z", 2, "expected '~'"),
        ("y", 1, "missing '~'"),
        ("~ x | z", 0, "missing response"),
        ("y ~ ", 4, "missing term"),
        ("y ~ x + | z", 8, "missing term"),
        ("y ~ x |", 7, "missing term"),
        ("y ~ x | z +", 11, "missing term"),
        ("y ~ x ~ z", 6, "unexpected '~'"),
        ("y ~ x & z", 6, "unexpected character '&'"),
        ("y ~ x, z", 5, "unexpected character ','"),
        ("y ~ `x", 4, "unterminated backtick"),
        ("y ~ `` | z", 4, "empty backticked name"),
        ("y + w ~ x | z", 2, "expected '~'"),
    ],
)
def test_errors(formula: str, position: int, fragment: str) -> None:
    with pytest.raises(FormulaError) as info:
        parse_formula(formula)
    assert fragment in str(info.value)
    assert info.value.position == position
    assert f"position {position}" in str(info.value)


def test_unsupported_constructs_suggest_a_dataframe_column() -> None:
    with pytest.raises(FormulaError, match="compute the column in the dataframe"):
        parse_formula("y ~ log(x) | z")
    with pytest.raises(FormulaError, match="pass offset= instead"):
        parse_formula("y ~ x + offset(log(e)) | z")


def test_formula_error_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        parse_formula("y ~ 0 | z")


def test_formula_must_be_a_string() -> None:
    with pytest.raises(TypeError):
        parse_formula(42)  # type: ignore[arg-type]
