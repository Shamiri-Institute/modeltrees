"""Parser for the v0.1 formula subset (docs/api-design.md, section 1.2).

```
formula   := response "~" part [ "|" part ]
response  := name
part      := term { ("+" | "-") term }
term      := name | "1" | "0"
name      := identifier | "`" any text without a backtick "`"
```

A part may also start with a sign, as R allows (``y ~ -1 + x | z``). Every
construct outside the grammar raises :class:`FormulaError` naming it, with
the 0-based character position in the formula.
"""

from dataclasses import dataclass
from enum import Enum

from modeltrees.errors import FormulaError

_SUGGESTION = "compute the column in the dataframe first"


class _Kind(Enum):
    NAME = "name"
    NUMBER = "number"
    TILDE = "~"
    BAR = "|"
    PLUS = "+"
    MINUS = "-"
    END = "end"


@dataclass(frozen=True)
class _Token:
    kind: _Kind
    text: str
    position: int
    quoted: bool = False  # a backticked name


@dataclass(frozen=True)
class ParsedFormula:
    """A formula of the v0.1 subset.

    ``regressors`` and ``partitioning`` are column names in formula order,
    duplicates dropped. ``intercept`` says whether the regressor part has an
    intercept. ``expanded`` is the two-part form, for example
    ``"y ~ 1 | z1 + z2"`` for the one-part formula ``"y ~ z1 + z2"``.
    """

    response: str
    regressors: tuple[str, ...]
    intercept: bool
    partitioning: tuple[str, ...]
    expanded: str


def _is_identifier_start(c: str) -> bool:
    return c.isalpha() or c in "_."


def _is_identifier_char(c: str) -> bool:
    return c.isalnum() or c in "_."


_OPERATORS = {"~": _Kind.TILDE, "|": _Kind.BAR, "+": _Kind.PLUS, "-": _Kind.MINUS}

# Characters that start an unsupported construct, with its name.
_UNSUPPORTED = {
    ":": "interaction ':'",
    "*": "interaction '*'",
    "/": "nesting '/'",
    "^": "power '^'",
    "%": "operator '%...%'",
    "(": "parentheses",
    ")": "parentheses",
}


def _tokenize(formula: str) -> list[_Token]:
    tokens: list[_Token] = []
    i, n = 0, len(formula)
    while i < n:
        c = formula[i]
        if c.isspace():
            i += 1
        elif c in _OPERATORS:
            tokens.append(_Token(_OPERATORS[c], c, i))
            i += 1
        elif c == "`":
            end = formula.find("`", i + 1)
            if end < 0:
                raise FormulaError(
                    f"unterminated backtick at position {i} in {formula!r}", i
                )
            if end == i + 1:
                raise FormulaError(
                    f"empty backticked name at position {i} in {formula!r}", i
                )
            tokens.append(_Token(_Kind.NAME, formula[i + 1 : end], i, quoted=True))
            i = end + 1
        elif c.isdigit():
            j = i
            while j < n and _is_identifier_char(formula[j]):
                j += 1
            tokens.append(_Token(_Kind.NUMBER, formula[i:j], i))
            i = j
        elif _is_identifier_start(c):
            j = i
            while j < n and _is_identifier_char(formula[j]):
                j += 1
            if formula[j:].lstrip().startswith("("):
                raise _call(formula[i:j], i, formula)
            tokens.append(_Token(_Kind.NAME, formula[i:j], i))
            i = j
        elif c in _UNSUPPORTED:
            raise _unsupported(_UNSUPPORTED[c], i, formula)
        else:
            raise FormulaError(
                f"unexpected character {c!r} at position {i} in {formula!r}", i
            )
    tokens.append(_Token(_Kind.END, "", n))
    return tokens


def _call(name: str, position: int, formula: str) -> FormulaError:
    """A name followed by "(": a function call or transformation."""
    if name == "offset":
        return FormulaError(
            f"offset(...) at position {position} is not supported inside the "
            f"formula ({formula!r}); pass offset= instead",
            position,
        )
    if name == "cbind":
        return _unsupported("a cbind(...) response", position, formula)
    return _unsupported(f"function call {name}(...)", position, formula)


def _unsupported(
    construct: str, position: int, formula: str, suggestion: str = _SUGGESTION
) -> FormulaError:
    return FormulaError(
        f"{construct} at position {position} is not supported in formulas "
        f"({formula!r}); {suggestion}",
        position,
    )


class _Parser:
    def __init__(self, formula: str) -> None:
        self.formula = formula
        self.tokens = _tokenize(formula)
        self.i = 0

    def peek(self, offset: int = 0) -> _Token:
        return self.tokens[min(self.i + offset, len(self.tokens) - 1)]

    def next(self) -> _Token:
        token = self.peek()
        self.i += 1
        return token

    def error(self, message: str, token: _Token) -> FormulaError:
        return FormulaError(
            f"{message} at position {token.position} in {self.formula!r}",
            token.position,
        )

    def name(self, token: _Token, where: str) -> str:
        if token.kind is _Kind.NAME:
            if token.text == "." and not token.quoted:
                raise _unsupported(
                    "the '.' shorthand",
                    token.position,
                    self.formula,
                    "list the columns by name",
                )
            return token.text
        if token.kind is _Kind.END:
            raise self.error(f"missing {where}", token)
        raise self.error(
            f"expected a column name for the {where}, got {token.text!r}", token
        )

    def parse(self) -> ParsedFormula:
        first = self.peek()
        if first.kind is _Kind.TILDE:
            raise self.error("missing response before '~'", first)
        response = self.name(self.next(), "response")
        tilde = self.next()
        if tilde.kind is not _Kind.TILDE:
            if tilde.kind is _Kind.END:
                raise self.error("missing '~'", tilde)
            raise self.error(
                f"expected '~' after the response, got {tilde.text!r}", tilde
            )
        parts = [self.part()]
        while self.peek().kind is _Kind.BAR:
            bar = self.next()
            if len(parts) == 2:
                raise _unsupported(
                    "a third '|' part",
                    bar.position,
                    self.formula,
                    "use y ~ x1 + x2 | z1 + z2",
                )
            parts.append(self.part())
        end = self.peek()
        if end.kind is not _Kind.END:
            raise self.error(f"unexpected {end.text!r}", end)

        if len(parts) == 1:
            # glmtree.R:11-18 and lmtree.R:200-207 rewrite y ~ z to y ~ 1 | z.
            regressors: list[_Token] = []
            intercept = True
            z_terms, z_intercepts = parts[0]
        else:
            terms, intercept_tokens = parts[0]
            regressors = terms
            intercept = True
            # R's terms(): the last intercept token wins (y ~ 0 + 1 + x has one).
            for sign, _ in intercept_tokens:
                intercept = sign is _Kind.PLUS
            if not intercept and not regressors:
                raise self.error(
                    "the regressor part has no columns (no intercept and no terms)",
                    self.tokens[0] if not intercept_tokens else intercept_tokens[-1][1],
                )
            z_terms, z_intercepts = parts[1]
        if z_intercepts:
            hint = (
                " (a one-part formula y ~ z1 + z2 lists only partitioning variables)"
                if len(parts) == 1
                else ""
            )
            raise self.error(
                "intercept terms are not allowed among the partitioning variables"
                + hint,
                z_intercepts[0][1],
            )
        if not z_terms:
            raise self.error("no partitioning variables", self.tokens[-1])
        for token in [*regressors, *z_terms]:
            if token.text == response:
                raise self.error(
                    f"the response {response!r} appears on the right-hand side", token
                )
        x = tuple(dict.fromkeys(t.text for t in regressors))
        z = tuple(dict.fromkeys(t.text for t in z_terms))
        return ParsedFormula(
            response, x, intercept, z, _expanded(response, x, intercept, z)
        )

    def part(self) -> tuple[list[_Token], list[tuple[_Kind, _Token]]]:
        """Column terms, and the intercept tokens with their signs."""
        terms: list[_Token] = []
        intercepts: list[tuple[_Kind, _Token]] = []
        sign = _Kind.PLUS
        if self.peek().kind in (_Kind.PLUS, _Kind.MINUS):
            sign = self.next().kind
        while True:
            token = self.next()
            if token.kind is _Kind.NUMBER:
                if token.text not in ("0", "1"):
                    raise self.error(
                        f"numeric constant {token.text!r} is not a term (names "
                        "that start with a digit need backticks)",
                        token,
                    )
                if sign is _Kind.MINUS and token.text == "0":
                    raise _unsupported(
                        "'- 0'",
                        token.position,
                        self.formula,
                        "write '+ 1' for an intercept or '- 1' for none",
                    )
                # "0" and "- 1" remove the intercept, "1" adds it.
                is_one = token.text == "1"
                intercepts.append(
                    (
                        _Kind.PLUS if is_one and sign is _Kind.PLUS else _Kind.MINUS,
                        token,
                    )
                )
            elif token.kind is _Kind.NAME:
                name = self.name(token, "term")
                if sign is _Kind.MINUS:
                    raise _unsupported(
                        f"removing a term ('- {name}'; a name containing '-' "
                        "needs backticks)",
                        token.position,
                        self.formula,
                    )
                terms.append(token)
            elif token.kind in (_Kind.END, _Kind.BAR):
                raise self.error("missing term", token)
            else:
                raise self.error(f"expected a term, got {token.text!r}", token)
            if self.peek().kind not in (_Kind.PLUS, _Kind.MINUS):
                return terms, intercepts
            sign = self.next().kind


def _quote(name: str) -> str:
    simple = (
        name
        and _is_identifier_start(name[0])
        and all(_is_identifier_char(c) for c in name)
        and name != "."
    )
    return name if simple else f"`{name}`"


def _expanded(
    response: str, x: tuple[str, ...], intercept: bool, z: tuple[str, ...]
) -> str:
    if not x:
        rhs = "1"
    else:
        rhs = " + ".join(_quote(v) for v in x) + ("" if intercept else " - 1")
    return f"{_quote(response)} ~ {rhs} | {' + '.join(_quote(v) for v in z)}"


def parse_formula(formula: str) -> ParsedFormula:
    """Parse a formula of the v0.1 subset, or raise :class:`FormulaError`."""
    if not isinstance(formula, str):
        raise TypeError(f"formula must be a string, got {type(formula).__name__}")
    return _Parser(formula).parse()
