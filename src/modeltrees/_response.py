"""Model-specific coding of the response (docs/api-design.md, section 4.3).

The design builder returns the response raw: float64 values (a Boolean as
0/1) or 0-based codes into the levels kept after missing-data removal.
Each entry point turns it into what its node model takes, and records the
level order on the tree as ``response_levels``:

- ``binomial``: a categorical response by R's rule, the first level failure
  (0) and every other level success (1); a numeric response must be 0/1, or
  proportions in [0, 1] with weights as the number of trials.
- ``safelogit``: as binomial, but proportions are not allowed.
- ``multinom``: a categorical response (or an integer one with ``levels``)
  as codes; the first kept level is the reference class.
- ``gaussian`` (lmtree and glmtree's gaussian family): numeric.
- ``poisson`` and ``negbin``: non-negative integer counts.

A Boolean counts as numeric 0/1 everywhere but multinom. Violations raise
:class:`~modeltrees.errors.DataError` naming the column.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from modeltrees._design import Response
from modeltrees.errors import DataError

ResponseRule = Literal[
    "gaussian", "binomial", "poisson", "negbin", "safelogit", "multinom"
]


@dataclass(frozen=True)
class CodedResponse:
    """The response as the node model takes it.

    ``values`` is float64, or 0-based class codes (intp) for multinom.
    ``levels`` is the level order of a categorical response (the tree's
    ``response_levels``), None for a numeric one.
    """

    values: NDArray[Any]
    levels: tuple[str, ...] | None


def code_response(
    response: Response, rule: ResponseRule, *, weighted: bool
) -> CodedResponse:
    """Code ``response`` for a model. ``weighted`` says whether the user
    passed weights, which binomial proportions need as trials."""
    term = response.term
    name = term.name
    categorical = term.levels is not None
    if rule == "multinom":
        if not categorical:
            raise DataError(
                f"the response {name!r} of a multinomial tree must be categorical "
                f"(string, categorical or Enum), or integer with levels=; got "
                f"{term.storage} values"
            )
        assert term.levels is not None
        if len(term.levels) < 2:
            raise DataError(
                f"the response {name!r} has {len(term.levels)} level(s) after "
                "missing-data removal; a multinomial tree needs at least two"
            )
        return CodedResponse(np.asarray(response.values, dtype=np.intp), term.levels)

    if rule in ("binomial", "safelogit"):
        if categorical:
            assert term.levels is not None
            # R's binomial()$initialize: y != levels(y)[1].
            values = (np.asarray(response.values) != 0).astype(np.float64)
            return CodedResponse(values, term.levels)
        y = np.asarray(response.values, dtype=np.float64)
        if np.any((y < 0) | (y > 1)):
            raise DataError(
                f"the response {name!r} has values outside [0, 1]; a binomial "
                "response must be 0/1, Boolean, categorical or proportions"
            )
        proportions = bool(np.any((y > 0) & (y < 1)))
        if proportions and rule == "safelogit":
            raise DataError(
                f"the response {name!r} has values strictly between 0 and 1; "
                "safelogittree needs a binary response (0/1, Boolean or "
                "categorical)"
            )
        if proportions and not weighted:
            raise DataError(
                f"the response {name!r} has proportions (values strictly between "
                "0 and 1); pass the number of trials as weights="
            )
        return CodedResponse(y, None)

    if categorical:
        raise DataError(
            f"the response {name!r} is categorical ({term.storage}); this model "
            "needs a numeric response"
        )
    y = np.asarray(response.values, dtype=np.float64)
    if rule in ("poisson", "negbin"):
        if np.any(y < 0):
            raise DataError(
                f"the response {name!r} has negative values; counts must be "
                "non-negative"
            )
        if np.any(y != np.round(y)):
            raise DataError(
                f"the response {name!r} has non-integer values; counts must be "
                "whole numbers"
            )
    return CodedResponse(y, None)
