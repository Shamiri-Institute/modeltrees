"""Post-pruning rules for MOB.

A rule receives, for an inner node whose kids are all terminal, the pair
``objfun = (node objective, sum of the kids' objectives)``,
``df = (k, n_kids * k + dfsplit)`` and ``nobs = (node nobs, n)``, with ``k``
the root's coefficient count and ``n`` the size of the full sample
(``mob_prune``, reference/partykit/R/modelparty.R:625-676). It returns True
to prune the kids.

``mob()`` installs rules for a negative log-likelihood objective
(modelparty.R:101-119), ``lmtree()`` rules for a residual sum of squares
(lmtree.R:12-29). A model can choose its rules with a ``prune_rule(name)``
method, as :class:`modeltrees.models.lm.LmModel` does.
"""

import math
from collections.abc import Callable

PruneRule = Callable[[tuple[float, float], tuple[int, int], tuple[float, float]], bool]


def _mob_aic(
    objfun: tuple[float, float], df: tuple[int, int], nobs: tuple[float, float]
) -> bool:
    return (2 * objfun[0] + 2 * df[0]) < (2 * objfun[1] + 2 * df[1])


def _mob_bic(
    objfun: tuple[float, float], df: tuple[int, int], nobs: tuple[float, float]
) -> bool:
    # R quirk, replicated: mob()'s BIC takes log(n) of the full sample
    # (modelparty.R:110), not of the node being pruned. nobs[1] is that n.
    log_n = math.log(nobs[1])
    return (2 * objfun[0] + log_n * df[0]) < (2 * objfun[1] + log_n * df[1])


def _lm_aic(
    objfun: tuple[float, float], df: tuple[int, int], nobs: tuple[float, float]
) -> bool:
    return (nobs[0] * math.log(objfun[0]) + 2 * df[0]) < (
        nobs[0] * math.log(objfun[1]) + 2 * df[1]
    )


def _lm_bic(
    objfun: tuple[float, float], df: tuple[int, int], nobs: tuple[float, float]
) -> bool:
    # As in lmtree.R:19, the penalty uses log of the full sample size.
    log_n = math.log(nobs[1])
    return (nobs[0] * math.log(objfun[0]) + log_n * df[0]) < (
        nobs[0] * math.log(objfun[1]) + log_n * df[1]
    )


def _lookup(name: str, aic: PruneRule, bic: PruneRule) -> PruneRule | None:
    key = name.lower()
    if key == "aic":
        return aic
    if key == "bic":
        return bic
    if key == "none":
        return None
    raise ValueError(f"unknown prune rule {name!r}; use 'aic', 'bic' or 'none'")


def mob_prune_rule(name: str) -> PruneRule | None:
    """``mob()``'s "aic", "bic" or "none" rule for a negative log-likelihood."""
    return _lookup(name, _mob_aic, _mob_bic)


def lm_prune_rule(name: str) -> PruneRule | None:
    """``lmtree()``'s "aic", "bic" or "none" rule for a residual sum of squares."""
    return _lookup(name, _lm_aic, _lm_bic)
