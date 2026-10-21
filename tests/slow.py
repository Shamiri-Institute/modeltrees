"""The ``slow`` marker for tests excluded from the default run.

pyproject.toml registers the marker and deselects it in ``addopts``;
``pytest -m ""`` runs everything. A test is marked when it or its
parametrized group takes more than about 5 seconds: growing the larger
negative binomial and multinomial fixture trees, the parallel fixture
trees and the benchmark tree.
"""

from collections.abc import Iterable
from typing import Any

import pytest

SLOW = pytest.mark.slow

# Fixture trees whose growth takes about 4 seconds or more (measured with
# --durations on the whole suite).
SLOW_TREES = frozenset(
    {
        "sim_negbin_weights",
        "sim_negbin_cluster",
        "sim_negbin_onesplit",
        "sim_cluster_models_negbin_cluster",
        "sim_multinom_onesplit",
        "sim_multinom_cluster",
        "sim_multinom_noref_onesplit",
        "sim_cluster_models_multinom_cluster",
    }
)


def mark_slow_trees(params: Iterable[Any], *, name_at: int = 0) -> list[Any]:
    """The parameters, with ``SLOW`` on those naming a tree in ``SLOW_TREES``.

    ``params`` may hold scalars, tuples or ``pytest.param`` objects;
    ``name_at`` is the position of the tree name among the values. Ids and
    existing marks are kept.
    """
    out = []
    for p in params:
        if hasattr(p, "values") and hasattr(p, "marks"):
            values, marks, id_ = tuple(p.values), list(p.marks), p.id
        elif isinstance(p, tuple):
            values, marks, id_ = p, [], None
        else:
            values, marks, id_ = (p,), [], None
        if values[name_at] in SLOW_TREES:
            marks.append(SLOW)
        out.append(pytest.param(*values, marks=marks, id=id_))
    return out
