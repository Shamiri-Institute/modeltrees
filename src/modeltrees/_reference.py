"""A switch that runs the exact reference implementation.

The speedups in split search and in the QR (docs/performance.md) must not
change results. The code they replace stays in the library as the
reference implementation, and ``reference_path()`` selects it everywhere:
the per-candidate split search on fancy-indexed rows, full fits in split
search, and the dqrdc2 port for every least-squares solve. Tests grow trees
both ways and compare them.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_REFERENCE: ContextVar[bool] = ContextVar("modeltrees_reference", default=False)


@contextmanager
def reference_path() -> Iterator[None]:
    """Use the exact reference implementation inside the ``with`` block."""
    token = _REFERENCE.set(True)
    try:
        yield
    finally:
        _REFERENCE.reset(token)


def reference_active() -> bool:
    return _REFERENCE.get()
