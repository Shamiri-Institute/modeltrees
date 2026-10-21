"""Parallel split search: exact candidate fits in worker processes.

docs/api-design.md, section 6. One :class:`SplitPool` per tree, created by
``tree.mob`` when ``n_jobs`` gives more than one worker and shut down when the
tree is grown. Split search hands it the exact fits of many candidates; it
cuts them into about four contiguous chunks per worker, and the objectives
come back by candidate index, so the assembled vector is the one the serial
loop builds whatever the chunking or completion order.

Workers start with the BLAS thread variables set to 1 in their environment
(``_PINNED``). A BLAS library reads them once, when numpy loads it, so they
are set in ``os.environ`` for as long as the pool lives: the forkserver (or
the spawned interpreter) inherits them when it starts, before any import.
"""

import contextlib
import ctypes
import itertools
import multiprocessing
import os
import pickle
import sys
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray

from modeltrees.split_search import SideFit, _pair_objective

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# A search runs in parallel only when its candidates times the node's rows
# exceed this (docs/performance.md, "Parallel split search"). It changes
# time, never results. Tests lower it with parallel_threshold().
_THRESHOLD = 2e6

# Chunks per worker, for load balance.
_CHUNKS_PER_WORKER = 4

_PINNED = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


@contextlib.contextmanager
def parallel_threshold(value: float) -> Iterator[None]:
    """Test hook: run searches in parallel above ``value`` instead."""
    global _THRESHOLD
    old = _THRESHOLD
    _THRESHOLD = value
    try:
        yield
    finally:
        _THRESHOLD = old


def resolve_n_jobs(n_jobs: int | None) -> int:
    """The number of workers: None and 1 give 1 (serial), -1 gives
    ``os.process_cpu_count()``, k > 1 gives k."""
    if n_jobs is not None and (
        not isinstance(n_jobs, (int, np.integer)) or isinstance(n_jobs, bool)
    ):
        raise TypeError(f"n_jobs must be an int or None, got {n_jobs!r}")
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs == -1:
        return os.process_cpu_count() or 1
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be None, -1 or a positive int, got {n_jobs}")
    return int(n_jobs)


def _context() -> Any:
    # fork with a running BLAS thread pool is unsafe; forkserver is Linux's
    # default from Python 3.14, spawn the default elsewhere.
    method = "forkserver" if sys.platform.startswith("linux") else "spawn"
    return multiprocessing.get_context(method)


_env_lock = threading.Lock()
_env_users = 0
_env_saved: dict[str, str | None] = {}


def _pin_env() -> None:
    global _env_users
    with _env_lock:
        if _env_users == 0:
            for name in _PINNED:
                _env_saved[name] = os.environ.get(name)
                os.environ[name] = "1"
        _env_users += 1


def _unpin_env() -> None:
    global _env_users
    with _env_lock:
        _env_users -= 1
        if _env_users == 0:
            for name, value in _env_saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            _env_saved.clear()


_search_ids = itertools.count()


class SplitPool:
    """The worker processes of one tree, started at the first parallel
    search (``executor``) and stopped by ``shutdown``."""

    def __init__(self, n_workers: int) -> None:
        self.n_workers = n_workers
        self._executor: ProcessPoolExecutor | None = None

    def use(self, n_candidates: int, n_rows: int) -> bool:
        """Whether a search with this many candidate fits runs in parallel."""
        return n_candidates >= 2 and float(n_candidates) * n_rows > _THRESHOLD

    def executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            _pin_env()
            try:
                self._executor = ProcessPoolExecutor(
                    self.n_workers, mp_context=_context()
                )
            except BaseException:
                _unpin_env()
                raise
        return self._executor

    def shutdown(self) -> None:
        """Wait for the workers to exit; nothing is left running."""
        executor, self._executor = self._executor, None
        if executor is None:
            return
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        finally:
            _unpin_env()

    def pairs(
        self,
        fit: SideFit,
        z: NDArray[np.generic],
        lefts: FloatArray | Sequence[NDArray[np.intp]],
    ) -> tuple[FloatArray, BoolArray]:
        """Both sides of every candidate, fitted in the workers.

        ``z`` holds the node's values of the variable; candidate ``i`` sends
        left the rows with ``z <= lefts[i]`` (numeric, ``lefts`` a float
        array) or ``z`` in ``lefts[i]`` (factor level codes), as the serial
        search does. Returns the summed objectives in candidate order and
        whether either side failed (the objective is then undefined).
        """
        try:
            payload = pickle.dumps((fit, z), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise TypeError(f"n_jobs > 1 needs a picklable node model: {exc}") from exc
        key = f"{os.getpid()}-{next(_search_ids)}"
        n = len(lefts)
        n_chunks = min(n, _CHUNKS_PER_WORKER * self.n_workers)
        bounds = np.linspace(0, n, n_chunks + 1).round().astype(np.intp)
        executor = self.executor()
        futures: list[tuple[int, int, Future[tuple[FloatArray, BoolArray]]]] = []
        for a, b in itertools.pairwise(bounds.tolist()):
            chunk: FloatArray | list[NDArray[np.intp]] = (
                lefts[a:b] if isinstance(lefts, np.ndarray) else list(lefts[a:b])
            )
            futures.append((a, b, executor.submit(_fit_chunk, key, payload, chunk)))
        objective = np.empty(n)
        failed = np.empty(n, dtype=bool)
        try:
            for a, b, future in futures:
                # A worker's exception is re-raised here with the worker's
                # traceback as its __cause__ (concurrent.futures).
                objective[a:b], failed[a:b] = future.result()
        except BaseException:
            for _, _, future in futures:
                future.cancel()
            raise
        return objective, failed


# Worker side: the last search's fit and values, unpickled once per worker.
_cached: tuple[str, SideFit, NDArray[np.generic]] | None = None


def _fit_chunk(
    key: str, payload: bytes, lefts: FloatArray | list[NDArray[np.intp]]
) -> tuple[FloatArray, BoolArray]:
    global _cached
    if _cached is None or _cached[0] != key:
        fit, z = pickle.loads(payload)
        _cached = (key, fit, z)
    _, fit, z = _cached
    numeric = isinstance(lefts, np.ndarray)
    objective = np.zeros(len(lefts))
    failed = np.zeros(len(lefts), dtype=bool)
    for j, left_def in enumerate(lefts):
        # The serial search's masks: zv <= uz[i], or np.isin(codes, levels).
        left = z <= left_def if numeric else np.isin(z, left_def)
        pair = _pair_objective(fit(left, None), fit(~left, None))
        if pair is None:
            failed[j] = True
        else:
            objective[j] = pair
    return objective, failed


def blas_threads() -> dict[str, int]:
    """The thread count of each OpenBLAS library loaded in this process, by
    file name (Linux only; empty elsewhere). For tests and diagnostics."""
    try:
        with open("/proc/self/maps") as f:
            paths = sorted({line.split()[-1] for line in f if "openblas" in line})
    except OSError:
        return {}
    out: dict[str, int] = {}
    for path in paths:
        lib = ctypes.CDLL(path)
        for symbol in (
            "scipy_openblas_get_num_threads64_",
            "scipy_openblas_get_num_threads",
            "openblas_get_num_threads64_",
            "openblas_get_num_threads",
        ):
            getter = getattr(lib, symbol, None)
            if getter is not None:
                out[os.path.basename(path)] = int(getter())
                break
    return out
