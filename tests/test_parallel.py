"""Parallel split search (docs/api-design.md, section 6; task 6f)."""

import dataclasses
import json
import multiprocessing
import os
import pickle
import struct
import subprocess
import sys
import warnings
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from typing import Any
from unittest import mock

import numpy as np
import pytest
from fixtures_io import FIXTURES
from grown_trees import control, fixture, model, tree_inputs
from tree_fixtures import cluster, variables

import modeltrees as mt
from modeltrees import _parallel
from modeltrees.errors import (
    GlmFitWarning,
    ModelTreesWarning,
    TooFewClustersWarning,
)
from modeltrees.models import (
    GlmModel,
    LmModel,
    MultinomModel,
    NegBinModel,
    SafeLogitModel,
)
from modeltrees.node_model import FitResult, NodeModel, failed_fit
from modeltrees.tree import GrownTree, MobControl, Node, mob

# The five built-in models, each with the kind of response it fits.
BUILT_IN: list[tuple[NodeModel, str]] = [
    (LmModel(coef_names=["(Intercept)", "x"]), "gaussian"),
    (GlmModel("binomial", coef_names=["(Intercept)", "x"]), "binary"),
    (GlmModel("poisson", epsilon=1e-10, maxit=50), "count"),
    (NegBinModel(coef_names=["(Intercept)", "x"]), "count"),
    (SafeLogitModel(separation_tol=1e-5), "binary"),
    (MultinomModel(("lo", "mid", "hi"), coef_names=["(Intercept)", "x"]), "class"),
]


@pytest.mark.parametrize(
    ("model", "response"), BUILT_IN, ids=lambda m: type(m).__name__
)
def test_built_in_models_are_picklable(model: NodeModel, response: str) -> None:
    copy = pickle.loads(pickle.dumps(model))
    assert type(copy) is type(model)
    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(60), rng.standard_normal(60)])
    y = {
        "gaussian": rng.standard_normal(60),
        "binary": rng.integers(0, 2, 60).astype(float),
        "count": rng.negative_binomial(2, 0.4, 60).astype(float),
        "class": rng.integers(0, 3, 60),
    }[response]
    kw: dict[str, Any] = {
        "start": None,
        "weights": np.ones(60),
        "offset": None,
        "cluster": None,
        "caseweights": True,
        "estfun": True,
        "vcov": True,
    }
    a = model.fit(y, x, **kw)
    b = copy.fit(y, x, **kw)
    _same(a, b, "fit")


def test_resolve_n_jobs() -> None:
    assert _parallel.resolve_n_jobs(None) == 1
    assert _parallel.resolve_n_jobs(1) == 1
    assert _parallel.resolve_n_jobs(3) == 3
    assert _parallel.resolve_n_jobs(np.int64(2)) == 2  # type: ignore[arg-type]
    assert _parallel.resolve_n_jobs(-1) == (os.process_cpu_count() or 1)
    for bad in (0, -2):
        with pytest.raises(ValueError, match="n_jobs"):
            _parallel.resolve_n_jobs(bad)
    for wrong in (True, 2.0, "2"):
        with pytest.raises(TypeError, match="n_jobs"):
            _parallel.resolve_n_jobs(wrong)  # type: ignore[arg-type]


_WORKER_REPORT = """
import json, os
from modeltrees import _parallel
import numpy, scipy.linalg

pool = _parallel.SplitPool(2)
try:
    ex = pool.executor()
    worker = ex.submit(_parallel.blas_threads).result()
    env = ex.submit(os.getenv, "OPENBLAS_NUM_THREADS").result()
finally:
    pool.shutdown()
print(json.dumps({
    "parent": _parallel.blas_threads(),
    "worker": worker,
    "worker_env": env,
    "parent_env_after": os.environ.get("OPENBLAS_NUM_THREADS"),
}))
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_a_worker_reports_one_blas_thread() -> None:
    # A fresh interpreter without the thread variables, so that the parent
    # has its default BLAS threads and the workers must get theirs pinned.
    env = {k: v for k, v in os.environ.items() if k not in _parallel._PINNED}
    out = subprocess.run(
        [sys.executable, "-c", _WORKER_REPORT],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    report = json.loads(out.stdout)
    assert report["worker"], "no OpenBLAS library found in the worker"
    assert set(report["worker"].values()) == {1}
    assert report["worker_env"] == "1"
    # The parent's environment is restored when the pool shuts down.
    assert report["parent_env_after"] is None
    if (os.process_cpu_count() or 1) > 1:
        assert max(report["parent"].values()) > 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_the_test_process_has_one_blas_thread() -> None:
    # tests/conftest.py sets the variables before numpy is loaded.
    threads = _parallel.blas_threads()
    assert threads and set(threads.values()) == {1}


# Step 2: parallel trees equal serial trees bit for bit.


def _same(a: Any, b: Any, path: str = "tree") -> None:
    """Exact equality, bit for bit for floats and numeric arrays."""
    assert type(a) is type(b), f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        for f in dataclasses.fields(a):
            _same(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        if a.dtype.kind == "O":
            for i, (u, v) in enumerate(zip(a.ravel(), b.ravel(), strict=True)):
                _same(u, v, f"{path}[{i}]")
        else:
            assert a.tobytes() == b.tobytes(), f"{path} differs"
    elif isinstance(a, float):
        assert struct.pack("<d", a) == struct.pack("<d", b), f"{path}: {a!r} {b!r}"
    elif isinstance(a, Mapping):
        assert list(a) == list(b), path
        for k in a:
            _same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (u, v) in enumerate(zip(a, b, strict=True)):
            _same(u, v, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _same_tree(serial: GrownTree, par: GrownTree) -> None:
    # Node.kids holds the subtrees, so comparing the root compares every node.
    _same(serial, par)


def _grow_fixture(directory: str, name: str, n_jobs: int | None) -> GrownTree:
    f = fixture(directory, name)
    y, x, w, off = tree_inputs(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlmFitWarning)
        warnings.simplefilter("ignore", TooFewClustersWarning)
        return mob(
            model(f),
            y,
            x,
            variables(f),
            weights=w,
            offset=off,
            cluster=cluster(f),
            control=control(f),
            n_jobs=n_jobs,
        )


NON_LM_TREES = [
    (d, n)
    for d in ("trees", "extension_trees", "cluster_trees")
    for n in sorted(p.stem for p in (FIXTURES / d).glob("*.json"))
    if fixture(d, n)["function_name"] != "lmtree" and not n.endswith("_norestart")
]


def test_the_fixture_list() -> None:
    names = {n for _, n in NON_LM_TREES}
    assert len(NON_LM_TREES) == 34
    for expected in (
        "sim_cluster_glmtree_binomial_cluster_weights",  # clustered, weighted
        "sim_negbin_weights",
        "sim_multinom_noref_onesplit",  # reference-absent
        "sim_cluster_models_safelogit_cluster",
        "sim_separation_safelogit",
    ):
        assert expected in names


@cache
def _serial(directory: str, name: str) -> GrownTree:
    return _grow_fixture(directory, name, None)


@pytest.mark.parametrize("n_jobs", [2, 4])
@pytest.mark.parametrize(("directory", "name"), NON_LM_TREES)
def test_fixture_trees_equal_the_serial_tree(
    directory: str, name: str, n_jobs: int
) -> None:
    calls = []
    real = _parallel.SplitPool.pairs

    def counting(self: _parallel.SplitPool, *args: Any) -> Any:
        calls.append(len(args[2]))
        return real(self, *args)

    with (
        _parallel.parallel_threshold(0),
        mock.patch.object(_parallel.SplitPool, "pairs", counting),
    ):
        par = _grow_fixture(directory, name, n_jobs)
    serial = _serial(directory, name)
    _same_tree(serial, par)
    # Every search with at least two candidate fits ran in the workers
    # (pruning removes searched nodes, so only counted without it).
    assert serial.control.prune is not None or calls == [
        m for m in map(_n_fitted, serial.nodes) if m is not None and m >= 2
    ]
    assert multiprocessing.active_children() == []


def _n_fitted(node: Node) -> int | None:
    """The candidates split search fitted at a node (admissible ones: a
    finite or NaN objective, or a failed fit), None for no search or a
    single factor partition (R's objective 1 without fitting)."""
    s = node.split_search
    if s is None or s.objective.shape[0] < 2 and s.values is None:
        return None
    return int(np.sum(s.objective != np.inf)) + s.n_failed


def _simulated(n: int = 400, levels: int = 10) -> dict[str, Any]:
    """Binary data with a numeric and a nominal partitioning variable whose
    levels split the slope."""
    rng = np.random.default_rng(20261021)
    xv = rng.standard_normal(n)
    z1 = rng.uniform(size=n)
    z2 = rng.integers(0, levels, n)
    slope = np.where(np.isin(z2, [0, 3, 4, 7]), 2.0, -1.0)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-slope * xv))).astype(float)
    names = tuple(f"l{i}" for i in range(levels))
    return {
        "y": y,
        "x": np.column_stack([np.ones(n), xv]),
        "z": {
            "z1": ("numeric", z1),
            "z2": ("nominal", z2, names),
        },
    }


def _grow_simulated(model: NodeModel, n_jobs: int | None, **kw: Any) -> GrownTree:
    d = _simulated(**kw)
    return mob(model, d["y"], d["x"], d["z"], n_jobs=n_jobs)


class _SometimesFailingGlm(GlmModel):
    """A binomial glm whose split-search fits fail for some side sizes."""

    def fit(self, y: Any, x: Any, **kw: Any) -> FitResult:
        if not kw.get("estfun") and y.shape[0] % 7 == 3:
            return failed_fit(tuple(self.coef_names or ()), "side size 3 mod 7")
        return super().fit(y, x, **kw)


class _Boom(Exception):
    pass


class _RaisingGlm(GlmModel):
    """A binomial glm whose split-search fits raise for some side sizes."""

    def fit(self, y: Any, x: Any, **kw: Any) -> FitResult:
        if not kw.get("estfun") and y.shape[0] == 333:
            raise _Boom("a candidate fit with 333 rows")
        return super().fit(y, x, **kw)


def test_nominal_partitions_with_many_levels() -> None:
    model = GlmModel("binomial", coef_names=["(Intercept)", "x"])
    serial = _grow_simulated(model, None)
    root = serial.root
    assert root.split_search is not None
    assert serial.variables[root.split_search.variable].name == "z2"
    # 10 levels: 2^9 - 1 = 511 candidate partitions.
    assert root.split_search.objective.shape == (511,)
    calls = []
    real = _parallel.SplitPool.pairs

    def counting(self: _parallel.SplitPool, *args: Any) -> Any:
        calls.append(len(args[2]))
        return real(self, *args)

    with (
        _parallel.parallel_threshold(0),
        mock.patch.object(_parallel.SplitPool, "pairs", counting),
    ):
        par = _grow_simulated(model, 3)
    _same_tree(serial, par)
    assert calls[0] == _n_fitted(root) and calls[0] > 400


@pytest.mark.parametrize("n_jobs", [2, 4])
def test_failed_candidates_are_counted_as_serially(n_jobs: int) -> None:
    model = _SometimesFailingGlm("binomial", coef_names=["(Intercept)", "x"])
    serial = _grow_simulated(model, None)
    failed = [n.split_search.n_failed for n in serial.nodes if n.split_search]
    assert sum(failed) > 0
    with _parallel.parallel_threshold(0):
        par = _grow_simulated(model, n_jobs)
    _same_tree(serial, par)


def test_a_worker_error_propagates_and_the_pool_shuts_down() -> None:
    model = _RaisingGlm("binomial", coef_names=["(Intercept)", "x"])
    with pytest.raises(_Boom, match="333 rows"):
        _grow_simulated(model, None)
    with _parallel.parallel_threshold(0), pytest.raises(_Boom) as info:
        _grow_simulated(model, 2)
    assert "333 rows" in str(info.value)
    # The worker's traceback is the cause.
    cause = info.value.__cause__
    assert cause is not None and type(cause).__name__ == "_RemoteTraceback"
    assert "_fit_chunk" in str(cause) and "raise _Boom" in str(cause)
    # No pool is left running and the environment is restored.
    assert multiprocessing.active_children() == []
    assert _parallel._env_users == 0
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"  # tests/conftest.py


def test_an_unpicklable_model_fails_in_the_parent() -> None:
    model = GlmModel("binomial", coef_names=["(Intercept)", "x"])
    model.hook = lambda: None  # type: ignore[attr-defined]
    with _parallel.parallel_threshold(0), pytest.raises(TypeError, match="picklable"):
        _grow_simulated(model, 2)
    assert multiprocessing.active_children() == []


def _nested_grow() -> tuple[GrownTree, int]:
    """Grow in parallel inside a worker of another pool."""
    model = GlmModel("binomial", coef_names=["(Intercept)", "x"])
    with _parallel.parallel_threshold(0):
        tree = _grow_simulated(model, 2, n=200)
    return tree, len(multiprocessing.active_children())


def test_nested_use_leaves_no_pool_running() -> None:
    model = GlmModel("binomial", coef_names=["(Intercept)", "x"])
    serial = _grow_simulated(model, None, n=200)
    ctx = multiprocessing.get_context(
        "forkserver" if sys.platform.startswith("linux") else "spawn"
    )
    with ProcessPoolExecutor(1, mp_context=ctx) as outer:
        tree, running = outer.submit(_nested_grow).result()
    _same_tree(serial, tree)
    assert running == 0
    # Two parallel trees in a row in this process, each with its own pool.
    with _parallel.parallel_threshold(0):
        for _ in range(2):
            _same_tree(serial, _grow_simulated(model, 2, n=200))
            assert multiprocessing.active_children() == []


def test_n_jobs_minus_one() -> None:
    directory, name = "trees", "sim_binomial_glmtree_multisplit"
    with _parallel.parallel_threshold(0):
        par = _grow_fixture(directory, name, -1)
    _same_tree(_serial(directory, name), par)


def test_restart_false_runs_serially_with_one_warning() -> None:
    directory, name = "trees", "sim_binomial_glmtree_multisplit_norestart"
    serial = _grow_fixture(directory, name, None)
    with (
        _parallel.parallel_threshold(0),
        mock.patch.object(
            _parallel.SplitPool, "executor", side_effect=AssertionError("pool")
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        par = _grow_fixture(directory, name, 4)
    _same_tree(serial, par)
    ours = [c for c in caught if c.category is ModelTreesWarning]
    assert len(ours) == 1
    assert "restart=False" in str(ours[0].message)
    assert "n_jobs=4 is ignored" in str(ours[0].message)


def test_restart_false_warns_once_through_the_entry_points() -> None:
    d = _simulated(n=200)
    model = GlmModel("binomial", coef_names=["(Intercept)", "x"])
    with pytest.warns(ModelTreesWarning, match="restart=False") as caught:
        mt.mob(
            model,
            d["y"],
            d["x"],
            d["z"],
            control=MobControl(restart=False),
            n_jobs=2,
        )
    assert sum(c.category is ModelTreesWarning for c in caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelTreesWarning)
        mt.mob(model, d["y"], d["x"], d["z"], control=MobControl(restart=False))
