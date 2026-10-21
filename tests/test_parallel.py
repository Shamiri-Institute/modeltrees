"""Parallel split search (docs/api-design.md, section 6; task 6f)."""

import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from modeltrees import _parallel
from modeltrees.models import (
    GlmModel,
    LmModel,
    MultinomModel,
    NegBinModel,
    SafeLogitModel,
)
from modeltrees.node_model import NodeModel

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
        "count": rng.poisson(3.0, 60).astype(float),
        "class": rng.integers(0, 3, 60),
    }[response]
    w = np.ones(60)
    a = model.fit(y, x, weights=w)
    b = copy.fit(y, x, weights=w)
    np.testing.assert_array_equal(a.coefficients, b.coefficients)
    assert a.objfun == b.objfun


def test_resolve_n_jobs() -> None:
    assert _parallel.resolve_n_jobs(None) == 1
    assert _parallel.resolve_n_jobs(1) == 1
    assert _parallel.resolve_n_jobs(3) == 3
    assert _parallel.resolve_n_jobs(np.int64(2)) == 2
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
