"""The speedups against the exact reference path (docs/performance.md).

Each fast path is compared with the code it replaces, which stays in the
library: the LAPACK least squares with the dqrdc2 port, objective-only glm
fits with glm_fit, find_split with find_split_reference, and whole trees with trees grown
under ``reference_path()``. The R fixtures are checked by the other test
modules, which run the fast paths.
"""

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from test_tree import ORACLE_TREES, _control, _grown
from tree_fixtures import cluster, inputs, model, tree, variables

from modeltrees import reference_path
from modeltrees.instability import TooFewClustersWarning, VariableKind
from modeltrees.models import _qr
from modeltrees.models.families import GlmFitWarning, get_family
from modeltrees.models.glm import GlmModel
from modeltrees.models.glm_fit import glm_fit, glm_objective
from modeltrees.models.lm import LmModel
from modeltrees.node_model import FitResult
from modeltrees.split_search import (
    SplitSearch,
    find_split,
    find_split_reference,
)
from modeltrees.tree import ModelTree, mob

QR_RTOL = 1e-10


# 1. LAPACK least squares against the dqrdc2 port.


def _assert_qr_close(x: np.ndarray, y: np.ndarray, tol: float = 1e-7) -> bool:
    """dqrls equals the port; returns whether the fast path was taken."""
    fast = _qr.dqrls(x, y, tol)
    ref = _qr.dqrls_reference(x, y, tol)
    assert fast.rank == ref.rank
    np.testing.assert_array_equal(fast.pivot, ref.pivot)
    scale = max(float(np.linalg.norm(y)), 1.0)
    np.testing.assert_allclose(
        fast.coefficients, ref.coefficients, rtol=QR_RTOL, atol=QR_RTOL * scale
    )
    np.testing.assert_allclose(
        fast.residuals, ref.residuals, rtol=0, atol=QR_RTOL * scale
    )
    np.testing.assert_allclose(fast.effects, ref.effects, rtol=0, atol=QR_RTOL * scale)
    return fast.qr is None


@pytest.mark.parametrize("seed", range(20))
def test_lapack_matches_port_on_random_data(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 400))
    p = int(rng.integers(1, min(n, 8)))
    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    x *= rng.uniform(0.01, 100, p)  # column scales do not matter
    y = x @ rng.standard_normal(p) + rng.standard_normal(n)
    assert _assert_qr_close(x, y)


@pytest.mark.parametrize("seed", range(10))
def test_lapack_matches_port_on_weighted_data(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.integers(0, 3, n)])
    w = np.sqrt(rng.exponential(size=n))
    y = rng.standard_normal(n)
    assert _assert_qr_close(x * w[:, None], y * w)


def test_lapack_matches_port_on_tied_columns() -> None:
    # Many tied values in a column, and a duplicated column (aliased).
    rng = np.random.default_rng(3)
    n = 100
    tied = rng.integers(0, 2, n).astype(float)
    y = rng.standard_normal(n)
    assert _assert_qr_close(np.column_stack([np.ones(n), tied]), y)
    dup = np.column_stack([np.ones(n), tied, tied])
    assert not _assert_qr_close(dup, y)
    assert _qr.dqrls(dup, y, 1e-7).rank == 2


def test_lapack_declines_collinear_and_near_singular() -> None:
    rng = np.random.default_rng(4)
    n = 100
    x1 = rng.standard_normal(n)
    y = rng.standard_normal(n)
    for delta, fast_expected in [
        (0.0, False),
        (1e-9, False),
        (1e-6, False),
        (1e-2, True),
    ]:
        x = np.column_stack([np.ones(n), x1, x1 + delta * rng.standard_normal(n)])
        assert _assert_qr_close(x, y) is fast_expected
    # A dependent column in the middle, as in test_glm_fit.
    x = np.column_stack([np.ones(n), 2 * x1 + 1, x1, rng.standard_normal(n)])
    assert not _assert_qr_close(x, y)


def test_lapack_threshold_is_stricter_than_r_tolerance() -> None:
    # A column keeping 1e-5 of its norm: dqrdc2 keeps it (1e-5 > 1e-7), the
    # fast path declines (1e-5 < 1e-4) and the port decides.
    rng = np.random.default_rng(5)
    n = 200
    q, _ = np.linalg.qr(rng.standard_normal((n, 2)))
    x = np.column_stack([q[:, 0], q[:, 0] + 1e-5 * q[:, 1]])
    y = rng.standard_normal(n)
    assert _qr.dqrls_lapack(x, y) is None
    assert _qr.dqrls(x, y, 1e-7).rank == 2
    x_ok = np.column_stack([q[:, 0], q[:, 0] + 1e-3 * q[:, 1]])
    assert _qr.dqrls_lapack(x_ok, y) is not None


def test_lapack_declines_degenerate_shapes() -> None:
    y = np.arange(3.0)
    for x in (
        np.ones((3, 3)),
        np.ones((2, 3))[:, :2].repeat(1, 0)[:2],
        np.zeros((3, 1)),
    ):
        yy = y[: x.shape[0]]
        assert _qr.dqrls_lapack(x, yy) is None
        _assert_qr_close(x, yy)


def test_reference_path_turns_the_lapack_path_off() -> None:
    rng = np.random.default_rng(6)
    x = np.column_stack([np.ones(50), rng.standard_normal(50)])
    y = rng.standard_normal(50)
    assert _qr.dqrls(x, y, 1e-7).qr is None
    with reference_path():
        assert _qr.dqrls(x, y, 1e-7).qr is not None


# 2. Objective-only glm fits against glm_fit.


def _glm_cases() -> list[Any]:
    rng = np.random.default_rng(7)
    n = 300
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.uniform(size=n)])
    eta = x @ np.array([0.3, 1.0, -0.5])
    cases = [
        ("binomial", x, (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)),
        ("poisson", x, rng.poisson(np.exp(eta)).astype(float)),
        ("gaussian", x, eta + rng.standard_normal(n)),
        # Separation, aliased column, non-integer successes.
        ("binomial", x, (x[:, 1] > 0).astype(float)),
        ("poisson", np.column_stack([x, x[:, 1]]), rng.poisson(2.0, n).astype(float)),
        ("binomial", x, rng.uniform(size=n)),
    ]
    return [pytest.param(*c, id=f"{c[0]}-{i}") for i, c in enumerate(cases)]


@pytest.mark.parametrize(("family", "x", "y"), _glm_cases())
def test_glm_objective_equals_glm_fit(
    family: str, x: np.ndarray, y: np.ndarray
) -> None:
    rng = np.random.default_rng(8)
    w = rng.integers(1, 4, y.shape[0]).astype(float)
    fam = get_family(family)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlmFitWarning)
        full = glm_fit(x, y, fam, weights=w)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the objective path is quiet
        obj = glm_objective(x, y, fam, weights=w)
    np.testing.assert_array_equal(obj.coefficients, full.coefficients)
    assert obj.aic == full.aic
    assert obj.rank == full.rank
    assert obj.converged == full.converged


def test_glm_model_objective_fit_equals_full_fit() -> None:
    rng = np.random.default_rng(9)
    n = 200
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = (rng.uniform(size=n) < 0.4).astype(float)
    w = rng.integers(0, 3, n).astype(float)
    m = GlmModel("binomial")
    kw: dict[str, Any] = {
        "weights": w,
        "offset": None,
        "cluster": None,
        "caseweights": True,
        "vcov": False,
    }
    # Same QR path: the objective-only fit is the full fit, bit for bit.
    full = m.fit(y, x, start=None, estfun=True, **kw)
    fast = m.fit(y, x, start=None, estfun=False, **kw)
    np.testing.assert_array_equal(fast.coefficients, full.coefficients)
    assert fast.objfun == full.objfun
    assert fast.converged == full.converged
    # The reference path also uses the dqrdc2 port: equal to rounding.
    with reference_path():
        ref = m.fit(y, x, start=None, estfun=False, **kw)
    np.testing.assert_allclose(fast.coefficients, ref.coefficients, rtol=1e-12)
    assert math.isclose(fast.objfun, ref.objfun, rel_tol=1e-12)


# 3. find_split against find_split_reference.


def _side_fitter(
    m: Any, y: np.ndarray, x: np.ndarray | None, w: np.ndarray, caseweights: bool
) -> Callable[[np.ndarray, np.ndarray | None], FitResult]:
    def fit(mask: np.ndarray, start: np.ndarray | None) -> FitResult:
        return m.fit(
            y[mask],
            None if x is None else x[mask],
            start=start,
            weights=w[mask],
            offset=None,
            cluster=None,
            caseweights=caseweights,
            estfun=False,
            vcov=False,
        )

    return fit


def _assert_same_search(
    fast: SplitSearch, ref: SplitSearch, rtol: float = 1e-10
) -> None:
    assert fast.best == ref.best
    assert fast.split == ref.split
    assert fast.minsize == ref.minsize
    np.testing.assert_array_equal(fast.values, ref.values)
    np.testing.assert_array_equal(np.isinf(fast.objective), np.isinf(ref.objective))
    finite = np.isfinite(ref.objective)
    scale = float(np.max(np.abs(ref.objective[finite]))) if finite.any() else 1.0
    np.testing.assert_allclose(
        fast.objective[finite], ref.objective[finite], rtol=rtol, atol=rtol * scale
    )
    if fast.best is not None:
        # The winner's objective is an exact fit in both; the two paths use
        # different QR code, so they agree to rounding.
        assert math.isclose(
            fast.objective[fast.best], ref.objective[ref.best], rel_tol=1e-12
        )


def _scenario(seed: int, kind: str) -> dict[str, Any]:
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(30, 400))
    z = rng.uniform(-1, 1, n)
    if kind in ("ties", "tiny"):
        z = np.round(z, 1)
    xv = rng.standard_normal(n)
    y = np.where(z <= 0.1, 1 + xv, -1 - 0.5 * xv) + rng.standard_normal(n)
    w = np.ones(n)
    minsize = 10.0
    if kind == "weights":
        w = rng.uniform(0.1, 3, n)
    elif kind == "intweights":
        w = rng.integers(1, 5, n).astype(float)
    elif kind == "zeroweights":
        w = rng.integers(0, 3, n).astype(float)
    elif kind == "tiny":
        n = 12
        z, xv, y, w = z[:n], xv[:n], y[:n], w[:n]
        minsize = 2.0
    x = np.column_stack([np.ones(n), xv])
    if kind == "constcol":
        # x is constant left of 0: those sides are rank-deficient.
        x = np.column_stack([np.ones(n), np.where(z <= 0, 1.0, xv)])
    if kind == "binary":
        x = np.column_stack([np.ones(n), (z > 0.3).astype(float)])
    return {"z": z, "y": y, "x": x, "w": w, "minsize": minsize}


KINDS = [
    "plain",
    "ties",
    "weights",
    "intweights",
    "zeroweights",
    "constcol",
    "binary",
    "tiny",
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("caseweights", [True, False])
def test_find_split_lm_matches_reference(
    kind: str, seed: int, caseweights: bool
) -> None:
    s = _scenario(seed, kind)
    m = LmModel()
    fit = _side_fitter(m, s["y"], s["x"], s["w"], caseweights)
    args = (0, VariableKind.NUMERIC, s["z"], None, s["w"], fit)
    kw: dict[str, Any] = {"minsize": s["minsize"], "caseweights": caseweights}
    with reference_path():
        ref = find_split_reference(*args, **kw)
    _assert_same_search(find_split(*args, **kw), ref)


@pytest.mark.parametrize("kind", ["plain", "ties", "intweights", "zeroweights"])
@pytest.mark.parametrize("seed", range(3))
def test_find_split_glm_matches_reference(kind: str, seed: int) -> None:
    s = _scenario(seed, kind)
    yb = (s["y"] > 0).astype(float)
    m = GlmModel("binomial")
    fit = _side_fitter(m, yb, s["x"], s["w"], True)
    args = (0, VariableKind.NUMERIC, s["z"], None, s["w"], fit)
    for restart in (True, False):
        kw: dict[str, Any] = {"minsize": s["minsize"], "restart": restart}
        with reference_path(), warnings.catch_warnings():
            warnings.simplefilter("ignore", GlmFitWarning)
            ref = find_split_reference(*args, **kw)
        _assert_same_search(find_split(*args, **kw), ref)


def test_find_split_center_and_factor_match_reference() -> None:
    s = _scenario(0, "ties")
    m = LmModel()
    fit = _side_fitter(m, s["y"], s["x"], s["w"], True)
    args = (0, VariableKind.NUMERIC, s["z"], None, s["w"], fit)
    with reference_path():
        ref = find_split_reference(*args, minsize=10, numsplit="center")
    _assert_same_search(find_split(*args, minsize=10, numsplit="center"), ref)
    codes = np.digitize(s["z"], [-0.5, 0, 0.5])
    levels = ("a", "b", "c", "d")
    for kind in (VariableKind.NOMINAL, VariableKind.ORDERED):
        fargs = (0, kind, codes, levels, s["w"], fit)
        with reference_path():
            ref = find_split_reference(*fargs, minsize=10)
        got = find_split(*fargs, minsize=10)
        assert got.split == ref.split and got.best == ref.best
        assert got.left_levels == ref.left_levels
        np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-12)


def test_admissibility_at_the_minsize_boundary_with_fractional_weights() -> None:
    # 0.1 is not exact in binary: cumulative sums and masked sums of these
    # weights differ in the last bits right at minsize.
    n = 300
    rng = np.random.default_rng(11)
    z = np.arange(n, dtype=float)
    w = np.full(n, 0.1)
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    m = LmModel()
    fit = _side_fitter(m, y, x, w, True)
    for minsize in (3.0, 2.9999999999999996, 3.0000000000000004, 10.0):
        args = (0, VariableKind.NUMERIC, z, None, w, fit)
        with reference_path():
            ref = find_split_reference(*args, minsize=minsize)
        _assert_same_search(find_split(*args, minsize=minsize), ref)


# 4. Whole trees: fast path against the reference path.


def _grow_fixture(name: str) -> ModelTree:
    fixture = tree(name)
    y, x, w, off = inputs(fixture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TooFewClustersWarning)
        warnings.simplefilter("ignore", GlmFitWarning)
        return mob(
            model(fixture),
            y,
            x,
            variables(fixture),
            weights=w,
            offset=off,
            cluster=cluster(fixture),
            control=_control(fixture),
        )


def _assert_same_tree(fast: ModelTree, ref: ModelTree) -> None:
    assert len(fast.nodes) == len(ref.nodes)
    np.testing.assert_array_equal(fast.fitted_node, ref.fitted_node)
    for a, b in zip(fast.nodes, ref.nodes, strict=True):
        assert a.id == b.id and a.stop_reason == b.stop_reason
        assert a.split == b.split
        np.testing.assert_array_equal(a.rows, b.rows)
        np.testing.assert_allclose(a.fit.coefficients, b.fit.coefficients, rtol=1e-10)
        assert math.isclose(a.fit.objfun, b.fit.objfun, rel_tol=1e-10)
        if b.split_search is not None:
            assert a.split_search is not None
            _assert_same_search(a.split_search, b.split_search, rtol=1e-9)


@pytest.mark.parametrize("name", ORACLE_TREES)
def test_fixture_tree_equals_reference_path(name: str) -> None:
    fast = _grown(name)  # shared with test_tree, which checks it against R
    with reference_path():
        ref = _grow_fixture(name)
    _assert_same_tree(fast, ref)


@pytest.mark.parametrize("model_name", ["lm", "glm"])
def test_benchmark_tree_equals_reference_path(model_name: str) -> None:
    rng = np.random.default_rng(13)
    n = 600
    xv = rng.standard_normal(n)
    z1, z2 = rng.uniform(-1, 1, n), np.round(rng.uniform(-1, 1, n), 2)
    eta = np.where(z1 <= 0, 1 + xv, -1 - xv)
    x = np.column_stack([np.ones(n), xv])
    w = rng.integers(0, 3, n).astype(float)
    z = {"z1": ("numeric", z1), "z2": ("numeric", z2)}
    m: Any
    if model_name == "lm":
        y, m = eta + rng.standard_normal(n), LmModel()
    else:
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
        m = GlmModel("binomial")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlmFitWarning)
        fast = mob(m, y, x, z, weights=w)
        with reference_path():
            ref = mob(m, y, x, z, weights=w)
    assert len(fast.nodes) > 1
    _assert_same_tree(fast, ref)
