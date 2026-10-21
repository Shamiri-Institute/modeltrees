"""The sandwich of docs/api-design.md, section 5.1, on random data.

``modeltrees._vcov.sandwich`` against the formula written out densely:
``H^-1 (c sum_g s_g s_g') H^-1`` with explicit loops over clusters and rows,
``np.linalg.inv`` and the adjustments of sandwich::meatCL
(vcovCL.R:130-176). The comparison with R's vcovCL on the fixture trees is
in tests/test_tree_vcov.py.
"""

import itertools

import numpy as np
import pytest

from modeltrees._vcov import sandwich

RTOL = 1e-12


def _data(
    seed: int, n: int = 60, k: int = 3, n_clusters: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A positive definite Hessian, scores with weights inside, integer
    weights with zeros, and cluster ids."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((k, k))
    h = a @ a.T + k * np.diag(rng.uniform(1, 100, k))
    w = rng.integers(0, 4, n).astype(np.float64)
    u = rng.standard_normal((n, k)) * rng.uniform(0.1, 10, k)
    cl = rng.integers(0, n_clusters, n)
    return h, w[:, None] * u, w, cl


def _dense(
    h: np.ndarray,
    ef: np.ndarray,
    w: np.ndarray,
    cl: np.ndarray | None,
    caseweights: bool,
    type: str,
    cadjust: bool,
) -> np.ndarray:
    k = ef.shape[1]
    rows = [i for i in range(ef.shape[0]) if w[i] > 0]
    n = sum(w[i] for i in rows) if caseweights else len(rows)
    meat = np.zeros((k, k))
    if cl is None:
        for i in rows:
            s = ef[i] / np.sqrt(w[i]) if caseweights else ef[i]
            meat += np.outer(s, s)
        g = n
    else:
        groups = sorted({cl[i] for i in rows})
        for gid in groups:
            s = sum(ef[i] for i in rows if cl[i] == gid)
            meat += np.outer(s, s)
        g = len(groups)
    c = (g / (g - 1) if cadjust else 1.0) * ((n - 1) / (n - k) if type == "HC1" else 1)
    hinv = np.linalg.inv(h)
    out: np.ndarray = hinv @ (c * meat) @ hinv
    return out


CASES = list(
    itertools.product(
        [0, 1, 2], [True, False], [True, False], ["HC0", "HC1"], [True, False]
    )
)


@pytest.mark.parametrize(("seed", "clustered", "caseweights", "type", "cadjust"), CASES)
def test_against_the_dense_formula(
    seed: int, clustered: bool, caseweights: bool, type: str, cadjust: bool
) -> None:
    h, ef, w, cl = _data(seed)
    got = sandwich(
        h,
        ef,
        weights=w,
        cluster=cl if clustered else None,
        caseweights=caseweights,
        type=type,  # type: ignore[arg-type]
        cadjust=cadjust,
    )
    expected = _dense(h, ef, w, cl if clustered else None, caseweights, type, cadjust)
    assert got.status == "ok" and got.reason is None
    np.testing.assert_allclose(got.matrix, expected, rtol=RTOL, atol=0)
    np.testing.assert_array_equal(got.matrix, got.matrix.T)
    positive = w > 0
    if clustered:
        assert got.n_clusters == len(set(cl[positive]))
    else:
        assert got.n_clusters == (w.sum() if caseweights else positive.sum())
    assert not got.too_few_clusters
    d = np.sqrt(np.diag(h))
    assert got.condition == pytest.approx(np.linalg.cond(h / np.outer(d, d)), rel=1e-10)


def test_frequency_weights_equal_replicated_rows() -> None:
    # Integer case weights give the sandwich of the data with each row
    # repeated w_i times, with and without clusters (same clusters for the
    # copies), and n the sum of the weights in the HC1 factor.
    h, ef, w, cl = _data(3)
    unit = np.divide(ef, w[:, None], out=np.zeros_like(ef), where=w[:, None] > 0)
    reps = w.astype(int)
    ef_r = np.repeat(unit, reps, axis=0)
    cl_r = np.repeat(cl, reps)
    for cluster, cluster_r in ((None, None), (cl, cl_r)):
        for type in ("HC0", "HC1"):
            a = sandwich(
                h,
                ef,
                weights=w,
                cluster=cluster,
                caseweights=True,
                type=type,  # type: ignore[arg-type]
            )
            b = sandwich(
                h,
                ef_r,
                weights=np.ones(ef_r.shape[0]),
                cluster=cluster_r,
                caseweights=True,
                type=type,  # type: ignore[arg-type]
            )
            np.testing.assert_allclose(a.matrix, b.matrix, rtol=RTOL)
            assert a.n_clusters == b.n_clusters


def test_zero_weight_rows_equal_deleted_rows() -> None:
    h, ef, w, cl = _data(4)
    keep = w > 0
    for caseweights, cluster in itertools.product((True, False), (True, False)):
        a = sandwich(
            h, ef, weights=w, cluster=cl if cluster else None, caseweights=caseweights
        )
        b = sandwich(
            h,
            ef[keep],
            weights=w[keep],
            cluster=cl[keep] if cluster else None,
            caseweights=caseweights,
        )
        np.testing.assert_array_equal(a.matrix, b.matrix)
        assert a.n_clusters == b.n_clusters


def test_a_cluster_with_only_zero_weights_is_not_counted() -> None:
    h, ef, w, cl = _data(5)
    cl = cl.copy()
    w = w.copy()
    cl[:3] = 99
    w[:3] = 0.0
    ef[:3] = 0.0
    got = sandwich(h, ef, weights=w, cluster=cl, caseweights=True)
    assert got.n_clusters == len(set(cl[w > 0])) and 99 not in set(cl[w > 0])
    expected = _dense(h, ef, w, cl, True, "HC1", True)
    np.testing.assert_allclose(got.matrix, expected, rtol=RTOL)


def test_cluster_labels_of_any_type() -> None:
    h, ef, w, cl = _data(6)
    labels = np.array([f"school {c}" for c in cl], dtype=object)
    a = sandwich(h, ef, weights=w, cluster=cl, caseweights=True)
    b = sandwich(h, ef, weights=w, cluster=labels, caseweights=True)
    np.testing.assert_array_equal(a.matrix, b.matrix)


def test_too_few_clusters_is_flagged() -> None:
    # G = k: the summed scores of a fit are zero, so the meat has rank at
    # most G - 1 < k, but the matrix is still returned, as R does.
    h, ef, w, _ = _data(7, k=3)
    cl = np.arange(ef.shape[0]) % 3
    got = sandwich(h, ef, weights=w, cluster=cl, caseweights=True)
    assert got.status == "ok" and got.too_few_clusters and got.n_clusters == 3
    np.testing.assert_allclose(
        got.matrix, _dense(h, ef, w, cl, True, "HC1", True), rtol=RTOL
    )
    four = sandwich(
        h, ef, weights=w, cluster=np.arange(ef.shape[0]) % 4, caseweights=True
    )
    assert not four.too_few_clusters


WEIGHT_CASES = {
    "unit": lambda rng, n: np.ones(n),
    "equal": lambda rng, n: np.full(n, 3.0),
    "unequal": lambda rng, n: rng.integers(1, 5, n).astype(np.float64),
}


@pytest.mark.parametrize("caseweights", [True, False])
@pytest.mark.parametrize("weights", list(WEIGHT_CASES))
@pytest.mark.parametrize(("n_clusters", "too_few"), [(2, True), (3, True), (4, False)])
def test_too_few_clusters_under_any_weights(
    weights: str, caseweights: bool, n_clusters: int, too_few: bool
) -> None:
    """Unlike the instability tests (which sum estfun / sqrt(w) and run at
    G = k under unequal case weights), the meat sums estfun itself, whose
    cluster sums add up to the summed scores, zero at the estimate. So G = k
    is short of rank under every weighting, and the flag is G <= k."""
    rng = np.random.default_rng(20 + n_clusters)
    n, k = 60, 3
    h, _, _, _ = _data(9, n=n, k=k)
    w = WEIGHT_CASES[weights](rng, n)
    u = rng.standard_normal((n, k))
    u -= (w @ u) / w.sum()
    ef = w[:, None] * u
    cl = np.arange(n) % n_clusters
    got = sandwich(h, ef, weights=w, cluster=cl, caseweights=caseweights)
    assert got.status == "ok" and got.too_few_clusters is too_few
    lam = np.linalg.eigvalsh(got.matrix)
    rank = int(np.sum(lam > 1e-12 * lam[-1]))
    assert rank == min(k, n_clusters - 1)


@pytest.mark.parametrize("cadjust", [True, False])
def test_a_single_cluster_gives_nan(cadjust: bool) -> None:
    h, ef, w, _ = _data(8)
    got = sandwich(
        h,
        ef,
        weights=w,
        cluster=np.zeros(ef.shape[0]),
        caseweights=True,
        cadjust=cadjust,
    )
    assert got.status == "unavailable" and got.reason is not None
    assert got.n_clusters == 1 and got.too_few_clusters
    assert got.matrix.shape == (3, 3) and np.all(np.isnan(got.matrix))
    assert np.isfinite(got.condition)


def test_singular_hessian() -> None:
    h, ef, w, cl = _data(9)
    v = np.array([1.0, -2.0, 0.5])
    singular = np.outer(v, v) + np.outer([0, 1.0, 1.0], [0, 1.0, 1.0])
    got = sandwich(singular, ef, weights=w, cluster=cl, caseweights=True)
    assert got.status == "singular" and got.reason is not None
    assert np.all(np.isnan(got.matrix)) and got.condition > 1e12
    zero_diag = h.copy()
    zero_diag[1, :] = zero_diag[:, 1] = 0.0
    got = sandwich(zero_diag, ef, weights=w, cluster=cl, caseweights=True)
    assert got.status == "singular" and got.condition == np.inf


def test_ill_conditioned_but_positive_definite_is_ok() -> None:
    # The scaling removes a bad scale: only the scaled condition counts.
    h, ef, w, cl = _data(10)
    s = np.array([1e-6, 1.0, 1e6])
    got = sandwich(h * np.outer(s, s), ef * s, weights=w, cluster=cl, caseweights=True)
    ref = sandwich(h, ef, weights=w, cluster=cl, caseweights=True)
    assert got.status == "ok"
    assert got.condition == pytest.approx(ref.condition, rel=1e-10)
    np.testing.assert_allclose(got.matrix, ref.matrix / np.outer(s, s), rtol=1e-9)


def test_unavailable_without_hessian_or_scores() -> None:
    h, ef, w, _ = _data(11)
    for hessian, estfun in ((None, ef), (h, None), (None, None)):
        got = sandwich(hessian, estfun, weights=w, cluster=None, caseweights=True)
        assert got.status == "unavailable" and np.isnan(got.condition)
        assert np.all(np.isnan(got.matrix))


def test_precision_weights_keep_the_weights_in_the_scores() -> None:
    # caseweights=False: no division by the weights, n and G count rows.
    h, ef, w, _ = _data(12)
    got = sandwich(h, ef, weights=w, cluster=None, caseweights=False, type="HC1")
    assert got.n_clusters == int(np.sum(w > 0))
    np.testing.assert_allclose(
        got.matrix, _dense(h, ef, w, None, False, "HC1", True), rtol=RTOL
    )


def test_input_checks() -> None:
    h, ef, w, cl = _data(13)
    with pytest.raises(ValueError, match="type"):
        sandwich(h, ef, weights=w, cluster=cl, caseweights=True, type="HC3")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Hessian"):
        sandwich(h[:2, :2], ef, weights=w, cluster=cl, caseweights=True)
    with pytest.raises(ValueError, match="weights"):
        sandwich(h, ef, weights=w[:-1], cluster=cl, caseweights=True)
    with pytest.raises(ValueError, match="cluster"):
        sandwich(h, ef, weights=w, cluster=cl[:-1], caseweights=True)
