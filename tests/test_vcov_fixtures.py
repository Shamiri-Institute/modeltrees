"""ModelTree.vcov against R and statsmodels (docs/api-design.md, section 5;
docs/sandwich-fixtures.md).

1. sandwich::vcovCL on every leaf of the vcovcl fixtures, with ``type`` and
   ``cadjust`` crossed, with the school clusters and without clusters.
2. What R cannot check: G <= k, G = 1, a completely separated leaf, case
   weights (against replicated rows), zero weights (against deleted rows).
3. statsmodels' ``cov_type="cluster"``, ``"HC0"`` and a coarser cluster on
   single-node trees.
"""

import warnings
from functools import cache
from typing import Any

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm  # type: ignore[import-untyped]
from fixture_frames import polars_frame
from fixtures_io import FIXTURES, load_json
from grown_trees import fixture, grow, tree_inputs
from slow import mark_slow_trees
from test_api_fixtures import public
from test_multinom import _r_offset

import modeltrees as mt
from modeltrees._vcov import sandwich
from modeltrees.models.families import Binomial, Family, NegativeBinomial, Poisson
from modeltrees.models.glm import GlmModel
from modeltrees.models.multinom import (
    multinom_evaluate,
    multinom_hessian,
    multinom_scores,
)

RTOL = 1e-8

VCOVCL_FILES = sorted(p.stem for p in (FIXTURES / "vcovcl").glob("*.json"))


@cache
def _vcovcl(name: str) -> dict[str, Any]:
    f: dict[str, Any] = load_json(f"vcovcl/{name}.json")
    return f


def _tree_of(name: str) -> tuple[str, str]:
    directory, file = _vcovcl(name)["tree_fixture"].split("/")
    return directory, file.removesuffix(".json")


def _entries(*models: str) -> list[Any]:
    return [
        pytest.param(name, leaf["id"], j, id=f"{name}-{leaf['id']}-{j}")
        for name in VCOVCL_FILES
        if _vcovcl(name)["model"] in models
        for leaf in _vcovcl(name)["leaves"]
        for j in range(len(leaf["vcovcl"]))
    ]


def _leaf(name: str, node_id: int) -> dict[str, Any]:
    return next(lf for lf in _vcovcl(name)["leaves"] if lf["id"] == node_id)


def _vcov(name: str, node_id: int, entry: dict[str, Any]) -> mt.LeafVcov:
    tree = public(*_tree_of(name))
    with warnings.catch_warnings():
        # G <= k is tested on its own below.
        warnings.simplefilter("ignore", mt.TooFewClustersWarning)
        return tree.vcov(
            node_id,
            cluster=None if entry["cluster"] == "school" else False,
            type=entry["type"],
            cadjust=entry["cadjust"],
        )[node_id]


def test_fixture_coverage() -> None:
    models = {_vcovcl(n)["model"] for n in VCOVCL_FILES}
    assert models == {
        "lm",
        "glm_binomial",
        "glm_poisson",
        "negbin",
        "safelogit",
        "multinom",
    }
    for name in VCOVCL_FILES:
        for leaf in _vcovcl(name)["leaves"]:
            combos = {(e["cluster"], e["type"], e["cadjust"]) for e in leaf["vcovcl"]}
            assert combos == {
                (c, t, a)
                for c in ("school", None)
                for t in ("HC0", "HC1")
                for a in (True, False)
            }


# 1. Against sandwich::vcovCL.


@pytest.mark.parametrize(
    ("name", "node_id", "j"),
    mark_slow_trees(
        _entries("lm", "glm_binomial", "glm_poisson", "negbin", "safelogit")
    ),
)
def test_vcovcl(name: str, node_id: int, j: int) -> None:
    leaf = _leaf(name, node_id)
    entry = leaf["vcovcl"][j]
    got = _vcov(name, node_id, entry)
    assert got.status == "ok"
    assert got.names == tuple(leaf["coefficient_names"])
    np.testing.assert_allclose(got.matrix, np.array(entry["matrix"]), rtol=RTOL)
    if entry["cluster"] == "school":
        assert got.n_clusters == entry["n_clusters"] == leaf["n_clusters"]
    else:
        # sandwich's cluster = NULL is 1:n, G = n (unit weights).
        assert got.n_clusters is None and entry["n_clusters"] == leaf["n"]


def _moved_sandwich(name: str, node_id: int, entry: dict[str, Any]) -> np.ndarray:
    """Our sandwich at our coefficients moved by R's convergence offset
    ``-H^-1 g_R`` (docs/extensions-port.md, section 7)."""
    directory, tree_name = _tree_of(name)
    f = fixture(directory, tree_name)
    node = next(n for n in f["nodes"] if n["id"] == node_id)
    fit = grow(directory, tree_name).node(node_id).fit
    y, x, w, _ = tree_inputs(f)
    assert x is not None
    rows = np.array(node["rows"], dtype=np.intp)
    xr, yr, wr = x[rows], y[rows], w[rows]
    offset = _r_offset(fit, xr, yr, wr, np.array(node["estfun"]))
    p = xr.shape[1]
    beta = fit.coefficients.reshape(-1, p) + offset.reshape(-1, p)
    _, prob = multinom_evaluate(beta, xr, yr, wr)
    school = pl.read_csv(FIXTURES / f["data_csv"])["school"].to_numpy()[rows]
    moved = sandwich(
        multinom_hessian(prob, xr, wr),
        multinom_scores(prob, xr, yr, wr),
        weights=wr,
        cluster=school if entry["cluster"] == "school" else None,
        caseweights=True,
        type=entry["type"],
        cadjust=entry["cadjust"],
    )
    return moved.matrix


@pytest.mark.parametrize(
    ("name", "node_id", "j"), mark_slow_trees(_entries("multinom"))
)
def test_vcovcl_multinom(name: str, node_id: int, j: int) -> None:
    """nnet's estimate is off the MLE by ``-H^-1 g_R``, so R's matrix is
    ours at the moved coefficients. The tolerance is twice the change of our
    sandwich under that move, plus RTOL of the largest entry
    (docs/api-port.md, section 16)."""
    leaf = _leaf(name, node_id)
    entry = leaf["vcovcl"][j]
    got = _vcov(name, node_id, entry)
    assert got.status == "ok"
    expected = np.array(entry["matrix"])
    moved = _moved_sandwich(name, node_id, entry)
    tol = 2 * np.abs(moved - got.matrix) + RTOL * np.max(np.abs(expected))
    assert np.all(np.abs(got.matrix - expected) <= tol)
    # The move explains R's matrix far better than the tolerance needs.
    assert np.max(np.abs(moved - expected)) < 0.05 * np.max(
        np.abs(got.matrix - expected)
    ) + RTOL * np.max(np.abs(expected))


# 2. What R cannot check.


def test_too_few_clusters() -> None:
    # sim_cluster_few node 3: G = 3 schools for k = 4. R returns the
    # rank-deficient matrix without a warning (compared in test_vcovcl);
    # ours warns and gives the same matrix.
    name = "sim_cluster_few_lmtree_cluster"
    leaf = _leaf(name, 3)
    assert leaf["n_clusters"] == 3 and leaf["k"] == 4 and leaf["clusters_le_k"]
    tree = public(*_tree_of(name))
    for entry in leaf["vcovcl"]:
        if entry["cluster"] != "school":
            continue
        with pytest.warns(mt.TooFewClustersWarning, match="node 3 has 3 clusters"):
            got = tree.vcov(3, type=entry["type"], cadjust=entry["cadjust"])[3]
        np.testing.assert_allclose(got.matrix, np.array(entry["matrix"]), rtol=RTOL)
        # Rank G - 1 = 2: two eigenvalues at rounding level, as R's.
        lam = np.linalg.eigvalsh(got.matrix)
        assert np.all(np.abs(lam[:2]) < 1e-12 * lam[-1]) and lam[2] > 1e-6 * lam[-1]


def test_single_cluster_is_nan_where_r_stops() -> None:
    # R (sandwich 3.1-3) stops with "non-conformable arrays" for one
    # cluster; ours gives NaN with status unavailable (documented
    # divergence, docs/api-port.md, section 16).
    name = "sim_cluster_glmtree_binomial_cluster"
    single = _vcovcl(name)["single_cluster"]
    assert {e["cadjust"] for e in single} == {True, False}
    assert all(e["error"] == "non-conformable arrays" for e in single)
    tree = public(*_tree_of(name))
    for e in single:
        with pytest.warns(mt.TooFewClustersWarning, match="node 2 has 1 clusters"):
            got = tree.vcov(
                e["node"],
                cluster=np.ones(tree.n_rows),
                type=e["type"],
                cadjust=e["cadjust"],
            )[e["node"]]
        assert got.status == "unavailable" and got.n_clusters == 1
        assert got.reason is not None and "single cluster" in got.reason
        assert np.all(np.isnan(got.matrix)) and np.isfinite(got.condition)


def test_completely_separated_leaf_is_unavailable() -> None:
    tree = public("extension_trees", "sim_separation_safelogit")
    separated = [
        i for i in tree.leaves if tree.node(i).extras["separation"] == "complete"
    ]
    assert separated
    got = tree.vcov()
    for i in tree.leaves:
        expected = "unavailable" if i in separated else "ok"
        assert got[i].status == expected
    for i in separated:
        assert np.all(np.isnan(got[i].matrix))


# (directory, tree, weights column): the leaves of the tree give the row
# sets; the multinomial tree is unweighted, so the data's w0 (0 to 3) is used.
WEIGHTED = [
    ("trees", "sim_cluster_lmtree_cluster_weights", "w"),
    ("trees", "sim_cluster_glmtree_binomial_cluster_weights", "w"),
    ("extension_trees", "sim_negbin_weights", "w"),
    ("extension_trees", "sim_multinom_onesplit", "w0"),
]

# glm.fit's starting values depend on the weights, so weighted and
# replicated fits stop at different iterates within epsilon, and their
# working weights lag the estimate by different amounts. Without a
# convergence test (epsilon 0) IRLS runs to its fixed point, where both are
# the MLE to rounding; glm.fit then warns that it did not converge.
FIXED_POINT: dict[str, Any] = {"epsilon": 0.0, "maxit": 50}


def _single_node(
    f: dict[str, Any],
    df: pl.DataFrame,
    cluster: str | None,
    weights: str | None,
    theta: float | None = None,
) -> mt.ModelTree:
    """A one-node tree (maxdepth 1) of the fixture's model on ``df``.

    For negative binomial, ``theta`` fixes theta: a GlmModel with
    ``NegativeBinomial(theta)``, whose scores and Hessian are NegBinModel's
    at that theta (tests/test_hessians.py), grown by mob(). theta.ml's own
    tolerance would otherwise let weighted and replicated fits differ in
    theta by about 1e-7.
    """
    kw: dict[str, Any] = {"maxdepth": 1, "weights": weights, "cluster": cluster}
    formula = f["formula"]
    if f["function_name"] == "lmtree":
        return mt.lmtree(formula, df, **kw)
    if f["function_name"] == "glmtree":
        return mt.glmtree(
            formula, df, family=f["family"]["family"], **FIXED_POINT, **kw
        )
    if f["fit_function"] == "negbinfit":
        assert theta is not None and f["offset"] == "log(exposure)"
        x = np.column_stack([np.ones(df.height), df["x"].to_numpy()])
        return mt.mob(
            GlmModel(NegativeBinomial(theta), **FIXED_POINT),
            df[f["response"]].to_numpy().astype(np.float64),
            x,
            {"z1": ("numeric", df["z1"].to_numpy())},
            weights=None if weights is None else df[weights].to_numpy(),
            offset=np.log(df["exposure"].to_numpy()),
            cluster=None if cluster is None else df[cluster].to_numpy(),
            control=mt.MobControl(maxdepth=1),
        )
    return mt.multinomtree(formula, df, **kw)


def _leaf_frame(directory: str, name: str, node_id: int) -> pl.DataFrame:
    df = polars_frame(fixture(directory, name))
    return df[grow(directory, name).node(node_id).rows]


@pytest.mark.parametrize(
    ("directory", "name", "weights"), mark_slow_trees(WEIGHTED, name_at=1)
)
def test_frequency_weights_equal_replicated_rows(
    directory: str, name: str, weights: str
) -> None:
    """Integer case weights against the leaf's rows repeated w_i times,
    with the same clusters, and without clusters: the same covariance, so
    the n of the HC1 factor is the sum of the weights. Fitted to IRLS's
    fixed point, the two agree to rounding, so 1e-10 relative to each entry
    or to the largest."""
    f = fixture(directory, name)
    for node in grow(directory, name).nodes:
        if not node.terminal:
            continue
        df = _leaf_frame(directory, name, node.id)
        w = df[weights].to_numpy()
        assert np.all(w == np.round(w)) and np.any(w > 1)
        replicated = df.select(
            pl.all().repeat_by(pl.col(weights).cast(pl.UInt32))
        ).explode(df.columns)
        theta = node.fit.extras.get("theta")
        for cl in ["school", None] if "school" in df.columns else [None]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", mt.ModelTreesWarning)
                weighted = _single_node(f, df, cl, weights, theta)
                unit = _single_node(f, replicated, cl, None, theta)
            for type in ("HC0", "HC1"):
                a = weighted.vcov(1, type=type)[1]  # type: ignore[arg-type]
                b = unit.vcov(1, type=type)[1]  # type: ignore[arg-type]
                assert a.status == b.status == "ok"
                np.testing.assert_allclose(
                    a.matrix,
                    b.matrix,
                    rtol=1e-10,
                    atol=1e-10 * np.max(np.abs(b.matrix)),
                )
                assert a.n_clusters == b.n_clusters
            assert weighted.node(1).nobs == float(w.sum()) == unit.node(1).nobs


@pytest.mark.parametrize(
    ("directory", "name"),
    [
        ("trees", "sim_gaussian_lmtree_zero_weights"),
        ("trees", "sim_cluster_glmtree_binomial_cluster"),
    ],
)
def test_zero_weights_equal_deleted_rows(directory: str, name: str) -> None:
    f = fixture(directory, name)
    df = polars_frame(f)
    if f["weights"] is None:
        rng = np.random.default_rng(1)
        df = df.with_columns(pl.Series("w0", (rng.random(df.height) > 0.2) * 1.0))
        weights = "w0"
    else:
        weights = f["weights"]
    w = df[weights].to_numpy()
    assert np.any(w == 0)
    cl = "school" if "school" in df.columns else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mt.ModelTreesWarning)
        full = _single_node(f, df, cl, weights)
        kept = _single_node(f, df.filter(pl.col(weights) > 0), cl, weights)
    for type in ("HC0", "HC1"):
        a = full.vcov(1, type=type)[1]  # type: ignore[arg-type]
        b = kept.vcov(1, type=type)[1]  # type: ignore[arg-type]
        np.testing.assert_allclose(a.matrix, b.matrix, rtol=1e-12)
        assert a.n_clusters == b.n_clusters


# 3. statsmodels on single-node trees.


@cache
def _models_frame() -> pl.DataFrame:
    return polars_frame(
        fixture("cluster_trees", "sim_cluster_models_glmtree_poisson_cluster")
    )


@cache
def _cluster_frame() -> pl.DataFrame:
    return polars_frame(fixture("trees", "sim_cluster_lmtree_cluster"))


def _glm_exact(
    family: Family,
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    off: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """The exact GLM scores and expected information at ``beta``."""
    eta = x @ beta + (0 if off is None else off)
    mu = family.linkinv(eta)
    d = family.mu_eta(eta)
    v = family.variance(mu)
    scores = ((y - mu) * d / v)[:, None] * x
    h = x.T @ ((d**2 / v)[:, None] * x)
    return h, scores


CASES = ["lm", "binomial", "poisson", "negbin", "safelogit", "multinom"]


class _Case:
    """A single-node tree on school-clustered data, the statsmodels model
    of the same fit, and for the glm.fit-based models the family and data
    to bound the lag of glm.fit's working weights."""

    def __init__(self, kind: str) -> None:
        one: dict[str, Any] = {"maxdepth": 1, "cluster": "school"}
        df = _cluster_frame() if kind in ("lm", "binomial") else _models_frame()
        self.school = df["school"].to_numpy()
        self.x = np.column_stack([np.ones(df.height), df["x"].to_numpy()])
        self.off: np.ndarray | None = None
        self.family: Family | None = None
        if kind == "lm":
            self.tree = mt.lmtree("y_gauss ~ x | z1", df, **one)
            self.y = df["y_gauss"].to_numpy().astype(np.float64)
            self.model = sm.OLS(self.y, self.x)
        elif kind == "binomial":
            self.tree = mt.glmtree("y_bin ~ x | z1", df, family="binomial", **one)
            self.y = _binary(df["y_bin"], self.tree.response_levels)
            self.family = Binomial()
            self.model = sm.GLM(self.y, self.x, family=sm.families.Binomial())
        elif kind == "poisson":
            self.tree = mt.glmtree(
                "y_pois ~ x | z1", df, family="poisson", offset="log_exposure", **one
            )
            self.y = df["y_pois"].to_numpy().astype(np.float64)
            self.off = df["log_exposure"].to_numpy()
            self.family = Poisson()
            self.model = sm.GLM(
                self.y, self.x, family=sm.families.Poisson(), offset=self.off
            )
        elif kind == "negbin":
            self.tree = mt.negbintree("y_nb ~ x | z1", df, offset="log_exposure", **one)
            self.y = df["y_nb"].to_numpy().astype(np.float64)
            self.off = df["log_exposure"].to_numpy()
            theta = self.tree.node(1).core.fit.extras["theta_irls"]
            self.family = NegativeBinomial(theta)
            self.model = sm.GLM(
                self.y,
                self.x,
                family=sm.families.NegativeBinomial(alpha=1 / theta),
                offset=self.off,
            )
        elif kind == "safelogit":
            self.tree = mt.safelogittree("y_sl ~ x | z1", df, **one)
            self.y = _binary(df["y_sl"], self.tree.response_levels)
            self.family = Binomial()
            self.model = sm.Logit(self.y, self.x)
        else:
            self.tree = mt.multinomtree("y_mn ~ x | z1", df, **one)
            levels = self.tree.response_levels
            assert levels is not None
            labels = df["y_mn"].cast(pl.String).to_list()
            self.y = np.array([levels.index(v) for v in labels], dtype=np.float64)
            self.model = sm.MNLogit(self.y, self.x)

    def statsmodels(self, cov_type: str, groups: np.ndarray | None) -> np.ndarray:
        kw: dict[str, Any] = {"cov_type": cov_type}
        if groups is not None:
            kw["cov_kwds"] = {"groups": groups}
        if isinstance(self.model, sm.GLM):
            ref = self.model.fit(tol=1e-13, **kw)
        elif isinstance(self.model, sm.OLS):
            ref = self.model.fit(**kw)
        else:
            ref = self.model.fit(
                method="newton", tol=1e-14, maxiter=100, disp=False, **kw
            )
        return np.asarray(ref.cov_params())

    def comparable(self, got: np.ndarray) -> np.ndarray:
        """Our matrix in statsmodels' convention. For the negative binomial
        (log link, not canonical) statsmodels' robust bread is the observed
        information, R's ``bread.glm`` and ours the expected one
        (docs/api-port.md, section 16): keep our meat ``H V H`` and swap
        the bread for the observed information at our estimate."""
        if not isinstance(self.family, NegativeBinomial):
            return got
        fit = self.tree.model.fit(
            self.y,
            self.x,
            start=None,
            weights=np.ones(self.y.shape[0]),
            offset=self.off,
            cluster=None,
            caseweights=True,
            estfun=True,
            vcov=True,
        )
        assert fit.hessian is not None
        h = fit.hessian
        h_obs_inv = np.linalg.inv(h + self._observed_term(fit.coefficients))
        out: np.ndarray = h_obs_inv @ (h @ got @ h) @ h_obs_inv
        return out

    def _observed_term(self, beta: np.ndarray) -> np.ndarray:
        """Observed minus expected information of the negative binomial
        with log link: ``sum_i (y_i - mu_i) (mu_i / theta) / (1 + mu_i /
        theta)^2 x_i x_i'``."""
        assert isinstance(self.family, NegativeBinomial) and self.off is not None
        theta = self.family.theta
        mu = np.exp(self.x @ beta + self.off)
        c = (self.y - mu) * (mu / theta) / (1 + mu / theta) ** 2
        out: np.ndarray = self.x.T @ (c[:, None] * self.x)
        return out

    def tolerance(
        self, got: np.ndarray, cluster: np.ndarray | None, type: str, cadjust: bool
    ) -> np.ndarray:
        """Twice the difference between our sandwich and the one from the
        exact scores and information at our estimate (none for lm and
        multinomial, whose Hessians and scores are exact), plus RTOL of the
        largest entry. statsmodels computes the exact one at its own tight
        estimate. Both are in statsmodels' convention (``comparable``)."""
        base = np.full(got.shape, RTOL * np.max(np.abs(got)))
        if self.family is None:
            return base
        beta = self.tree.node(1).core.fit.coefficients
        h, scores = _glm_exact(self.family, self.y, self.x, beta, self.off)
        if isinstance(self.family, NegativeBinomial):
            h = h + self._observed_term(beta)
        exact = sandwich(
            h,
            scores,
            weights=np.ones(self.y.shape[0]),
            cluster=cluster,
            caseweights=True,
            type=type,  # type: ignore[arg-type]
            cadjust=cadjust,
        ).matrix
        out: np.ndarray = 2 * np.abs(self.comparable(got) - exact) + base
        return out


def _binary(column: pl.Series, levels: tuple[str, ...] | None) -> np.ndarray:
    if levels is None:
        return column.to_numpy().astype(np.float64)
    out: np.ndarray = (column.cast(pl.String).to_numpy() != levels[0]) * 1.0
    return out


@cache
def _cases(kind: str) -> _Case:
    return _Case(kind)


@pytest.mark.parametrize("kind", CASES)
def test_against_statsmodels_cluster(kind: str) -> None:
    """``cov_type="cluster"`` with its default correction, G / (G - 1)
    times (n - 1) / (n - k): our default HC1 with cadjust."""
    c = _cases(kind)
    got = c.tree.vcov(1)[1]
    assert got.status == "ok" and got.n_clusters == len(np.unique(c.school))
    expected = c.statsmodels("cluster", c.school)
    tol = c.tolerance(got.matrix, c.school, "HC1", True)
    assert np.all(np.abs(c.comparable(got.matrix) - expected) <= tol)


@pytest.mark.parametrize("kind", CASES)
def test_against_statsmodels_hc0_without_clusters(kind: str) -> None:
    c = _cases(kind)
    got = c.tree.vcov(1, cluster=False, type="HC0", cadjust=False)[1]
    assert got.status == "ok" and got.n_clusters is None
    expected = c.statsmodels("HC0", None)
    tol = c.tolerance(got.matrix, None, "HC0", False)
    assert np.all(np.abs(c.comparable(got.matrix) - expected) <= tol)


@pytest.mark.parametrize("kind", CASES)
def test_against_statsmodels_coarser_clusters(kind: str) -> None:
    c = _cases(kind)
    coarse = c.school // 3
    got = c.tree.vcov(1, cluster=coarse)[1]
    assert got.n_clusters == len(np.unique(coarse)) < len(np.unique(c.school))
    expected = c.statsmodels("cluster", coarse)
    tol = c.tolerance(got.matrix, coarse, "HC1", True)
    assert np.all(np.abs(c.comparable(got.matrix) - expected) <= tol)
