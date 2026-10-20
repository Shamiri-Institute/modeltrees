"""Prediction and routing of new data (docs/api-design.md, section 4.2)."""

import warnings
from functools import cache
from typing import Any

import numpy as np
import polars as pl
import pytest

import modeltrees as mt
from modeltrees.models import FitResult, LmModel, failed_fit


@cache
def _data() -> pl.DataFrame:
    """z1 splits at exactly 0.5 (values on a 0.01 grid); right of it f
    splits {a} against {b, c}, and level d occurs only left of 0.5."""
    rng = np.random.default_rng(7)
    n = 800
    z1 = np.round(rng.uniform(size=n), 2)
    f = np.where(
        z1 <= 0.5, rng.choice(["a", "b", "c", "d"], n), rng.choice(["a", "b", "c"], n)
    )
    x = rng.normal(size=n)
    slope = np.where(z1 <= 0.5, -2.0, np.where(f == "a", 2.0, 5.0))
    y = 1 + slope * x + rng.normal(size=n)
    return pl.DataFrame({"y": y, "x": x, "z1": z1, "f": f, "noise": rng.normal(size=n)})


@cache
def _tree() -> mt.ModelTree:
    tree = mt.lmtree("y ~ x | z1 + f + noise", _data())
    # The structure the routing tests rely on.
    assert [n.rule for n in tree.nodes] == [
        (),
        ("z1 <= 0.5",),
        ("z1 > 0.5",),
        ("z1 > 0.5", "f in {a}"),
        ("z1 > 0.5", "f in {b, c}"),
    ]
    split = tree.node(3).split
    assert split is not None and split.absent_levels == ("d",)
    assert tree.node(4).nobs < tree.node(5).nobs
    return tree


def _new(**columns: list[Any]) -> pl.DataFrame:
    n = len(next(iter(columns.values())))
    base: dict[str, Any] = {"x": [1.0] * n, "z1": [0.2] * n, "f": ["a"] * n}
    base["noise"] = [0.0] * n
    base.update(columns)
    return pl.DataFrame(
        base,
        schema={"x": pl.Float64, "z1": pl.Float64, "f": pl.String, "noise": pl.Float64},
    )


def _leaf_value(tree: mt.ModelTree, node: int, x: float) -> float:
    c = tree.node(node).coefficients
    return c["(Intercept)"] + c["x"] * x


def test_training_routing_and_values() -> None:
    tree = _tree()
    df = _data()
    nodes = tree.predict(df, type="node")
    assert nodes.dtype == np.int64
    assert np.array_equal(nodes, tree.fitted_node)
    assert np.array_equal(tree.predict(type="node"), tree.fitted_node)
    x = df["x"].to_numpy()
    expected = np.array(
        [_leaf_value(tree, int(k), xi) for k, xi in zip(nodes, x, strict=True)]
    )
    np.testing.assert_allclose(tree.predict(df), expected, rtol=1e-12)
    np.testing.assert_array_equal(tree.predict(df), tree.predict())


def test_exact_break_goes_left() -> None:
    tree = _tree()
    new = _new(z1=[0.5, float(np.nextafter(0.5, 1)), -np.inf, np.inf], f=["b"] * 4)
    assert tree.predict(new, type="node").tolist() == [2, 5, 2, 5]


def test_level_absent_from_the_node_goes_to_the_larger_kid() -> None:
    tree = _tree()
    new = _new(z1=[0.3, 0.8, 0.8, 0.8], f=["d", "d", "a", "c"])
    assert tree.predict(new, type="node").tolist() == [2, 5, 4, 5]
    values = tree.predict(new)
    assert values[1] == pytest.approx(_leaf_value(tree, 5, 1.0), rel=1e-12)


def test_unseen_level_raises_by_default() -> None:
    tree = _tree()
    new = _new(z1=[0.3, 0.8], f=["e", "a"])
    with pytest.raises(mt.UnseenLevelError, match=r"f: 'e'"):
        tree.predict(new)
    with pytest.raises(mt.UnseenLevelError):
        tree.predict(new, type="node")


def test_unseen_level_null_gives_no_prediction() -> None:
    tree = _tree()
    # The unseen level is off the path of the first row (z1 <= 0.5) but the
    # row still gets no prediction.
    new = _new(z1=[0.3, 0.8, 0.8], f=["e", "e", "a"])
    assert tree.predict(new, type="node", unseen="null").tolist() == [0, 0, 4]
    values = tree.predict(new, unseen="null")
    assert np.isnan(values[:2]).all()
    assert values[2] == pytest.approx(_leaf_value(tree, 4, 1.0), rel=1e-12)


def test_missing_split_variable_on_and_off_the_path() -> None:
    tree = _tree()
    new = pl.DataFrame(
        {
            "x": [1.0, 1.0, 1.0, 1.0, None],
            "z1": [None, 0.3, 0.8, 0.8, 0.8],
            "f": ["a", None, None, "b", "b"],
            "noise": [0.0, 0.0, 0.0, None, 0.0],
        },
        schema={"x": pl.Float64, "z1": pl.Float64, "f": pl.String, "noise": pl.Float64},
    )
    # Row 0: z1 missing at the root. Row 1: f missing, but off the path.
    # Row 2: f missing on the path. Row 3: noise is never split on.
    # Row 4: the regressor is missing; the node is still returned.
    assert tree.predict(new, type="node").tolist() == [0, 2, 0, 5, 5]
    values = tree.predict(new)
    assert np.isnan(values[[0, 2, 4]]).all()
    assert values[1] == pytest.approx(_leaf_value(tree, 2, 1.0), rel=1e-12)
    assert values[3] == pytest.approx(_leaf_value(tree, 5, 1.0), rel=1e-12)


def test_float_nan_counts_as_missing() -> None:
    tree = _tree()
    new = _new(z1=[float("nan"), 0.3])
    assert tree.predict(new, type="node").tolist() == [0, 2]


def test_node_type_does_not_need_the_regressors() -> None:
    tree = _tree()
    new = pl.DataFrame({"z1": [0.3, 0.8], "f": ["a", "b"], "noise": [0.0, 0.0]})
    assert tree.predict(new, type="node").tolist() == [2, 5]
    with pytest.raises(mt.DataError, match="'x'"):
        tree.predict(new)


def test_intercept_only_row_count_comes_from_newdata() -> None:
    tree = mt.lmtree("y ~ z1 + f", _data())
    new = _new(z1=[0.3, 0.8, 0.9])
    values = tree.predict(new)
    assert values.shape == (3,)
    for v, k in zip(values, tree.predict(new, type="node"), strict=True):
        assert v == tree.node(int(k)).coefficients["(Intercept)"]


def test_argument_errors() -> None:
    tree = _tree()
    with pytest.raises(ValueError, match="type"):
        tree.predict(type="mean")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unseen"):
        tree.predict(unseen="drop")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'link' or 'response'"):
        tree.predict(type="prob")
    with pytest.raises(TypeError, match="mob"):
        tree.predict(z={"z1": np.array([0.3])})
    with pytest.raises(TypeError, match="newdata"):
        tree.predict(offset=np.zeros(800))


def test_offset_column_is_read_from_newdata() -> None:
    rng = np.random.default_rng(5)
    n = 500
    z = rng.uniform(size=n)
    exposure = rng.uniform(1, 3, size=n)
    y = rng.poisson(exposure * np.exp(0.5 + (z > 0.5)))
    df = pl.DataFrame({"y": y, "z": z, "log_exposure": np.log(exposure)})
    tree = mt.glmtree("y ~ z", df, family="poisson", offset="log_exposure")
    nodes = tree.predict(df, type="node")
    beta = np.array([tree.node(int(k)).coefficients["(Intercept)"] for k in nodes])
    eta = beta + np.log(exposure)
    np.testing.assert_allclose(tree.predict(df, type="link"), eta, rtol=1e-12)
    np.testing.assert_allclose(tree.predict(df), np.exp(eta), rtol=1e-12)
    np.testing.assert_array_equal(tree.predict(df), tree.predict())
    # An explicit offset overrides the column; a missing one gives NaN.
    zero = tree.predict(df, type="link", offset=np.zeros(n))
    np.testing.assert_allclose(zero, beta, rtol=1e-12)
    no_col = df.drop("log_exposure")
    with pytest.raises(mt.DataError, match="log_exposure"):
        tree.predict(no_col)
    assert tree.predict(no_col, type="node").tolist() == nodes.tolist()
    holes = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(None)
        .otherwise(pl.col("log_exposure"))
        .alias("log_exposure")
    )
    values = tree.predict(holes)
    assert np.isnan(values[0]) and not np.isnan(values[1:]).any()


def test_multinomial_types() -> None:
    rng = np.random.default_rng(11)
    n = 900
    z = rng.uniform(size=n)
    x = rng.normal(size=n)
    eta = np.column_stack([np.zeros(n), 0.5 + x * (z > 0.5), -0.5 - x * (z > 0.5)])
    p = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    y = np.array(["lo", "mid", "hi"])[[rng.choice(3, p=pi) for pi in p]]
    df = pl.DataFrame({"y": y, "x": x, "z": z}).with_columns(
        pl.col("y").cast(pl.Enum(["lo", "mid", "hi"]))
    )
    tree = mt.multinomtree("y ~ x | z", df)
    assert tree.response_levels == ("lo", "mid", "hi")
    prob = tree.predict(df, type="prob")
    link = tree.predict(df, type="link")
    assert prob.shape == (n, 3) and link.shape == (n, 2)
    np.testing.assert_allclose(prob.sum(axis=1), 1.0, rtol=1e-12)
    labels = tree.predict(df, type="class")
    assert labels.dtype == object
    assert labels.tolist() == [("lo", "mid", "hi")[i] for i in np.argmax(prob, axis=1)]
    with pytest.raises(ValueError, match="'prob'.*'class'"):
        tree.predict(df)
    new = df.head(2).with_columns(pl.Series("x", [None, 1.0], dtype=pl.Float64))
    assert tree.predict(new, type="class").tolist()[0] is None
    assert np.isnan(tree.predict(new, type="prob")[0]).all()


class _FailingChildren(LmModel):
    """Least squares whose node fits fail below 500 rows, so that the
    children of the root are failed leaves (split search still works)."""

    def fit(self, y: Any, x: Any, **kwargs: Any) -> FitResult:
        if kwargs.get("estfun") and np.shape(y)[0] < 500:
            return failed_fit(("(Intercept)", "x"), "test failure")
        return super().fit(y, x, **kwargs)


def test_failed_leaves_predict_nan_but_route() -> None:
    df = _data()
    y, x, z1 = df["y"].to_numpy(), df["x"].to_numpy(), df["z1"].to_numpy()
    xm = np.column_stack([np.ones_like(x), x])
    model = _FailingChildren(coef_names=["(Intercept)", "x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mt.NodeFitFailedWarning)
        tree = mt.mob(model, y, xm, {"z1": ("numeric", z1)}, control=mt.MobControl())
    assert [n.status for n in tree.nodes] == ["ok", "failed", "failed"]
    nodes = tree.predict(type="node")
    assert set(nodes.tolist()) == {2, 3}
    assert np.isnan(tree.predict()).all()
    new = tree.predict(z={"z1": np.array([0.1, 0.9])}, x=np.array([[1.0, 0.0]] * 2))
    assert np.isnan(new).all()
    assert tree.predict(z={"z1": np.array([0.1, 0.9])}, type="node").tolist() == [2, 3]


def test_no_estfun_leaves_predict_from_their_coefficients() -> None:
    # A constant-response group: the safe logit fit there has no scores.
    rng = np.random.default_rng(2)
    n = 400
    g = rng.choice(["g1", "g2"], n)
    x = rng.normal(size=n)
    y = np.where(g == "g1", 0, (rng.uniform(size=n) < 1 / (1 + np.exp(-x)))).astype(int)
    df = pl.DataFrame({"y": y, "x": x, "g": g})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mt.GlmFitWarning)
        tree = mt.safelogittree("y ~ x | g", df)
    statuses = {n.id: n.status for n in tree.nodes if n.is_leaf}
    assert "no_estfun" in statuses.values()
    nodes = tree.predict(df, type="node")
    eta = np.array(
        [
            tree.node(int(k)).coefficients["(Intercept)"]
            + tree.node(int(k)).coefficients["x"] * xi
            for k, xi in zip(nodes, x, strict=True)
        ]
    )
    np.testing.assert_allclose(tree.predict(df, type="link"), eta, rtol=1e-12)
    assert not np.isnan(tree.predict(df)).any()


def test_mob_tree_takes_codes_or_labels() -> None:
    df = _data()
    levels = ("a", "b", "c", "d")
    codes = np.array([levels.index(v) for v in df["f"].to_list()])
    y, x, z1 = df["y"].to_numpy(), df["x"].to_numpy(), df["z1"].to_numpy()
    xm = np.column_stack([np.ones_like(x), x])
    tree = mt.mob(
        LmModel(coef_names=["(Intercept)", "x"]),
        y,
        xm,
        {"z1": ("numeric", z1), "f": ("nominal", codes, levels)},
    )
    by_codes = tree.predict(z={"z1": z1, "f": codes}, type="node")
    by_labels = tree.predict(z={"z1": z1, "f": df["f"].to_numpy()}, type="node")
    assert np.array_equal(by_codes, tree.fitted_node)
    assert np.array_equal(by_labels, tree.fitted_node)
    with pytest.raises(mt.UnseenLevelError):
        tree.predict(z={"z1": z1[:1], "f": np.array(["e"])}, type="node")
    with pytest.raises(ValueError, match="pass x"):
        tree.predict(z={"z1": z1, "f": codes})
    np.testing.assert_array_equal(
        tree.predict(z={"z1": z1, "f": codes}, x=xm), tree.predict()
    )
