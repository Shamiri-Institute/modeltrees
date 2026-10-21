"""Coefficient tables, likelihood, printing and summary (docs/api-design.md,
sections 3.3, 3.4 and 3.6)."""

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest

import modeltrees as mt
from modeltrees.models import FitResult, LmModel, failed_fit


def _data(n: int = 600, seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.uniform(size=n)
    x = rng.normal(size=n)
    y = 1 + 2 * x * (z > 0.5) + rng.normal(size=n)
    return {"y": y, "x": x, "z": z}


def test_wide_and_long_layouts() -> None:
    tree = mt.lmtree("y ~ x | z", pl.DataFrame(_data()))
    wide = tree.coef_table()
    assert isinstance(wide, pl.DataFrame)
    assert wide.columns == [
        "node",
        "n_rows",
        "nobs",
        "status",
        "stop_reason",
        "(Intercept)",
        "x",
    ]
    assert wide["node"].to_list() == list(tree.leaves)
    for row in wide.iter_rows(named=True):
        info = tree.node(row["node"])
        assert row["(Intercept)"] == info.coefficients["(Intercept)"]
        assert row["x"] == info.coefficients["x"]
        assert row["n_rows"] == info.n_rows and row["nobs"] == info.nobs
    long = tree.coef_table(layout="long")
    assert long.columns == ["node", "term", "estimate"]
    assert long.height == 2 * len(tree.leaves)


def test_backend_rules() -> None:
    data = _data()
    # A pandas frame gives pandas, a polars frame polars, a dict polars.
    assert isinstance(
        mt.lmtree("y ~ x | z", pd.DataFrame(data)).coef_table(), pd.DataFrame
    )
    tree = mt.lmtree("y ~ x | z", data)
    assert isinstance(tree.coef_table(), pl.DataFrame)
    assert isinstance(tree.coef_table(backend="pandas"), pd.DataFrame)
    arrays = tree.coef_table(backend="numpy")
    assert isinstance(arrays, dict) and isinstance(arrays["x"], np.ndarray)
    assert arrays["status"].dtype == object
    with pytest.raises(ValueError, match="backend"):
        tree.coef_table(backend="excel")  # type: ignore[arg-type]
    with pytest.raises(ImportError, match="pyarrow"):
        tree.coef_table(backend="pyarrow")


def test_argument_errors() -> None:
    tree = mt.lmtree("y ~ x | z", _data())
    with pytest.raises(ValueError, match="se"):
        tree.coef_table(se="model")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layout"):
        tree.coef_table(layout="tall")  # type: ignore[arg-type]


def test_aic_and_bic() -> None:
    tree = mt.lmtree("y ~ x | z", _data())
    ll = tree.loglik()
    assert ll.df == 3 * len(tree.leaves) + len(tree.nodes) - len(tree.leaves)
    assert ll.nobs == 600
    assert tree.aic() == -2 * ll.value + 2 * ll.df
    assert tree.bic() == -2 * ll.value + math.log(600) * ll.df


class _FailingChildren(LmModel):
    def fit(self, y: Any, x: Any, **kwargs: Any) -> FitResult:
        if kwargs.get("estfun") and np.shape(y)[0] < 500:
            return failed_fit(("(Intercept)", "x"), "test failure")
        return super().fit(y, x, **kwargs)


def _failed_tree() -> mt.ModelTree:
    d = _data()
    xm = np.column_stack([np.ones(600), d["x"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mt.NodeFitFailedWarning)
        return mt.mob(
            _FailingChildren(coef_names=["(Intercept)", "x"]),
            d["y"],
            xm,
            {"z": ("numeric", d["z"])},
        )


def test_failed_leaf_makes_the_likelihood_nan() -> None:
    tree = _failed_tree()
    assert [i.status for i in tree.nodes if i.is_leaf] == ["failed", "failed"]
    with pytest.warns(mt.NodeFitFailedWarning, match="leaf 2, 3"):
        ll = tree.loglik()
    assert math.isnan(ll.value) and ll.df == 3 + 3 + 1
    with pytest.warns(mt.NodeFitFailedWarning):
        assert math.isnan(tree.aic())
    with pytest.warns(mt.NodeFitFailedWarning):
        assert math.isnan(tree.bic())


def test_failed_leaf_prints_its_reason() -> None:
    text = str(_failed_tree())
    assert "[2] z <= " in text and "[fit failed: test failure]" in text
    assert (
        text.splitlines()[0] == "Model-based recursive partitioning (_FailingChildren)"
    )


def test_summary_tables() -> None:
    data = _data()
    data["x"] = data["x"].copy()
    data["x"][:4] = np.nan
    with pytest.warns(mt.MissingDataWarning):
        tree = mt.lmtree("y ~ x | z", data)
    s = tree.summary()
    assert s.dropped_rows == 4 and s.missing_counts == {"x": 4}
    assert s.nodes["node"].to_list() == [i.id for i in tree.nodes]
    assert s.variables["variable"].to_list() == ["z"]
    tested = [i for i in tree.nodes if i.tests is not None]
    assert s.tests.height == sum(len(i.tests or {}) for i in tested)
    assert s.coefficients.equals(tree.coef_table())
    assert str(s).startswith(str(tree))
    assert "Parameter instability tests:" in str(s)
    assert "MissingDataWarning: 4 of 600 rows dropped" in str(s)
    assert "n = 596 (4 rows dropped for missing values: x: 4)" in str(tree)
