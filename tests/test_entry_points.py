"""The entry points (docs/api-design.md, sections 1.1, 7.3 and 8)."""

import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest

import modeltrees as mt
from modeltrees import models
from modeltrees.errors import GlmFitWarning, MissingDataWarning
from modeltrees.tree import mob as core_mob


def _gaussian(n: int = 300, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.uniform(size=n)
    x = rng.normal(size=n)
    y = 1 + 2 * x * (z > 0.5) + rng.normal(size=n)
    return pl.DataFrame({"y": y, "x": x, "z": z})


def _separated(n: int = 600, seed: int = 3) -> pl.DataFrame:
    """Binary data separated by x in two regions of z, so that glm.fit
    warns in more than one node."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(size=n)
    x = rng.normal(size=n)
    noise = rng.integers(0, 2, n).astype(float)
    y = np.where(z > 0.66, x > 0, np.where(z < 0.33, x < 0, noise)).astype(float)
    return pl.DataFrame({"y": y, "x": x, "z": z})


def test_public_names() -> None:
    expected = {
        "lmtree",
        "glmtree",
        "negbintree",
        "safelogittree",
        "multinomtree",
        "mob",
        "ModelTree",
        "NodeInfo",
        "SplitRule",
        "TestResult",
        "LogLik",
        "LeafVcov",
        "TreeSummary",
        "RecordedWarning",
        "MobControl",
        "ModelTreesError",
        "FormulaError",
        "DataError",
        "UnseenLevelError",
        "FitError",
        "ModelTreesWarning",
        "NodeFitFailedWarning",
        "TooFewClustersWarning",
        "MissingDataWarning",
        "GlmFitWarning",
        "reference_path",
        "__version__",
    }
    assert set(mt.__all__) == expected
    assert not hasattr(mt, "hello")
    for name in mt.__all__:
        assert hasattr(mt, name)
    assert set(models.__all__) == {
        "LmModel",
        "GlmModel",
        "NegBinModel",
        "SafeLogitModel",
        "MultinomModel",
        "NodeModel",
        "FitResult",
        "FitStatus",
        "failed_fit",
        "PredictingModel",
        "LogLikModel",
        "ReportingModel",
        "BatchSplitModel",
    }


def test_control_options_and_control_object_agree() -> None:
    df = _gaussian()
    a = mt.lmtree("y ~ x | z", df, alpha=0.01, minsize=40)
    b = mt.lmtree("y ~ x | z", df, control=mt.MobControl(alpha=0.01, minsize=40))
    assert a.control == b.control == mt.MobControl(alpha=0.01, minsize=40)
    assert [n.rule for n in a.nodes] == [n.rule for n in b.nodes]


def test_control_errors() -> None:
    df = _gaussian()
    with pytest.raises(TypeError, match="not both"):
        mt.lmtree("y ~ x | z", df, control=mt.MobControl(), alpha=0.1)
    with pytest.raises(TypeError, match="alhpa"):
        mt.lmtree("y ~ x | z", df, alhpa=0.1)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="MobControl"):
        mt.lmtree("y ~ x | z", df, control={"alpha": 0.1})  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="mtry"):
        mt.lmtree("y ~ x | z", df, mtry=2)


@pytest.mark.parametrize("n_jobs", [2, -1, 0])
def test_n_jobs_other_than_one_is_not_implemented(n_jobs: int) -> None:
    with pytest.raises(NotImplementedError, match="n_jobs"):
        mt.lmtree("y ~ x | z", _gaussian(), n_jobs=n_jobs)


def test_n_jobs_none_and_one_run_serially() -> None:
    df = _gaussian()
    a = mt.lmtree("y ~ x | z", df, n_jobs=1)
    b = mt.lmtree("y ~ x | z", df)
    assert [n.objective for n in a.nodes] == [n.objective for n in b.nodes]
    with pytest.raises(TypeError, match="n_jobs"):
        mt.lmtree("y ~ x | z", df, n_jobs=True)


def test_family_out_of_scope() -> None:
    with pytest.raises(NotImplementedError, match="Gamma"):
        mt.glmtree("y ~ x | z", _gaussian(), family="Gamma")  # type: ignore[arg-type]


def test_model_names_and_formula() -> None:
    df = _gaussian()
    assert mt.lmtree("y ~ z", df).formula == "y ~ 1 | z"
    assert mt.lmtree("y ~ z", df).model_name == "lmtree"
    assert mt.glmtree("y ~ x | z", df).model_name == "glmtree(gaussian)"
    counts = df.with_columns(pl.col("y").abs().round().cast(pl.Int64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlmFitWarning)
        tree = mt.negbintree("y ~ x | z", counts, maxdepth=1)
    assert tree.model_name == "negbintree"


def test_mob_wraps_the_core_tree() -> None:
    df = _gaussian()
    y, x, z = df["y"].to_numpy(), df["x"].to_numpy(), df["z"].to_numpy()
    xm = np.column_stack([np.ones_like(x), x])
    model = models.LmModel(coef_names=["(Intercept)", "x"])
    tree = mt.mob(model, y, xm, {"z": ("numeric", z)})
    core = core_mob(model, y, xm, {"z": ("numeric", z)})
    assert isinstance(tree, mt.ModelTree)
    assert tree.formula is None and tree.model_name == "mob(LmModel)"
    assert np.array_equal(tree.fitted_node, core.fitted_node)
    assert [n.objective for n in tree.nodes] == [n.fit.objfun for n in core.nodes]
    with pytest.raises(NotImplementedError):
        mt.mob(model, y, xm, {"z": ("numeric", z)}, n_jobs=4)


def _grow_separated(
    **kwargs: Any,
) -> tuple[mt.ModelTree, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tree = mt.glmtree("y ~ x | z", _separated(), family="binomial", **kwargs)
    return tree, caught


def test_warnings_are_recorded_merged_and_reemitted_once() -> None:
    tree, caught = _grow_separated()
    assert tree.warnings, "the data should make glm.fit warn"
    assert any(r.count > 1 for r in tree.warnings)
    # One re-emitted warning per distinct message, with the count.
    assert len(caught) == len(tree.warnings)
    for r, w in zip(tree.warnings, caught, strict=True):
        assert w.category is r.category is GlmFitWarning
        suffix = "" if r.count == 1 else f" ({r.count} times)"
        assert str(w.message) == r.message + suffix
        # Attributed to the caller of the entry point.
        assert w.filename == __file__


def test_user_error_filter_applies_to_reemitted_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", GlmFitWarning)
        with pytest.raises(GlmFitWarning):
            mt.glmtree("y ~ x | z", _separated(), family="binomial")


def test_missing_data_warning_is_emitted_once_and_recorded() -> None:
    df = _gaussian().with_columns(
        pl.when(pl.int_range(pl.len()) < 7).then(None).otherwise(pl.col("x")).alias("x")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tree = mt.lmtree("y ~ x | z", df)
    missing = [w for w in caught if w.category is MissingDataWarning]
    assert len(missing) == 1
    assert str(missing[0].message) == "7 of 300 rows dropped for missing values (x: 7)"
    assert tree.warnings == (
        mt.RecordedWarning(
            MissingDataWarning, "7 of 300 rows dropped for missing values (x: 7)", 1
        ),
    )
    assert tree.n_rows == 300 and tree.n_used == 293
    assert np.all(tree.fitted_node[:7] == 0) and np.all(tree.fitted_node[7:] > 0)


def test_warnings_before_an_error_are_still_emitted() -> None:
    # The categorical response has missing values: the drop is warned
    # about, then lmtree rejects the response.
    f = ["a", "b"] * 150
    df = _gaussian().with_columns(
        pl.Series("f", [None, None, None, *f[3:]], dtype=pl.String)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(mt.DataError, match="'f'.*categorical"):
            mt.lmtree("f ~ x | z", df)
    assert [w.category for w in caught] == [MissingDataWarning]


# Exceptions from the frontend (section 7.1).


def test_formula_error_with_position() -> None:
    with pytest.raises(mt.FormulaError, match="log") as info:
        mt.lmtree("y ~ log(x) | z", _gaussian())
    assert info.value.position == 4
    assert isinstance(info.value, ValueError)


def test_data_errors() -> None:
    df = _gaussian()
    with pytest.raises(mt.DataError, match="'w'"):
        mt.lmtree("y ~ x | z", df, weights="w")
    with pytest.raises(mt.DataError, match="collect"):
        mt.lmtree("y ~ x | z", df.lazy())  # type: ignore[arg-type]
    with pytest.raises(mt.DataError, match="negative"):
        mt.lmtree("y ~ x | z", df, weights=np.full(300, -1.0))
    holes = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 2).then(None).otherwise(pl.col("z")).alias("z")
    )
    with pytest.raises(mt.DataError, match="na_action='fail'"):
        mt.lmtree("y ~ x | z", holes, na_action="fail")
    infinite = df.with_columns(pl.col("y").replace(df["y"][0], float("inf")))
    with pytest.raises(mt.DataError, match="infinite"):
        mt.lmtree("y ~ x | z", infinite)
    counts = df.with_columns((pl.col("y") - 10).round().alias("y"))
    with pytest.raises(mt.DataError, match="negative"):
        mt.glmtree("y ~ x | z", counts, family="poisson")


def test_fit_error_reaches_the_user_with_its_type() -> None:
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"y": rng.poisson(2, 200), "z": rng.uniform(size=200)})
    with pytest.raises(mt.FitError, match="first iteration failed") as info:
        mt.glmtree("y ~ z", df, family="poisson", maxit=0)
    assert type(info.value).__name__ == "GlmFitError"


def test_type_errors() -> None:
    with pytest.raises(TypeError, match="eager dataframe"):
        mt.lmtree("y ~ x | z", [1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="offset"):
        mt.multinomtree("y ~ x | z", _gaussian(), offset="x")
