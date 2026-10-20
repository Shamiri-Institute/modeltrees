"""Model-specific response coding (docs/api-design.md, section 4.3)."""

import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest

from modeltrees._design import build_design
from modeltrees._response import CodedResponse, ResponseRule, code_response
from modeltrees.errors import DataError, MissingDataWarning


def _code(
    data: Any, rule: ResponseRule, *, weighted: bool = False, **kwargs: Any
) -> CodedResponse:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingDataWarning)
        design = build_design("y ~ z", data, **kwargs)
    return code_response(design.response, rule, weighted=weighted)


Z = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


# binomial


def test_binomial_categorical_first_level_is_failure() -> None:
    df = pl.DataFrame({"y": ["no", "yes", "maybe", "no", "yes", "no"], "z": Z})
    r = _code(df, "binomial")
    # Code-point sort: maybe, no, yes; only "maybe" is failure.
    assert r.levels == ("maybe", "no", "yes")
    assert np.array_equal(r.values, [1, 1, 0, 1, 1, 1])
    assert r.values.dtype == np.float64


def test_binomial_enum_order_decides_failure() -> None:
    y = pl.Series("y", ["no", "yes", "yes", "no", "yes", "no"])
    df = pl.DataFrame({"y": y.cast(pl.Enum(["yes", "no"])), "z": Z})
    r = _code(df, "binomial")
    assert r.levels == ("yes", "no")
    assert np.array_equal(r.values, [1, 0, 0, 1, 0, 1])


def test_binomial_unused_level_is_dropped_before_coding() -> None:
    y = pl.Series("y", ["b", "c", "c", "b", "c", "b"])
    df = pl.DataFrame({"y": y.cast(pl.Enum(["a", "b", "c"])), "z": Z})
    r = _code(df, "binomial")
    # "a" never occurs, so "b" is the first level and the failure.
    assert r.levels == ("b", "c")
    assert np.array_equal(r.values, [0, 1, 1, 0, 1, 0])


def test_binomial_numeric_and_boolean() -> None:
    r = _code({"y": np.array([0.0, 1, 1, 0, 1, 0]), "z": Z}, "binomial")
    assert r.levels is None and np.array_equal(r.values, [0, 1, 1, 0, 1, 0])
    r = _code(
        {"y": np.array([False, True, True, False, True, False]), "z": Z}, "binomial"
    )
    assert r.levels is None and np.array_equal(r.values, [0, 1, 1, 0, 1, 0])


def test_binomial_proportions_need_weights() -> None:
    y = np.array([0.0, 0.5, 1, 0.25, 1, 0])
    data = {"y": y, "z": Z, "n": np.full(6, 4.0)}
    with pytest.raises(DataError, match="'y'.*weights"):
        _code(data, "binomial")
    r = _code(data, "binomial", weighted=True, weights="n")
    assert np.array_equal(r.values, y)


def test_binomial_values_outside_unit_interval() -> None:
    with pytest.raises(DataError, match=r"'y'.*outside \[0, 1\]"):
        _code({"y": np.array([0.0, 2, 1, 0, 1, 0]), "z": Z}, "binomial")


# safelogit


def test_safelogit_codes_as_binomial() -> None:
    df = pl.DataFrame({"y": ["no", "yes", "yes", "no", "yes", "no"], "z": Z})
    r = _code(df, "safelogit")
    assert r.levels == ("no", "yes")
    assert np.array_equal(r.values, [0, 1, 1, 0, 1, 0])


def test_safelogit_rejects_proportions_even_with_weights() -> None:
    data = {"y": np.array([0.0, 0.5, 1, 0.25, 1, 0]), "z": Z, "n": np.full(6, 4.0)}
    with pytest.raises(DataError, match="'y'.*binary"):
        _code(data, "safelogit", weighted=True, weights="n")


# multinom


def test_multinom_categorical_records_order() -> None:
    y = pl.Series("y", ["hi", "lo", "mid", "lo", "hi", "mid"])
    df = pl.DataFrame({"y": y.cast(pl.Enum(["lo", "mid", "hi"])), "z": Z})
    r = _code(df, "multinom")
    assert r.levels == ("lo", "mid", "hi")
    assert r.values.dtype == np.intp
    assert np.array_equal(r.values, [2, 0, 1, 0, 2, 1])


def test_multinom_drops_unused_levels() -> None:
    y = pl.Series("y", ["hi", "mid", "mid", "hi", "hi", "mid"])
    df = pl.DataFrame({"y": y.cast(pl.Enum(["lo", "mid", "hi"])), "z": Z})
    r = _code(df, "multinom")
    # "lo" never occurs: "mid" becomes the reference class.
    assert r.levels == ("mid", "hi")
    assert np.array_equal(r.values, [1, 0, 0, 1, 1, 0])


def test_multinom_integer_with_levels() -> None:
    data = {"y": np.array([3, 1, 2, 1, 3, 2]), "z": Z}
    r = _code(data, "multinom", levels={"y": [1, 2, 3]})
    assert r.levels == ("1", "2", "3")
    assert np.array_equal(r.values, [2, 0, 1, 0, 2, 1])


def test_multinom_rejects_numeric_and_single_class() -> None:
    with pytest.raises(DataError, match="'y'.*categorical"):
        _code({"y": np.array([3, 1, 2, 1, 3, 2]), "z": Z}, "multinom")
    with pytest.raises(DataError, match="'y'.*at least two"):
        _code({"y": np.array(["a"] * 6), "z": Z}, "multinom")


# numeric models


@pytest.mark.parametrize("rule", ["gaussian", "poisson", "negbin"])
def test_numeric_models_reject_categorical(rule: ResponseRule) -> None:
    with pytest.raises(DataError, match="'y'.*categorical"):
        _code({"y": np.array(["a", "b"] * 3), "z": Z}, rule)


def test_gaussian_takes_any_finite_number() -> None:
    y = np.array([-1.5, 2.0, 3.25, 0.0, 1e6, -7.0])
    r = _code({"y": y, "z": Z}, "gaussian")
    assert r.levels is None and np.array_equal(r.values, y)


@pytest.mark.parametrize("rule", ["poisson", "negbin"])
def test_counts(rule: ResponseRule) -> None:
    y = np.array([0, 3, 1, 0, 12, 2])
    r = _code({"y": y, "z": Z}, rule)
    assert np.array_equal(r.values, y) and r.values.dtype == np.float64
    with pytest.raises(DataError, match="'y'.*negative"):
        _code({"y": np.array([0, -1, 1, 0, 2, 2]), "z": Z}, rule)
    with pytest.raises(DataError, match="'y'.*non-integer"):
        _code({"y": np.array([0, 1.5, 1, 0, 2, 2]), "z": Z}, rule)
