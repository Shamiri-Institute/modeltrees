"""The optional node-model protocols: loglik, report and predict.

docs/api-design.md, sections 3.5, 3.6 and 4.2. The tree log-likelihood is
checked against R's ``logLik(tree)`` recorded in every tree fixture's
summary. Predictions are checked against the linear predictor and inverse
link computed by hand from each leaf's coefficients.
"""

import copy
import math
from typing import Any

import numpy as np
import pytest
from grown_trees import ALL_TREES, fixture, grow, model, tree_inputs

from modeltrees._loglik import tree_loglik
from modeltrees.models.glm import GlmModel
from modeltrees.models.lm import LmModel
from modeltrees.models.multinom import MultinomModel
from modeltrees.models.negbin import NegBinModel
from modeltrees.models.safelogit import SafeLogitModel
from modeltrees.node_model import failed_fit

RTOL = 1e-8
PREDICT_RTOL = 1e-12
DBL_EPSILON = float(np.finfo(np.float64).eps)

# partykit's tree on zero weights is not an oracle (docs/decisions.md): it
# fails the root test, while the port treats zero weights as deleted rows.
NOT_ORACLE = "sim_gaussian_lmtree_zero_weights"
DEVIANCE = ("extension_trees", "quine_negbin_intercept_deviance")
SEPARATED = ("extension_trees", "sim_separation_safelogit")

REPORT_KEYS: dict[type, tuple[str, ...]] = {
    LmModel: (),
    GlmModel: ("converged",),
    NegBinModel: ("theta", "theta_se", "theta_warning", "converged", "n_warnings"),
    SafeLogitModel: ("separation", "n_boundary", "converged", "n_warnings"),
    MultinomModel: ("fit_status", "classes_present", "converged"),
}


def _trees_with(method: str, *, oracle: bool = False) -> list[Any]:
    return [
        pytest.param(d, n, id=f"{d}/{n}")
        for d, n in ALL_TREES
        if hasattr(model(fixture(d, n)), method) and not (oracle and n == NOT_ORACLE)
    ]


# logLik.


@pytest.mark.parametrize(("directory", "name"), _trees_with("loglik", oracle=True))
def test_tree_loglik_matches_r(directory: str, name: str) -> None:
    f = fixture(directory, name)
    ll = tree_loglik(grow(directory, name), model(f))  # type: ignore[arg-type]
    assert ll.value == pytest.approx(f["summary"]["loglik"], rel=RTOL, abs=0)
    assert ll.df == f["summary"]["loglik_df"]
    assert ll.nobs == f["nobs"]


def test_deviance_objective_loglik_is_not_minus_objfun() -> None:
    """quine with objective = "deviance": the leaf loglik is negbin's
    log-likelihood at the leaf's theta, not minus the deviance."""
    grown = grow(*DEVIANCE)
    m = model(fixture(*DEVIANCE))
    assert isinstance(m, NegBinModel)
    for leaf in (node for node in grown.nodes if node.terminal):
        value, df = m.loglik(leaf.fit)
        assert df == 2
        assert not math.isclose(value, -leaf.fit.objfun, rel_tol=1e-3)
        assert value == leaf.fit.extras["twologlik"] / 2


def _failed_names(m: Any) -> tuple[str, ...]:
    if isinstance(m, MultinomModel):
        return ("b:(Intercept)", "b:x", "c:(Intercept)", "c:x")
    return ("(Intercept)", "x")


MODELS = [
    LmModel(),
    GlmModel("binomial"),
    GlmModel("gaussian"),
    NegBinModel(),
    NegBinModel(objective="deviance"),
    SafeLogitModel(),
    MultinomModel(["a", "b", "c"]),
]


def _models_with(method: str) -> list[Any]:
    return [
        pytest.param(m, id=f"{type(m).__name__}-{i}")
        for i, m in enumerate(MODELS)
        if hasattr(m, method)
    ]


@pytest.mark.parametrize("m", _models_with("loglik"))
def test_failed_fit_loglik_is_nan(m: Any) -> None:
    names = _failed_names(m)
    value, df = m.loglik(failed_fit(names, "why"))
    assert math.isnan(value)
    assert df >= len(names)


def test_failed_leaf_makes_the_tree_loglik_nan() -> None:
    directory, name = "trees", "sim_gaussian_lmtree_onesplit"
    grown = copy.deepcopy(grow(directory, name))
    leaf = next(node for node in grown.nodes if node.terminal)
    leaf.fit = failed_fit(leaf.fit.coef_names, "why")
    m = model(fixture(directory, name))
    ll = tree_loglik(grown, m)  # type: ignore[arg-type]
    assert math.isnan(ll.value)
    assert ll.df == fixture(directory, name)["summary"]["loglik_df"]


# Reports.


@pytest.mark.parametrize(("directory", "name"), _trees_with("report"))
def test_report_keys_are_fixed_per_model(directory: str, name: str) -> None:
    m = model(fixture(directory, name))
    expected = REPORT_KEYS[type(m)]
    for node in grow(directory, name).nodes:
        report = m.report(node.fit)  # type: ignore[attr-defined]
        assert tuple(report) == expected
        for value in report.values():
            assert value is None or isinstance(value, float | int | str | bool)


@pytest.mark.parametrize("m", _models_with("report"))
def test_failed_fit_report_has_the_same_keys(m: Any) -> None:
    report = m.report(failed_fit(_failed_names(m), "why"))
    assert tuple(report) == REPORT_KEYS[type(m)]


# Prediction.


def _inverse_link(m: Any, eta: np.ndarray) -> np.ndarray:
    """The inverse links written out, with R's bounds (family.c and
    make.link("log"))."""
    if isinstance(m, LmModel):
        return eta
    family = "binomial" if isinstance(m, SafeLogitModel) else None
    if isinstance(m, GlmModel):
        family = m.family.family
    if family == "gaussian":
        return eta
    if family == "binomial":
        e = np.exp(np.clip(eta, -30, 30))
        e = np.where(eta < -30, DBL_EPSILON, np.where(eta > 30, 1 / DBL_EPSILON, e))
        return e / (1 + e)
    # poisson and negative binomial, log link
    return np.maximum(np.exp(eta), DBL_EPSILON)


@pytest.mark.parametrize(("directory", "name"), _trees_with("predict"))
def test_predict_matches_the_hand_computed_leaf_fit(directory: str, name: str) -> None:
    f = fixture(directory, name)
    m = model(f)
    y, x, _, off = tree_inputs(f)
    xm = np.ones((y.shape[0], 1)) if x is None else x
    for leaf in (node for node in grow(directory, name).nodes if node.terminal):
        rows = leaf.rows
        xr = xm[rows]
        offr = None if off is None else off[rows]
        coef = leaf.fit.coefficients
        keep = ~np.isnan(coef)
        eta = xr[:, keep] @ coef[keep] + (0 if offr is None else offr)
        link = m.predict(leaf.fit, xr, offr, "link")  # type: ignore[attr-defined]
        np.testing.assert_allclose(link, eta, rtol=PREDICT_RTOL, atol=0)
        response = m.predict(leaf.fit, xr, offr, "response")  # type: ignore[attr-defined]
        np.testing.assert_allclose(
            response, _inverse_link(m, eta), rtol=PREDICT_RTOL, atol=0
        )


def test_intercept_only_predict_needs_x_or_an_offset() -> None:
    fit = LmModel().fit(np.array([1.0, 2.0, 4.0]), None, weights=np.ones(3))
    with pytest.raises(ValueError, match="needs an offset"):
        LmModel().predict(fit, None, None, "response")
    got = LmModel().predict(fit, None, np.array([0.0, 1.0]), "link")
    np.testing.assert_allclose(got, [7 / 3, 10 / 3], rtol=PREDICT_RTOL)
    with pytest.raises(ValueError, match="not 'prob'"):
        LmModel().predict(fit, np.ones((2, 1)), None, "prob")
