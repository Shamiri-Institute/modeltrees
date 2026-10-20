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
from modeltrees.node_model import FitStatus, failed_fit

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


def test_every_model_implements_the_protocols() -> None:
    for d, n in ALL_TREES:
        m = model(fixture(d, n))
        for method in ("loglik", "report", "predict"):
            assert callable(getattr(m, method, None)), (type(m).__name__, method)


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


def test_separated_leaf_loglik() -> None:
    """The completely separated safe logit leaf counts its two coefficients
    and its objective, though it has no scores."""
    grown = grow(*SEPARATED)
    m = model(fixture(*SEPARATED))
    assert isinstance(m, SafeLogitModel)
    separated = [n for n in grown.nodes if n.fit.status is FitStatus.NO_ESTFUN]
    assert any(n.fit.extras["separation"] == "complete" for n in separated)
    for node in separated:
        value, df = m.loglik(node.fit)
        assert value == -node.fit.objfun
        assert df == 2


def test_the_special_leaves_are_in_the_loglik_comparison() -> None:
    compared = {tuple(p.values) for p in _trees_with("loglik", oracle=True)}
    assert DEVIANCE in compared and SEPARATED in compared


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
        if isinstance(m, MultinomModel):
            _check_multinom(m, leaf.fit, xr)
            continue
        keep = ~np.isnan(coef)
        eta = xr[:, keep] @ coef[keep] + (0 if offr is None else offr)
        link = m.predict(leaf.fit, xr, offr, "link")  # type: ignore[attr-defined]
        np.testing.assert_allclose(link, eta, rtol=PREDICT_RTOL, atol=0)
        response = m.predict(leaf.fit, xr, offr, "response")  # type: ignore[attr-defined]
        np.testing.assert_allclose(
            response, _inverse_link(m, eta), rtol=PREDICT_RTOL, atol=0
        )


def _check_multinom(m: MultinomModel, fit: Any, xr: np.ndarray) -> None:
    if fit.extras["status"] == "reference_absent":
        # No coefficients without the reference class, so no prediction
        # (docs/api-port.md 3.3, item 5).
        for kind in ("link", "prob", "class"):
            assert np.all(np.isnan(m.predict(fit, xr, None, kind))), kind
        return
    n_classes = len(m.levels)
    p = xr.shape[1]
    beta = fit.coefficients.reshape(n_classes - 1, p)
    absent = np.all(np.isnan(beta), axis=1)
    eta = np.zeros((xr.shape[0], n_classes))
    eta[:, 1:] = xr @ np.where(np.isnan(beta), 0, beta).T
    expo = np.exp(eta)
    expo[:, 1:][:, absent] = 0.0
    prob = expo / expo.sum(axis=1, keepdims=True)
    link = m.predict(fit, xr, None, "link")
    expected_link = eta[:, 1:].copy()
    expected_link[:, absent] = np.nan
    np.testing.assert_allclose(link, expected_link, rtol=PREDICT_RTOL, atol=0)
    got = m.predict(fit, xr, None, "prob")
    np.testing.assert_allclose(got, prob, rtol=PREDICT_RTOL, atol=0)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-14)
    cls = m.predict(fit, xr, None, "class")
    np.testing.assert_array_equal(cls, np.argmax(prob, axis=1).astype(np.float64))
    with pytest.raises(ValueError, match="'prob'.*'class'"):
        m.predict(fit, xr, None, "response")


def test_intercept_only_predict_needs_x_or_an_offset() -> None:
    fit = LmModel().fit(np.array([1.0, 2.0, 4.0]), None, weights=np.ones(3))
    with pytest.raises(ValueError, match="needs an offset"):
        LmModel().predict(fit, None, None, "response")
    got = LmModel().predict(fit, None, np.array([0.0, 1.0]), "link")
    np.testing.assert_allclose(got, [7 / 3, 10 / 3], rtol=PREDICT_RTOL)
    with pytest.raises(ValueError, match="not 'prob'"):
        LmModel().predict(fit, np.ones((2, 1)), None, "prob")


# Degenerate multinomial fits (docs/api-port.md, section 3.3).

LEVELS = ["lo", "mid", "hi"]
X3 = np.column_stack([np.ones(6), np.linspace(-1, 1, 6)])


def _multinom_fit(y: list[int]) -> Any:
    return MultinomModel(LEVELS).fit(np.array(y), X3, weights=np.ones(6))


def test_multinom_absent_class() -> None:
    m = MultinomModel(LEVELS)
    fit = _multinom_fit([0, 1, 0, 1, 1, 0])  # "hi" absent
    prob = m.predict(fit, X3, None, "prob")
    assert np.all(prob[:, 2] == 0)
    np.testing.assert_allclose(prob.sum(axis=1), 1.0, rtol=1e-14)
    link = m.predict(fit, X3, None, "link")
    assert np.all(np.isnan(link[:, 1])) and np.all(np.isfinite(link[:, 0]))
    value, df = m.loglik(fit)
    assert value == -fit.objfun and df == 2
    assert m.report(fit) == {
        "fit_status": "ok",
        "classes_present": "lo, mid",
        "converged": True,
    }


def test_multinom_single_class() -> None:
    m = MultinomModel(LEVELS)
    fit = _multinom_fit([1] * 6)
    assert fit.status is FitStatus.NO_ESTFUN
    np.testing.assert_array_equal(m.predict(fit, X3, None, "prob"), [[0, 1, 0]] * 6)
    np.testing.assert_array_equal(m.predict(fit, X3, None, "class"), [1.0] * 6)
    assert np.all(np.isnan(m.predict(fit, X3, None, "link")))
    assert m.loglik(fit) == (0.0, 0)


def test_multinom_reference_absent() -> None:
    m = MultinomModel(LEVELS)
    fit = _multinom_fit([1, 2, 1, 2, 2, 1])
    assert fit.extras["status"] == "reference_absent"
    for kind in ("link", "prob", "class"):
        assert np.all(np.isnan(m.predict(fit, X3, None, kind)))
    value, df = m.loglik(fit)
    assert value == -fit.objfun and df == 2


def test_multinom_failed_fit_predicts_nan() -> None:
    m = MultinomModel(LEVELS)
    fit = failed_fit(m.names(["(Intercept)", "x"]), "why")
    assert np.all(np.isnan(m.predict(fit, X3, None, "prob")))
    assert np.all(np.isnan(m.predict(fit, X3, None, "class")))
    with pytest.raises(ValueError, match="offset"):
        m.predict(fit, X3, np.zeros(6), "prob")
