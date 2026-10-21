"""The fit failure protocol (docs/model-interface.md, section 4.2).

A toy model wraps LmModel and fails, or returns no scores, on chosen subsets
of the rows. The rows are identified through the regressor column, whose
values are unique, so the model knows each subset's values of the split
variable z1. There is no R oracle: R's fit functions stop mob() instead.
"""

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from modeltrees import reference_path
from modeltrees.instability import VariableKind
from modeltrees.models.lm import LmModel
from modeltrees.node_model import (
    FitResult,
    FitStatus,
    extra_df,
    failed_fit,
    uses_batch_split,
)
from modeltrees.split_search import find_split, find_split_reference
from modeltrees.tree import (
    MobControl,
    NodeFitFailedWarning,
    StopReason,
    mob,
)

N = 300
# Decides from a subset's z1 values (and whether scores were asked for)
# what the fit returns.
Rule = Callable[[np.ndarray, bool], FitStatus]


def _data(seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z1 = rng.uniform(size=N)
    z2 = rng.uniform(size=N)
    x1 = rng.standard_normal(N)
    slope = np.where(z1 <= 0.5, 1.0, -1.0)
    y = 0.5 + slope * x1 + rng.standard_normal(N) * 0.5
    return {"z1": z1, "z2": z2, "x1": x1, "y": y}


class ToyModel(LmModel):
    """LmModel that fails, or has no scores, where ``rule`` says so."""

    def __init__(
        self,
        d: dict[str, np.ndarray],
        rule: Rule,
        *,
        batch_split: bool = True,
        extra_df: int = 0,
    ) -> None:
        super().__init__()
        self.z_of_x = dict(zip(d["x1"].tolist(), d["z1"].tolist(), strict=True))
        self.rule = rule
        self.batch_split = batch_split
        self.extra_df = extra_df
        self.starts: list[np.ndarray | None] = []

    def fit(self, y: Any, x: Any, **kw: Any) -> FitResult:
        self.starts.append(kw.get("start"))
        z = np.array([self.z_of_x[v] for v in x[:, 1].tolist()])
        status = self.rule(z, kw["estfun"])
        result = super().fit(y, x, **kw)
        if status is FitStatus.FAILED:
            return failed_fit(result.coef_names, "toy failure", rows=z.shape[0])
        if status is FitStatus.NO_ESTFUN:
            return FitResult(
                result.coefficients,
                result.coef_names,
                result.objfun,
                None,
                status=FitStatus.NO_ESTFUN,
                reason="toy scores",
            )
        return result


def _x(d: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.ones(N), d["x1"]])


def _ok(z: np.ndarray, estfun: bool) -> FitStatus:
    return FitStatus.OK


def _search(
    model: LmModel, d: dict[str, np.ndarray], *, reference: bool, **kw: Any
) -> Any:
    x = _x(d)
    w = np.ones(N)

    def side(mask: np.ndarray, start: np.ndarray | None) -> FitResult:
        return model.fit(
            d["y"][mask],
            x[mask],
            start=start,
            weights=w[mask],
            offset=None,
            cluster=None,
            caseweights=True,
            estfun=False,
            vcov=False,
        )

    batch = None
    if not reference and uses_batch_split(model):

        def batch(order: np.ndarray, cuts: np.ndarray) -> Any:
            return model.split_objectives(
                d["y"],
                x,
                weights=w,
                offset=None,
                cluster=None,
                caseweights=True,
                order=order,
                cuts=cuts,
            )

    args = (0, VariableKind.NUMERIC, d["z1"], None, w, side)
    if reference:
        return find_split_reference(*args, minsize=20, **kw)
    return find_split(*args, minsize=20, batch=batch, **kw)


def _right_side_above(bound: float) -> Rule:
    """Fails the right side of every candidate at or above ``bound``."""

    def rule(z: np.ndarray, estfun: bool) -> FitStatus:
        if not estfun and z.shape[0] < N and z.min() > bound:
            return FitStatus.FAILED
        return FitStatus.OK

    return rule


def _left_side_ending_in(lo: float, hi: float) -> Rule:
    """Fails the left side of every candidate with a break in (lo, hi)."""

    def rule(z: np.ndarray, estfun: bool) -> FitStatus:
        if not estfun and z.shape[0] < N and z.min() == ZMIN and lo < z.max() < hi:
            return FitStatus.FAILED
        return FitStatus.OK

    return rule


ZMIN = float(_data()["z1"].min())


# Split search.


def test_failed_fit_result() -> None:
    r = failed_fit(("a", "b"), "why", theta=1.0)
    assert r.failed and r.status is FitStatus.FAILED and r.reason == "why"
    assert math.isinf(r.objfun) and r.estfun is None and not r.converged
    assert np.all(np.isnan(r.coefficients)) and r.extras == {"theta": 1.0}
    ok = FitResult(np.zeros(1), ("a",), 0.0, None)
    assert ok.status is FitStatus.OK and not ok.failed and ok.extras == {}


@pytest.mark.parametrize("reference", [True, False])
def test_failed_candidates_are_inadmissible_and_counted(reference: bool) -> None:
    d = _data()
    plain = _search(LmModel(), d, reference=True)
    toy = ToyModel(d, _right_side_above(0.7), batch_split=False)
    got = _search(toy, d, reference=reference)
    failing = np.isfinite(plain.objective) & (plain.values >= 0.7)
    # The last candidates below 0.7 have right sides that start above it.
    failing |= np.isfinite(plain.objective) & np.array(
        [d["z1"][d["z1"] > v].min(initial=1.0) > 0.7 for v in plain.values]
    )
    assert failing.sum() > 10
    assert got.n_failed == int(failing.sum())
    assert np.all(np.isinf(got.objective[failing]))
    keep = ~failing
    np.testing.assert_allclose(
        got.objective[keep], plain.objective[keep], rtol=1e-10, atol=1e-10
    )
    assert got.split == plain.split  # the best candidate does not fail


@pytest.mark.parametrize("batch_split", [True, False])
@pytest.mark.parametrize("reference", [True, False])
def test_failing_best_candidate_moves_the_split(
    reference: bool, batch_split: bool
) -> None:
    d = _data()
    plain = _search(LmModel(), d, reference=True)
    assert plain.split is not None and 0.45 < plain.split.breaks < 0.55
    toy = ToyModel(d, _left_side_ending_in(0.45, 0.55), batch_split=batch_split)
    got = _search(toy, d, reference=reference)
    expected = plain.objective.copy()
    band = (plain.values > 0.45) & (plain.values < 0.55) & np.isfinite(expected)
    expected[band] = np.inf
    assert got.n_failed == int(band.sum())
    assert got.best == int(np.argmin(expected))
    assert got.split is not None and got.split.breaks == plain.values[got.best]
    # With the batch scorer the failed refits force exact fits of every
    # candidate, so all objectives are exact.
    np.testing.assert_allclose(got.objective, expected, rtol=1e-10, atol=1e-10)


def test_every_candidate_failing_gives_no_split() -> None:
    d = _data()

    def rule(z: np.ndarray, estfun: bool) -> FitStatus:
        return FitStatus.OK if estfun or z.shape[0] == N else FitStatus.FAILED

    toy = ToyModel(d, rule, batch_split=False)
    got = _search(toy, d, reference=False)
    assert got.split is None and got.best is None
    assert got.n_failed == int(
        np.sum(np.isfinite(_search(LmModel(), d, reference=True).objective))
    )


def test_restart_false_skips_the_starts_of_failed_candidates() -> None:
    d = _data()
    toy = ToyModel(d, _left_side_ending_in(0.45, 0.55), batch_split=False)
    got = _search(toy, d, reference=True, restart=False)
    assert got.n_failed > 0
    # Each start is the coefficients of the previous successful candidate,
    # never the NaN coefficients of a failed one.
    starts = [s for s in toy.starts if s is not None]
    assert starts and all(np.all(np.isfinite(s)) for s in starts)
    toy_fast = ToyModel(d, _left_side_ending_in(0.45, 0.55), batch_split=False)
    fast = _search(toy_fast, d, reference=False, restart=False)
    np.testing.assert_array_equal(fast.objective, got.objective)
    assert fast.n_failed == got.n_failed


def test_batch_split_opt_out() -> None:
    d = _data()
    assert uses_batch_split(LmModel())
    assert uses_batch_split(ToyModel(d, _ok))
    assert not uses_batch_split(ToyModel(d, _ok, batch_split=False))
    assert not uses_batch_split(object())


# Trees.


def _grow(model: LmModel, d: dict[str, np.ndarray], **control: Any) -> Any:
    return mob(
        model,
        d["y"],
        _x(d),
        {"z1": ("numeric", d["z1"]), "z2": ("numeric", d["z2"])},
        control=MobControl(minsize=20, **control),
    )


def test_tree_with_failed_candidates_matches_reference() -> None:
    d = _data()
    rule = _left_side_ending_in(0.45, 0.55)
    got = _grow(ToyModel(d, rule), d)
    with reference_path():
        ref = _grow(ToyModel(d, rule), d)
    assert [n.id for n in got.nodes] == [n.id for n in ref.nodes]
    root, root_ref = got.root, ref.root
    assert root.split_search is not None and root_ref.split_search is not None
    assert root.split_search.n_failed == root_ref.split_search.n_failed > 0
    assert root.split == root_ref.split
    assert root.split is not None and not 0.45 < root.split.breaks < 0.55
    np.testing.assert_array_equal(got.fitted_node, ref.fitted_node)


def test_failed_node_fit_is_terminal_with_a_warning() -> None:
    d = _data()

    def rule(z: np.ndarray, estfun: bool) -> FitStatus:
        # Only the node fit (estfun = True) of the right kid fails.
        return FitStatus.FAILED if estfun and z.min() > 0.4 else FitStatus.OK

    plain = _grow(LmModel(), d)
    assert plain.root.split is not None
    with pytest.warns(NodeFitFailedWarning, match="node 3: the model fit failed"):
        got = _grow(ToyModel(d, rule), d)
    right = got.root.kids[1]
    assert right.terminal and right.stop_reason is StopReason.FIT_FAILED
    assert right.fit.failed and right.fit.reason == "toy failure"
    assert right.test is None and not right.test_failed
    assert right.fit.extras == {"rows": right.rows.shape[0]}
    # The rest of the tree is unchanged.
    assert got.root.split == plain.root.split
    np.testing.assert_array_equal(got.root.kids[0].rows, plain.root.kids[0].rows)


def test_failed_root_fit() -> None:
    d = _data()

    def rule(z: np.ndarray, estfun: bool) -> FitStatus:
        return FitStatus.FAILED

    with pytest.warns(NodeFitFailedWarning, match="node 1"):
        got = _grow(ToyModel(d, rule), d)
    assert len(got.nodes) == 1
    assert got.root.stop_reason is StopReason.FIT_FAILED
    assert np.all(got.fitted_node == 1)


def test_no_estfun_node_is_terminal_without_tests() -> None:
    d = _data()

    def rule(z: np.ndarray, estfun: bool) -> FitStatus:
        return FitStatus.NO_ESTFUN if estfun and z.min() > 0.4 else FitStatus.OK

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning, as in partykit
        got = _grow(ToyModel(d, rule), d)
    right = got.root.kids[1]
    assert right.terminal and right.stop_reason is StopReason.NO_ESTFUN
    # partykit stores test = NA, as for a failed test.
    assert right.test is None and right.test_failed and right.p_value is None
    assert right.fit.status is FitStatus.NO_ESTFUN


def test_no_estfun_after_the_stopping_rules() -> None:
    # partykit checks minsize and maxdepth before the scores (:536-550).
    d = _data()

    def rule(z: np.ndarray, estfun: bool) -> FitStatus:
        return FitStatus.NO_ESTFUN if estfun else FitStatus.OK

    got = _grow(ToyModel(d, rule), d, maxdepth=1)
    assert got.root.stop_reason is StopReason.MAXDEPTH and not got.root.test_failed


def test_no_estfun_candidates_stay_admissible() -> None:
    d = _data()

    def rule(z: np.ndarray, estfun: bool) -> FitStatus:
        return FitStatus.NO_ESTFUN if z.shape[0] < N else FitStatus.OK

    plain = _search(LmModel(), d, reference=True)
    got = _search(ToyModel(d, rule, batch_split=False), d, reference=False)
    assert got.n_failed == 0
    np.testing.assert_allclose(got.objective, plain.objective, rtol=1e-10)


# Pruning df.


def test_extra_df_does_not_count_in_pruning() -> None:
    """Pruning counts coefficients only, as partykit's mob_prune
    (modelparty.R:646-647); extra_df counts in the log-likelihood df."""
    d = _data()
    assert extra_df(LmModel()) == 0
    assert extra_df(ToyModel(d, _ok, extra_df=1)) == 1
    seen: list[tuple[int, int]] = []

    def rule(
        objfun: tuple[float, float], df: tuple[int, int], nobs: tuple[float, float]
    ) -> bool:
        seen.append(df)
        return False

    for extra in (0, 1, 3):
        seen.clear()
        _grow(ToyModel(d, _ok, extra_df=extra), d, prune=rule)
        assert seen and all(df == (2, 5) for df in seen)


def test_extra_df_does_not_change_the_aic_decision() -> None:
    d = _data()
    kept = _grow(ToyModel(d, _ok), d, prune="aic")
    assert len(kept.nodes) > 1
    # An extra_df large enough to prune the root split if it counted (the
    # gain in lmtree's AIC, the toy model has LmModel's prune rules) leaves
    # the tree unchanged.
    root, kids = kept.root.fit.objfun, sum(k.fit.objfun for k in kept.root.kids)
    gain = kept.root.nobs * (math.log(root) - math.log(kids))
    extra = math.ceil(gain / 2)
    same = _grow(ToyModel(d, _ok, extra_df=extra), d, prune="aic")
    assert len(same.nodes) == len(kept.nodes)
    assert same.root.stop_reason is kept.root.stop_reason
