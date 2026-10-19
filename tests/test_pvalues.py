"""P-value conversion against partykit, tests/fixtures/pvalues.json."""

import math
from typing import Any

import numpy as np
import pytest
from fixtures_io import load_json, nan_array

from modeltrees._hansen_table import MOB_BETA_SUPLM
from modeltrees.pvalues import (
    adjust_pvalues,
    log_chisq_sf,
    logp_chisq,
    logp_suplm,
    suplm_tau,
    suplm_trimming,
)

PVALUES = load_json("pvalues.json")
RTOL = 1e-10
ATOL = 1e-12


def _case_id(case: dict[str, Any]) -> str:
    return f"trim={case['trim']:.4g}-k={case['k']}"


def test_hansen_table_is_identical_to_partykit() -> None:
    expected = np.array(PVALUES["hansen_table"]["values"], dtype=np.float64)
    assert MOB_BETA_SUPLM.shape == (1000, 3)
    np.testing.assert_array_equal(MOB_BETA_SUPLM, expected)


@pytest.mark.parametrize("case", PVALUES["suplm"], ids=_case_id)
def test_suplm_trimming(case: dict[str, Any]) -> None:
    t = suplm_trimming(case["nobs"], case["trim"], case["minsize"])
    assert t.start == case["from"]
    assert t.end == case["to"]
    assert t.lam == pytest.approx(case["lambda"], rel=1e-15)
    assert suplm_tau(t.lam) == pytest.approx(case["tau"], rel=1e-15)


@pytest.mark.parametrize("case", PVALUES["suplm"], ids=_case_id)
def test_logp_suplm(case: dict[str, Any]) -> None:
    lam = suplm_trimming(case["nobs"], case["trim"], case["minsize"]).lam
    got = np.array([logp_suplm(s, case["k"], lam) for s in case["statistic"]])
    np.testing.assert_allclose(
        got, nan_array(case["log_p_value"]), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(
    "case", PVALUES["chisq"], ids=lambda c: f"k={c['k']}-levels={c['levels']}"
)
def test_logp_chisq(case: dict[str, Any]) -> None:
    assert case["df"] == case["k"] * (case["levels"] - 1)
    got = np.array([logp_chisq(s, case["df"]) for s in case["statistic"]])
    np.testing.assert_allclose(
        got, nan_array(case["log_p_value"]), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("case", PVALUES["bonferroni"])
def test_adjust_pvalues(case: dict[str, Any]) -> None:
    got = adjust_pvalues(nan_array(case["p_value_raw"]))
    np.testing.assert_allclose(
        got, nan_array(case["p_value_adjusted"]), rtol=RTOL, atol=0, equal_nan=True
    )


# Values below come from R 4.6.1: pchisq(x, df, lower.tail = FALSE, log.p = TRUE).
@pytest.mark.parametrize(
    ("x", "df", "expected"),
    [
        (300.0, 1.0, -153.08098859174726),
        (1500.0, 2.0, -750.0),
        (5000.0, 3.0, -2495.9669948169017),
        (1e-3, 5.0, -1.6814877203843298e-09),
        (0.0, 3.0, 0.0),
        (1e5, 40.0, -49833.763716718786),
        (2000.0, 1.2573241, -1002.9210149201368),
    ],
)
def test_log_chisq_sf_matches_r_in_the_far_tail(
    x: float, df: float, expected: float
) -> None:
    assert log_chisq_sf(x, df) == pytest.approx(expected, rel=RTOL, abs=ATOL)


class TestLogpSuplmEdgeCases:
    """Behaviour away from the fixture grid, derived from modelparty.R:268-303."""

    @pytest.mark.parametrize("k", [1, 2, 5, 20, 40])
    @pytest.mark.parametrize("trim", [0.005, 0.1, 0.3, 0.49, 0.495, 0.5])
    def test_tiny_statistic_gives_log_p_near_zero(self, k: int, trim: float) -> None:
        # Not clipped at 0: R returns values like 4e-16 from rounding near tau = 0.49.
        lam = suplm_trimming(1000, trim, 1).lam
        logp = logp_suplm(1e-12, k, lam)
        assert math.isfinite(logp)
        assert -1e-5 < logp <= 1e-15

    def test_zero_statistic(self) -> None:
        lam = suplm_trimming(1000, 0.1, 1).lam
        assert logp_suplm(0.0, 1, lam) == 0.0
        assert logp_suplm(0.0, 60, lam) == 0.0

    @pytest.mark.parametrize("k", [1, 2, 10, 40])
    @pytest.mark.parametrize("trim", [0.005, 0.1, 0.3, 0.495, 0.5])
    def test_huge_statistic_is_finite_and_tiny(self, k: int, trim: float) -> None:
        lam = suplm_trimming(1000, trim, 1).lam
        logp = logp_suplm(5000.0, k, lam)
        assert math.isfinite(logp)
        assert logp < -1000.0

    @pytest.mark.parametrize("k", [1, 3, 40])
    @pytest.mark.parametrize("trim", [0.1, 0.25, 0.4])
    def test_monotone_in_statistic(self, k: int, trim: float) -> None:
        lam = suplm_trimming(1000, trim, 1).lam
        stats = np.linspace(0.5, 400.0, 400)
        logp = np.array([logp_suplm(s, k, lam) for s in stats])
        assert np.all(np.diff(logp) <= 1e-12)

    def test_tau_boundary_001_uses_last_table_row(self) -> None:
        # tau <= 0.01 returns pp[25]; just above it interpolates rows 24 and 25.
        lam = 0.01
        assert suplm_tau(lam) == 0.01
        beta = MOB_BETA_SUPLM[24]
        x = 30.0
        expected = log_chisq_sf(max(beta[0] + beta[1] * x, 0.0), beta[2])
        assert logp_suplm(x, 1, lam) == expected
        above = logp_suplm(x, 1, 0.0101)
        assert above == pytest.approx(expected, abs=0.05)

    def test_tau_boundary_049_and_05(self) -> None:
        x, k = 12.0, 3
        chisq = log_chisq_sf(x, k)
        # tau = 0.5 exactly (lambda = 1) is the plain chi-squared tail.
        assert logp_suplm(x, k, 1.0) == chisq
        # tau = 0.49 is the first table row: the chi-squared weight is zero.
        beta = MOB_BETA_SUPLM[(k - 1) * 25]
        row1 = log_chisq_sf(max(beta[0] + beta[1] * x, 0.0), beta[2])
        assert logp_suplm(x, k, 0.49) == pytest.approx(row1, rel=1e-12)
        # Between 0.49 and 0.5 the result moves from row 1 toward chi-squared.
        mid = logp_suplm(x, k, 0.495)
        assert min(row1, chisq) <= mid <= max(row1, chisq)

    def test_interpolation_hits_table_rows_exactly(self) -> None:
        # tau = 0.51 - j / 50 lands on row j (1-based), weight 1.
        k, x = 4, 20.0
        for j in (2, 10, 24):
            tau = 0.51 - j / 50
            beta = MOB_BETA_SUPLM[(k - 1) * 25 + j - 1]
            row = log_chisq_sf(max(beta[0] + beta[1] * x, 0.0), beta[2])
            assert logp_suplm(x, k, tau) == pytest.approx(row, rel=1e-9)

    def test_one_parameter_large_lambda(self) -> None:
        # lambda > 1 maps to tau = 1 / (1 + sqrt(lambda)); trim 0.1 gives 81.
        lam = suplm_trimming(1000, 0.1, 1).lam
        assert lam == 81.0
        assert suplm_tau(lam) == 0.1

    def test_k40_to_k41_switches_to_estrella(self) -> None:
        lam = suplm_trimming(1000, 0.1, 1).lam
        x = 100.0
        estrella = (
            -math.lgamma(41 / 2)
            + 41 / 2 * math.log(x / 2)
            - x / 2
            + math.log(abs(math.log(lam) * (1 - 41 / x) + 2 / x))
        )
        assert logp_suplm(x, 41, lam) == pytest.approx(estrella, rel=1e-14)
        assert logp_suplm(x, 40, lam) != pytest.approx(estrella, rel=1e-3)

    def test_many_parameters_power_interpolation(self) -> None:
        # Below 1.5 * k, (x / (1.5 k))^sqrt(k) * estrella(1.5 k).
        k = 100
        lam = suplm_trimming(1000, 0.1, 1).lam
        at_knot = logp_suplm(1.5 * k, k, lam)
        assert logp_suplm(75.0, k, lam) == pytest.approx(
            0.5 ** math.sqrt(k) * at_knot, rel=1e-14
        )
        assert logp_suplm(0.0, k, lam) == 0.0

    def test_zero_trimming_gives_infinite_lambda(self) -> None:
        # from = 0 makes lambda = Inf and tau = 0, so the last table row is used.
        t = suplm_trimming(10, 0.0, 0)
        assert t.lam == math.inf
        assert suplm_tau(t.lam) == 0.0
        beta = MOB_BETA_SUPLM[2 * 25 - 1]
        expected = log_chisq_sf(max(beta[0] + beta[1] * 3.0, 0.0), beta[2])
        assert logp_suplm(3.0, 2, t.lam) == expected

    def test_nan_lambda_raises_like_r(self) -> None:
        # R stops at `if(lambda < 1)` with a missing value.
        with pytest.raises(ValueError):
            logp_suplm(3.0, 2, math.nan)

    def test_absolute_trim(self) -> None:
        t = suplm_trimming(200, 15, 1)
        assert (t.start, t.end) == (15.0, 185.0)
        t = suplm_trimming(200, 0.1, 30)
        assert (t.start, t.end) == (30.0, 170.0)


class TestAdjustPvaluesEdgeCases:
    def test_threshold_is_strict(self) -> None:
        # p = 0.001 is not above the threshold, so it is Bonferroni-adjusted.
        got = adjust_pvalues([0.001, 0.5])
        assert got[0] == 0.002
        assert got[1] == 1 - 0.5**2

    def test_all_nan(self) -> None:
        got = adjust_pvalues([np.nan, np.nan])
        assert np.all(np.isnan(got))

    def test_caps_at_one(self) -> None:
        got = adjust_pvalues([0.0009] * 2000)
        assert np.all(got == 1.0)
