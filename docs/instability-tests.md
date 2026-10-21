# Parameter instability tests

This document maps the parameter instability tests and p-values that
partykit 1.3-0 runs inside `mob()` to the Python port in
`src/modeltrees/instability.py` and `src/modeltrees/pvalues.py`. It covers
phase 2 and its cluster extension. No tree-growing code is involved.

Line numbers are `file:line` in the pinned sources under `reference/`.
`modelparty.R` means `reference/partykit/R/modelparty.R`.

## 1. The R code involved

All the code is in partykit. It sits in closures inside `mob_partynode()`, so
it is not reachable from outside the package.

| Piece | Location | Ported as |
| --- | --- | --- |
| `minsize` coerced to integer | `modelparty.R:167-168` | caller passes `minsize: int` |
| `root.matrix()`, "from strucchange" | `modelparty.R:194-203` | `instability.root_matrix` |
| `mob_grow_fluctests()` | `modelparty.R:209-381` | `instability.fluctuation_tests` |
| variables to test, `mtry` | `modelparty.R:218` | all variables (default `mtry = Inf`) |
| drop zero-weight rows, `nobs` | `modelparty.R:220-227` | `instability.decorrelated_process` (rows dropped consistently, see section 4, item 6) |
| scale by `sqrt(nobs)` | `modelparty.R:230` | same |
| meat, `vcov = "opg"`, with `cluster` sums | `modelparty.R:231-246` | same (cluster branch `:243-244`, section 4, item 14) |
| `J12 = root.matrix(chol2inv(chol(meat)))`, decorrelate | `modelparty.R:247-252` | same |
| `parm` column selection | `modelparty.R:254-259` | not ported (default `NULL`) |
| trimming `from`, `to`, `lambda` | `modelparty.R:261-265` | `pvalues.suplm_trimming` |
| `logp.supLM()` | `modelparty.R:267-303` | `pvalues.logp_suplm` |
| Estrella (2003) branch, `k > 40` | `modelparty.R:270-277` | same |
| Hansen (1997) branch, `k <= 40` | `modelparty.R:278-301` | same |
| Hansen table `mob_beta_suplm` | `reference/partykit/R/mob-pvalue.R:1-674` | `_hansen_table.MOB_BETA_SUPLM` |
| skip variables with < 2 distinct values | `modelparty.R:308` | same |
| factor test, segment weights | `modelparty.R:309-326` | `instability._chisq_statistic` |
| ordered factor switch `iord` | `modelparty.R:313` | ordered = nominal under `"chisq"` |
| ordered `"max"` test (mvtnorm) | `modelparty.R:331-338` | not ported |
| ordered `"L2"` test (strucchange) | `modelparty.R:339-343` | not ported |
| chi-squared statistic and p-value | `modelparty.R:344-346` | `_chisq_statistic`, `pvalues.logp_chisq` |
| numeric ordering, `breakties` | `modelparty.R:349-355` | stable sort (default `breakties = FALSE`) |
| supLM statistic | `modelparty.R:356-367` | `instability._suplm_statistic` |
| supLM p-value call | `modelparty.R:368` | `instability.statistics_to_log_pvalues` |
| `best = which.min(pval)`, `exp(pval)` | `modelparty.R:372-380` | `instability.best_variable` |
| call under `try()`, failure handling | `modelparty.R:550`, `:569-573` | raises `InstabilityTestError` |
| Bonferroni/Sidak adjustment | `modelparty.R:552-557` | `pvalues.adjust_pvalues` |
| alpha decision, `p.value` | `modelparty.R:559-561` | tree code, not in this phase |
| stored `test` matrix (adjusted p) | `modelparty.R:576` | `FluctuationTests.p_value_adjusted` |
| `mob_control()` defaults | `modelparty.R:726-756` | keyword defaults `trim = 0.1`, `bonferroni = True` |

### 1.1 What the code computes

**Scores to process.** Rows with weight 0 are dropped (`:222-224`).
`nobs = sum(weights)` if any weight differs from 1, else the number of rows
left (`:227`). The process is `estfun / sqrt(nobs)` (`:230`). With case
weights the weights are already inside the scores, so the meat is
`crossprod(process / sqrt(weights))` (`:241`). The process is decorrelated by
the symmetric root of the inverse meat: `process %*% t(J12)` (`:247-252`).

**Numeric variables, supLM.** Rows are ordered by the variable with R's
stable `order()` (`:354`). The process is cumulated by column. The time index
`tt0` is `1:n`, or the cumulative weights when any weight differs from 1
(`:358`). The statistic is the maximum over `from <= tt0 <= to` of
`rowSums(cumproc^2) / (t * (1 - t))`, `t = tt0 / nobs` (`:359-367`). With no
index in the window the statistic is 0 and the p-value `NA` (`:360-368`).

**Trimming.** `from = trim` if `trim > 1`, otherwise `ceiling(nobs * trim)`.
Then `from = max(from, minsize)`, `to = nobs - from` and
`lambda = ((nobs - from) * to) / (from * (nobs - to))` (`:262-265`). Here
`minsize` is `mob_partynode`'s variable: the user's value coerced with
`as.integer` (`:168`), or the root default from `mob_grow` (`:533`).

**supLM p-value, `k <= 40`.** `tau = lambda` if `lambda < 1`, else
`1 / (1 + sqrt(lambda))` (`:281`). The 25 table rows for `k` give
`pp[j] = log P(chi2(df_j) > max(0, b0_j + b1_j * x))` (`:282-285`). Then:

- `tau == 0.5`: the chi-squared(k) log tail of `x` (`:286-287`);
- `tau <= 0.01`: `pp[25]` (`:288-289`);
- `0.49 <= tau < 0.5`: `log((exp(log(0.5 - tau) + pp[1]) +
  exp(log(tau - 0.49) + chisq)) * 100)`, falling back to the plain mean of
  the two log values when that is not finite (`:290-293`);
- otherwise linear interpolation between `pp[floor(taua)]` and
  `pp[floor(taua) + 1]` with `taua = (0.51 - tau) * 50`, computed in the
  same naive way with the same fallback (`:294-299`).

**supLM p-value, `k > 40`.** Estrella (2003):
`-lgamma(k/2) + k/2 log(x/2) - x/2 + log|log(lambda)(1 - k/x) + 2/x|`. For
`x <= 1.5 k` partykit uses `(x / (1.5 k))^sqrt(k)` times the value at
`1.5 k` (`:270-277`).

**Factors.** Rows are ordered by the factor, unused levels dropped, and the
segment weights are the per-level weight sums divided by `nobs`
(`:310-321`). The statistic is `sum_j sum_levels (sum of process_j in
level)^2 / segweight`, and the p-value is the chi-squared tail with
`k * (levels - 1)` df (`:345-346`). Ordered factors take this path under the
default `ordinal = "chisq"` (`:313`).

**Adjustment.** With `m` the number of non-`NA` raw p-values,
`p_adj = 1 - (1 - p)^m` for `p > 0.001` and `min(1, m p)` otherwise
(`:553-557`). `best` is chosen from the raw log p-values before adjustment,
`NA` ignored, ties to the first (`:373`).

## 2. Is strucchange called?

Only under one non-default option. The full list of strucchange and mvtnorm
uses in partykit 1.3-0 `R/` and `NAMESPACE`:

| Use | Where | When |
| --- | --- | --- |
| `strucchange::ordL2BB(...)$computePval` | `modelparty.R:342` | ordered factor and `mob_control(ordinal = "L2")` |
| `mvtnorm::pmvnorm` | `modelparty.R:333-338` | ordered factor and `mob_control(ordinal = "max")` |
| `importFrom("strucchange", "sctest")` | `NAMESPACE:18-20` | only to register `sctest.modelparty` (`modelparty.R:943-956`), which returns the stored `info$test`. Not called during growth. |
| `root.matrix` | `modelparty.R:194-203` | a local copy of strucchange code; strucchange itself is not called |

`ordL2BB` is defined in `reference/strucchange/R/zzz.R:265`. It simulates
Brownian bridges with R's RNG, `nrep = 10000` by default, so it cannot be
matched exactly from Python.

No other `mob_control` option reaches strucchange. In particular:

- numeric partitioning variables always use partykit's own `logp.supLM` and
  `mob_beta_suplm` table, under every option;
- ordered factors with the default `ordinal = "chisq"` use `pchisq`;
- `vcov`, `trim`, `breakties`, `parm`, `caseweights`, `mtry` and `cluster`
  change the inputs to these computations, not the p-value code.

`mob_control()` lowercases `ordinal` before `match.arg` (`:756`), so `"L2"`
is stored as `"l2"`; `mob_grow_fluctests` then tests `ordinal == "max"` and
sends every other non-`"chisq"` value to `ordL2BB` (`:331`, `:339-342`).

## 3. Scope

Ported: everything the default `mob_control()` needs, that is `vcov = "opg"`,
`caseweights = TRUE`, `ordinal = "chisq"`, `trim` (fraction or absolute),
`minsize`, `bonferroni` on or off, `mtry = Inf`, `breakties = FALSE`,
`parm = NULL`, and `cluster` (section 4, item 14). `caseweights = FALSE`
was added with the tree code (docs/tree-growing.md): `nobs` is the row count,
the meat does not divide by the weights, the supLM time index is `1:n` and the
factor segment weights are level counts (`:227`, `:241`, `:320`, `:358`).
There is no R fixture for it.

Out of scope because they depend on strucchange or on random numbers:

- `ordinal = "L2"` (strucchange `ordL2BB`, simulated);
- `ordinal = "max"` (mvtnorm `pmvnorm`, randomized quasi-Monte Carlo);
- `breakties = TRUE` (`runif` jitter, `:349-352`);
- `mtry < Inf` (`sample`, `:218`).

Out of scope for now because default `mob()` does not need them, though they
do not depend on strucchange: `vcov = "info"` and `"sandwich"` (`:233-251`)
and `parm` (`:254-259`). `fluctuation_tests` takes keyword arguments so these
can be added without changing existing calls.

## 4. Decisions and ambiguities

1. **The Hansen table is R's in-memory value, not the source text.** R's
   parser is not correctly rounded. It reads the literal `-8.56206391`
   (R row 808, column 1) as `-8.5620639099999991117`, one ulp below the
   double Python's `float()` gives. This was confirmed in R 4.6.1:
   `partykit:::mob_beta_suplm[808, 1] == -8.56206391` is `TRUE` and prints as
   `-8.5620639099999991117`. `_hansen_table.py` was generated from the source
   text by a one-off script, with this single entry replaced by
   `-8.562063909999999`. A test checks that all 3000 entries equal the
   fixture copy exactly.
2. **Chi-squared log tails.** R's `pchisq(..., log.p = TRUE)` stays finite far
   past double underflow (for example -750 at x = 1500, df = 2), but scipy's
   `chi2.logsf` returns `-inf` there. `pvalues.log_chisq_sf` uses
   `log1p(-gammainc)` when the lower tail is below 0.5, `log(gammaincc)` while
   the upper tail is above 1e-280, and a log-space continued fraction
   (modified Lentz) beyond that. The p-value fixture reaches log p of about
   -230; the far tail is tested against values printed from R.
3. **The naive weighted averages are kept.** partykit forms
   `log(exp(a) + exp(b))` directly and falls back to a plain mean of the log
   values when it underflows. A log-sum-exp would give different, though
   better, answers in that region, so the port computes it the R way.
4. **No clipping.** `logp_suplm` can return slightly positive values (about
   4e-16 near `tau = 0.49`), and Estrella's approximation is not bounded by
   0. R does not clip, so neither does the port.
5. **NaN lambda raises.** `lambda` is `NaN` only when `nobs = 0`. R fails at
   `if(lambda < 1)`; `logp_suplm` raises `ValueError`. `trim = 0` with
   `minsize = 0` gives `lambda = Inf` and `tau = 0`, which R and the port both
   map to the last table row.
6. **Zero case weights: the port diverges from partykit, by decision.**
   partykit 1.3-0 drops zero-weight rows from the process but divides by the
   unfiltered weights at `:241`. R then recycles the weight vector over the
   column-major matrix (with a warning), or errors if the vector is longer
   than the matrix. When the vector is recycled every weight is used at
   least once, so a zero weight puts `Inf` or `NaN` into the meat and `chol()`
   fails. With `caseweights = TRUE`, any zero weight therefore makes the
   tests fail and the node terminal. The fixture
   `sim_gaussian_lmtree_zero_weights.json` (5(a)) shows this ("the leading
   minor of order 2 is not positive"). docs/decisions.md says not to
   replicate it. The port drops zero-weight rows from the scores, weights and
   partitioning variables before any computation (`decorrelated_process`,
   `fluctuation_statistics`), so `nobs`, the meat, the supLM cumulative
   weights `tt0` and the factor segment weights (`:320`, `:358`) all see the
   positive-weight rows only. A node with zero weights then tests like the
   data with those rows deleted. The oracle is
   `sim_gaussian_lmtree_zero_weights_deleted.json` (5(b)): the same data and
   settings with the 57 zero-weight rows removed before fitting
   (`removed_rows`, 0-based into `sim_gaussian`). The root test of 5(a), run
   by the port with zero weights, matches the 5(b) root at the tolerances in
   section 5. The recycled-division arithmetic survives only in
   `tests/test_instability.py`, where it shows that R's meat is not finite.

   Candidate split values: the instability tests use the variable values only
   to order the positive-weight rows, so zero-weight values cannot affect
   them (tested by overwriting those values). No split search exists yet.
   When it is written, the candidate values must come from positive-weight
   rows only; otherwise a candidate from a zero-weight row can win a tie and
   shift the break. This is recorded in docs/decisions.md.
7. **`best` uses log p-values.** `which.min` runs on the raw log p-values
   (`:373`) before `exp()`, so ties after underflow to 0 resolve as in R.
8. **What "tested" means in the fixtures.** The fixture field `tested` is
   membership in `mtest` (`:218`), which is every variable when
   `mtry = Inf`. The port's `FluctuationStatistics.computed` is narrower: it
   is `False` where R leaves the p-value `NA`, either because the variable
   has fewer than two distinct values (`:308`) or because no supLM index is
   in the trimming window (`:360`). Tests compare `computed` with the
   non-`NA` raw p-values. For example, node 5 of
   `sim_binomial_glmtree_multisplit.json` has z2 constant after the split on
   z2.
9. **Sorting.** R's `order()` uses a stable radix sort for these inputs, so
   tied values keep row order. The port uses `np.argsort(kind="stable")`.
   Ties matter because the supLM maximum also runs over indices inside tied
   blocks. The integer variable `z4` in the simulated data has many ties
   and matches R.
10. **Linear algebra.** `chol2inv(chol(meat))` is done with scipy's
    `cho_factor`/`cho_solve` and `root.matrix` with `numpy.linalg.eigh`. The
    eigenvector signs can differ from R's `eigen`, but `V sqrt(D) V'` does not
    depend on them. Any negative eigenvalue raises, as in R, even a tiny one
    from rounding.
11. **Factor representation.** Factors are passed as integer codes in R's
    level order. Only the set of levels present among positive-weight rows
    matters, so codes for unused levels are never needed.
12. **Fractional `minsize`.** `mob_partynode` coerces `minsize` with
    `as.integer` before the trimming uses it (`:168`), so `minsize = 0.05`
    becomes 0 there. The port's `minsize` is an `int`, and converting it is
    the caller's job, which mirrors the R behaviour.
13. **Project setup.** `pyproject.toml` named a missing `README.md`, which
    stopped the package from building. That line was removed, and numpy and
    scipy were added as runtime dependencies, with pytest, mypy, ruff and
    scipy-stubs in the dev group. `pyproject.toml` and `uv.lock` are
    committed.
14. **Clusters.** With `cluster`, the scaled scores (`process/sqrt(weights)`,
    already divided by `sqrt(nobs)`) are summed within clusters and the meat is
    the cross product of those sums (`:243-244`). Only the meat changes. The
    process, the supLM and chi-squared statistics and the p-value code use the
    row-level decorrelated process as before. `decorrelated_process`,
    `fluctuation_statistics` and `fluctuation_tests` take `cluster` as one id
    per row of the node: any sortable dtype, grouped by value like
    `as.numeric(cluster)`, so labels and row order do not matter. Missing ids
    and a wrong length raise `ValueError` (R's `model.frame` removes missing
    values before `mob_partynode` sees them).
    - *Zero weights.* The cluster vector is filtered with the same mask as
      the scores, so a node with zero weights tests like the data with those
      rows deleted, clusters included. A cluster with no positive weight does
      not count toward the number of clusters. partykit filters the scores
      but not the cluster vector, which on top of the weight bug in item 6
      would make `tapply` see more ids than rows. There is no R oracle for
      this (R fails), so the tests check equivalence with the deleted-rows
      data, whose cluster result matches the R fixture.
    - *Case weights with clusters.* The port follows `:244` literally. The
      fixtures `sim_cluster_lmtree_cluster_weights` and
      `sim_cluster_glmtree_binomial_cluster_weights`, and the blind cases
      `blind_03_lm` and `blind_13_glm`, check the combination against R.
    - *Too few clusters.* As decided in docs/decisions.md,
      `decorrelated_process` emits `TooFewClustersWarning` and raises
      `TooFewClustersError`, a subclass of `InstabilityTestError`, for fewer
      clusters than parameters, and for exactly as many when the cluster sums
      are constrained (next items),
      so the tree code that treats `InstabilityTestError` as partykit's
      `try()` failure makes the node terminal. The count is of clusters among
      positive-weight rows and `k` is the number of columns of the node's
      scores (aliased columns excluded). With fewer clusters than parameters
      R fails too, inside `chol()` ("the leading minor of order 4 is not
      positive" in `sim_cluster_few_lmtree_cluster.json`, node 3: 3 clusters,
      4 coefficients), and `try()` swallows it, so partykit stops without any
      message. That failure is decided by rounding: with `G < k` the meat
      has rank `G` and is exactly singular under any weights, and `chol()`
      fails or not on its rounding noise. In the second blind run (task 6l,
      tests/fixtures/blind_cases/README.md) R failed at 68 of the 70 such
      nodes it tested and succeeded at 2 (`blind_22_glm` node 9,
      `blind_52_multinom` node 6), where it split on statistics of order
      1e16. The port stops at `G < k` always, which is R's usual outcome.
      With one cluster and more than one parameter R fails earlier,
      at `J12 %*% t(process)` (`:252`): `tapply` over one cluster makes
      `apply` return a vector, so the meat is 1 by 1. The outcome is the
      same, the node is terminal.
    - *Exactly as many clusters as parameters, constant scaling: the port
      diverges from partykit, by decision.* Fitted scores (weights inside)
      sum to zero over a node. The meat sums `process / sqrt(weights)`
      within clusters, so when `sqrt(weights)` is constant over the node's
      positive-weight rows (unit or equal case weights) or the meat does not
      divide by it (`caseweights = FALSE`), the cluster sums sum to zero too
      and with `k` clusters the meat has rank at most `k - 1`. Whether
      partykit's `chol()` then fails depends on rounding: in a quick R run
      (lmtree, y on random x, 2 to 4 clusters, 8 seeds each) `try()` caught
      a failure in 12 of 24 fits and the other 12 returned p-values from a
      numerically singular meat. The port does not leave this to rounding:
      `n_clusters == k` raises `TooFewClustersError` there, so such a node
      is always terminal. Where R happens to get p-values there, the port
      stops instead; those p-values come from a singular meat and are not
      meaningful. This includes one cluster and one parameter, which R runs
      on a 1 by 1 meat of rounding noise.
      `test_exactly_as_many_clusters_as_parameters_raises` (unit weights),
      `test_exactly_as_many_clusters_with_equal_weights_raises` and
      `test_exactly_as_many_clusters_without_caseweights_raises` cover
      `k = 1, 2, 4`, and `test_one_more_cluster_than_parameters_runs` the
      first count that is tested.
    - *Exactly as many clusters as parameters, unequal case weights: as
      partykit.* Dividing the scores by unequal `sqrt(weights)` breaks the
      zero sum: the cluster sums are unconstrained, the meat has rank `G` and
      is regular at `G = k`, and the tests run (a failing Cholesky is still
      an `InstabilityTestError`). The blind run (task 6j) found two such
      nodes, where the earlier `n_clusters <= k` guard stopped while R
      split: `blind_03_lm` node 20 (`G = k = 3`) and `blind_13_glm` node 13
      (`G = k = 2`). On R's own scores the meat's smallest/largest
      eigenvalue is 8.9e-4 and 3.7e-3, far from rounding, and the port's
      statistics and log p-values equal R's to all 17 digits (9957.818866179252
      and 6584.768483180042; 1902.6518537080092 and 737.5969983956378).
      `test_as_many_clusters_as_parameters_with_case_weights_matches_r`
      checks both nodes against R,
      `test_exactly_as_many_clusters_with_unequal_weights_runs` random
      scores for `k = 1, 2, 4`, and
      `test_fewer_clusters_than_parameters_with_unequal_weights_raises` that
      `G < k` still stops. The cluster-robust `vcov` keeps `G <= k` for its
      warning: it sums the scores themselves, whose cluster sums add up to
      zero under any weights (docs/api-port.md, section 19).
    - *Fixtures.* `sim_cluster` (30 schools of 8 to 40 pupils, random intercept
      sd 1.5, school-level `zs` and `zt`) gives four trees: lmtree and binomial
      glmtree, each with and without `cluster`. Without it, `zs` and `zt`
      have root p-values below 1e-3 in both models; with it, above 0.05. The
      clustered binomial tree still splits on `z2` and `z3` at nodes 3 and 4
      (raw p of about 0.002 and 0.0001), which is what partykit
      computes, not a planted effect. `sim_cluster_few` (21 ordinary schools
      and 3 large high-`zs` schools whose intercept and `x1` slope differ by
      6, `y ~ x1 + x2 + x3 | zs`) is the few-clusters case. Its seed
      (20261022) was the first of a short scan in which the root split on
      `zs` and the child failed; seeds 20261023 and 20261025 to
      20261029 did not split at the root, because a cluster-summed test has
      little power with 24 clusters. The generator asserts that the fixture
      contains a failed node with fewer clusters than coefficients.

## 5. Tests

`tests/test_pvalues.py`, against `tests/fixtures/pvalues.json`:

- the Hansen table, exact equality on all 3000 entries;
- trimming (`from`, `to`, `lambda`, `tau`) for 16 trims crossed with 21
  values of `k`;
- `logp_suplm` on the full grid: 336 (trim, k) cases by 22 statistics,
  covering every branch (`tau = 0.5`, `tau <= 0.01`, `[0.49, 0.5)`,
  interpolation, `k = 39, 40, 41, 45, 60`);
- `logp_chisq` for 16 (k, levels) cases, and the Bonferroni/Sidak vectors,
  including `NA` and the 0.001 threshold;
- edge cases: statistics of 0 and 1e-12, a statistic of 5000 (far tail),
  monotonicity, the `tau = 0.01`, `0.49` and `0.5` boundaries, exact hits on
  table rows, `k = 40` versus `41`, the Estrella power interpolation at
  `k = 100`, `lambda = Inf` and `NaN`, absolute trim and `minsize`
  trimming.

`tests/test_instability.py`, against every node with a test in the 27 tree
fixtures (every node with status `ok`, and the 2 `failed` ones: zero weights and few clusters). R's `estfun` for the node
is the input, so model fitting is not involved:

- the statistic alone: `nobs`, `k`, trimming, which variables have a
  p-value, and the statistics;
- the p-value conversion alone: R's statistic in, R's raw log p-value out;
- the full test: raw log p-values, raw and adjusted p-values, `best`, and
  the node's `p.value`;
- zero weights: R's failure on 5(a) as a documented divergence, the 5(a) root
  against the 5(b) oracle, and invariance to values in zero-weight rows;
- clusters, against the three clustered tree fixtures (`sim_cluster_lmtree_cluster`,
  `sim_cluster_glmtree_binomial_cluster`, `sim_cluster_few_lmtree_cluster`),
  which go through the same statistic, p-value conversion and full-test tests
  as every other fixture, with the node's cluster ids read from the data CSV;
  the effect of `cluster` on the root p-values of the school-level variables
  (clustered and unclustered fixtures, in R and in the port); the few-clusters
  warning and error on R's failed node; cluster-count thresholds (fewer than,
  exactly and one more than `k`); zero-weight
  rows in the cluster vector (extra zero-weight rows, including a school with
  only zero-weight rows, leave R's result unchanged; zero weights equal
  deleted rows); relabelling; singleton clusters equal no clusters; input
  validation;
- edge cases: one parameter (scalar root, numeric and factor), many
  parameters (`k = 39, 40, 41, 45`), constant variables, an empty trimming
  window, singular scores.

Tolerances are relative 1e-9 (absolute 1e-12) for statistics and log
p-values, and relative 1e-8 for p-values on the probability scale. The
largest observed relative error on the p-value grid is about 1.2e-14.
