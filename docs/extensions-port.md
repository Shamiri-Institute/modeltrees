# Porting the extension models

How the extension models of docs/extensions-spec.md are ported to Python,
the decisions taken where the spec or R left a choice, and how the port
compares with the R fixtures. Task 5a covers the fit failure protocol and
the negative binomial model (sections 2 to 5), task 5b the safe logit
(section 6) and multinomial (section 7) models.

Line numbers `negbin.R:N` refer to `reference/MASS/R/negbin.R` (MASS
7.3-66).

## 1. What is ported where

| R | Python |
| --- | --- |
| `MASS::negative.binomial(theta)` with the log link (`negbin.R:213-267`) | `models/families.py`: `NegativeBinomial` |
| `glm.fit(..., etastart)` | `models/glm_fit.py`: `glm_fit(etastart=...)` |
| `MASS::theta.ml` (`negbin.R:328-366`) | `models/negbin.py`: `theta_ml`, `ThetaML` |
| `glm.nb`'s internal `loglik` (`negbin.R:82-84`) | `models/negbin.py`: `negbin_loglik` |
| `MASS::glm.nb(y ~ 0 + x, ...)` (`negbin.R:76-211`) | `models/negbin.py`: `glm_nb`, `NegBinFit` |
| `negbinfit` (`r/extensions/negbin.R`) | `models/negbin.py`: `NegBinModel` |
| `safelogitfit` (`r/extensions/safelogit.R`) | `models/safelogit.py`: `SafeLogitModel`, `classify_separation` |
| `multinomfit` (`r/extensions/multinom.R`, nnet's `multinom`) | `models/multinom.py`: `MultinomModel`, `multinom_newton`, `multinom_response` |
| fit failures, `estfun = NULL`, theta in the log-likelihood df | `node_model.FitStatus`, `failed_fit`, `extra_df`, `uses_batch_split`; `tree.StopReason.FIT_FAILED`, `NO_ESTFUN` (docs/model-interface.md, section 4.2) |

## 2. Decisions on the failure protocol

1. **Models report failures; `mob()` does not catch exceptions.** A model
   returns `failed_fit(...)` for a fit it cannot complete. An exception
   from a model still propagates, so `LmModel` and `GlmModel` keep
   partykit's behavior (an error stops the tree) and programming errors are
   not hidden.
2. **Input errors still raise.** Negative counts raise `ValueError` in
   `NegBinModel`, as R's family `initialize` stops. They are a property of
   the data, not of a node, so turning them into failed fits would only
   hide them.
3. **`restart = FALSE`.** A failed candidate's coefficients are not passed
   on as starting values; the next candidate starts from the last
   successful one. R has no counterpart.
4. **Batch split objectives.** A failed refit makes split search fit every
   remaining admissible candidate exactly, so the winner is the exact
   search's even when the best approximation fails. Models can also opt
   out with `batch_split = False`.
5. **Order of the checks at a node.** A failed node fit is reported before
   the `minsize` and `maxdepth` rules, so a small node whose fit failed shows
   `fit_failed` rather than `minsize`. `no_estfun` comes after them, where
   partykit checks for `NULL` scores, so the fixtures' stop reasons are
   reproduced.
6. **`extra_df` counts in the log-likelihood df only.** Pruning counts the
   coefficients only, as partykit's `mob_prune` does (modelparty.R:646-647),
   and the default `minsize` uses the coefficient count, as partykit and
   the thesis code do (docs/extensions-spec.md, section 3). `extra_df` (1
   for theta) is what `NegBinModel.loglik` adds to the rank, so
   `tree.loglik()`, `aic()` and `bic()` count theta as `logLik.negbin`
   does: the tree fixtures' `loglik_df` (rank + 1 per leaf, plus one per
   split) is checked on every negative binomial tree. The blind run (task
   6j) decided the pruning convention against R: at `blind_36_negbin` node
   18, R's own objectives with R's df (3, 7) keep the split by 0.86 AIC,
   and with theta counted, (4, 9), it would be pruned by 1.14; R keeps it.
   This is the convention partykit already follows for the gaussian
   dispersion, which `logLik.lm` counts and pruning does not. So `aic()` of
   a pruned negative binomial tree is not the quantity the pruning
   compared. Task 5a had counted theta in pruning without an oracle.

## 3. Decisions on the negative binomial model

1. **The loop is copied, not re-derived.** `glm_nb` follows `negbin.R:122-171`
   step by step: the Poisson fit from `start`, `theta.ml` at its means with
   the attributes dropped (`as.vector`), the alternation with
   `etastart = log(mu)`, `theta.ml` at the means *before* the refit's
   update, the convergence rule `|Lm0 - Lm| / d1 + |del| / d2 > epsilon`
   with `d1` from the Poisson fit's residual df, and the
   "alternation limit reached" warning when the counter exceeds `maxit`.
   glm.fit runs with `intercept = FALSE`, as glm.nb does for `y ~ 0 + X`;
   it only changes the null deviance, which is not used.
2. **`etastart` in the glm.fit port.** R evaluates `etastart` before
   `start` (`eta <- etastart %||% {...}`), so `start` is then neither
   length-checked nor used as `coefold`, only as the base of the coefficient
   update. The port does the same.
3. **What the extras are.** `theta` and `theta_se` come from the last
   `theta.ml` call (`SE` is `sqrt(1 / info)` at the start of its last Newton
   step), `theta_irls` is the theta of the last glm.fit call,
   `theta_warning` is R's `th.warn` (the last `theta.ml` warning attribute,
   or "alternation limit reached"), and `converged` and `iter` are the last
   glm.fit call's, as in R's `z$converged` and `z$iter`. `warnings` adds
   every warning of the fit in R's order. If the alternation never runs
   (`maxit < 1`), `theta_se` and `theta_warning` are None, as R's
   attributes are missing.
4. **Deviance objective.** `objective = "deviance"` is the last glm.fit
   call's deviance, R's `z$deviance`, which is evaluated at `theta_irls`,
   not at the returned `theta`. They differ by at most `epsilon` after
   convergence.
5. **Warnings.** Node fits (`estfun=True`) emit each warning as a
   `GlmFitWarning` with R's message; split-search fits are quiet, as
   `GlmModel`'s are, and keep their warnings in `extras["warnings"]`.
6. **Zero weights.** Rows with zero weight are left out of the fit and get
   zero scores, as in `GlmModel`. In glm.nb they contribute nothing to any
   sum (glm.fit drops them from the solve, `theta.ml` and `loglik` multiply
   by the weight), so this changes nothing but rounding, and avoids
   `0 * Inf` in `theta.ml` for a zero-weight row with a zero mean.
7. **Errors.** glm.fit's `stop()` (`GlmFitError`) and the errors `theta.ml`
   and glm.nb stop with (`NegBinError`) make a failed fit (Q6). R stops in
   `theta.ml`'s `while` with "missing value where TRUE/FALSE needed" when a
   Newton step is NaN; the port raises `NegBinError` with that message at
   the same point. Scalar arithmetic in `theta.ml` and `loglik` uses numpy
   floats so that division by zero and `log(0)` give R's Inf and NaN
   instead of Python exceptions.
8. **Special functions.** `digamma` and `trigamma` are scipy's `digamma`
   and `zeta(2, .)` (the value `polygamma(1, .)` returns, bit for bit). They
   agree with R's to rounding. Both are evaluated once per distinct count
   and then indexed, which gives the same values and makes `theta.ml` about
   twice as fast.
9. **Family name.** R's `"Negative Binomial(<format(round(theta, 4))>)"` is
   approximated with seven significant digits. Only the prefix is used, by
   the unit dispersion of the scores.

## 4. Comparison with R

Tests in `tests/test_negbin.py`, in the order of CLAUDE.md, at
`rtol = 1e-8`, `atol = 1e-10` except where noted:

1. **Node fits** (`node_fits/negbin_*`, 5 fixtures): coefficients, names,
   objective, scores, theta, theta_se, theta_irls, theta_warning,
   converged and iter. Observed agreement: about 1e-14 relative or better.
2. **Split search** on every fixture node's rows in R's selected variable:
   candidates, effective `minsize`, the Inf pattern, every objective, the
   chosen split, and no failed candidates.
3. **Trees** (`extension_trees/`, the 7 negbin trees, including weights,
   offset, clusters and the deviance objective): ids, depths, parents,
   kids, stop reasons, rows, nobs, splits, coefficients, objectives, scores,
   model info, test statistics, adjusted p-values, `best`, node p-values,
   split-search objectives in the grown tree, `fitted_node`, and the
   summary counts, summed leaf objective and `loglik_df`.
4. **Thesis** (`thesis_quine_crosscheck.json`): the deviance-objective quine
   tree against partykit's values at 1e-8 and against the printed thesis
   values at their printed precision (split variables, statistics,
   1 - p, leaf sizes, intercepts, theta).

Every comparison passes; no quantity diverges, so there is no first
diverging quantity to report. Observed agreement in the trees, relative:
coefficients, objectives and scores 1e-14 or better, except up to 4e-11 in
`sim_negbin_weights`, whose two near-Poisson leaves carry the largest
differences; split-search objectives 4e-12 or better; theta 5e-14 or better
in ordinary nodes.

**Near-Poisson theta.** Two leaves of `sim_negbin_weights` (nodes 10 and
15, theta about 1.7e4 and 8.4e4) stop `theta.ml` at its iteration limit.
As the spec says, theta is compared there at `rtol = 1e-3`; the observed
differences are 2.0e-7 and 7.4e-7 relative. theta_irls and theta_se use the same loose
tolerance in those nodes; coefficients, objectives and scores use the
usual one and pass.

**Checked by hand in R** (R 4.6.1, MASS 7.3-66; printed values in the
tests, not fixtures):

- 30 zero counts, intercept only: R stops in `theta.ml` with "missing value
  where TRUE/FALSE needed"; the port returns a failed fit with that reason.
- 29 zeros and a one: `-twologlik / 2 = 4.40122018151033` and
  theta = 730.987499089747 with "iteration limit reached" twice; the port
  gives the objective to 6e-12 and theta to 1.2e-8 relative.

**Known knife-edge, not a test.** A constant positive response (30 values
of 3) is degenerate: the Poisson means equal y up to rounding, so
`theta.ml`'s starting value `n / sum((y / mu - 1)^2)` divides by a sum of
rounding errors. R's Poisson fit returns 3.0000000000000324 and a start of
about 8.6e27, and glm.nb finishes; the port's returns exactly 3, the start
is Inf, the first Newton step is NaN, and the fit fails as R does in that
case. The outcome depends on the last bit of the Poisson fit, so it is
recorded here and not tested. Such a node has no overdispersion to
estimate and no information in its scores.

## 5. Speed

Growing the 7 negbin fixture trees takes about 28 s, almost all in
`sim_negbin_weights` (15 nodes, 1171 weighted observations) and the two
600-row trees with one split. Each candidate fit is a full glm.nb: a
Poisson fit, then usually one or two alternations, each a glm.fit and a
`theta.ml` call. There is no batch scorer (`split_objectives`) for this
model.

## 6. Safe logit

`SafeLogitModel` (docs/extensions-spec.md, section 4, provisional) is
glm.fit with `binomial("logit")` on the positive-weight rows, as
`GlmModel("binomial")` fits, plus the three "safe" rules of
`safelogitfit`.

1. **Warnings.** glm.fit's warnings are recorded, never emitted, in node
   fits and split-search fits alike, and returned in `extras["warnings"]`
   with R's `unique()` (first occurrences, in order). For this,
   `glm_fit(emit_warnings=False)` and `glm_objective(record_warnings=True)`
   record without emitting, and `GlmFitError.warnings` carries the warnings
   given before an error, as R's `warns` does.
2. **Errors.** A `GlmFitError` is a failed fit (`failed_fit`) with
   `extras["status"] = "error"` and the message in `reason` and
   `extras["error"]`; `separation`, `n_boundary` and `boundary` are None
   (R's NA). A non-binary response still raises `ValueError`, as
   safelogitfit stops before its `tryCatch`.
3. **Separation.** `classify_separation` counts the positive-weight rows
   with `|y - p| < separation_tol` among glm.fit's fitted values. The
   objective-only fits of split search (`glm_objective`, which now also
   returns the fitted values and `boundary`) classify it too, so a
   completely separated fit has status `NO_ESTFUN` everywhere, and the
   `reference_path()` fit agrees.

**Separated candidates are admissible.** docs/decisions.md says a
candidate whose child "is separated, for safe logit" is inadmissible. The
R oracle does not do this: safelogitfit returns glm.fit's finite objective
for a completely separated fit, and mob() uses it. The fixture
`sim_separation_safelogit` depends on it: the root split
`{g1, g2, g3} | {g4}` wins with objective 208.99, node 2's objective plus
about 4e-10 for the separated `g4` group, and node 2's split creates a
completely separated child (node 4). So `NO_ESTFUN` fits count in split
search with their objective, as docs/model-interface.md, section 4.2, says,
and only a glm.fit error (status `failed`) makes a candidate inadmissible.
The decisions entry is read that way here; its wording should be
clarified.

**Comparison with R** (`tests/test_safelogit.py`, `rtol = 1e-8`,
`atol = 1e-10`):

1. Node fits (`node_fits/safelogit_*`, 4 fixtures): coefficients,
   objective, scores, status, separation, `n_boundary`, `converged`,
   `boundary` and warnings. Observed: coefficients within 1e-13 absolute,
   objectives and scores within 4e-15. The coefficients of the two
   completely separated fits are compared at `rtol = 1e-3`, as the spec
   asks; they agree to 1.4e-10 absolute (slope about 952). The
   objective-only fit gives the same coefficients, objective, status and
   extras bit for bit.
2. Without separation the fit is bit for bit `GlmModel("binomial")`'s:
   on `safelogit_none`, and on the whole `sim_separation_safelogit_plain`
   tree (rows, splits, coefficients, objectives, scores, test statistics,
   split-search objectives).
3. Split search on every fixture node, and both trees
   (`sim_separation_safelogit`, `sim_separation_safelogit_plain`): the
   items of section 4, item 3, plus the separation info and the
   `no_estfun` stops of nodes 4 and 5. No warning is emitted while
   growing, as in R.

Every comparison passes; no quantity diverges. A unit test checks that
plain `GlmModel("binomial")` on `sim_separation` grows the same first two
splits and then splits further, as the spec describes. The glm.fit error
path has no R fixture; a unit test triggers it with a `start` of the wrong
length.

## 7. Multinomial logit

`MultinomModel` (docs/extensions-spec.md, section 5, provisional) fits the
baseline-category logit by Newton's method in numpy and scipy, not by a
copy of nnet's BFGS (decisions Q10).

1. **Response and layout.** `MultinomModel(levels)` takes the full level
   order, whose first level is the reference class; `y` holds 0-based codes.
   `multinom_response(y, levels)` maps labels to codes, and with
   `codes=True` checks integer codes. The order is always explicit, because
   R sorts factor levels by the locale. Coefficients are `(K - 1) p`,
   class-major, named `"<level>:<regressor>"`; the scores are
   `w_i (1{y_i = l_k} - p_ik) x_ij` in the same order.
2. **Newton's method** (`multinom_newton`). From zero, or from `start` with
   NaN set to 0 (R's `multinom_start_wts`), each iteration solves with the
   Hessian of minus the log-likelihood (`scipy.linalg.solve`,
   `assume_a="pos"`) and halves the step until the objective does not
   increase by more than `1e-12 (1 + |objective|)`. Without that rounding
   allowance, steps near the optimum, whose objective change is below its
   rounding error, were halved for nothing. Convergence is on the
   gradient: every entry at most `gtol = 1e-12` times `sum_i w_i |x_ij|`,
   the largest it can be. Quadratic convergence makes the tight `gtol`
   cost at most one step; with `1e-10` the estimate stopped about 5e-10
   from statsmodels' on random data. `maxit = 100`. A fit that stops
   without converging (singular Hessian, no decreasing step, `maxit`) keeps
   its estimate, with `converged = False`; under complete separation the
   coefficients grow until the gradient criterion is met, as glm.fit's do.
3. **Degenerate nodes.** Classes count as present with positive total
   weight. One class present: objective 0, NaN coefficients, `NO_ESTFUN`,
   so its objective counts in split search, as R's does. A non-reference
   class absent: its coefficients are NaN, it has no score columns, and
   `minsize` and the pruning df still count it (`len(coefficients)`). The
   reference class absent (`extras["status"] = "reference_absent"`): NaN
   coefficients, no scores and `NO_ESTFUN`, with the objective R computes,
   fitted with the first present class as reference, in both `objfun` and
   `extras["value"]`. As in R, such a candidate stays admissible in split
   search, competes with its true objective and can win; the child is then
   terminal. None of the fixture trees has such a candidate.
4. **Rank.** A design whose positive-weight rows, scaled by `sqrt(w)`, have
   rank below `p` under `dqrdc2` with lm.fit's tolerance `1e-7` is a failed
   fit with `extras["status"] = "rank_deficient"` (decisions Q9). An offset
   raises `ValueError`, as multinomfit stops.
5. **Extras.** `status` (`ok`, `single_class`, `reference_absent`,
   `rank_deficient`), `classes_present` (labels), `converged`,
   `iterations`, `value` and `max_rel_gradient`. Zero-weight rows are left
   out of the fit and get zero scores, as in multinomfit.

**Tolerances from R's gradient.** nnet's BFGS (`reltol = 1e-16`) stops with
a gradient `g_R` that the fixtures record as the column sums of R's
scores, up to 1.6e-8 of the column sums of absolute scores. Newton's
method converges to the MLE, from which R's coefficients differ by
`-H^-1 g_R` to first order, with `H` the Hessian of minus the
log-likelihood at the MLE. On every node fit this offset predicts R's
coefficients to within 5 percent of its size, or 1e-10
(`test_r_offset_explains_the_difference`; the observed residuals are
about 1e-14 except 8e-11 for `multinom_absent_class`, whose gradient is
tiny). The tests therefore use, per fixture and per tree node:

- coefficients: `atol = 2 max |H^-1 g_R| + 1e-10`, required to be below
  1e-6;
- scores: per entry `w_i |x_ij| ||x_i||_1 coef_tol / 2`, plus 1e-10, the
  change the coefficient tolerance can cause, since
  `sum_l |d p_ik / d eta_il| = 2 p_ik (1 - p_ik) <= 1/2`;
- test statistics and adjusted p-values: twice their change when our
  coefficients are moved by `-H^-1 g_R` and the tests recomputed, plus
  `1e-8` relative;
- objectives, split-search objectives and everything structural: the
  usual `rtol = 1e-8`, or exact.

The values that result:

| fixture | max \|H^-1 g_R\| | coefficient atol | observed |
| --- | --- | --- | --- |
| `node_fits/multinom_full` | 4.7e-8 | 9.4e-8 | 4.7e-8 |
| `node_fits/multinom_start` | 7.5e-9 | 1.5e-8 | 7.5e-9 |
| `node_fits/multinom_weights_zero` | 4.9e-8 | 9.7e-8 | 4.9e-8 |
| `node_fits/multinom_absent_class` | 5.3e-10 | 1.2e-9 | 5.3e-10 |
| `sim_multinom_nosplit`, node 1 | 1.1e-8 | 2.2e-8 | 1.1e-8 |
| `sim_multinom_onesplit`, nodes 1, 2, 3 | 4.7e-8, 6.8e-12, 2.0e-12 | 9.4e-8, 1.1e-10, 1.0e-10 | 4.7e-8, 6.8e-12, 8.3e-12 |
| `sim_multinom_cluster`, nodes 1, 2, 3 | 5.1e-12, 1.0e-7, 1.1e-14 | 1.1e-10, 2.1e-7, 1.0e-10 | 5.0e-12, 1.0e-7, 1.1e-12 |

Where R's gradient is largest (onesplit node 1, cluster node 2) the test
statistics differ from R's by up to 6.3e-7 relative and the adjusted
p-values by up to 2.1e-6. Two checks show that this is R's optimisation
error and not the port: the instability tests run on R's own scores
reproduce R's statistics and p-values to 3e-15 relative
(`test_tests_on_r_scores`, at the usual tolerance), and the statistics
recomputed at our coefficients moved by `-H^-1 g_R` reproduce R's to
2e-13.

**Comparison with R** (`tests/test_multinom.py`):

1. Node fits (`node_fits/multinom_*`, 6 fixtures): names, coefficients,
   objective (observed 2e-15 relative), scores, status and classes
   present; the reference-absent fit has status `no_estfun`, no scores and
   R's objective in `objfun` and `extras["value"]`, and the single-class
   fit has objective 0 and no scores. Starting values change the estimate by less than 1e-12.
2. Split search on every fixture node: candidates, the Inf pattern, every
   objective (observed 1e-13 relative), the chosen split, no failed
   candidates.
3. Trees (`sim_multinom_nosplit`, `sim_multinom_onesplit`,
   `sim_multinom_cluster`): ids, depths, parents, kids, stop reasons, rows,
   splits (exact), coefficients, objectives, scores, model info, test
   statistics, adjusted p-values, `best`, split-search objectives,
   `fitted_node` and the summary.

Every comparison passes at the tolerances above; no quantity diverges.

**Without an R oracle** (unit tests): statsmodels' `MNLogit` on random data
(coefficients, log-likelihood and per-row scores, observed 4e-16),
frequency weights against replicated rows, zero weights against deleted
rows, a class present only with zero weight, the rank-deficient design
(also only on the positive-weight rows, and as inadmissible split
candidates), reference-absent candidates (admissible, with the sum of both
sides' objectives) next to single-class sides (admissible, objective 0), a
split whose reference-absent child gives the smallest objective and is
chosen, with that child terminal by `no_estfun`, the offset error, the default
`minsize` `ceiling(10 (K - 1) p)` and the pruning df with an absent class,
a single-class node stopping with `no_estfun`, convergence from a far
start, and complete separation.

**Speed.** `sim_multinom_onesplit` and `sim_multinom_cluster` take about
4 s each: 1,565 Newton fits per tree with 4 coefficients, about 5
iterations each. There is no batch scorer.
