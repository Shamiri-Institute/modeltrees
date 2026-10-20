# Porting the extension models

How the extension models of docs/extensions-spec.md are ported to Python,
the decisions taken where the spec or R left a choice, and how the port
compares with the R fixtures. Task 5a covers the fit failure protocol and
the negative binomial model. Safe logit and multinomial follow.

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
| fit failures, `estfun = NULL`, theta in the pruning df | `node_model.FitStatus`, `failed_fit`, `extra_df`, `uses_batch_split`; `tree.StopReason.FIT_FAILED`, `NO_ESTFUN` (docs/model-interface.md, section 4.2) |

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
6. **`extra_df` counts in pruning only.** The default `minsize` still uses
   the coefficient count, as partykit and the thesis code do
   (docs/extensions-spec.md, section 3). The tree fixtures' `loglik_df`,
   which partykit computes from `logLik.negbin` (rank + 1 per leaf, plus
   one per split), equals the pruning df with `extra_df = 1` on every
   negative binomial tree, which is a partial R check of the convention.

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
