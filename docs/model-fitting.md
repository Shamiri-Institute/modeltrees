# Node model fitting

This document maps R's `glm.fit`, `lm.fit` and `lm.wfit`, and partykit's
`lmfit` and `glmfit` node models, to the Python port in
`src/modeltrees/models/`. It records the decisions and ambiguities met while
porting. No tree-growing or split-search code is involved.

R sources are R 4.6.1 (`src/library/stats/R/glm.R`, `lm.R`, `family.R`,
`src/library/stats/src/family.c`, `lm.c`, `src/appl/dqrdc2.f`, `dqrls.f`,
`dqrsl.f`). partykit line numbers are `file:line` in
`reference/partykit/R/`.

## 1. What is ported where

| R | Python |
| --- | --- |
| `.Call(C_Cdqrls, x, y, tol)`: `dqrls`, `dqrdc2`, `dqrsl` job 1110 | `models/_qr.py`: `dqrls`, `dqrdc2` |
| `gaussian()`, `binomial()`, `poisson()` with canonical links | `models/families.py`: `Gaussian`, `Binomial`, `Poisson`, `get_family` |
| `stats::glm.fit` | `models/glm_fit.py`: `glm_fit`, `GlmFit` |
| `stats::lm.fit`, `stats::lm.wfit` | `models/lm_fit.py`: `lm_fit`, `lm_wfit` |
| partykit `lmfit` (`lmtree.R:59-104`) | `models/lm.py`: `LmModel` |
| partykit `glmfit` (`glmtree.R:57-133`) | `models/glm.py`: `GlmModel` |
| factor response, `binomial()$initialize` | `models/glm.py`: `binomial_response` |
| `FitResult`, `NodeModel` and the optional protocols | `node_model.py`, as in docs/model-interface.md, section 4 |

## 2. Decisions and ambiguities

1. **R's QR is ported, not replaced.** Which coefficient is aliased depends
   on `dqrdc2`'s limited pivoting: a column whose remaining norm falls below
   `tol` times its original norm moves to the end, so the later of two
   dependent columns is dropped. numpy's and scipy's QR pivot differently.
   `dqrdc2` and the `dqrsl` solve are ported line by line, with the column
   updates vectorised. Tolerances are R's: `1e-7` for `lm.fit` and
   `lm.wfit`, `min(1e-7, epsilon / 1000)` in `glm.fit`. Non-finite input
   raises `ValueError("NA/NaN/Inf in 'x'")`, as Cdqrls does.
2. **Family details from R's C code.** The logit inverse link and its
   derivative clamp outside `|eta| > 30` (`family.c`), and poisson clamps
   `exp(eta)` at `DBL_EPSILON`. These decide the fitted values in the
   near-separation fixture, which matches R to about 3e-14 relative.
3. **Log densities in the AIC.** R's `dbinom` and `dpois` use Loader's
   saddle-point algorithm. The port uses the closed forms with `gammaln`,
   `xlogy` and `xlog1py`. They agree to rounding: the AIC matches every
   fixture to better than 1e-15 relative. `dpois` of a non-integer count is
   `-Inf`, as in R.
4. **Warnings and errors.** Each R `warning()` in `glm.fit` and
   `binomial()$initialize` becomes a `GlmFitWarning` with R's message, also
   kept in `GlmFit.warnings` in R's order. The fixture warning lists match
   exactly. R's `stop()` raises `GlmFitError` (a `ValueError`), and family
   input checks raise `ValueError`. The fits in split search are quiet: `GlmModel.fit` with
   `estfun=False` uses `glm_objective`, the same IRLS without warnings, null
   deviance or residuals (docs/performance.md, Step 1b), so a candidate whose
   fit does not converge no longer warns. Node fits still warn. Under
   `modeltrees.reference_path()` split search uses `glm_fit` as before.
5. **Not ported.** `etastart`, `mustart`, `trace`, `singular.ok = FALSE`,
   two-column binomial responses and non-canonical links. `intercept` is
   kept, defaulting to `TRUE` as in `glmfit`, which never sets it.
6. **Zero weights in `GlmModel`: the port diverges from partykit for
   gaussian, by decision.** R's `gaussian()$aic` is
   `nobs * (log(2 * pi * dev / nobs) + 1) + 2 - sum(log(wt))` with `nobs`
   counting zero-weight rows, so a single zero weight makes the AIC, and so
   `glmfit`'s objective, `Inf` (checked in R 4.6.1:
   `glmfit(..., weights = c(0, rep(1, 19)))$objfun` is `Inf`). Split search
   would then see every candidate as `Inf`. docs/decisions.md says not to
   replicate zero-weight bugs, and the instability tests already treat zero
   weights as deleted rows. `GlmModel` therefore fits on the positive-weight
   rows and returns zero scores for the others. For binomial and poisson this
   gives exactly R's coefficients, objective and scores, since zero-weight
   rows contribute nothing there. For gaussian the objective is that of the
   data with those rows deleted. `glm_fit` itself keeps R's `Inf`.
   `LmModel` needs no change: its objective `sum(w * r^2)` ignores
   zero-weight rows, and `lm.wfit` handles them as in R.
7. **glmtree's scores are not the exact score.** `glm.fit` returns the
   working weights from the start of its last iteration (`wt[good] <- w^2`)
   and the working residuals at the final fit, and `glmfit` multiplies the
   two (`glmtree.R:97`). The port does the same and matches R to 1e-14. The
   exact score at the final coefficients differs by the last IRLS step.
   Because the convergence test is on the deviance, which is quadratic near
   the optimum, that step can be about `sqrt(epsilon)`, and at
   `epsilon = 1e-8` the scores differ from statsmodels' `score_obs` by up to
   1e-4 relative. The statsmodels cross-check of the score formula therefore
   refits with `epsilon = 1e-13`, where the gap is below 1e-5; R's own scores
   are checked against the fixtures at 1e-8.
8. **Step halving has no R fixture** (docs/decisions.md). The unit test uses
   poisson data (seed 20261030, `x = 0..100`, 40 points) and the start
   `(-5, -2)`, whose first IRLS step overflows `exp()`. The fit then halves
   back, converges after 267 iterations with one "step size truncated due to
   divergence" warning, and equals statsmodels' MLE to 1e-15. The same call
   was run by hand in R 4.6.1 and gives the same iteration count, warning and
   coefficients. R's inner halving loop is capped at `maxit`: here 26
   halvings are needed, so `maxit = 26` fails with "inner loop 1; cannot
   correct step size" and `maxit = 27` does not, in R and in the port. The
   R coefficients at `maxit = 27` are in the test, printed from R, as the
   far-tail p-values are in the instability tests. They are not a fixture.
   "Inner loop 2" (invalid `mu`) cannot be reached by these three families,
   because a non-finite `mu` makes the deviance non-finite first.
9. **Coefficient names.** `NodeModel.fit` receives a plain array, so the
   names are given to the model's constructor (`coef_names`). Without them
   the names are `(Intercept)` for `x = None` and `x1, x2, ...` otherwise,
   as `lm.fit` names unnamed columns. The tree code will pass the names from
   the model matrix.
10. **Binomial response at the boundary.** `binomial_response(y, levels)`
    applies R's rule `y != levels(y)[1]`: the first level is failure. The
    level order is `levels` if given, else the categories of a pandas-style
    categorical (`.cat.categories` or `.categories`, read by duck typing so
    pandas is never imported), else the sorted unique values. R sorts factor
    levels by the locale's collation, which can differ from Python's sort for
    mixed-case strings, so callers should pass `levels` when it matters. The
    node models accept numeric or boolean `y` only and raise `TypeError` for
    strings, pointing to `binomial_response`.
11. **Arguments that do not change the fit.** As in R, `LmModel` ignores
    `start`, and both models ignore `cluster`. `caseweights` changes only the
    gaussian dispersion in `GlmModel`'s scores (`glmtree.R:102-107`). Both
    models raise `NotImplementedError` for `vcov=True`, since `"info"` and
    `"sandwich"` are out of the v0.1 scope.
12. **`lmfit` chooses `lm.fit` when every weight is exactly 1** and `lm.wfit`
    otherwise (`lmtree.R:68-73`). The port does the same, so the arithmetic
    path matches R's.

## 3. Tests

Tolerances are `rtol = 1e-8`, `atol = 1e-10` against R. The observed
agreement is about 1e-13 relative or better.

`tests/test_glm_fit.py`, against the 11 glm.fit fixtures:

- iteration count, convergence and boundary flags, rank and degrees of
  freedom, exactly;
- coefficients (NaN pattern included), deviance, null deviance, AIC, fitted
  values, linear predictors, working weights, working residuals and prior
  weights;
- R's warnings, in order, both as `GlmFitWarning`s and in the result;
- partykit's `glmfit` objective and scores, and `lmfit` coefficients,
  objective and scores for the gaussian cases;
- the cases named in the task: near separation (14 iterations, fitted
  probabilities numerically 0 or 1), zero-heavy poisson with and without
  offset and weights, the rank-deficient gaussian (x3 aliased), maxit
  (3 iterations, not converged) and a user start (16 iterations);
- step halving against statsmodels and its `maxit` bound (item 8);
- the QR port: a dependent column in the middle is pivoted to the end and
  aliased, a zero column, full rank against `numpy.linalg.lstsq`, and
  non-finite input; `lm.wfit` residuals for zero-weight rows; binomial
  zero-weight rows; gaussian AIC `Inf` with a zero weight.

`tests/test_node_models.py`, against every node of the 27 tree fixtures
(237 nodes): the node's rows are taken from the data CSV and refit, and the
coefficients and names, objective and scores are compared with the fixture.
The response, regressors, weights and offset (`log(exposure)`) are rebuilt
from the CSV, and the factor response goes through `binomial_response`.
Every node is also cross-checked against statsmodels: `WLS` coefficients
and residual sum of squares (lmtree) or negative log-likelihood (gaussian
glmtree), and `GLM` with frequency weights for binomial and poisson:
coefficients, negative log-likelihood and, on a tight-epsilon refit, the
scores (item 7). Unit tests cover zero weights equal deleted rows (and
partykit's objective where it is finite), the dispersion with and without
case weights, aliased columns, the intercept-only model, `vcov`, names and
`binomial_response`.

`tests/test_imports.py` imports `modeltrees` and every submodule in a fresh
interpreter and checks that neither pandas nor statsmodels was loaded.
