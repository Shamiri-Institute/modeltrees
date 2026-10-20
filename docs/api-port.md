# Porting the user-facing layer

Deviations from docs/api-design.md, and the decisions taken where the design
left a choice, recorded as the layer is implemented. Task 6b covers the
core preparation: the error classes (design section 7), the rename of the
core tree class (section 12, item 5) and the optional model protocols
(sections 3.5, 3.6 and 4.2).

## 1. Errors and warnings

1. **Where the classes live.** Every public class of section 7 is defined
   in `modeltrees/errors.py` and exported from `modeltrees`. The three
   existing warning classes moved there; `modeltrees.instability`,
   `modeltrees.tree` and `modeltrees.models.families` import them, so the
   old import paths give the same class objects. `GlmFitError` and
   `NegBinError` stay in `models/glm_fit.py` and `models/negbin.py`, with
   `FitError` as their new base (still a `ValueError`).
2. **`FormulaError.position`** is an optional constructor argument
   (`FormulaError(message, position=None)`).
3. **`modeltrees.__all__`** has the error and warning classes and
   `reference_path`. The entry points and result classes of section 8.1
   join it when they exist; `hello()` stays until the section 8.1 surface
   replaces it.
4. **Internal exceptions.** `tests/test_errors.py` grows every tree and
   extension tree fixture (44 trees, `sim_cluster_few_lmtree_cluster`
   among them, whose nodes fail the tests with too few clusters) and
   checks that no `InstabilityTestError` or `TooFewClustersError` leaves
   `mob()`.

## 2. `GrownTree`

The core class is renamed with no alias. `scripts/bench_tree.py` and
docs/tree-growing.md, which named it, are updated in the same change.

## 3. Model protocols

The protocols are in `modeltrees/node_model.py`: `LogLikModel`,
`ReportingModel` (with the `Report` mapping type) and `PredictingModel`, plus
two helpers the models share, `rank(result)` (the number of non-NaN
coefficients) and `linear_predictor(result, x, offset)`. The tree sum is
`modeltrees._loglik.tree_loglik(tree, model)`, returning
`LogLik(value, df, nobs)`.

### 3.1 Log-likelihood

1. **What the result must carry.** `loglik(result)` sees only the
   `FitResult`. Two models need more than the objective:
   - `LmModel` stores `n_positive` (rows with positive weight) and
     `sum_log_weights` in `extras`, for `logLik.lm`'s
     `0.5 (sum(log w) - N (log(2 pi) + 1 - log N + log RSS))`. As in R, `N`
     counts rows, not the sum of case weights, and rows with zero weight
     are excluded. A node whose weights are all 1 is fitted by `lm.fit`
     in R and has no weights, which gives the same value.
   - `NegBinModel` stores `twologlik` (R's `z$twologlik`) in `extras`, so
     the deviance objective's leaves report the log-likelihood at the
     leaf's theta.
2. **A failed leaf** has value NaN, and its df counts every coefficient
   (it has no rank), plus the dispersion or theta where the model counts
   one. The tree's value is then NaN; its df stays defined.
3. **Multinomial df** is `(K' - 1) p` for the `K'` classes present, which
   is the number of non-NaN coefficients for an ordinary fit. A fit
   without the reference class has NaN coefficients but R's objective,
   and R's df (`multinom` fitted on the present classes), so the df comes
   from `classes_present`, not from the NaN count. One class present gives
   value 0 and df 0.
4. **The fixtures.** Every tree and extension tree fixture is compared,
   except `sim_gaussian_lmtree_zero_weights`, whose R tree is not an oracle
   (docs/decisions.md): R stops at the root, the port grows the tree of the
   deleted-rows fixture, which is compared.

### 3.2 Reports

1. **Value types.** `classes_present` is the present labels joined with
   `", "` (for example `"lo, mid, hi"`), since report values are scalars.
2. **Failed fits** give the same keys, with None where the fit has no
   value. For negative binomial, a failed fit carries no warnings, so
   `n_warnings` is None; for safe logit the warnings before the error are
   counted.
3. **`converged`** is `FitResult.converged` for glm, negative binomial and
   safe logit, and the Newton fit's flag in `extras` for multinomial, None
   when no Newton fit ran (one class present, rank-deficient design).

### 3.3 Prediction

1. **Aliased coefficients count as 0** in the linear predictor, as
   `predict.lm` and `predict.glm` drop aliased columns.
2. **Inverse links keep R's bounds**: the binomial logit's
   `DBL_EPSILON` and `1 / DBL_EPSILON` beyond `|eta| > 30` (family.c), and
   `pmax(exp(eta), DBL_EPSILON)` for the log link, as `predict.glm` uses
   the family's `linkinv`. The tests' hand-computed inverse links apply
   the same bounds.
3. **`x = None`** is an intercept-only model; the number of rows then
   comes from `offset`, and without an offset `predict` raises
   `ValueError`. The frontend always passes the intercept column.
   Multinomial needs `x` (it has no offset).
4. **`kind="class"`** returns the 0-based code of the most probable class
   as float64, NaN where no probability is defined; the frontend maps the
   codes to labels (`response_levels`). Ties go to the first class.
5. **Degenerate multinomial fits.** An absent class has probability 0 and
   a NaN link column. A fit with one class present gives that class
   probability 1 and a NaN link. A fit without the reference class has no
   coefficients, so every kind is NaN; so is a failed fit.
6. **`kind="response"` for multinomial** raises `ValueError` naming
   `"prob"` and `"class"` (design section 13, decision 8). Other models
   raise `ValueError` for `"prob"` and `"class"`.

## 4. Formulas

Task 6c, `modeltrees/_formula.py`, `parse_formula(formula) -> ParsedFormula`.

1. **Conflicting intercept tokens.** Section 1.2 says the regressor part
   has an intercept unless it contains `0`, `-1` or `+ 0`; it does not say
   what `y ~ 0 + 1 + x | z` means. The parser follows R's `terms()`: the
   last intercept token wins (`0 + 1 + x` has an intercept, `1 + 0 + x` and
   `1 - 1` do not), checked with R 4.6.1.
2. **A leading sign** is accepted in either part (`y ~ -1 + x | z`,
   `y ~ +x | z`), as R accepts it; the grammar of 1.2 has no leading sign.
3. **`- 0`** raises, as "`-` of anything but `1`" (R reads it as `+ 1`).
4. **One part with only intercept tokens** (`y ~ 1`) raises: it has no
   partitioning variables.
5. **Positions** are 0-based indices into the formula string, of the
   offending token (a call's name, a backtick, an operator). Messages
   repeat the position and the formula.
6. **`expanded`** is the two-part form for every formula, written with
   `- 1` for a part without an intercept (`y ~ x - 1 | z`), `1` for an
   intercept-only part, and backticks around names that are not
   identifiers.
7. **Numbers.** A digit sequence other than `0` and `1`, including a name
   that starts with a digit (`1x`), raises with a hint to use backticks.
