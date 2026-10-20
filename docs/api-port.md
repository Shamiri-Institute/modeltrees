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

## 5. The dataframe boundary

Task 6c, `modeltrees/_frame.py`: `Frame(data)`, `Frame.column(name) ->
Column`, `array_column(name, values)` (the mapping path, also used for
arrays passed as weights, offset or cluster) and `read_columns`.

1. **Storage types.** Every column becomes one of `float` (float64; floats
   and decimals), `integer` (the native integer dtype), `boolean`,
   `string` or `categorical` (Enum and Categorical). Other dtypes (dates,
   lists, structs) raise `DataError` when the column is used; unused
   columns are never converted.
2. **Labels are strings.** Categorical values and categories are
   converted with `str()`, so a pandas categorical of integers has levels
   `"3"`, `"1"`. Object columns must hold only strings and None.
3. **Null and NaN.** `Column.null` and `Column.nan` are separate masks.
   pandas reports NaN as null (`is_null()` is true), so a pandas NaN is
   under `null`; polars and numpy NaN are under `nan`. Missing data
   (section 4.1) is their union.
4. **Kind candidates.** A numeric storage type (float, integer, Boolean)
   may be declared `nominal` or `ordered` in `kinds`; string and
   categorical columns cannot be `numeric`.
5. **A polars Categorical has no declared levels**, so its levels come
   from the code-point sort (section 4.3, rule 4). An unordered pandas
   categorical keeps its declared categories (section 13, decision 3).
6. **Input types.** Anything that is not a `Mapping`, an eager narwhals
   frame or a lazy frame raises `TypeError`. A mapping's values go through
   `numpy.asarray`, so lists work; they must be 1-D and of equal length.

## 6. The design builder

Task 6c, `modeltrees/_design.py`, `build_design(formula, data, *, weights,
offset, cluster, levels, kinds, na_action) -> Design`.

1. **Output.** `Design` holds the parsed formula, the raw `Response`
   (float64 values, a Boolean as 0/1, or 0-based codes with levels), the
   regressor matrix `x` (always with the intercept column when the formula
   has one, so an intercept-only model is an `(n, 1)` column of ones, not
   `None`), `coef_names`, the partitioning variables as the core's
   `Variable`, weights (ones when not given), offset and cluster (None when
   not given), `used_rows` and `missing_counts`. Every array covers the
   used rows only. `Term` describes the response, each regressor and each
   partitioning variable (kind, storage, kept and dropped levels, coding
   and matrix columns) for a later predict step.
2. **R's coefficient names keep backticks.** R writes a non-syntactic
   name with backticks in `model.matrix` column names (`` `my x` ``,
   `` `my f`b ``, `` `b b`TRUE ``), so the port does too (`r_name`):
   a name is syntactic if it starts with a letter or a dot not followed by
   a digit, has only letters, digits, `.` and `_`, and is not reserved.
3. **Boolean regressors** follow R's logical exactly: a factor with levels
   `FALSE` and `TRUE`, both always kept (R keeps an unused one, so a
   constant `TRUE` column gives an all-one `bTRUE`). With an intercept
   that is the single `<col>TRUE` column of section 1.3; without one, as
   the first factor term, it gets `<col>FALSE` and `<col>TRUE`, as
   `model.matrix(y ~ 0 + b)` does. `kinds={"b": "numeric"}` makes it a
   0/1 column named `b`; `"nominal"` an ordinary factor of the observed
   levels.
4. **`levels` makes a numeric column categorical** unless `kinds` says
   otherwise (section 4.3 allows a multinomial "integer with `levels`").
   `levels` with `kinds="numeric"` raises.
5. **Numeric columns made categorical.** Integer, float and Boolean
   columns may be `nominal` or `ordered` through `kinds`. Their level
   labels are R's `as.character`: integers in decimal, Booleans `FALSE` and
   `TRUE`, doubles with 15 significant digits in the shorter of fixed and
   scientific notation (`"1e+05"`, `"2.5"`, `"10"`). Rule 3 of section 4.3
   (numeric order) applies to all of them, so an ordered integer or float
   needs no `levels`.
6. **Values outside `levels`** are checked over every non-missing row of
   the column, not only the used rows. User levels are converted with the
   same label rule, so `levels={"y": [0, 1, 2]}` matches integer codes.
7. **Unknown names in `kinds` or `levels`** (columns the formula does not
   use) raise `DataError`, to catch typos.
8. **Weights and offset** must be integer or float columns or arrays.
   Arrays passed for weights, offset or cluster go through the mapping path
   of `_frame` and are named `(weights)`, `(offset)` and `(cluster)` in
   missing-value counts. Cluster ids are passed through unchanged (numbers,
   or strings for string and categorical columns); the core recodes them.
9. **Missing data.** `missing_counts` counts each used column's missing
   values over all input rows; the warning lists them in the order
   response, regressors, partitioning variables, weights, offset, cluster.
   Inf checks run on the used rows only.

## 7. Fixture parity of the design

`tests/test_design_fixtures.py` builds the design of every tree and
extension tree fixture (44) from a polars frame, of six from a pandas
frame and of six from a dict of numpy arrays, and compares the regressor
matrix, partitioning variables, weights, offset, cluster, raw response and
coefficient names with `tests/tree_fixtures.py` exactly (`np.array_equal`).

1. **Every fixture formula is in the subset.** The only expression is the
   `offset = log(exposure)` argument of the poisson and negative binomial
   fixtures, which is not part of the formula. A keyword offset is a
   column name or an array, so the test precomputes a column
   `log_exposure` and passes `offset="log_exposure"`.
2. **Intercept-only models.** `tree_fixtures` passes `x = None`; the design
   always has the `(Intercept)` column, so the test compares with a column
   of ones (section 6, item 1).
3. **Multinomial coefficient names** in the fixtures are class-major
   (`mid:(Intercept)`); the design's `coef_names` are the regressor columns
   (`(Intercept)`, `x`), which the model flattens per class.
4. **Dict input** passes `levels` for ordered factors and for factors whose
   schema order is not the code-point sort, so the dict subset also covers
   rule 4 of section 4.3.

## 8. Response coding

Task 6d, `modeltrees/_response.py`, `code_response(response, rule, *,
weighted) -> CodedResponse(values, levels)`, applied by the entry points to
the raw response of the design.

1. **Counts are whole numbers.** Section 4.3 says only "non-negative
   counts" for poisson and negbin. R stops on negative values and only
   warns on non-integer ones, but our poisson log-density (as R's `dpois`)
   is -Inf for them, so every objective would be Inf. Non-integer values
   raise `DataError` too.
2. **Binomial proportions.** Values strictly between 0 and 1 without
   `weights=` raise `DataError` ("pass the number of trials as weights").
   With weights, a non-integer number of successes keeps R's
   `GlmFitWarning` from the family; it is not checked here. Values outside
   [0, 1] raise `DataError` (R stops with "y values must be 0 <= y <= 1").
3. **Booleans** are numeric 0/1 for every model but multinom, which needs
   a categorical response.
4. **`response_levels`** is the kept level order of a categorical response
   (after unused levels are dropped), for binomial and safe logit as well
   as multinom; None for a numeric or Boolean response. A binomial or
   safe logit response with a single kept level is allowed (every row is a
   failure), as in R; multinom needs two.
