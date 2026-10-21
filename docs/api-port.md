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
   joined it in task 6d, which removed `hello()` (section 10, item 1).
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

## 9. The public result classes

Task 6d, `modeltrees/_api.py`: `ModelTree`, `NodeInfo`, `SplitRule`,
`TestResult`, `RecordedWarning` and `TreeSummary`; `LogLik` stays in
`modeltrees/_loglik.py` and is re-exported.

1. **`SplitRule`** has the five fields of section 3.2 plus
   `absent_levels` (the training levels absent from the node at a nominal
   split, which `predict` sends to the larger kid) and `boolean` (a
   Boolean partitioning variable). `threshold` is the break for a numeric
   split, the last level that goes left (a label) for an ordered one, and
   None for a nominal one. `conditions()` gives the left and right
   condition strings.
2. **Rule strings** follow `print.partynode`: a numeric break is
   `round(break, 5)` written as R's `as.character` (`z1 <= 3.99578`), an
   ordered split `z3 <= L2` / `z3 > L2`. A nominal split is written
   `z2 in {a, c}` as section 3.2 shows (R prints `z2 in a, c`). A Boolean
   variable gives `b = false` / `b = true` (section 1.4).
3. **`TestResult.p_value`** is the adjusted p-value, None for a variable
   that was not tested (R's `NA`); the statistic is then 0, as in R.
4. **`NodeInfo.extras`** is the model's report (section 3.5) plus
   `"warnings"`, the fit's warning texts, for the models that keep them
   (negative binomial, safe logit). A model without `report` gives an
   empty dict.
5. **`ModelTree.depth`** is the largest node depth with the root at depth
   1, so that `maxdepth` bounds it. partykit's `depth()` is one less (0
   for a root-only tree).
6. **`ModelTree.model`** exposes the node model, read-only, in addition to
   the fields of section 3.1. `used_rows` and `fitted_node` are read-only
   numpy arrays; setting an attribute raises `AttributeError`.

## 10. Entry points

Task 6d, `modeltrees/_api.py`: `lmtree`, `glmtree`, `negbintree`,
`safelogittree`, `multinomtree` and the public `mob`.

1. **Exports.** `modeltrees.__all__` is section 8.1's list without
   `LeafVcov`, which joins it with `vcov` (a later task). `__version__`
   comes from the installed package's metadata. `modeltrees.models`
   exports section 8.2's names; `BatchSplitModel`, specified in
   docs/model-interface.md (section 4.1) but never defined, is now a
   protocol in `modeltrees/node_model.py`.
2. **Control options** are a `TypedDict` (`ControlOptions`) unpacked into
   the entry points. Python accepts any keyword at runtime, so the names
   are checked: an unknown one raises `TypeError` listing the options,
   and so does `control` together with any option, or a `control` that is
   not a `MobControl`. The public `mob` keeps section 1.1's signature:
   `control` only, no control options.
3. **`n_jobs`** accepts None or 1 (an int, not a Boolean); other ints
   raise `NotImplementedError`, anything else `TypeError`. Task 6f made
   parallel split search available (section 17): `NotImplementedError` is
   gone, and 0 or a negative int other than -1 raises `ValueError`.
4. **Recorded warnings.** The whole fit is recorded: the design build
   (so the `MissingDataWarning` is in `tree.warnings` too, section 4.1),
   the response coding and the growth. Inside, the filter is `"always"`,
   so every occurrence is counted and user filters apply only to the
   re-emitted warnings. A warning recorded `k > 1` times is re-emitted
   once as `"<message> (k times)"`, so a filter matching the message from
   its start still matches. Warnings recorded before an exception are
   re-emitted before it propagates. The re-emitted warnings are
   attributed to the caller of the entry point. `MissingDataWarning`'s
   message is unchanged.
5. **`multinomtree(offset=...)`** raises `TypeError`, since the model has
   no offset (section 4.3's multinom rules).
6. **`mob()` trees** have `formula = None`, `model_name =
   "mob(<model class>)"`, every row used, and `response_levels` from the
   model's `levels` attribute if it has one (the multinomial model).

## 11. Prediction

Task 6d, `modeltrees/_predict.py` and `ModelTree.predict`.

1. **Coding new data.** The design's `Term`s are the `xlevels`: a numeric
   term needs a numeric (or Boolean) column, a categorical one is matched
   by label (R's `as.character` labels, so an integer column made nominal
   in training matches integer values). A level dropped in training
   (declared but unused) has no matrix column and no kid, so it counts as
   unseen.
2. **Which columns are needed.** `type="node"` reads only the
   partitioning variables; the other types also read the regressors and
   the offset. A needed column that is absent raises `DataError`.
3. **Unseen levels.** `unseen="error"` raises for an unseen level in any
   column read, on the row's path or not, as R's `model.frame(xlev = ...)`
   does. With `unseen="null"` a row with an unseen level gets no
   prediction at all (node 0, NaN, None), even if the level is in a
   variable off its path: section 4.2's "no prediction for those rows",
   taken literally.
4. **Routing.** Numeric breaks as `.bincode(right = TRUE)`: `z <= break`
   goes to kid 1 and `Inf` to kid 2; NaN and null are missing. R routes
   `-Inf` at random: `.bincode(x, c(-Inf, b, Inf), right = TRUE)` has the
   intervals `(-Inf, b]` and `(b, Inf]`, so `-Inf` falls in neither and
   partykit draws a kid (docs/sandwich-fixtures.md 2.2). The port sends
   `-Inf` to kid 1, as `-Inf <= b` says, a documented divergence; the
   predict fixtures mark those rows random and give no expected value. A
   level absent from the node goes to the kid with the larger `nobs`
   (`w2n`), kid 1 on a tie (R routes such a row at random too).
5. **Offset.** With `newdata`, `offset=` (a column name or an array)
   wins; otherwise a tree grown with an offset column reads that column
   from the new data, and a tree grown with an offset array raises
   `DataError` asking for `offset=`. A missing offset value gives NaN for
   that row (the node is still returned). `offset=` without `newdata`
   raises `TypeError`: the training rows use the training offset.
6. **Training rows.** `predict()` without data uses `fitted_node` and the
   stored design, so it equals `predict(training_frame)` bit for bit;
   rows dropped for missing values get no prediction.
7. **Trees grown by `mob()`** take `z` (a mapping of every partitioning
   variable to its values: floats; for factors integer codes into the
   training levels, or labels), `x` and `offset` as arrays, instead of
   `newdata`. `x` is required for types other than `"node"` unless the
   model was intercept-only (`x=None` in training). Without `z` the
   training rows are predicted.
8. **Validation order.** The type and `unseen` are checked first, then
   the data is coded (missing columns, wrong storage), then unseen levels,
   then the model is asked for an empty prediction of the type, so a type
   the model does not define (multinomial `"response"`, `"prob"` for the
   others) raises its `ValueError` before any leaf is computed.

## 12. Tables, likelihood and printing

Task 6d, `modeltrees/_report.py` and the `ModelTree` methods.

1. **`coef_table` backends.** `backend` is None, "polars", "pandas",
   "pyarrow", "modin", "cudf" or "numpy"; anything else raises
   `ValueError`, a missing library `ImportError`. The frame is built
   through narwhals in `_frame.build_frame`, so no other module imports a
   dataframe library. In the "numpy" dict, columns of strings, Booleans
   with None or mixed values are object arrays. `summary()` takes the same
   `backend` argument for its tables.
2. **`se="robust"`** raises `NotImplementedError` until `vcov` exists;
   any other value but None raises `ValueError`.
3. **`loglik()`** needs a model with `loglik` (the `LogLikModel`
   protocol), else `TypeError`. A failed leaf makes the value NaN and
   emits `NodeFitFailedWarning` naming the leaves on every call of
   `loglik`, `aic` or `bic`; df stays defined (section 3.1, item 2).
4. **Printing.** `str(tree)` uses 4 significant digits for the
   coefficients and theta; `tree.print(digits=...)` changes it (section
   3.4's `tree.print`). Each value is formatted on its own (`%.4g`), not
   with R's common formatting of a vector. The footer's parameter count
   and objective use R's 7 digits. A leaf's `n` is its `nobs`, as
   `print.modelparty` prints `x$nobs`. The header has a title per model
   ("Linear model tree", "Generalized linear model tree (family: ...)",
   "Negative binomial regression tree", "Safe logistic regression tree",
   "Multinomial logit tree", "Model-based recursive partitioning
   (<model class>)" for `mob()`), the formula, the partitioning variables
   with their kinds (a Boolean one as "numeric, Boolean") and
   `n = <used rows>`, with the dropped rows and per-column counts when
   rows were dropped. The objective's label is the residual sum of
   squares for lmtree, the deviance for negbintree with
   `objective="deviance"`, none for `mob()`, else the negative
   log-likelihood.
5. **`summary()` tables.** `variables`: variable, kind, levels and
   dropped levels (joined with ", "). `nodes`: node, parent, depth,
   n_rows, nobs, stop_reason, status, the variable with the smallest
   p-value (also when not significant), its statistic, the adjusted
   p-value, and `split`, the left kid's condition. `tests`: node,
   variable, statistic and adjusted p-value (NaN where not tested) for
   every node with tests. `coefficients`: `coef_table()`. The text is
   the tree print followed by each tested node's test table, the dropped
   levels and the recorded warnings.

## 13. Tests of the public layer

Task 6d. `tests/test_api_fixtures.py` grows all 44 tree and extension
tree fixtures through their entry point from a polars frame (the frame
helpers moved from `tests/test_design_fixtures.py` to
`tests/fixture_frames.py`).

1. **Equal to the array path, bit for bit.** Every node's rows, nobs,
   stop reason, p-value, the whole `FitResult` (coefficients, objective,
   scores, extras), the instability tests, the split search and the
   split, and the tree's minsize, variables and fitted nodes, compared
   with exact equality (NaN equal to NaN), for all 44 fixtures including
   `sim_gaussian_lmtree_zero_weights`.
2. **Against R.** `fitted_node`, `predict(type="node")` and the routing of
   the training frame as new data equal the fixture's `fitted_node`, and
   `loglik()` its `summary.loglik` (rtol 1e-8) and `loglik_df`, for the
   43 oracle fixtures. The zero-weights fixture's routing is compared with
   the array path instead.
3. **Predictions** equal the leaf model refitted on the leaf's rows and
   the linear predictor and inverse link (softmax for multinomial)
   written out, at rtol 1e-12, for every type the model has; the
   prediction of the training rows equals that of the training frame bit
   for bit.
4. **Coefficient table.** Bit for bit the array-path leaf coefficients,
   and against the fixture with the tolerances the model tests already
   use for the same coefficients (no new ones): rtol 1e-8 with atol
   1e-10, `SEPARATED_RTOL` from `tests/test_safelogit.py` for completely
   separated safe logit leaves, and the convergence-offset bound of
   `tests/test_multinom.py` for multinomial leaves.
5. **Missing data** against the same frame with the rows deleted by hand:
   the grown trees are identical, and the dropped rows have node 0 and NaN
   predictions.

## 14. The reference-absent tree in `extension_trees/`

Task 6e, step 0. `sim_multinom_noref_onesplit` moved from
`reference_absent_trees/` to `extension_trees/` (the one-line change in
`r/generate_all.R`; two regenerations gave byte-identical output, and the
JSON is unchanged). The suites that glob `extension_trees/` now grow it.

1. **Counts.** There are 13 extension tree fixtures, so the public layer and
   the design parity tests cover 45 trees, 44 of them oracles. The counts
   in sections 1, 3.1, 7 and 13 are those of task 6d.
2. **What the standing tests check on it.** Structure, splits, every
   split-search objective, the instability tests, the fitted nodes and
   logLik, as for every multinomial tree; the leaf without the reference
   class has R's objective, NaN coefficients, no scores and
   `StopReason.NO_ESTFUN`, its coefficient table row is NaN, and every
   prediction in it is NaN (section 3.3, item 5).
   `test_reference_absent_sides` fits both sides of every admissible root
   candidate and compares their classes, status and objective with the
   fixture's `reference_absent` field.
3. **The fixture README** still introduces the file under its own heading
   ("Reference-absent multinomial tree"). Task 6f, step 0 rewrote the
   paragraph in `r/generate_all.R` that said it lived in its own directory:
   it now says the file is in `extension_trees/` and is listed separately
   because one leaf has no scores. Two regenerations changed only that
   paragraph of `tests/fixtures/README.md` (eight lines, rewrapped) and were
   byte-identical to each other; no JSON or CSV changed.

## 15. Hessians and `vcov`

Task 6e, steps 1 to 3: `FitResult.hessian` (design sections 5.2 and 5.3),
`modeltrees/_vcov.py` (5.1) and `ModelTree.vcov` with
`coef_table(se="robust")`.

### 15.1 The Hessians

1. **`FitResult.hessian`** is the last field, default None, so existing
   constructors are unchanged. `check_vcov` is removed: every model
   answers `vcov=True`, and the core still passes `vcov=False` everywhere.
2. **Exactly symmetric.** `X' (w X)` is symmetric only to rounding, so the
   models return `(H + H') / 2` (`models._common.weighted_crossprod`; the
   multinomial model symmetrizes `multinom_hessian`'s result, leaving
   Newton's method untouched).
3. **`vcov=True` takes the full glm.fit path** in `GlmModel` and
   `SafeLogitModel`, as `estfun=True` does; the split-search shortcut is
   used only when neither is asked for. The Hessian of `vcov=True` alone
   equals that of `estfun=True, vcov=True`.
4. **Which weights.** The glm.fit-based models use `GlmFit.weights`, the
   working weights from the start of the last IRLS iteration, as their
   scores and R's `summary.glm` do. The Hessian is therefore not exactly
   the derivative of the scores at the estimate: the tests bound the
   difference by `|X|' |W_early - W_exact| |X| / dispersion` ("the lag")
   and check that the lag is below 1e-4 of the largest entry on converged
   fits (observed up to 4e-5, `glm_poisson_zeros_offset_weighted`). The
   lag accounts for the whole finite-difference gap on every fixture.
5. **Negative binomial.** The log link is not canonical, so the Hessian
   (vcov(glm.nb)'s bread, the expected information at `theta_irls`) is
   the derivative of the summed scores with the response held at the
   fitted means. The derivative at the data adds `sum_i w_i (y_i - mu_i)
   (mu_i / theta) / (1 + mu_i / theta)^2 x_i x_i'`; both are tested.
6. **Quasi-separation does not show in the condition number.** In
   `node_fits/safelogit_quasi` the diverging `d4` coefficient has
   information 2e-6 against 57 and 42 for the others, but its row is
   nearly uncorrelated with them, so the Hessian scaled by its diagonal has
   condition number 1.5 (unscaled 2.7e7). `LeafVcov.status` is then "ok"
   with a huge variance for `d4`; design section 5.3's "LeafVcov.status
   says so" holds only when the separation also makes the scaled matrix
   ill conditioned. The separation is reported in the leaf's `extras`.
7. **Tests** (`tests/test_hessians.py`): every model against a central
   difference of its summed scores written out (relative step 1e-5,
   tolerance 1e-7 of the largest entry, plus the lag), on the glm.fit
   fixtures (lm and glm), `node_fits/negbin_*`, `safelogit_*` and
   `multinom_*`; the inverse against statsmodels' `cov_params()` (WLS, GLM,
   GLM with `NegativeBinomial(alpha = 1 / theta_irls)`, Logit, MNLogit) at
   its dispersion, within twice the first-order change of the inverse by
   the lag plus 1e-8; and the Hessian and inverse against the vcovcl
   fixtures' `hessian` and `vcov_model` (rtol 1e-8; for multinomial within
   twice the change from R's convergence offset, as in section 16).

### 15.2 The sandwich

1. **`sandwich()` does not warn.** It returns a `Sandwich` with
   `too_few_clusters`; `ModelTree.vcov` emits the `TooFewClustersWarning`
   naming the node, and only when clusters are used (without clusters G is
   the number of rows).
2. **Singular.** Besides a failed Cholesky factorization (or a
   non-positive diagonal), a scaled condition number above `1 / eps`
   counts as singular: LAPACK's Cholesky succeeds on a rank-deficient
   matrix whose last pivot is rounding noise, and the inverse would have
   no correct digit.
3. **G = 1** gives NaN and status "unavailable" for both `cadjust` values:
   the summed scores are zero at the estimate, so the meat is empty
   whether or not `G / (G - 1)` is applied. HC1 with `n <= k` is
   "unavailable" too.
4. **Without clusters, G is the leaf's `nobs`**: the sum of the case
   weights under case weights (the rows repeated `w_i` times are each a
   cluster), the positive-weight rows otherwise. So `cadjust` multiplies by
   `nobs / (nobs - 1)`, which is sandwich's `n / (n - 1)` for unit weights.
5. **`caseweights=False`**: the scores are not divided by the weights and
   `n` and G count the positive-weight rows, as `w2n` does.

### 15.3 `ModelTree.vcov`

1. **Signature.** As design section 5.1: `node_id` positional, `cluster`,
   `type` and `cadjust` keyword-only. `cluster=True` is accepted and
   requires the tree's clusters (ValueError otherwise).
2. **Other clusters by name** are read from the training data, so the
   tree keeps a reference to it (the narwhals wrapper of the user's frame,
   or the mapping), not a copy. A changed frame gives the changed column.
   An array has one value per input row. Missing values on the rows the
   tree used raise `DataError`; on dropped rows they are allowed. A tree
   grown by `mob()` takes arrays only (`TypeError` for a name).
3. **What the tree keeps** for the refits: the coded response, the
   weights and the tree's clusters over the used rows (new keyword
   arguments of the `ModelTree` constructor, None in trees built without
   them, whose `vcov` raises `TypeError`).
4. **Refit warnings.** `GlmFitWarning`s of the refit are suppressed: the
   same fit warned while the tree was grown and is in `tree.warnings`.
5. **The refit check** compares each coefficient with the stored one
   relative to the larger of the two magnitudes (1e-10), and the NaN
   pattern; a difference emits one `ModelTreesWarning` per node naming it,
   and the covariance is the refit's. Every node of every fixture tree
   passes without a warning.
6. **`n_clusters`** is counted on the node's positive-weight rows; it is
   None without clusters, also for unavailable fits.
7. **`reason`** for a fit without scores is the fit's status and reason
   (`"the fit has status no_estfun: complete separation: ..."`).
8. **`coef_table(se="robust")`** uses `vcov()` with its defaults; it has
   no arguments for other clusters or types (use `vcov` for those). In the
   wide layout the `std_error_<term>` columns follow the coefficient
   columns, before the report columns; in the long layout `std_error`
   follows `estimate`. Section 12, item 2 (NotImplementedError) no longer
   applies.
9. **`LeafVcov`** joins `modeltrees.__all__` (section 10, item 1). It is
   defined in `modeltrees/_vcov.py` and has a `std_errors` property.
   `matrix` is read-only.
10. **Quasi-separation** (task 6f, step 0). When a safe logit fit's
    `extras["separation"]` is "quasi", `vcov()` emits a `ModelTreesWarning`
    "node N is quasi-separated: coefficient 'T' diverges, so its variance
    is not reliable" and puts the same text in `LeafVcov.reason` (after
    the sandwich's own reason, if any, with "; "). The status is not
    changed (item 6 of 15.1). The diverging coefficient is the estimated
    one with the least information per unit of its regressor, `H_jj /
    sum_i w_i x_ij^2`, a weighted mean working weight that does not depend
    on the regressor's scale (`models.safelogit.diverging_coefficient`).
    On `node_fits/safelogit_quasi` it is `d4` (ratio 2.8e-8 against 0.17
    and 0.12). The report names only the smallest; several diverging
    coefficients would show one of them.

## 16. `vcov` and `predict` against R

Task 6e, step 4: `tests/test_vcov_fixtures.py` (tests/fixtures/vcovcl/,
docs/sandwich-fixtures.md section 1) and `tests/test_predict_fixtures.py`
(tests/fixtures/predict/, section 2).

### 16.1 sandwich::vcovCL

1. **Every leaf, every combination.** For each leaf of the seven vcovcl
   fixtures, `vcov(leaf, cluster=None or False, type, cadjust)` is
   compared with the eight recorded matrices (school clusters and
   `cluster = NULL`, HC0 and HC1, `cadjust` TRUE and FALSE). lm, glm
   binomial, glm poisson, negbin and the non-separated safe logit pass at
   rtol 1e-8; the largest difference observed is 5e-15 relative to the
   largest entry. G equals the fixture's `n_clusters`; R's `cluster =
   NULL` has G = n, ours `n_clusters = None` with G = nobs (section 15.2,
   item 4), the same for these unweighted trees.
2. **Multinomial tolerance.** nnet's estimate is off the MLE by `-H^-1
   g_R` (docs/extensions-port.md, section 7), so R's matrix is ours at the
   moved coefficients. The tolerance is twice the change of our sandwich
   (Hessian and scores at the moved coefficients, the same clusters and
   adjustments) plus 1e-8 of the largest entry. Observed, relative to the
   largest entry:

   | leaf | ours vs R | tolerance | moved vs R |
   | --- | --- | --- | --- |
   | `sim_cluster_models_multinom_cluster`, node 2 | 3.9e-15 | 1.0e-8 | 6.7e-15 |
   | `sim_cluster_models_multinom_cluster`, node 3 | 2.7e-8 | 6.3e-8 | 7.1e-15 |

   The move explains R's matrix to rounding, so the difference is nnet's
   convergence, not the sandwich.
3. **G <= k** (`sim_cluster_few_lmtree_cluster`, node 3, G = 3, k = 4): R
   returns the rank-deficient matrix without a warning; ours equals it at
   rtol 1e-8 and emits `TooFewClustersWarning` naming the node. Two of its
   eigenvalues are at rounding level, as R's.
4. **G = 1, a documented divergence.** sandwich 3.1-3 stops ("non-
   conformable arrays", `single_cluster` in the binomial fixture); ours
   gives status "unavailable", NaN and a `TooFewClustersWarning`
   (section 15.2, item 3).
5. **A completely separated safe logit leaf** is "unavailable"; R has no
   fixture for it (its `estfun` is NULL).
6. **Weighted leaves.** Weighted multinomial vcov has no R oracle.
   sandwich's `estfun.multinom` reads the weights from `model.frame()` of
   the refit, which has none, so its scores lack the case weights and
   `vcovCL` differs from the formula of docs/api-design.md 5.1 by up to
   0.92 (the second blind run, task 6l,
   tests/fixtures/blind_cases/README.md). Weighted multinomial leaves are
   checked against that formula evaluated on our own scores only. For
   weighted leaves of the other models HC0 equals `vcovCL`, while HC1
   differs: sandwich's n is the number of rows, ours is the sum of the
   weights (16.2, item 1).

### 16.2 Weights, without R

1. **Frequency weights equal replicated rows.** sandwich's `meatCL` does
   not divide the scores by the weights and counts rows, so a weighted R
   fit is not an oracle (docs/sandwich-fixtures.md 1.5). Instead, on every
   leaf of `sim_cluster_lmtree_cluster_weights`,
   `sim_cluster_glmtree_binomial_cluster_weights`, `sim_negbin_weights`
   and `sim_multinom_onesplit` (with the data's `w0`, which has zeros), a
   one-node tree on the leaf's rows with integer weights equals one on the
   rows repeated `w_i` times, with the same clusters and without clusters,
   for HC0 and HC1, to 1e-10. So **the n of the HC1 factor `(n - 1) / (n -
   k)` is the sum of the weights** (the leaf's `nobs`), and without
   clusters G is too.
   - glm.fit's starting values depend on the weights, so the two fits stop
     at different iterates within epsilon and their working weights lag
     by different amounts (1.6e-5 relative in the binomial sandwich at
     epsilon 1e-8, 1e-8 at 1e-14). The test fits with epsilon 0 and 50
     iterations, IRLS's fixed point.
   - theta.ml's own tolerance lets the weighted and replicated negative
     binomial fits differ in theta by up to 7e-8, which moves the
     covariance (it is conditional on theta), and epsilon 0 makes theta.ml
     fail on one leaf. The negative binomial check therefore holds theta
     at the leaf's value: a `GlmModel` with `NegativeBinomial(theta)`
     grown by `mob()`, whose scores and Hessian are `NegBinModel`'s at that
     theta (`tests/test_hessians.py`).
2. **Zero weights equal deleted rows** (an lm and a binomial one-node
   tree), to 1e-12.

### 16.3 statsmodels

One-node trees with `cluster = "school"` on `sim_cluster` (lm `y_gauss`,
binomial `y_bin`) and `sim_cluster_models` (poisson and negbin with the
offset, safe logit `y_sl`, multinomial `y_mn`), against `OLS`, `GLM`,
`Logit` and `MNLogit` with `cov_type="cluster"` (default correction: our
HC1 with `cadjust`), `"HC0"` without clusters (our HC0 without `cadjust`)
and the coarser cluster `school // 3` (11 instead of 30 clusters on
`sim_cluster`, 13 instead of 36 on `sim_cluster_models`).

1. **Tolerance.** statsmodels uses the exact scores and information at its
   tight estimate; ours uses glm.fit's working weights from the start of
   its last iteration (section 15.1, item 4). The tolerance is twice the
   difference between our sandwich and the exact one at our estimate,
   plus 1e-8 of the largest entry; lm and multinomial (exact Hessians)
   get 1e-8 alone. Observed with the school clusters, relative to the
   largest entry: lm 8e-16, binomial 2e-10, poisson 9e-7 (tolerance
   1.9e-6), negbin 8e-9, safe logit 3e-6 (tolerance 6.1e-6), multinomial
   1e-14.
2. **The negative binomial bread differs.** For the log link, which is not
   canonical for the negative binomial, statsmodels' robust covariance
   uses the observed information as bread; R's `bread.glm` (and ours, as
   design section 5.2 specifies) uses the expected one. The difference is
   about 30 percent on `sim_cluster_models`. The test keeps our meat `H V
   H` and swaps the bread for the observed information at our estimate
   before comparing; it checks the scores, clusters and adjustments, and
   R checks the bread (16.1).

### 16.4 predict

On the rows R routes deterministically, `predict(newdata)` equals R's
`node` exactly and `response` and `link` at rtol 1e-8 (observed 1e-14),
with R's NA (a missing regressor or offset) as NaN, for the three predict
fixtures. lmtree's `"link"` has no R value (R stops, `failing_calls`) and
equals `"response"`. On the rows R routes at random: a missing split
variable on the path gives node 0 and NaN, and `-Inf` goes to the first
kid of the split, as the most negative finite value does (section 11,
item 4). Unseen levels raise `UnseenLevelError` both in a split variable
(R stops too) and in a partitioning variable no split uses (R ignores it),
and give node 0 with `unseen="null"`.

## 17. Parallel split search

Task 6f, step 1: `modeltrees/_parallel.py`, `find_split(parallel=...)` and
`n_jobs` in `tree.mob` and every entry point (design section 6).

1. **`n_jobs`.** None and 1 run today's serial code; nothing of the
   parallel path is touched (no pool, no environment change). -1 is
   `os.process_cpu_count()` (1 if it returns None), k > 1 is k workers. 0
   and negative values other than -1 raise `ValueError` (joblib's -2 and
   below are not offered); a Boolean or a non-int `TypeError`. The check
   runs before the data is read. A value that resolves to one worker (-1
   on one CPU) is serial.
2. **What runs in workers.** In `find_split`, with `restart=True`: the
   exact fits of a numeric search without a batch scorer, or whose batch
   scorer declined (returned None), and the fits of a factor search, when
   `parallel.use(candidates, node rows)` holds. Ordered factors take the
   same route as nominal ones; their candidate count (levels - 1) keeps
   them below the threshold in practice. A factor with a single candidate
   partition keeps R's objective-1 quirk on the serial path. lm's batch
   path, including its refits and the "fit them all after a failed refit"
   fallback, stays serial, as does everything under `reference_path()`
   (`find_split_reference` has no parallel path).
3. **Threshold.** `_parallel._THRESHOLD`, candidates (admissible ones,
   the fits actually made) times the node's rows (zero-weight rows
   included), strictly above. `_parallel.parallel_threshold(value)` is the
   test hook, a context manager. Task 6f, step 3 set it to 5e5 instead of
   the design's starting 2e6 (docs/performance.md, Step 5: twice the
   break-even of a glm tree's first parallel search, which pays the worker
   start-up; 2e6 left glm trees at n = 1000 serial).
4. **Pool.** `SplitPool` in `tree.mob`, created only for `n_workers > 1`
   and `restart=True`; its `ProcessPoolExecutor` starts at the first
   parallel search, and `mob` shuts it down (`wait=True`,
   `cancel_futures=True`) in a `finally` right after growing, before
   pruning, so also on errors and interrupts. A tree whose searches all
   stay below the threshold never starts a process.
5. **BLAS variables.** `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and
   `MKL_NUM_THREADS` are set to "1" in `os.environ` from the pool's start
   to its shutdown and then restored (a counter shared by concurrent
   pools). `ProcessPoolExecutor` starts workers on demand, so the variables
   must be in place whenever one starts, not only at creation. The parent's
   own BLAS is already loaded and does not reread them. Two consequences:
   - the forkserver is a process-wide singleton that keeps the
     environment of its start. If the pool starts it, it keeps the
     variables at 1, so later forkserver children of the same process
     (also the user's own) get one BLAS thread. If something else started
     it earlier without them, our workers inherit that environment, and the
     variables do not reach them. `tests/test_parallel.py` checks a worker
     in a fresh interpreter;
   - while a tree is grown with `n_jobs > 1`, other threads of the process
     see the three variables at 1.
6. **What a search sends.** The side fit (`tree._NodeSideFit`, a frozen
   dataclass with the model and the node's arrays, which replaces the
   closure of `_node_fitters`; the serial path calls the same object) and
   the node's values of the variable are pickled once per search in the
   parent. The bytes go with each chunk, and a worker unpickles them once
   per search (cached by a search key). So the arrays cross the pipe once
   per chunk, about `4 k` times per search, not once per worker as the
   design says; at n = 20000 that is about 1 MB per chunk against seconds
   of fitting. An unpicklable model fails in the parent with `TypeError`
   before any worker starts.
7. **Chunks and assembly.** The admissible candidates, in candidate
   order, are cut into `min(m, 4 k)` contiguous chunks with boundaries
   `round(linspace(0, m, chunks + 1))`. A worker rebuilds each mask
   exactly as the serial loop does (`zv <= uz[i]`, `np.isin(codes,
   levels)`), fits left then right with no start, and returns the summed
   objective and a failed flag per candidate. The parent writes them into
   the objective vector at the candidates' indices (failed ones stay Inf,
   `n_failed` is the count of flags) and then applies the unchanged
   first-minimum rule and split construction (`_factor_split`, now shared
   with the reference).
8. **`restart=False`** with more than one worker grows serially (no pool)
   and `tree.mob` emits one `ModelTreesWarning` per call, recorded and
   re-emitted by the entry points like every warning.
9. **Errors.** Futures are read in submission order; the first exception
   cancels the rest and is raised in the parent with its own type and the
   worker's traceback as `__cause__` (`concurrent.futures`' remote
   traceback). A worker that dies gives `BrokenProcessPool`. The pool is
   shut down by `mob`'s `finally` either way.
10. **Start method.** `forkserver` on Linux, `spawn` elsewhere. Both import
    the caller's `__main__` in the workers, so a script that grows a tree
    with `n_jobs > 1` needs the `if __name__ == "__main__":` guard (without
    it, multiprocessing raises at worker start and the search fails with
    `BrokenProcessPool`). Custom models must be importable by the workers
    (defined in a module, not in `__main__` of an interactive session).

11. **Tests** (task 6f, step 2; `tests/test_parallel.py`).
    `tests/conftest.py` sets the three BLAS variables to 1 for the whole
    test process before numpy loads (the suite's results are unchanged; a
    test checks that every loaded OpenBLAS reports one thread). With the
    threshold at 0, every glm, negbin, safe logit and multinomial tree
    fixture in `trees/`, `extension_trees/` and `cluster_trees/` (34 trees:
    the clustered, the weighted, `sim_cluster_glmtree_binomial_cluster_weights`
    with both, and the reference-absent `sim_multinom_noref_onesplit`;
    the `restart=False` fixture is tested separately) grown with
    `n_jobs=2` and `4` equals the serial tree exactly: every dataclass
    field of every node, floats and numeric arrays compared by their bytes
    (so NaN payloads and signed zeros too), including `fit`, `test`,
    `split_search.objective` and `n_failed`, and `fitted_node`. Every
    search with at least two candidate fits went through the pool (counted
    on unpruned trees). No fixture has a failed candidate, so a model whose
    split fits fail for side sizes 3 mod 7 checks `n_failed` and the
    objectives. Also: a nominal variable with 10 levels (511 partitions);
    an exception raised in a worker reaching the caller with its type,
    message and remote traceback, with no child process left and the
    environment restored; an unpicklable model; `mob(n_jobs=2)` inside a
    worker of another process pool, and two parallel trees in a row; `n_jobs=-1`;
    `restart=False` with one warning and no pool. No parallel tree differed
    from its serial tree in any bit.

## 18. The formula features against R

Task 6i, step B: `tests/test_formula_fixtures.py` against the eleven
fixtures of `tests/fixtures/formula_trees/` (docs/formula-fixtures.md).
Each fixture's data is read into a polars frame (factors as `pl.Enum` with
the schema's levels, ordered factors through `kinds`, logicals as
Booleans, `NA` as null) and grown by the entry point with the fixture's
formula, weights and control; seven are repeated from a pandas frame
(categoricals, NaN for missing numbers) and from a dict of numpy arrays
(labels with None, `levels=` for the declared orders), which give the
polars tree bit for bit.

1. **No parity bug.** On every oracle fixture the used-row mask equals the
   rows R kept (`na_action`), the coefficient names equal R's exactly
   (`fb`, `fa`, `fc` for a factor declared d, b, a, c; `fd` ... `fc`
   without an intercept; `bTRUE`; `bFALSE`, `bTRUE`, `x`; `(Intercept)`
   for the one-part formulas), the kept and dropped levels equal the model
   frame's, and structure, splits, rows, nobs, stop reasons, coefficients,
   objectives, scores, raw and adjusted p-values, statistics, split-search
   objectives, logLik, `fitted_node` and `predict(type="node")` on the
   training rows match at the tolerances of `tests/test_tree.py` (rtol
   1e-8, atol 1e-10). `predict` on the new rows (a subset of the levels,
   the data's declared levels) equals R's node exactly and response at rtol
   1e-8, as `tests/test_predict_fixtures.py`. `_design.py`, `_formula.py`
   and `_predict.py` are unchanged.
2. **A split on a logical partitioning variable: R stops, the port
   splits.** partykit 1.3-0 tests a logical as numeric but searches its
   split as a factor and then fails in `kidids_split` ("variable 2 is not
   integer", docs/formula-fixtures.md, section 3, item 2). The port keeps
   design section 1.4 (a Boolean is the numeric 0/1, printed `zb = false` /
   `zb = true`), a documented deliberate difference. The test asserts R's
   error and that the Boolean tree equals R's tree on the same data with
   the variable stored as 0/1 (`sim_formula_logical_numeric_lmtree_split`)
   and the port's tree of that 0/1 column bit for bit. Where no split
   selects the logical (`sim_formula_logical_lmtree_logical`), its test
   statistics are R's.
3. **Levels dropped with missing rows are unseen at predict.** A new row
   with `f = e` or `zf = q`, levels that occur only in rows the model frame
   dropped, raises `UnseenLevelError`, as R stops ("factor f has new level
   e", "factor zf has new level q"; checked in R on complete rows, not
   recorded in a fixture). This is section 11, item 1. So the training
   frame predicted as new data needs `unseen="null"` when its dropped rows
   hold such levels.
4. **Printed splits** equal the lines of partykit's print, apart from the
   documented braces of a nominal split (section 9, item 2) and the
   Boolean conditions of item 2.

## 19. Fixes from the blind run

Task 6k fixes what the blind differential run of task 6j found
(tests/fixtures/blind_cases/README.md).

### Blind run

Task 6j grew 60 cases (12 per model: lm, glm, negbin, safe logit,
multinomial) that nobody shaped during development, with data and settings
drawn in R from fixed seeds, and compared the port's public entry points
with partykit 1.3-0 component by component. 54 matched on every component
and 6 did not, in three classes:

- **a port defect** (`blind_03_lm`, `blind_13_glm`): the cluster guard
  stopped at G = k also under unequal case weights, where partykit's meat
  is regular. Fixed in the core (item 1);
- **a documented decision that R contradicted** (`blind_36_negbin`): theta
  in the pruning df. Changed to partykit's coefficient count (item 4);
- **rounding in quantities the data do not determine** (`blind_29_negbin`,
  `blind_36_negbin` after the pruning change, `blind_43_safelogit`,
  `blind_47_safelogit`): a weakly identified theta and the slopes of a
  completely separated leaf. The blind-case comparison now compares what
  the data determine (items 5 and 6), with sensitivity tests (item 7); no
  standing tolerance changed.

No multinomial case diverged. All six cases are now standing tests that
match R on every component (item 8).

### Decisions

1. **Clusters with case weights at G = k.** The instability tests stop at
   `n_clusters < k` always and at `n_clusters == k` only when the cluster
   sums of the scaled scores are constrained to sum to zero, that is with
   `caseweights=False` or when every positive weight in the node is equal
   (docs/decisions.md; docs/instability-tests.md, section 4, item 14). With
   unequal case weights and G = k the tests run, as in partykit.
2. **The `vcov` warning keeps G <= k, a deliberate reading of "the same
   rule".** The rule is "warn at G = k when the cluster sums are
   constrained to sum to zero". `sandwich()` sums `estfun` itself within
   clusters (not `estfun / sqrt(w)`, sandwich's `meatCL` with frequency
   weights, section 16.2), and those sums add up to the summed scores, zero
   at the estimate under any weights. So the constraint always holds there
   and G = k is short of rank under unit, equal and unequal weights alike:
   on the leaves' own scores at `blind_03_lm` node 20 and `blind_13_glm`
   node 13 the vcov meat has smallest/largest eigenvalue -3e-17 and -1e-19
   (rank G - 1), where the instability meat has 8.9e-4 and 3.7e-3.
   `too_few_clusters` therefore stays `G <= k`, and
   `test_too_few_clusters_under_any_weights` (tests/test_vcov.py) and
   `test_too_few_clusters_warns_under_any_weights` (tests/test_tree_vcov.py)
   check the flag, the warning and the rank under each weighting, with and
   without `caseweights`.
3. **Standing tests.** `blind_03_lm` and `blind_13_glm` left the xfail set
   of `tests/test_blind_cases.py`: they are in `MATCHING` and must match R
   on every component.
4. **Pruning counts coefficients only.** `tree.mob` prunes with `k =
   len(root coefficients)` per node, as partykit's `mob_prune`
   (modelparty.R:646-647), for every model; `extra_df` (1 for negative
   binomial theta) counts only in `NegBinModel.loglik`'s df, and so in
   `tree.loglik()`, `aic()` and `bic()`, as `logLik.negbin` does. The oracle
   is `blind_36_negbin` node 18: with R's objectives and df (3, 7) the AIC
   rule keeps the split by 0.86, which R does; with theta counted, (4, 9),
   it would be pruned by 1.14
   (`test_aic_pruning_keeps_r_split_of_blind_36_node_18`). It is the
   convention partykit follows for the gaussian dispersion. A consequence:
   `aic()` of a pruned negative binomial tree is not the quantity the
   pruning compared (it adds one per leaf). With the extra df gone,
   `blind_36_negbin` first diverges at node 86's theta_se, a rounding
   difference (item 5).
5. **Weakly identified theta, in the blind-case comparison only.**
   `tests/blind_compare.py` compares theta, theta_irls and theta_se at
   `WEAK_THETA_RTOL = 1e-6` when R or the port reports glm.nb's
   "alternation limit reached" (theta.ml's "iteration limit reached" keeps
   the standing `THETA_LOOSE_RTOL`), and does not require the same
   alternation-limit flag when the final thetas agree to `SAME_THETA_RTOL =
   1e-10` relative. The standing fixture tests are unchanged. Evidence
   (log-likelihood at the port's means, at the port's and at R's theta):

   | node | theta | SE / theta | theta, theta_irls, theta_se vs R | log-likelihood difference |
   | --- | --- | --- | --- | --- |
   | `blind_29_negbin` 4 | 4924 | 16.6 | 4.8e-8, 2.8e-7, 2.0e-7 | 4.7e-10 (1.1e-12 relative) |
   | `blind_36_negbin` 86 | 272 | 14.1 | 3.2e-9, 3.9e-9, 1.4e-8 | 2.0e-11 (7.5e-14) |
   | `blind_36_negbin` 112 | 176 | 14.3 | 4.5e-15, 5.6e-10, 7.6e-10 | 1.1e-12 (1.4e-14) |

   The standard error is 14 to 17 times theta and the likelihood is flat
   in theta to rounding, so its digits beyond about 1e-7 are not
   determined by the data. At node 112 the port reports "alternation limit
   reached" and R does not: both trajectories jitter by 1e-8 to 2e-7 per
   alternation around glm.nb's stopping threshold `epsilon = 1e-8`, R's
   step happened to fall below it at alternation 24 and the port's did not
   by 25 (tests/fixtures/blind_cases/README.md), and the final thetas agree
   to 4.5e-15. Coefficients, objectives, scores, `converged` and `iter` keep
   the usual comparison and agree.
6. **Completely separated leaves, in the blind-case comparison only.** For
   a safe logit leaf with `separation == "complete"`, `blind_compare`
   compares the linear predictor `X beta` on the leaf's rows at
   `SEPARATED_RTOL` (the intercept direction, which the standing tests
   already compare at that tolerance), the fitted means at `rtol = 1e-8,
   atol = 1e-10`, and the objective at `rtol = 1e-8` without `atol` (it is
   about 8e-10, below `ATOL`, so this is a tightening), instead of each
   coefficient at `atol = 1e-10`. Evidence at node 3 (y = 1 on every row,
   intercept 26.566 after glm.fit's 25 iterations):
   - `blind_43_safelogit`: R's slopes move the linear predictor by at most
     2.2e-7, the port's by 8.5e-16; the linear predictors differ by 1.9e-8
     relative;
   - `blind_47_safelogit`: 4.4e-9 and 5.5e-10; 1.7e-7 relative;
   - in both, `1 - mu` is 2.9e-12 on every row, so the slopes change the
     fitted means by about 1e-19, below their rounding: the port's and R's
     fitted means are equal on every row, and the objectives are equal bit
     for bit.
7. **Sensitivity** (`tests/test_blind_cases.py`): a 1e-6 relative change
   of R's value is still caught for theta, theta_irls and theta_se in a
   well-identified node (`blind_29_negbin` node 3, theta 4.83, SE 1.65), for
   the objective of the separated leaf and for one fitted mean of it
   (`blind_43_safelogit`, `blind_47_safelogit` node 3).
8. **Standing tests.** All six blind cases match R on every component and
   are in `MATCHING`; the xfail set is empty.
9. **Fixture README.** The blind-case paragraph of `r/generate_all.R` now
   says the cases are standing tests; `Rscript r/generate_all.R` changed
   only that paragraph of `tests/fixtures/README.md` (no JSON or CSV
   changed).
