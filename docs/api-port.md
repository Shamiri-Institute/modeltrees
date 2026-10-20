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
   raise `NotImplementedError`, anything else `TypeError`.
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
   ("Reference-absent multinomial tree ... in its own directory"), since
   that text is in `r/generate_all.R` and the task allowed one changed
   line there.
