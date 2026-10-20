# User-facing API design (v0.1)

This document designs the layer that analysts and other users
call: formula entry points, dataframe input, the fitted-tree object,
prediction, cluster-robust leaf covariances, parallel split search, errors,
the exported names and the tests. It sits on top of the existing core,

```python
mob(model, y, x, z, *, weights=None, offset=None, cluster=None, control=None)
```

(docs/tree-growing.md), and changes the core only where section 12 says so.
The goal is a small, predictable surface: few entry points, one tree class,
one rule for each data question, and errors rather than guesses.

Every decision that was not forced lists its alternatives in a line or two and
says which was chosen and why. Section 13 records the decisions on the
questions this design left open. Nothing here is implemented yet.

Facts checked on 2026-10-20 for the decisions below:

- formulaic 1.2.2 (PyPI, 2026-06-02) has `pandas>=1.3`, `narwhals>=1.17`,
  `interface-meta`, `wrapt`, `typing-extensions`, `numpy` and `scipy` as
  hard runtime dependencies. Polars and pyarrow are optional extras.
- narwhals 2.27.1 has no runtime dependencies; every backend is an extra. It
  offers `narwhals.stable.v1` and `narwhals.stable.v2`, frozen APIs that will
  "never change or remove any public function" and are maintained
  indefinitely. `stable.v2` is recommended for new projects.
- Probed in this environment with `narwhals.stable.v2` and pandas 3.0.6: a
  pandas ordered categorical maps to `Enum(categories=[...])`, with the order
  kept in `dtype.categories`. An unordered pandas categorical maps to
  `Categorical`, and `Series.cat.get_categories()` returns pandas' declared
  category order. `is_null()` is true for pandas NaN. Polars is not
  installed here, and its behavior is taken from the narwhals documentation
  (Enum has `categories`; polars separates null from NaN).
- The tree fixtures' `summary` already records R's `logLik(tree)` and its df
  (`loglik`, `loglik_df`), for example 15 for the four-leaf
  `sim_gaussian_lmtree_multisplit` (4 leaves of 3 plus 3 splits) and 9 for
  `sim_multinom_onesplit` (2 leaves of 4 plus 1 split).

## 1. Entry points and formulas

### 1.1 Signatures

```python
def lmtree(formula: str, data: DataInput, *, weights=None, offset=None,
           cluster=None, levels=None, kinds=None, na_action="omit",
           n_jobs=None, control=None, **control_options) -> ModelTree
def glmtree(formula, data, *, family: Literal["gaussian", "binomial", "poisson"] = "gaussian",
            epsilon=1e-8, maxit=25, <shared keywords>) -> ModelTree
def negbintree(formula, data, *, objective: Literal["loglik", "deviance"] = "loglik",
               epsilon=1e-8, maxit=25, <shared keywords>) -> ModelTree
def safelogittree(formula, data, *, separation_tol=1e-6, epsilon=1e-8, maxit=25,
                  <shared keywords>) -> ModelTree
def multinomtree(formula, data, *, gtol=1e-12, maxit=100, <shared keywords>) -> ModelTree
def mob(model, y, x, z, *, weights=None, offset=None, cluster=None,
        control=None, n_jobs=None) -> ModelTree
```

The shared keywords:

| keyword | type | meaning |
| --- | --- | --- |
| `weights` | column name, array or None | case weights, as partykit's `weights` |
| `offset` | column name, array or None | added to the linear predictor |
| `cluster` | column name, array or None | cluster ids, used by the instability tests (and by `vcov`, section 5) |
| `levels` | `Mapping[str, Sequence]` | explicit level order per column (section 4.3) |
| `kinds` | `Mapping[str, "numeric" \| "ordered" \| "nominal"]` | overrides the inferred kind of a column (section 1.4) |
| `na_action` | `"omit"` or `"fail"` | section 4.1 |
| `n_jobs` | int or None | section 6 |
| `control` | `MobControl` | the ported `mob_control()` settings |
| `**control_options` | `MobControl` fields | shorthand, as R passes `...` to `mob_control()` |

`control_options` are the `MobControl` fields (`alpha`, `bonferroni`,
`minsize`, `maxdepth`, `trim`, `dfsplit`, `prune`, `restart`,
`caseweights`, `numsplit`, `catsplit` and the fields that raise
`NotImplementedError`), typed with a `TypedDict` and `Unpack`. Passing both
`control` and a control option raises `TypeError`. An unknown keyword raises
`TypeError` as usual.

- *Alternatives:* (a) only `control=MobControl(...)`; (b) only keywords; (c)
  both. **Chosen: (c).** `lmtree(f, df, alpha=0.01)` is what R users type,
  and `MobControl` stays the single place that validates them. Allowing both
  at once would need a merge rule, so it is an error.

Each entry point builds the design (section 1.3), the model
(`LmModel`, `GlmModel(family)`, `NegBinModel(objective=...)`,
`SafeLogitModel(...)`, `MultinomModel(levels)`) and calls the core. Model
tuning arguments are keywords of the entry point, not a model object, so
the user never meets the `NodeModel` protocol unless they call `mob`.

`glmtree`'s families are those `get_family` has: gaussian, binomial and
poisson with their canonical links. A family object or another link raises
`NotImplementedError` in v0.1.

`mob` is the array path for advanced users and custom models. It keeps the
current signature, adds `n_jobs`, and returns the same `ModelTree` class.

### 1.2 Supported formula syntax for v0.1

```
formula   := response "~" part [ "|" part ]
response  := name
part      := term { ("+" | "-") term }
term      := name | "1" | "0"
name      := identifier | "`" any text without a backtick "`"
identifier:= [letter or "_" or "."] { letter | digit | "_" | "." }
```

Rules, matching partykit's `Formula` handling where it applies:

1. **Two parts.** `y ~ x1 + x2 | z1 + z2`: `x1`, `x2` are regressors of the
   node model (with an intercept), `z1`, `z2` the partitioning variables.
2. **One part.** `y ~ z1 + z2` means an intercept-only node model
   partitioned by `z1` and `z2`, as `glmtree.R:11-18` and `lmtree.R:200-207`
   rewrite it to `y ~ 1 | z1 + z2`. This surprises users who expect `z1` to
   be a regressor, so the docstrings and the README show it explicitly, and
   `print` echoes the expanded formula `y ~ 1 | z1 + z2`.
3. **Intercept.** The regressor part has an intercept unless it contains
   `0`, `-1` or `+ 0`. `y ~ 1 | z` is intercept-only. `y ~ 0 | z` (no
   columns) raises `FormulaError`. Intercept tokens in the partitioning part
   raise `FormulaError`.
4. **Terms are column names.** Duplicated terms are dropped silently, as R
   does. A column may appear in both parts (common in MOB: `y ~ x | x + z`).
   The response may not appear on the right.
5. **Not supported in v0.1, each with a `FormulaError` that names the
   construct:** interactions (`:`, `*`, `/`, `^`), function calls and
   transformations (`log(x)`, `I(...)`, `C(...)`, `poly(...)`), `offset(...)`
   inside the formula, `cbind(...)` responses, the `.` shorthand, `-` of
   anything but `1`, and a third `|` part. The message suggests computing
   the column in the dataframe first (for an offset, `offset=`).
6. **Names.** Exact, case-sensitive column names. Names that are not
   identifiers (spaces, `-`, leading digits) use backticks, as in R.

*Alternatives for transformations:* (a) support `log()`, `I()` and
interactions now; (b) none in v0.1. **Chosen: (b).** Every transformation is
one dataframe expression for the user, while each one we support needs
naming rules, predict-time re-evaluation and R-compatible column names.
Interactions also change what "treatment coding" means. None of the fixtures
need them.

### 1.3 The regressor matrix

- Numeric columns (any integer or float dtype) enter as one column each.
- Categorical columns (string, categorical, enum, or a column declared
  `"nominal"` or `"ordered"` in `kinds`) use **treatment coding**: one
  indicator column per level except the first. With no intercept, the first
  categorical term gets all its levels, as R's `model.matrix` does.
  *Alternatives:* sum or Helmert coding, or R's `contr.poly` for ordered
  factors. **Chosen: treatment coding for every categorical regressor**,
  ordered or not. It is what analysts read off a table, and polars has no
  ordered factor to trigger polynomial contrasts. This diverges from R for
  pandas ordered categoricals used as regressors (R would use `contr.poly`);
  the README states it.
- Boolean columns are coded as R codes a logical: a factor with levels
  `FALSE` and `TRUE`, both always kept, even when one does not occur. With
  an intercept that is one indicator column named `<col>TRUE`. In a model
  without an intercept, a Boolean that is the first categorical term gets
  both indicator columns, `<col>FALSE` and `<col>TRUE`, as
  `model.matrix(y ~ 0 + b)` does (docs/api-port.md, section 6, item 3).
- Column names follow R so that coefficient names compare directly with R's:
  `(Intercept)`, `x` for a numeric column, `<col><level>` for an indicator
  (for example `z2b`), `<col>TRUE` for a Boolean.
- Column order: intercept, then terms in formula order, levels in level
  order.
- Levels that do not occur after missing-data removal are dropped before
  coding, as `mob()`'s model frame does (`drop.unused.levels = TRUE`,
  `modelparty.R:44`). The dropped levels are recorded in the design and
  shown by `summary()`.

The design keeps, per regressor, its kind and levels, so that `predict`
can rebuild the matrix for new data (R's `xlevels`).

### 1.4 Partitioning variable kinds

The kind of each partitioning variable comes from its dtype, then from
`kinds` if given:

| dtype (narwhals) | kind | levels |
| --- | --- | --- |
| integer, float | numeric | none |
| Boolean | numeric (0/1) | none |
| Enum, pandas ordered categorical | ordered if it is a pandas-like ordered categorical, else nominal | the dtype's categories |
| pandas-like unordered Categorical | nominal | `cat.get_categories()` (the declared order) |
| polars Categorical, String | nominal | code-point sort of the observed values |

- **Boolean as numeric.** partykit treats a logical partitioning variable
  as numeric, since `is.factor` is false (`modelparty.R:309`), so the
  supLM test runs on 0/1 values. The port matches R. The print shows the
  split as `z = false` / `z = true` instead of `z <= 0`.
- **Ordered only when declared.** A variable is ordinal if `kinds` says
  `"ordered"` or if it is a pandas-like ordered categorical. A polars Enum
  is nominal, because Enum fixes a set of categories but does not declare
  them ordinal. *Alternatives:* (a) every Enum is ordered; (b) only `kinds`
  makes a variable ordered; (c) declarations only, from either source.
  **Chosen: (c).** It mirrors R (an ordered factor is ordinal) and never
  infers ordinality from a storage type. Telling the two Enums apart uses
  narwhals' `Implementation` of the frame, the only backend-dependent rule
  in the design.
- An ordered variable needs a declared order: an Enum or categorical dtype,
  or `levels`. An ordered variable whose order would come from sorting
  strings raises `DataError`, because a code-point sort of `"10"` and `"2"`
  or of `"low"`, `"mid"`, `"high"` is wrong.
- `kinds={"school": "nominal"}` turns an integer column into a nominal
  variable, with levels sorted numerically unless `levels` gives them.

### 1.5 Formula library or own parser

- *Alternatives:*
  1. formulaic 1.2.2. A mature Wilkinson-formula implementation, with
     multi-part formulas and narwhals-based input. But it hard-requires
     pandas, which CLAUDE.md rules out at runtime, plus `interface-meta`
     and `wrapt`. Its model matrices are pandas or scipy sparse frames, its
     column names are not R's in every case, and most of its features
     (transforms, stateful transforms, interactions) are outside v0.1.
  2. patsy. In maintenance mode, pandas-oriented, no `|` parts.
  3. A small parser of our own for the grammar in 1.2.
- **Chosen: our own parser**, about 200 to 300 lines with the design
  builder: a tokenizer, the grammar above, and the treatment coding of 1.3.
  It needs no pandas, produces R's column names by construction, and the
  error messages can name the unsupported construct. If transformations or
  interactions are wanted later, formulaic should be re-evaluated first,
  in particular whether its pandas dependency has become optional.

## 2. Data input

`DataInput` is any eager dataframe narwhals supports (polars, pandas,
pyarrow, modin, cuDF), or a `Mapping[str, ArrayLike]` of equal-length 1-D
arrays. Lazy frames (polars `LazyFrame`, DuckDB relations) raise `DataError`
asking for `.collect()`. Columns are read once, converted to numpy, and the
core never sees a dataframe.

**Choice: narwhals (`narwhals.stable.v2`) as a runtime dependency.**

- *Alternatives:*
  1. narwhals. One code path for every backend, dtype inspection
     (Enum categories, Categorical, String, Boolean, null and NaN),
     `from_native(df, eager_only=True)` and construction of result frames in
     the caller's backend. It imports pandas or polars only when the caller's
     object already is one.
  2. Duck typing on `df[col].to_numpy()`. That works for numeric columns,
     but categorical dtypes, category order and null handling differ between
     pandas and polars, so it ends up re-implementing narwhals, worse.
  3. The dataframe interchange protocol (`__dataframe__`). pandas and polars
     are moving away from it in favor of the Arrow PyCapsule interface.
  4. The Arrow PyCapsule interface (`__arrow_c_stream__`). Consuming it in
     Python needs pyarrow, a large dependency.
  5. formulaic's frame handling. It brings pandas (see 1.5).
- **Justification for a third runtime dependency.** narwhals has no
  dependencies of its own, is pure Python, and its stable API is frozen
  with an indefinite maintenance promise, so the pin can be a floor
  (`narwhals>=2.0`). It is imported only in the boundary module
  (`modeltrees/_frame.py`), never in the core or the models, so the
  CLAUDE.md rule "numpy and scipy only" still holds for `src/` outside the
  boundary. narwhals is a required runtime dependency (section 13,
  decision 1), and CLAUDE.md's dependency paragraph says so.
- A `Mapping` input does not go through narwhals: float, integer and
  Boolean numpy arrays are numeric (Boolean as 0/1), and string or object
  arrays are nominal with code-point-sorted levels unless `levels` says
  otherwise.

polars is a dev dependency for the tests (section 9), and an optional one
for users (section 3.3). pandas is never imported by modeltrees.

## 3. The fitted tree

### 3.1 `ModelTree`

One public class for every entry point, `mob` included. It wraps the core's
result (renamed `GrownTree`, section 12) and the design.

```python
class ModelTree:
    formula: str | None              # expanded, e.g. "y ~ 1 | z1 + z2"; None from mob()
    model_name: str                  # "lmtree", "glmtree(binomial)", "negbintree", ...
    control: MobControl
    n_rows: int                      # rows in the input data
    n_used: int                      # rows after missing-data removal
    used_rows: NDArray[np.bool_]     # mask over the input rows
    fitted_node: NDArray[np.int64]   # leaf id per input row, 0 for dropped rows
    response_levels: tuple[str, ...] | None   # binomial or multinomial, else None
    warnings: tuple[RecordedWarning, ...]     # section 7.3

    def node(self, node_id: int) -> NodeInfo
    @property nodes -> tuple[NodeInfo, ...]   # id (preorder) order
    @property leaves -> tuple[int, ...]       # terminal node ids
    @property depth -> int
    def coef_table(self, *, layout="wide", se=None, backend=None) -> Frame
    def predict(self, newdata=None, *, type="response", offset=None,
                unseen="error", x=None, z=None) -> np.ndarray   # 4.2
    def loglik(self) -> LogLik                # value, df, nobs
    def aic(self) -> float
    def bic(self) -> float
    def vcov(self, node_id=None, *, cluster=None, type="HC1",
             cadjust=True) -> dict[int, LeafVcov]
    def summary(self) -> TreeSummary
    def __str__(self) -> str                  # the tree print, 3.4
```

`ModelTree` is immutable after construction. Every method is computed from
the stored nodes. `vcov` and `predict` refit nothing except as stated in
section 5.

### 3.2 Nodes

`node(i)` returns a frozen `NodeInfo` built from the core `Node`:

| field | content |
| --- | --- |
| `id`, `parent`, `kids`, `depth`, `is_leaf` | structure (ids 1-based, preorder, root depth 1, as partykit) |
| `n_rows`, `nobs` | rows in the node, and `w2n` (sum of weights under case weights) |
| `rule` | the path from the root as readable conditions, e.g. `("z1 <= 0.49", "z2 in {a, b}")` |
| `split` | `SplitRule(variable, kind, threshold, left_levels, right_levels)` or None |
| `coefficients` | `dict[str, float]`, NaN for aliased or unidentified |
| `objective` | the node's `objfun` |
| `status` | `"ok"`, `"failed"` or `"no_estfun"` |
| `stop_reason` | the `StopReason` value, e.g. `"not_significant"`, `"fit_failed"` |
| `reason` | why the fit failed or has no scores, or None |
| `p_value` | adjusted p-value of the selected variable, or None |
| `tests` | `dict[str, TestResult(statistic, p_value)]`, or None when not run or failed |
| `extras` | the model report (3.5) |

The core `Node` stays reachable as `NodeInfo.core` for advanced use, marked
unstable. *Alternatives:* expose the core `Node` directly (fewer types, but
it carries variable indices, level codes and split-search arrays that are
internal), or a view object. **Chosen: a frozen view with labels**, so the
public fields can stay fixed while the core changes.

### 3.3 Leaf coefficients table

`coef_table()` returns one row per leaf:

- `layout="wide"` (default): `node`, `n_rows`, `nobs`, `status`,
  `stop_reason`, one column per coefficient (R's `coef(tree)`), then the
  model report columns (3.5).
- `layout="long"`: `node`, `term`, `estimate`, plus `std_error` when
  `se="robust"` (section 5). No p-values: section 5.4.

The frame backend:

| `backend` | result |
| --- | --- |
| None, data was a dataframe | the caller's backend, built with narwhals |
| None, data was a Mapping or arrays | polars if importable, else `dict[str, np.ndarray]` |
| `"polars"`, `"pandas"`, `"pyarrow"` | that backend; `ImportError` if it is missing |
| `"numpy"` | `dict[str, np.ndarray]` |

*Alternatives:* always polars (simple, but forces polars on pandas users and
fails without it); always a dict (predictable, but not what analysts want);
the caller's backend. **Chosen: the caller's backend**, with polars as the
default for array input. A user who passed a frame gets the same kind back.

### 3.4 Printing

`str(tree)` follows `print.modelparty` (`modelparty.R:958-1016`): a header,
the tree with split rules, node sizes and leaf coefficients, and a footer.

```
Generalized linear model tree (family: binomial)

Model formula:
diabetes ~ glucose | pressure + triceps + insulin + mass + age

Fitted tree:
[1] root
|   [2] mass <= 26.3
|   |   [3] age <= 30: n = 120
|   |       (Intercept)     glucose
|   |       -9.1360          0.0571
...
Number of inner nodes:    2
Number of terminal nodes: 3
Number of parameters per node: 2
Objective function (negative log-likelihood): 385.03
```

- Numbers use `digits = 4` significant digits by default, as R's
  `getOption("digits") - 3`; `tree.print(digits=...)` changes it.
- Rows dropped for missing values are reported under the header:
  `n = 940 (60 rows dropped: missing values)`, with the count per column
  (4.1).
- The header lists every partitioning variable with its kind, for example
  `Partitioning variables: age (numeric), severity (ordered), region
  (nominal)`, so that an ordinal or nominal reading (1.4) is never
  implicit.
- A leaf with `status = "failed"` prints `n = 25 [fit failed: <reason>]`
  instead of coefficients. A `no_estfun` leaf prints its coefficients
  followed by `[no scores: <reason>]`, for example `complete separation`
  or `reference class 'lo' absent`.
- Model extras that matter at a glance are shown per leaf: theta for
  negative binomial, the separation status for safe logit when it is not
  `none`.

`summary()` returns a `TreeSummary` with a `__str__` and the tables behind
it as frames: the partitioning variables with their kind and levels, per
node (id, parent, depth, n, nobs, stop reason, status, selected variable,
statistic, adjusted p-value, split), per inner node the full
instability-test table, the leaf coefficients, the model reports, the
dropped levels and rows (with the missing-value count per column), and the
recorded warnings (7.3).

### 3.5 Model reports

Each model reports its extras through one optional method:

```python
class ReportingModel(NodeModel, Protocol):
    def report(self, result: FitResult) -> Mapping[str, float | int | str | bool | None]: ...
```

The keys are fixed per model, so the coefficient table has the same
columns in every leaf:

| model | report columns |
| --- | --- |
| lm | (none) |
| glm | `converged` |
| negbin | `theta`, `theta_se`, `theta_warning`, `converged`, `n_warnings` |
| safelogit | `separation` (`none`, `quasi`, `complete`, or null on error), `n_boundary`, `converged`, `n_warnings` |
| multinom | `fit_status` (`ok`, `single_class`, `reference_absent`, `rank_deficient`), `classes_present`, `converged` |

Full warning texts stay in `NodeInfo.extras["warnings"]` and in `summary()`.
A model without `report` contributes no columns.

### 3.6 logLik, AIC, BIC

As `logLik.modelparty` (`modelparty.R:907-920`):

- `loglik().value` is the sum of the leaf log-likelihoods,
- `df` is the sum of the leaf df plus `dfsplit` (0 or 1, from `control`)
  times the number of inner nodes,
- `nobs` is the sum of the leaf `nobs`.

`aic() = -2 value + 2 df`, `bic() = -2 value + log(nobs) df`.

Per leaf, through the existing `LogLikModel.loglik(result)` protocol, which
no model implements yet:

| model | leaf log-likelihood | leaf df |
| --- | --- | --- |
| lm | `logLik.lm`, per-leaf sigma (docs/decisions.md) | rank + 1 |
| glm | `-objfun` | rank, + 1 for gaussian |
| negbin | `-objfun` for `objective="loglik"`, `negbin_loglik` at the leaf's theta for `"deviance"` | rank + 1 (theta) |
| safelogit | `-objfun` | rank |
| multinom | `-objfun` | number of non-NaN coefficients |

These df reproduce the `loglik_df` of all 44 tree and extension tree
fixtures (checked with a script while writing this document), including the
completely separated safe logit leaf (rank 2) in `sim_separation_safelogit`
and every weighted tree. A failed leaf makes `loglik()`, `aic()` and `bic()` NaN, with a
`NodeFitFailedWarning` naming the leaf, since the tree's likelihood is then
undefined.

## 4. Missing data and levels

### 4.1 Missing data

R's `mob()` builds a model frame with the default `na.action = na.omit`,
which drops every row with `NA` in any variable of the formula, the
weights, the offset or the cluster.

**Rule: complete cases over the used columns.** With `na_action="omit"`
(default), a row is dropped if the response, any regressor, any
partitioning variable, the weight, the offset or the cluster is missing.
Missing means null, and NaN in a float column (R's `is.na` is true for
NaN). Columns that the formula does not use are ignored. With
`na_action="fail"` any missing value raises `DataError` naming the columns
and the count.

- Dropping emits one `MissingDataWarning` per fit, and only when rows are
  dropped, with the number of rows dropped and the count of missing values
  per column, for example `60 of 1000 rows dropped for missing values
  (age: 41, region: 25)`. The drop is also recorded: `n_used`,
  `used_rows`, the print header, `summary()` and the tree's recorded
  warnings (7.3). *Alternatives:* silent as in R, or default to `"fail"`.
  **Chosen: `"omit"` with one warning** (section 13, decision 5): R's
  default, but a dropped row changes the population a subgroup describes,
  so the user hears about it once per fit.
- Infinite values: allowed in numeric partitioning variables (R allows them,
  they sort to the ends). Inf in the response, a regressor, the weight or
  the offset raises `DataError`, as `lm.fit` and `glm.fit` stop.
- Levels are computed after the drop (1.3).

### 4.2 Prediction with missing or unseen values

`predict` returns numpy arrays, not frames (section 13, decision 4): node
ids as `int64` with 0 for "no node", `"link"`, `"response"` and `"prob"`
as `float64` with NaN where there is no prediction, and `"class"` labels as
an object array with None. The types:

| `type` | result |
| --- | --- |
| `"node"` | leaf id per row |
| `"link"` | the linear predictor `X beta + offset`; for multinomial an `(n, K - 1)` matrix of the non-reference classes |
| `"response"` | the inverse link of `"link"`; undefined for multinomial and raises `ValueError` pointing to `"prob"` and `"class"` |
| `"prob"` | multinomial only: the `(n, K)` probability matrix, columns in `response_levels` order |
| `"class"` | multinomial only: the most probable label, ties to the first level, as `which.max` |

`predict` routes each row from the root and stops as soon as it needs a
value:

- A missing value in a split variable on the row's path gives no node
  (node 0, NaN or None in the result). Variables off the path do not
  matter. R routes such rows at random (`node.R:189-194`), which the port
  does not do (docs/tree-growing.md, decision 3).
- A level seen in training but absent from the node at the split (the
  split's index is `NA` there) goes to the kid with the larger `w2n`,
  kid 1 on a tie: the rule the core already uses for zero-weight rows.
- A level never seen in training, in a partitioning variable or a
  categorical regressor: `unseen="error"` (default) raises
  `UnseenLevelError` naming the column and the levels, as R's
  `model.frame(xlev = ...)` stops with "factor has new levels".
  `unseen="null"` gives no prediction for those rows instead.
- A missing regressor or offset gives NaN for `"response"`, `"link"` and
  `"prob"` and None for `"class"`; `"node"` is still returned.

### 4.3 Level order

For the response, regressors and partitioning variables alike, the level
order comes from the first of:

1. `levels={"col": [...]}`. It must list every observed value; extra
   levels are allowed and dropped if unused (1.3). A value outside it
   raises `DataError`.
2. A declared dtype order: Enum categories, a pandas-like categorical's
   categories.
3. For an integer column made categorical by `kinds`: numeric order.
4. Otherwise (String, polars Categorical, string arrays): the code-point
   sort of the observed values, Python's `sorted` on `str`.

Never locale collation, never order of appearance, never polars'
physical category order. R sorts by the locale when it makes factors from
strings, so the README tells users with mixed-case or accented labels to
pass `levels` or use an Enum when R parity matters.

Response levels:

- **binomial** (`glmtree(family="binomial")`): a 0/1 or proportion numeric
  response is used as is (proportions need weights as trials, as in R). A
  Boolean response is false 0, true 1. A categorical response uses R's rule
  through `binomial_response`: the first level is failure, every other
  level success.
- **safelogit**: the same, but proportions raise `DataError`.
- **multinom**: the response must be categorical (or integer with
  `levels`). The first level is the reference class. Unused levels are
  dropped as R's model frame drops them, so a declared class that never
  occurs is not a class; `summary()` lists it. `response_levels` on the
  tree records the order used, and `predict(type="prob")` columns follow
  it.

## 5. Cluster-robust covariances for leaf coefficients

### 5.1 What is computed

The leaf fits ignore clusters (docs/decisions.md), so their standard errors
must come from a sandwich. For a leaf with non-aliased coefficients
`theta` (k of them):

```
V = H^-1 M H^-1,   M = c * sum_g s_g s_g',   s_g = sum_{i in g} estfun_i
```

- `estfun` is the leaf's score matrix, which every model already returns
  (weights included).
- `H` is the Hessian provided by the model (5.2).
- Without clusters, each row is its own cluster. Under case weights the
  meat is `sum_i estfun_i estfun_i' / w_i`, the HC0 meat of the data with
  each row repeated `w_i` times, as `mob_grow_fluctests` divides by
  `sqrt(weights)` (`modelparty.R:236-246`). With clusters the scores are
  summed within clusters first, as in the instability tests.
- Zero-weight rows contribute nothing and do not count as clusters.
- The adjustment `c` follows `sandwich::vcovCL`: `cadjust=True` multiplies
  by `G / (G - 1)`, and `type="HC1"` by `(n - 1) / (n - k)`, where `G` is
  the number of clusters with positive weight in the leaf and `n` the
  leaf's `nobs`. `type="HC0"` omits the second factor. The default is
  `HC1` with `cadjust=True` for every model, the extension models included
  (section 13, decision 7). It equals statsmodels' `cov_type="cluster"`
  with its default correction. sandwich::vcovCL defaults to HC1 only for
  plain lm fits and to HC0 for every other model, glm included
  (vcovCL.R:88-94). We default to HC1 with cadjust for every model, so the
  correction does not change silently when moving between lmtree and
  glmtree, and so the default matches statsmodels' cluster correction. For
  glmtree, negbintree, safelogittree and multinomtree, a user comparing
  against R's vcovCL must pass type="HC0". The docstring and the README say
  this. *Alternatives:* HC0 only (simplest), HC0 for the
  extension models, or statsmodels' naming. **Chosen: sandwich's names, one
  default for every model**, because R is the oracle and one default is
  easier to state.

```python
def vcov(self, node_id: int | None = None, *,
         cluster: str | ArrayLike | bool | None = None,
         type: Literal["HC0", "HC1"] = "HC1",
         cadjust: bool = True) -> dict[int, LeafVcov]

@dataclass(frozen=True)
class LeafVcov:
    node: int
    names: tuple[str, ...]        # all coefficient names
    matrix: FloatArray            # k_all x k_all, NaN rows/cols for aliased or unidentified
    n_clusters: int | None        # None without clusters
    status: Literal["ok", "singular", "unavailable"]
    reason: str | None
    condition: float              # condition number of the scaled Hessian
```

- `node_id=None` gives every leaf. An inner node is allowed; its fit and
  scores are stored too.
- `cluster=None` uses the tree's cluster if it was grown with one, else
  none. `cluster=False` forces the non-clustered sandwich. A column name or
  array gives other clusters (for example a coarser level than the one used
  in the tests); it goes through the same missing-data rule and must not
  be missing on used rows.
- The sandwich is pure linear algebra in a private function,
  `_vcov.sandwich(hessian, estfun, *, weights, cluster, caseweights, type,
  cadjust)`, testable on its own.

### 5.2 The Hessian in the node-model protocol

The protocol already has a `vcov: bool` flag in `fit` and a reserved
`FitResult.vcov` field (docs/decisions.md keeps both); every model now
raises on `vcov=True` (`check_vcov`).

**Design:** `fit(..., vcov=True)` asks the model for its Hessian, which it
returns in a new field `FitResult.hessian`:

- shape `(k_na, k_na)` over the non-aliased columns, in `estfun`'s column
  order;
- defined as `-d(sum_i estfun_i) / d theta`, the derivative of minus the
  summed scores, evaluated as the model's R counterpart evaluates its bread
  (the expected information for GLMs, which is the observed one for
  canonical links). Defining it against `estfun` rather than the objective
  makes the sandwich scale-free: lmtree's scores are `r w x` without
  `1 / sigma^2` and its objective is an RSS, so only this definition gives
  `H = X'WX` and a correct `V` without a dispersion factor.
- `None` when the fit is `failed` or `no_estfun`.

The core never requests it during growth, so trees, timings and fixtures
are unchanged. `vcov()` refits each requested node once with
`estfun=True, vcov=True` on the node's rows and takes the scores and the
Hessian from that refit. It checks that the refit's coefficients equal the
stored ones to a relative tolerance of 1e-10 (NaN in the same places) and
emits a `ModelTreesWarning` naming the node when they differ by more. The
fit is deterministic, but a bit-for-bit check would tie `vcov` to the
exact code path: split search and node fits may take different but
equivalent paths (docs/decisions.md, fast and reference paths agree to
about 1e-15 relative), and the BLAS may differ between growth and refit. *Alternatives:* (a) request the Hessian at every
node fit during growth and store it; (b) a separate `hessian(result, y, x,
...)` method; (c) refit on demand with the existing flag. **Chosen: (c).**
(a) costs time in every tree for a feature most trees do not use; (b)
needs the model to keep or recompute its working quantities from outside
the fit. A refit is what partykit's `refit.modelparty` does.

`FitResult.vcov` stays reserved for `vcov = "info"` and `"sandwich"` in the
instability tests (out of scope), which can later use the same Hessian.

### 5.3 Which models can supply it now

All five, from quantities they already compute:

| model | `H` | source |
| --- | --- | --- |
| `LmModel` | `X'WX` | the node's x and weights |
| `GlmModel` | `X' diag(working weights) X / dispersion` | `GlmFit.weights` of the last IRLS iteration, the dispersion already used for the scores; R's `summary.glm` bread |
| `NegBinModel` | as GlmModel, dispersion 1, at `theta_irls` | the last glm.fit call, whose scores the model returns |
| `SafeLogitModel` | as GlmModel with binomial | `None` for complete separation (`no_estfun`); with quasi-separation it is near singular, and `LeafVcov.status` says so |
| `MultinomModel` | `multinom_hessian(prob, x, w)` over present classes and non-NaN columns | already used by Newton's method |

Each needs only a few lines to return it; the fit-only paths of split
search are untouched.

Two limits, documented with `vcov`:

- **Negative binomial** covariances are conditional on theta, as
  `vcov(glm.nb)` is. Under the NB2 model beta and theta are orthogonal in
  the expected information, so this is asymptotically right, but small
  leaves understate the uncertainty.
- **Singular or near-singular `H`.** The Hessian is scaled by its diagonal
  and factored by Cholesky. If that fails, `status = "singular"` and the
  matrix is NaN. `condition` is always reported. If `G <= k`, the meat has
  rank below `k` and the robust variance is unreliable:
  `TooFewClustersWarning` names the leaf (the same class the tree uses).
  `G = 1` gives NaN, as `G / (G - 1)` is infinite.

### 5.4 Post-selection inference

The leaves were chosen by tests on the same data. Standard errors from any
covariance, robust or not, are conditional on the tree and too small for
inference about the subgroups. v0.1 therefore reports standard errors
(`coef_table(se="robust")`) but no p-values or confidence intervals. The
docstring and the README say why and point to sample splitting (grow on
one part, estimate on the other) as the honest route. Model-based
(non-robust) standard errors are deferred: they need a per-model
dispersion convention (section 13, decision 6).

## 6. Parallel split search

Option 2 of docs/performance.md: workers fit disjoint chunks of candidates
with the same code.

- `n_jobs=None` or `1`: serial, today's code path, unchanged.
  `n_jobs=k > 1`: up to k worker processes. `n_jobs=-1`:
  `os.process_cpu_count()`.
- **What runs in parallel.** The exact candidate fits of `find_split`:
  every numeric candidate for models without a batch scorer (glm, negbin,
  safelogit, multinom), and the nominal partitions when there are many
  (10 levels give 511). lm's batch path refits only a handful, so it stays
  serial. Instability tests, node fits and pruning stay serial.
- **Threshold.** A search is parallel only when the number of candidates
  times the node's rows exceeds a private constant (initially 2e6, tuned
  with `bench_tree.py`). It never changes results, only time.
- **`restart=False`** makes each candidate start from the previous one's
  coefficients, a serial chain. Split search then runs serially whatever
  `n_jobs` says. When `n_jobs > 1` is ignored for this reason, the entry
  point emits one `ModelTreesWarning` per fit (not per node) saying so
  (section 13, decision 10). Falling below the size threshold is not
  warned about: it is the normal case for small nodes.
- **Mechanics.** One `concurrent.futures.ProcessPoolExecutor` per tree,
  created at the first parallel search and shut down when the tree is
  grown, with the `forkserver` start method on Linux (the default from
  Python 3.14; `fork` with a running BLAS thread pool is unsafe) and
  `spawn` elsewhere. Models must be picklable; the five built-in ones are.
  Workers get `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and
  `MKL_NUM_THREADS` set to 1 in their environment before the worker
  process starts, so that k workers do not oversubscribe the cores
  (docs/performance.md, Step 1a). The variables are set in the environment
  the forkserver (or spawned interpreter) inherits, not in a pool
  `initializer`: a BLAS library reads them once, when numpy loads it, and
  the forkserver may already have imported numpy before an initializer
  runs. A test checks that a worker reports one BLAS thread. The node's arrays
  go to each worker once per search; candidates are cut into about
  `4 k` contiguous chunks for load balance.
- **Reproducibility rule.** The results with any `n_jobs` must be
  identical to `n_jobs=1`: the same tree, splits and ids, and every stored
  number (coefficients, objectives, split-search objectives, statistics)
  equal bit for bit. This holds by construction for the assembly: each
  chunk returns its objectives by candidate index, the full vector is
  assembled in candidate order, and the winner is taken by the same
  first-minimum rule over the same vector, so neither chunking nor
  completion order can matter. Failed candidates and their count are
  assembled the same way. What construction cannot promise is that a fit
  gives the same bits on one BLAS thread in a worker as on several in the
  parent. The LAPACK least-squares path of Step 1d stays on one thread at
  these sizes, and the prototype measured bit-identical objectives, but
  this depends on the BLAS. So the guarantee is stated as: identical to
  `n_jobs=1` run with single-threaded BLAS, which is also the setting
  docs/performance.md recommends for large fits. The tests run that way
  (section 9).
- Errors in a worker are re-raised in the parent with the worker's
  traceback attached as the cause. Warnings emitted in split-search fits
  are none (those fits are quiet in every model).

*Alternatives:* threads (no copies, but IRLS is mostly Python-level work
under the GIL, so gains are small unless the user runs a free-threaded
build; worth revisiting then), joblib (another dependency, same processes
underneath), parallelism over nodes (the two children of a split are
independent, but the tree is usually dominated by the root search).
**Chosen: processes over candidates**, measured at 2.9 to 3.9 times on 4 to
10 workers.

## 7. Errors and warnings

### 7.1 Exceptions

```
ModelTreesError(Exception)
├── FormulaError(ModelTreesError, ValueError)     parse errors, unsupported constructs; has .position
├── DataError(ModelTreesError, ValueError)        missing columns, wrong dtypes, Inf, negative weights,
│   │                                             missing values with na_action="fail", lazy frames
│   └── UnseenLevelError(DataError)               predict-time levels not seen in training
└── FitError(ModelTreesError, ValueError)         a node fit that cannot be completed and is not
    ├── GlmFitError(FitError)                     turned into a failed fit (lmtree, glmtree)
    └── NegBinError(FitError)
NotImplementedError                               options out of the v0.1 scope (MobControl, families)
TypeError                                         wrong argument types, control and control options together
```

Every exception subclasses `ValueError` where it does today, so existing
`except ValueError` code keeps working.

**Rule for core exceptions: nothing is wrapped.** Exceptions the core raises
on purpose derive from a public class (`GlmFitError` and `NegBinError`
become `FitError` subclasses by changing their base), so they reach the
user with their own type, message and traceback. Internal control-flow
exceptions (`InstabilityTestError`, `TooFewClustersError`) are always
caught in the core and must never escape; a test asserts it. Anything else
(`TypeError`, `IndexError`, `LinAlgError`, assertion errors) is a bug and
propagates unchanged. *Alternatives:* catch core exceptions at the boundary
and re-raise public ones with `from exc` (two types per error, and every
new core error must be mapped). **Chosen: shared base classes**, which need
no mapping and keep tracebacks short.

The frontend validates its inputs (columns, dtypes, lengths, weights) before
calling the core, so the core's own `ValueError` checks are a second line;
if one fires from a frontend call it is a frontend bug.

### 7.2 Warnings

```
ModelTreesWarning(UserWarning)
├── NodeFitFailedWarning        a node fit failed; the node is terminal
├── TooFewClustersWarning       a node has at most k clusters (tests) or G <= k (vcov)
├── MissingDataWarning          rows were dropped for missing values (4.1), counts per column
└── GlmFitWarning               glm.fit and glm.nb warnings, with R's messages
```

The existing three classes get the common base; `MissingDataWarning` is
the one new class. `ModelTreesWarning` itself is emitted for the ignored
`n_jobs` (6) and a refit that does not reproduce the stored coefficients
(5.2).

### 7.3 Recording

The entry points grow the tree inside `warnings.catch_warnings(record=True)`
and store each warning on the tree as `RecordedWarning(category, message,
count)` (identical messages merged), then re-emit each distinct warning
once, so a tree with 89 non-convergence warnings shows one line with the
count, not 89. Models that already keep warnings per fit (negbin,
safelogit) also have them in `NodeInfo.extras["warnings"]`, which
identifies the node. A user filter (`warnings.simplefilter("error")`)
still applies to the re-emitted warnings.

## 8. Public API surface

### 8.1 `modeltrees`

```python
__all__ = [
    # entry points
    "lmtree", "glmtree", "negbintree", "safelogittree", "multinomtree", "mob",
    # results and settings
    "ModelTree", "NodeInfo", "SplitRule", "TestResult", "LogLik", "LeafVcov",
    "TreeSummary", "RecordedWarning", "MobControl",
    # errors and warnings
    "ModelTreesError", "FormulaError", "DataError", "UnseenLevelError", "FitError",
    "ModelTreesWarning", "NodeFitFailedWarning", "TooFewClustersWarning",
    "MissingDataWarning", "GlmFitWarning",
    # the reference implementation switch (docs/decisions.md)
    "reference_path",
    "__version__",
]
```

`hello()` is removed.

### 8.2 `modeltrees.models`

For `mob()` and custom models: `LmModel`, `GlmModel`, `NegBinModel`,
`SafeLogitModel`, `MultinomModel`, `NodeModel`, `FitResult`, `FitStatus`,
`failed_fit`, and the optional protocols `PredictingModel`, `LogLikModel`,
`ReportingModel`, `BatchSplitModel`.

### 8.3 Private

Everything else, whatever its module name: `tree`, `split_search`,
`instability`, `pvalues`, `pruning`, `node_model` internals,
`models.glm_fit`, `models.families`, `models._qr`, the frontend modules
(`_formula`, `_frame`, `_design`, `_predict`, `_vcov`, `_report`). The
modules are not renamed now, because tests import them; `__all__` and the
documentation define the public set.

### 8.4 Stability for v0.1

- 0.x versioning: names in 8.1 and 8.2 may still change in 0.2. Where it is
  cheap, a renamed name keeps working for one minor release with a
  `DeprecationWarning`.
- Numerical results are the stable part: every R fixture match is a
  regression test, and a change that alters a tree needs a docs/decisions.md
  entry.
- Provisional, and marked so in the docs: the `NodeModel` protocol and its
  optional protocols, the report columns, and the safe logit and
  multinomial models' specifications (docs/extensions-spec.md).
- Printed output is for humans and not stable; tests compare `summary()`'s
  tables, not text.

## 9. Testing without new R fixtures

All through pytest, using `tests/tree_fixtures.py` for the existing array
path. polars joins statsmodels in the dev group.

1. **Formula path equals array path on every fixture.** For each of the 32
   tree fixtures and 12 extension tree fixtures (the non-oracle
   `sim_gaussian_lmtree_zero_weights` included, against the array path only): read the CSV with its
   `.schema.json` into a polars frame (factor columns as `pl.Enum` with the
   schema's levels; columns of type `ordered` declared with `kinds`), call
   the entry point with the fixture's `formula`, weights, cluster and
   `control`, and the offset as a precomputed column (`log(exposure)`
   becomes a frame column, since formulas have no functions). Then:
   - the design arrays equal `tree_fixtures.inputs()` and `variables()`
     exactly (`np.array_equal`), and the coefficient names equal the
     fixture's `coefficient_names`;
   - the grown tree equals the array-path tree exactly: structure, splits,
     every stored number bit for bit (same arrays, same code);
   - and so still matches R, by the existing tests.
   The same with a pandas frame for a subset (pandas as a test-only
   dependency is already present through statsmodels), and with a
   `dict` of numpy arrays.
2. **Formula and design units.** The grammar (each supported form and each
   `FormulaError`, with positions), one-part formulas, intercept removal,
   treatment coding with and without intercept, R's names, Boolean
   columns, dropped unused levels, the level-order rules of 4.3 (including
   a mixed-case example that R's locale would sort differently), and
   missing-data removal against a frame with the rows deleted by hand.
3. **`fitted_node` and `predict(type="node")` on the training data equal
   the fixture's `fitted_node`**, which is R's. This is an R check of
   routing with no new fixture.
4. **Predictions equal hand-computed leaf fits.** For each fixture tree
   and each leaf: fit the leaf's model on its rows directly, compute
   `X beta + offset` and the inverse link (softmax for multinomial), and
   compare `predict` at `rtol = 1e-12`. New-data routing: rows at exact
   break values (go left), levels absent from a node (larger kid), unseen
   levels (error and `"null"`), missing split variables on and off the
   path, failed and `no_estfun` leaves.
5. **logLik, AIC, BIC against R.** `loglik().value` and `df` equal each
   fixture's `summary.loglik` and `summary.loglik_df` (`rtol = 1e-8`), for
   lm (per-leaf sigma), every glm family, negbin, safe logit and
   multinomial. These are R values already in the fixtures.
   `quine_negbin_intercept_deviance` covers the deviance objective, where
   the loglik (-547.58) is not `-objfun` (165.37).
6. **Cluster-robust covariances against statsmodels, single node.** Grow
   one-node trees (`maxdepth=1`) on the fixture datasets that have a
   `school` column and compare `vcov(1)` with statsmodels
   (`cov_type="cluster"`, default correction, which is `HC1` with
   `cadjust`): `OLS` for lm, `GLM` for the three families, `GLM` with
   `NegativeBinomial(alpha = 1 / theta)` at our theta for negbin, `Logit`
   for safe logit, `MNLogit` (reordered to class-major) for multinomial.
   Tolerance from the coefficient agreement (about 1e-6 relative). Also:
   `HC0` without clusters against statsmodels' `HC0`; frequency weights
   against replicated rows (same clusters); `cluster=False` on a clustered
   tree; a separated safe logit leaf (`unavailable`); `G <= k` (warning)
   and `G = 1` (NaN).
7. **The Hessians.** For every model and every node fit fixture, `H`
   against a central finite difference of the summed scores, and `H^-1`
   against R's model-based covariance where the fixtures record it, else
   against statsmodels' `cov_params()` (scaled to the same dispersion).
   The refit check of 5.2 (relative tolerance 1e-10) on every fixture
   leaf, and its warning on a model whose refit differs.
8. **`n_jobs`.** With `OPENBLAS_NUM_THREADS=1` set for the test process,
   every glm, negbin, safe logit and multinomial fixture tree grown with
   `n_jobs=2` and `4` (threshold lowered by a test hook so the small
   fixtures run in parallel) equals the serial tree bit for bit, including
   `SplitSearch.objective` and failed-candidate counts. `restart=False`
   with `n_jobs=4` runs serially and emits one `ModelTreesWarning`. A
   worker reports one BLAS thread.
9. **Errors and warnings.** Each exception class from the frontend; the
   internal exceptions never escape; recorded and merged warnings; one
   `MissingDataWarning` with per-column counts; a user `"error"` filter.

R fixtures that would still be worth generating later:

- **predict output**: `predict(tree, newdata, type = "node" / "response" /
  "link")` from glmtree and lmtree, and `predict.multinom` probabilities
  per leaf, on new data with levels absent from some nodes (R gives `NA`
  kids, so only rows R routes deterministically).
- **reference-absent child**: a multinomial tree in which a candidate or
  leaf lacks the reference class (docs/extensions-port.md, 7.3, has no R
  tree with one).
- **sandwich::vcovCL** on the leaf fits of a clustered lmtree, glmtree
  (binomial, poisson), glm.nb and multinom. The fixture passes `type` and
  `cadjust` explicitly, never relying on vcovCL's defaults (which differ
  between lm and every other model, 5.1), and generates both
  `type = "HC0"` and `type = "HC1"` (with `cadjust = TRUE`) for every
  model.
- **formula features**: a tree with a factor regressor (treatment coding
  and its names), `y ~ 0 + f | z`, a logical partitioning variable, and a
  model frame with `NA` rows (na.omit).

## 10. Documented limitations (README)

The README must state:

1. **Safe logit and multinomial are this project's designs.** Rusch's
   thesis names safelogitModel and mlogitModel but gives no code, and the
   mobtools package is unavailable. Their rules (separation threshold,
   terminal separated nodes, degenerate classes, rank-deficient failures)
   are provisional (docs/extensions-spec.md, docs/decisions.md). Their
   oracle is this project's R code, not a published implementation.
2. **Negative binomial trees do not detect changes in overdispersion
   alone.** Only the regression coefficients are tested; theta is estimated
   per node but has no score column, as in the thesis.
   `sim_negbin_theta_only` does not split. Leaf covariances are
   conditional on theta.
3. **BLAS threads.** For large nodes, and whenever several fits run at
   once (`n_jobs`, or modeltrees inside a parallel job), set
   `OPENBLAS_NUM_THREADS=1` (or `OMP_NUM_THREADS=1`) before Python
   starts. With the default threads the reference path ran 2.3 times slower
   at n = 20000 (docs/performance.md, Step 1a). `n_jobs` workers pin
   themselves; the parent process is the user's to set.
4. **Split search cost.** Only the selected variable is searched at each
   node, but every candidate fits both sides. lmtree is near-linear in the
   node size (batch scorer); the other models are quadratic: about 16 s for
   a binomial glmtree at n = 5000 and about 3.5 minutes at n = 20000 on one
   core (docs/performance.md). negbin fits a whole glm.nb (Poisson fit plus
   alternations) per side and is several times slower than glm; multinomial
   fits about 5 Newton iterations per side. `n_jobs` gives 3 to 4 times on
   12 cores. Subsampling the candidates is not offered, because it changes
   split points.
5. **Inference.** Leaf standard errors are conditional on the selected
   tree; no p-values are reported for leaf coefficients (5.4).
6. **Scope.** Binary splits only; no surrogate splits or random routing of
   missing values; `vcov = "opg"` only in the tests; options needing R's
   random numbers (`mtry`, `breakties`, `ordinal = "L2"` / `"max"`) are not
   available; gaussian, binomial and poisson with canonical links only.
7. **Differences from R in data handling.** Strings are sorted by code
   point, not the locale (pass `levels`); ordered categoricals used as
   regressors are treatment-coded, not polynomial-coded; rows with missing
   split variables are not predicted rather than routed at random; zero
   weights act as deleted rows (docs/decisions.md).
8. **Unrelated to `py_mob`** on PyPI (monotonic binning).

## 11. Worked example

```python
import polars as pl
import modeltrees as mt

df = pl.read_csv("survey.csv").with_columns(
    pl.col("school").cast(pl.String),
    pl.col("arm").cast(pl.Enum(["control", "treatment"])),
    pl.col("severity").cast(pl.Enum(["none", "mild", "moderate", "severe"])),
)
tree = mt.glmtree(
    "improved ~ arm | age + sex + severity + region",
    df, family="binomial", cluster="school",
    kinds={"severity": "ordered"}, alpha=0.01, n_jobs=4,
)
print(tree)
tree.coef_table(layout="long", se="robust")   # polars frame
tree.predict(new_df, type="response")          # numpy array
```

## 12. Changes to the core this design needs

Listed so that the implementation tasks can be cut from them:

1. `FitResult.hessian` and each model's response to `vcov=True` (5.2, 5.3);
   `check_vcov` removed.
2. `LogLikModel.loglik` for the five models (3.6); `PredictingModel.predict`
   for the five models, kinds `"link"`, `"response"`, `"prob"` and
   `"class"` (4.2); `ReportingModel.report` (3.5).
3. `n_jobs` through `mob` into `find_split` (6), with the private threshold
   and a test hook.
4. Exception and warning base classes (7); `GlmFitError` and `NegBinError`
   re-based on `FitError`.
5. The core's `ModelTree` renamed `GrownTree` (internal); the public
   `ModelTree` wraps it. Tests that use the core class are updated in the
   same change.
6. New frontend modules: `_formula` (parser), `_frame` (narwhals boundary),
   `_design` (columns, levels, missing data, coding), `_predict`,
   `_vcov`, `_report`, `_api` (entry points and `ModelTree`).
7. `pyproject.toml`: `narwhals>=2.0` in `dependencies`, polars in the dev
   group (section 13, decision 1; CLAUDE.md already says so).

## 13. Decisions on the open questions

The questions this design left open, as decided. Sections 1 to 12 are
written to match.

1. **narwhals is a required runtime dependency**, imported only in
   `modeltrees/_frame.py`, the dataframe boundary. It is not an optional
   extra. The core and the models stay numpy and scipy only (section 2).
2. **Ordinal partitioning variables only when declared**: `kinds="ordered"`
   for the column, or a pandas-like ordered categorical. A polars Enum is
   nominal unless `kinds` says otherwise (1.4). `print` and `summary` list
   the kind of every partitioning variable (3.4), so the reading is never
   implicit.
3. **A declared pandas category order is honored** for an unordered pandas
   categorical, as for every declared order (4.3), even though the same
   data as a polars Categorical would be sorted.
4. **`predict` returns numpy arrays** (4.2): node ids with 0 for "no
   node", floats with NaN, labels as an object array with None.
5. **`na_action="omit"` by default**, with one `MissingDataWarning` (a
   `ModelTreesWarning` subclass) when rows are dropped, giving the count
   per column. The drop is also recorded on the tree and printed (4.1,
   3.4, 7.2).
6. **Model-based standard errors are deferred.** v0.1 has sandwich
   covariances only (5.4).
7. **`vcov` defaults to `type="HC1"` with `cadjust=True` for every model.**
   sandwich::vcovCL defaults to HC1 only for plain lm fits and to HC0 for
   every other model, glm included (vcovCL.R:88-94). One default keeps the
   correction from changing silently between lmtree and glmtree and
   matches statsmodels' cluster correction. A user comparing a glmtree,
   negbintree, safelogittree or multinomtree leaf with R's vcovCL must pass
   `type="HC0"`; the docstring and the README say this. The R fixture
   passes `type` and `cadjust` explicitly and generates both HC0 and HC1
   for every model (5.1, 9).
8. **Multinomial `predict(type="response")` is undefined and raises**,
   pointing to `type="prob"` (the probability matrix) and `type="class"`
   (the labels) (4.2).
9. **The `.` shorthand stays out of v0.1** (1.2).
10. **One `ModelTreesWarning` when `n_jobs > 1` is ignored because
    `restart=False`** (6). Running serially below the size threshold is
    not warned about.
