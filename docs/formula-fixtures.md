# R fixtures for the formula features

Task 6i. Decisions taken while generating the formula-feature fixtures
(task A) and while comparing the public layer with them (task B), and what
partykit 1.3-0 does with each feature. The generator is
`r/fixtures/formula.R`, called last by `r/generate_all.R`; the files are in
`tests/fixtures/formula_trees/` and `tests/fixtures/data/sim_formula_*`, and
tests/fixtures/README.md lists them.

## 1. Cases

Each fixture records its case number in `case`.

| case | fixture | formula | what it pins |
| --- | --- | --- | --- |
| 1 | `sim_formula_factor_lmtree_factor` | `y_gauss ~ x + f \| z1 + z2` | treatment coding of a factor declared d, b, a, c: `(Intercept)`, `x`, `fb`, `fa`, `fc` |
| 2 | `sim_formula_factor_glmtree_binomial_factor` | `y_bin ~ x + f \| z1 + z2` | the same for glmtree binomial, with a factor response declared stay, leave (stay is the failure) |
| 3 | `sim_formula_factor_lmtree_noint` | `y_cell ~ 0 + f \| z1 + z2` | no intercept: `fd`, `fb`, `fa`, `fc` |
| 4 | `sim_formula_logical_lmtree_logical` | `y_lreg ~ x + b \| z1 + zb` | a logical regressor with an intercept (`bTRUE`); a logical partitioning variable tested but not split on |
| 4 | `sim_formula_logical_lmtree_logical_noint` | `y_lreg ~ 0 + b + x \| z1 + zb` | the logical regressor without an intercept (`bFALSE`, `bTRUE`, `x`) |
| 4 | `sim_formula_logical_lmtree_logical_split` | `y_lsplit ~ x \| z1 + zb` | a planted split on the logical `zb`: **R stops** (section 3) |
| 4 | `sim_formula_logical_numeric_lmtree_split` | `y_lsplit ~ x \| z1 + zb` | the same data with `zb` stored as the double 0/1: R's split `zb <= 0` |
| 5 | `sim_formula_ordered_lmtree` | `y ~ x \| zo + zn + z1` | an ordered `zo` declared low, mid, high, top and a nominal `zn` with five levels declared v, r, t, p, s |
| 6 | `sim_formula_na_lmtree_weights` | `y ~ x + f \| z1 + zf`, `weights = w` | missing values in the response, a regressor, a numeric and a factor partitioning variable and the weights |
| 7 | `sim_formula_factor_lmtree_onepart` | `y_mean ~ z1 + z2` | lmtree's expansion to `y_mean ~ 1 \| z1 + z2` |
| 7 | `sim_formula_factor_glmtree_poisson_onepart` | `y_count ~ z1 + z2` | glmtree poisson's expansion to `y_count ~ 1 \| z1 + z2` |

Every tree uses the default `mob_control()` settings, written out in full
in `r_call` and `control` as in the other tree fixtures. Each dataset has a
planted split, and the generator (`check_formula_trees`) stops unless the
trees show it: cases 1 to 4 and 7 split on `z1` only, the numeric
companion of case 4 splits on `zb` at the root, case 5 splits on `zo` and
`zn`, case 6 on `z1` and `zf` and drops the levels `f = e` and `zf = q`.
It also checks the coefficient names above and R's error message.

## 2. Format

1. **Directory.** `formula_trees/`, so that the suites that glob `trees/`
   and `extension_trees/` (and assert their counts) are unchanged.
2. **Tree format.** The files are made by `tree_fixture()` of
   `r/fixtures/trees.R`, with the instrumented `mob_partynode` and the
   identity check against plain partykit, so they hold everything a tree
   fixture holds. The fields added by `formula.R` are listed in the
   README: `case`, `r_outcome`, `expanded_formula` (`tree$info$formula`,
   deparsed), `na_action`, `model_frame`, `print` and `predict`.
3. **Datasets.** Five CSVs with schema files, in the existing format, with
   two extensions that no other dataset uses: a `logical` column type,
   written `TRUE` and `FALSE`, and missing values, written `NA`, allowed
   only in a dataset built with `allow_na = TRUE` (`sim_formula_na`). The
   writer's other output is unchanged. `sim_formula_logical_numeric` is
   `sim_formula_logical` with `zb` replaced by `as.numeric(zb)`.
4. **Generated last.** The formula datasets and trees are built after the
   p-value fixture, just before the README, so that the random state seen by
   every earlier generator is what it was. Each dataset and each set of new
   rows sets its own seed (`set_fixture_seed`).
5. **Row indices.** As for the fixtures on reduced data, `rows` and
   `fitted_node` refer to the rows of the model frame, the input rows
   without `na_action.removed_rows`. `removed_rows`, `predict.regressor_rows`
   and `predict.partitioning_rows` are 0-based indices into the data CSV.
   `n` is the number of model frame rows; `nobs` the sum of the weights.

## 3. What R did

1. **Coefficient names.** `model.matrix` names an indicator
   `<variable><level>` in the declared level order, so case 1 has `fb`,
   `fa`, `fc` (reference `d`), case 3 all four levels `fd`, `fb`, `fa`, `fc`,
   a logical regressor `bTRUE`, and without an intercept `bFALSE`, `bTRUE`,
   `x` (the logical, the first factor-like term, gets both levels).
2. **A logical partitioning variable is numeric in the tests and
   categorical in the split search, and R stops when it would split on
   one.** `mob_grow_fluctests` branches on `is.factor(zi)`
   (modelparty.R:309), which is false for a logical, so the supLM test runs
   on the 0/1 order: in case 4 (`sim_formula_logical_lmtree_logical`) `zb`
   is tested at every node and never selected, and its statistics and
   p-values are in the fixture. `mob_grow_findsplit` branches on
   `is.numeric(zselect)` (modelparty.R:390), which is false for a logical
   too, so the selected logical goes through the factor branch:
   `factor(zselect)` has levels FALSE and TRUE, the single candidate
   partition gets objective 1 without fitting (the quirk of
   docs/decisions.md), and the split is an `index` over the logical's
   levels with `breaks = NULL` (`olevels` is NULL, so the index is built on
   an empty vector, modelparty.R:497-500). `kidids_split(sp, data = z)`
   (modelparty.R:596) then requires an integer variable for a split without
   breaks (split.R:140-141) and stops:
   `variable 2 is not integer` in `kidids_split(sp, data = z)`, `2` being
   `zb`'s position among the partitioning variables. The error is not
   caught (only the instability tests are in a `try()`), so `lmtree()`
   returns nothing. `sim_formula_logical_lmtree_logical_split` keeps the
   call, the message and the call that raised it, and names
   `sim_formula_logical_numeric_lmtree_split` as its oracle.
3. **The numeric companion.** With `zb` as the double 0/1, R splits at the
   break 0 and prints `zb <= 0` and `zb > 0` (the fixture's `print`); the
   split's `breaks` is 0 and `kind` numeric. This is the tree docs/api-design.md
   section 1.4 describes for a Boolean partitioning variable ("numeric
   (0/1)", printed `zb = false` / `zb = true`).
4. **Missing values.** `mob()` builds its model frame with the default
   `na.action = na.omit` and `drop.unused.levels = TRUE`. In case 6 the
   model frame drops the 40 rows with a missing value in `y`, `x`, `z1`,
   `zf` or `w` (two rows have two), its `na.action` attribute has class
   `omit`, and the generator checks that the model frame's row names are
   the input rows without them. The levels are dropped after the rows:
   `f = e` and `zf = q` occur only in dropped rows and are not in the model
   frame (`model_frame.dropped_levels`), so there is no `fe` coefficient
   and `zf` has three levels in the split.
5. **One-part formulas.** `tree$info$formula` is `y_mean ~ 1 | z1 + z2` and
   `y_count ~ 1 | z1 + z2`; each node has the single coefficient
   `(Intercept)`.
6. **No warnings** were raised by any of the eleven calls.

## 4. The new rows

1. **Construction.** For each tree, 12 new rows: row i takes the
   regressors, response and weights of training row `regressor_rows[i]` and
   the partitioning variables of training row `partitioning_rows[i]`, both
   drawn (with the fixture's seed) from the rows the tree used. Mixing two
   rows makes the regressors at a leaf differ from that leaf's training
   rows, so the response tests the rebuilt design, while the partitioning
   variables keep a combination that occurred, so every row is routed
   deterministically (checked with `route_rows` of `predict.R`).
2. **A subset of the levels.** Both rows are drawn from the used rows that
   avoid one level per factor (`excluded_levels`): the first level of a
   factor regressor (the reference, or the first indicator without an
   intercept), the last level of a factor partitioning variable, and FALSE
   for a logical regressor. `levels_present` lists what remains. The
   factor columns keep the data's declared levels, as R requires; in case 6
   these include the levels the model frame dropped (`e`, `q`), which R
   accepts: `predict.party` sends the frame through `model.frame(xlev =
   ...)`, which checks values, not declared levels.
3. **Expected values.** `predict(tree, newdata, type = "node")` and
   `type = "response"`. The generator checks the node against the routing
   and the response against the model matrix of the new rows times the
   leaf's coefficients and the inverse link (1e-12, observed at most 2.6e-16).

## 5. Reproducibility

`Rscript r/generate_all.R` was run twice; the two runs gave byte-identical
output (sha256 of every file under `tests/fixtures/`), and no file that
existed before changed except `tests/fixtures/README.md`, which gained the
convention for logical and missing values, the five datasets in the data
table and the section on the formula tree fixtures.
