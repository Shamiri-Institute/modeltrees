# Tree growing and pruning

This document maps the MOB tree-growing loop of partykit 1.3-0 to the Python
port in `src/modeltrees/tree.py`, `split_search.py` and `pruning.py`. It
records the decisions made, the R quirks replicated or deliberately not
replicated, and how the port compares with the R fixtures.

Line numbers are `file:line` in `reference/partykit/R/`. `modelparty.R`
means `reference/partykit/R/modelparty.R`.

## 1. What is ported where

| R | Python |
| --- | --- |
| `mob_partynode()` setup: `minsize` via `as.integer`, `w2n` (`modelparty.R:156-176`) | `tree._Grower.__init__`, `split_search.w2n` |
| `mob_grow()` (`modelparty.R:516-623`) | `tree._Grower.grow` |
| default `minsize`, set once at the root with `<<-` (`:533`) | `tree._Grower.grow` |
| stopping rules: `minsize`, `maxdepth` (`:536-547`) | same |
| `try(mob_grow_fluctests(...))` and failure handling (`:550`, `:569-573`) | `instability.fluctuation_tests`, `InstabilityTestError` caught |
| Bonferroni/Sidak adjustment, `alpha` decision, `p.value` (`:552-561`) | `pvalues.adjust_pvalues`, `tree._Grower.grow` |
| `mob_grow_findsplit()` (`:384-511`) | `split_search.find_split_reference`, and `find_split`, the faster equivalent used by `mob()` (docs/performance.md) |
| `mob_grow_getlevels()` (`:701-723`) | `split_search.getlevels` |
| `kidids_split()` (`split.R:132-160`) | `split_search.kid_ids`, `tree._route` |
| preorder node ids (`:600-615`) | `tree._Grower.grow` |
| `mob_prune()` (`:625-676`) | `tree._prune`, `tree._renumber` |
| `mob()` prune rules "aic", "bic" (`:101-119`) | `pruning.mob_prune_rule` |
| `lmtree()` prune rules (`lmtree.R:12-29`) | `pruning.lm_prune_rule`, `LmModel.prune_rule` |
| terminal node per row, `fitted_node()` (`mob()`, `:126-131`) | `ModelTree.fitted_node` |
| `mob_control()` (`:726-778`) | `tree.MobControl` |

The entry point is

```python
mob(model, y, x, z, *, weights=None, offset=None, cluster=None, control=None)
```

`model` is any `NodeModel`. `x` is the regressor matrix with its intercept
column, or `None` for an intercept-only model. `z` maps each partitioning
variable's name to its kind (`"numeric"`, `"ordered"` or `"nominal"`) and
its values. It accepts a `Variable`, a dict, or a tuple `(kind, values)` or
`(kind, values, levels)`. Factors are given either as integer codes with
`levels`, or as labels, which are mapped through `levels` or else through the
sorted unique values. There is no formula or dataframe layer. `mob` returns a
`ModelTree` holding the root `Node`, the nodes in id order, the `minsize`
used, and the terminal node id of every row.

Each `Node` keeps its rows, `nobs`, the `FitResult`, the instability tests,
the adjusted p-value of the selected variable, the split search with every
candidate and objective, the split, the kids and a `StopReason`. The stop
reasons are the same as the fixtures' `stop_reason`, plus `fit_failed`
for a node whose own fit failed, which partykit cannot produce
(docs/model-interface.md, section 4.2).

## 2. Options

Ported, as in `mob_control()`: `alpha`, `bonferroni`, `minsize`, `maxdepth`,
`trim`, `dfsplit`, `prune` (`"aic"`, `"bic"`, `"none"` or a function),
`restart`, `caseweights`, `numsplit` (`"left"`, `"center"`) and
`catsplit = "binary"`. Weights and clusters are arguments of `mob`.

`MobControl` raises `NotImplementedError` for the options excluded by
docs/decisions.md: `mtry < Inf`, `breakties = TRUE`, `ordinal = "L2"` or
`"max"`, `vcov = "info"` or `"sandwich"`, `parm`, and
`catsplit = "multiway"`. `verbose`, `terminal`, `inner`, `model`, `ytype`,
`xtype`, `applyfun` and `cores` have no counterpart. The port keeps the
scores and fit of every node, because it has no fitted R object to drop.

## 3. Decisions

1. **Factor levels and candidate values come from positive-weight rows.**
   Numeric candidates are the sorted unique values among rows with positive
   weight, as docs/decisions.md requires. Factor candidates are built in the
   same way. R's `factor(zselect)` drops unused levels (`:466`), and the port
   also drops levels that only zero-weight rows have. Otherwise such a level
   would add candidate partitions, change which candidate wins a tie, and
   could turn off the one-candidate rule, so a node with zero weights would
   no longer split like the data with those rows deleted.
2. **Ties in split objectives go to the first candidate**, as R's
   `which.min`. `NaN` objectives are skipped as R does (`np.nanargmin`).
   Numeric values are sorted by `np.unique`. Sort stability does not matter
   for candidate values. It matters only in the instability tests, which
   already use a stable sort (docs/instability-tests.md, item 9).
3. **Zero-weight rows whose nominal level has no kid.** A nominal split's
   index is `NA` for levels absent from the node (`:498-507`). With item 1 a
   level can be absent from the positive-weight rows but still occur in a
   zero-weight row. R would give such a row `NA` and route it at random with
   `sample()` (`node.R:189-194`). The port sends it to the kid with the larger
   `w2n`, kid 1 on a tie, so every row reaches a leaf and the routing is
   deterministic. These rows carry no weight, so nothing that is fitted or
   tested changes.
4. **Pruning `n` counts positive-weight rows.** `mob_prune` passes
   `nobs = c(node nobs, n)` with `n = nrow(Z)`, and `mob()`'s BIC closes over
   the same `n` (`:110`). The port uses the number of rows with positive
   weight. Without zero weights this is R's `n`. With zero weights it keeps
   the decision that zero weights act as deleted rows.
5. **Choosing the prune rules for a string `prune`.** partykit's `lmtree()`
   replaces `mob()`'s rules with RSS forms (`lmtree.R:12-29`). Without a
   formula layer the port asks the model: if the model has a
   `prune_rule(name)` method, its rules are used (`LmModel` returns lmtree's),
   otherwise `mob()`'s. An unknown string raises `ValueError`, where R warns
   and turns pruning off.
6. **Pruning order.** partykit repeatedly evaluates every inner node whose
   kids are all terminal and relabels the ids after each round. Each check
   depends only on the node and its kids' objectives, so the port prunes
   bottom-up, children first, and relabels once. The result is the same
   tree. A pruned node keeps its tests and split search, loses its split
   (R sets `split` to `NULL` for terminal nodes, `:683-687`), and gets
   `StopReason.PRUNED`.
7. **`minsize` is checked before `maxdepth`.** R evaluates both, and the
   fixtures record `minsize` when both apply (node 4 of
   `sim_binomial_glmtree_aic_pruned`). The port uses the same order.
8. **No selected variable.** If every raw p-value is `NA`, R's `best` is `NA`
   and the node is terminal (`:560`). The port records `not_significant` with
   `p_value = None`. No fixture has this case.
9. **`caseweights = FALSE`.** The instability tests now accept
   `caseweights` (docs/instability-tests.md, section 3). `w2n` is the count
   of positive weights, and the node models receive the flag as before.
   Oracle fixtures: `sim_gaussian_lmtree_weights_nocaseweights` and
   `sim_gaussian_glmtree_gaussian_weights_nocaseweights`. Extra Python tests
   check that unit weights give the same tree as the default, and that `nobs`
   counts rows.
10. **`restart = FALSE`** reuses the previous admissible candidate's
    coefficients as starting values in numeric split search (`:417-438`).
    R resets them when the names or lengths differ. With one model per tree
    they never do, so the reset is not ported. Factor search always starts
    from `NULL`, as in R. Oracle fixture:
    `sim_binomial_glmtree_multisplit_norestart`, whose split objectives differ
    from the `restart = TRUE` fixture's at node 2. Extra Python tests check
    that binomial trees match the `restart = TRUE` tree and that the starts
    are passed.
11. **`numsplit = "center"` past the last value.** R takes
    `mean(uz[i + 0:1])`, which is `NA` for the last candidate. The port gives
    `NaN`. The last candidate always leaves an empty right side, so it is
    inadmissible unless `minsize` is 0.
12. **Inputs.** Missing values in `z` raise `ValueError`, since R's model
    frame removes them before `mob_partynode` sees them. Negative or `NaN`
    weights also raise. Factor labels without `levels` are sorted with
    Python's sort, not R's locale collation, so pass `levels` when the order
    matters.

## 4. R quirks

Replicated, each with a code comment:

- **One-candidate factor objective.** When a factor has exactly one
  candidate partition and it is admissible, the objective is set to 1
  without fitting (`:475`). The `minsize` check comes first, as in R.
  Fixtures: node 14 of both weighted sim_gaussian trees, nodes 19 and 27 of
  `sim_gaussian_lmtree_zero_weights_deleted`, and node 22 of
  `sim_gaussian_glmtree_gaussian_weights`.
- **BIC with the full-sample `n`.** `mob()`'s BIC rule uses `log(n)` of the
  full sample, not of the node being pruned (`:110`). lmtree's BIC does the
  same through `nobs[2]` (`lmtree.R:19`). Fixture:
  `synth_diabetes_glmtree_binomial_bic`.
- **Sidak above 0.001, Bonferroni below** (`:553-557`), in
  `pvalues.adjust_pvalues` since phase 2.
- **`minsize` set once at the root.** `<<-` assigns the default from the
  root's coefficient count (`:533`), aliased coefficients included, and every
  node uses it. A user `minsize` is truncated by `as.integer` (`:168`), so
  `25.9` becomes 25.
- **Pruning `df` uses the root's coefficient count** for every node, aliased
  coefficients included (`:647`), plus `dfsplit`.
- **The local `minsize` reinterpretation** in split search (`:386-388`) is
  ported as written. Since `minsize` is already an integer there, it only
  changes 0, to `ceiling(0 * nobs) = 0`.

Deliberately not replicated:

- **`cluster[zs]` in the factor split search.** The right-hand fit receives
  the cluster ids of the complement. R passes `cluster[zs]` (`:479`).
  `LmModel` and `GlmModel` ignore `cluster`, so the fixtures cannot show a
  difference.
- **Zero case weights.** partykit fails the root test (docs/instability-
  tests.md, item 6). The port grows the tree of the data with those rows
  deleted (section 5).
- **Nodes with at most as many clusters as parameters** are terminal with a
  `TooFewClustersWarning` (docs/decisions.md). Fixture: node 3 of
  `sim_cluster_few_lmtree_cluster`, where R fails silently too.
- **Random routing of `NA` kids** (decision 3).
- **`n` in pruning counting zero-weight rows** (decision 4).

## 5. Comparison with R

All 31 oracle tree fixtures match R. `sim_gaussian_lmtree_zero_weights` is
not an oracle; see below. The tests ran in the order CLAUDE.md requires, and
every level passed before the next one was run:

1. Split search on each fixture node's rows in R's selected variable (134
   searches): the candidates and the effective `minsize` match exactly, and
   the objectives to `rtol = 1e-8`, with R's `Inf` in the same places.
2. The chosen split variable and break, or nominal index, match exactly.
   Pruned nodes, whose split R drops, find a split. The two
   `no_admissible_split` nodes find none.
3. In the grown trees, node rows, `n_rows` and `nobs` match exactly.
   Coefficients, their names and the objectives match to `rtol = 1e-8`.
4. Whole trees: node ids, parents, kids, depths, terminal flags, stop
   reasons, splits and the tree's `minsize` match exactly. The test
   statistics, adjusted p-values and `best` match, as do the node p-values
   (`rtol = 1e-8`). The split-search objectives inside the grown tree and
   `fitted_node` (the leaf of every row) match. The node, leaf and depth
   counts and the summed leaf objectives match the fixture `summary`. The
   pruned trees (`synth_diabetes_glmtree_binomial_aic`, `_aic_nodfsplit`,
   `_bic` and `sim_binomial_glmtree_aic_pruned`) match after pruning,
   including the renumbered ids.

No tree diverges, so there is no first diverging quantity to report. The
closest decisions in the fixtures (`decision_margins`) are a relative
split-objective gap of 4.3e-5 (`sim_gaussian_glmtree_gaussian_weights`) and
5.6e-5 (the synth_diabetes trees), and a p-value 9.2e-4 from `alpha`
(`sim_gaussian_glmtree_gaussian_weights`). The port's objectives agree with
R's far more closely than these gaps (1e-8 or better), so the matches do not
depend on luck at a near-tie.

**Zero weights.** `sim_gaussian` with weights `w0` (57 zero weights) is grown
with the zero weights in place. The tree equals
`sim_gaussian_lmtree_zero_weights_deleted`, which R grew on the data with
those rows deleted. Ids, parents, kids, stop reasons and splits match
exactly. Each node's positive-weight rows, mapped through `removed_rows`,
match exactly, as do `nobs`. Coefficients, objectives and split objectives
match to `rtol = 1e-8`. `fitted_node` matches on the kept rows, and every
zero-weight row reaches a leaf. The R fixture
`sim_gaussian_lmtree_zero_weights` itself, one node with `test_failed`, is
the documented divergence.

**Fixture data defect, fixed.** `sim_cluster.csv` once had `z1` in row 551
(0-based) one ulp away from R's value, because `format_double` checked the
round trip with R's `as.numeric`, which is not correctly rounded. The CSVs now
write every double with 17 significant digits, and the candidate-value test
compares exactly. `sim_cluster` passes.

**New fixtures.** `caseweights = FALSE` (lmtree and glmtree on weighted
`sim_gaussian`), `restart = FALSE` (glmtree on `sim_binomial`), and positive
weights with `cluster` (lmtree and glmtree on `sim_cluster`, which gained a
weight column `w`). All match at the same tolerances as the other fixtures, so
there is no first diverging quantity to report.

## 6. Tests

`tests/test_tree.py`:

- levels 1 to 4 above, for every oracle fixture, including the clustered,
  weighted and zero-weights-deleted ones;
- the few-clusters node: `TooFewClustersWarning`, terminal, `test_failed`;
- zero weights equal deleted rows. A zero-weight value between two
  candidates is not a candidate. A zero-weight row with a level absent from
  the positive rows goes to the larger kid;
- `getlevels` against R's enumeration for nominal and ordered factors;
- the default `minsize` and its truncation, `restart = FALSE`,
  `caseweights = FALSE` (Python consistency tests, plus oracle checks against
  their fixtures), `maxdepth`, `alpha`, `bonferroni = FALSE`,
  `numsplit = "center"`, the one-candidate objective, labels versus codes;
- the prune rules (mob and lmtree, AIC and BIC, `log(n)` of the full
  sample), lmtree's rules chosen through `LmModel.prune_rule`, `"none"`, a
  callable rule with the `df` passed with and without `dfsplit`, and preorder
  ids after pruning;
- `NotImplementedError` for every excluded option, and input validation.

`tests/tree_fixtures.py` rebuilds the model, inputs, partitioning variables
and clusters from a fixture. `tests/test_node_models.py` now uses it too.
