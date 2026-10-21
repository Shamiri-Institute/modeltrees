# R fixtures for leaf covariances, predict and a reference-absent multinomial tree

Task 6e-R. Decisions taken while generating the fixtures, and what partykit
1.3-0 and sandwich 3.1-3 do or do not expose. The generators are
`r/fixtures/sandwich.R` (cluster-robust covariances), `r/fixtures/predict.R`
(predict) and `r/fixtures/extensions.R` (the multinomial tree);
tests/fixtures/README.md lists the files.

## 1. sandwich::vcovCL on leaf fits

### 1.1 Trees

- **Reused**: `trees/sim_cluster_lmtree_cluster` (lm, 2 leaves),
  `trees/sim_cluster_glmtree_binomial_cluster` (binomial, 4 leaves) and
  `trees/sim_cluster_few_lmtree_cluster` (lm, 2 leaves, one with G = 3 and k =
  4).
- **New data** `sim_cluster_models` (seed 20261041): 36 schools of 8 to 40
  pupils, 830 rows, school-level random intercepts (sd 0.5; one per
  non-reference class for the multinomial), the school-level noise variables zs
  and zt, observation-level z1 (planted split at 5), z2 and z3, and an exposure.
  One response per model. Four trees, each with `cluster = school` and one split
  on z1 near 5: `sim_cluster_models_glmtree_poisson_cluster` (glmtree, poisson,
  offset `log(exposure)`), `sim_cluster_models_negbin_cluster` (negbinfit, same
  offset), `sim_cluster_models_safelogit_cluster` (safelogitfit, y_sl, no
  separation in either leaf; the generator checks it) and
  `sim_cluster_models_multinom_cluster` (multinomfit, 3 classes, k = 4).
- **Where the new trees live**: `tests/fixtures/cluster_trees/`, in the tree
  fixture format. They are not in `trees/` because `tests/test_node_models.py`
  asserts that `trees/` holds 32 files, and the task does not allow touching the
  Python tests. They are not in `extension_trees/` either, so the four trees
  stay together. A Python test of these trees has to glob `cluster_trees/`
  itself.

### 1.2 Leaf objects

partykit's leaf objects are `lm.fit` and `glm.fit` results with a patched class
and a call that cannot be re-evaluated, and safelogitfit and multinomfit return
lists. sandwich's methods need `model.matrix()`, `weights(x, "working")` and
`summary()` on the object. So **every leaf is refitted** on its rows with a
standard function:

| model        | refit                                                                                                          |
| ------------ | -------------------------------------------------------------------------------------------------------------- |
| lm           | `stats::lm(y ~ x, data = leaf)`                                                                                |
| glm binomial | `stats::glm(..., family = binomial(), control = glm.control(1e-8, 25))`                                        |
| glm poisson  | as binomial, `family = poisson()`, `offset = log(exposure)`                                                    |
| negbin       | `MASS::glm.nb(y_nb ~ x + offset(log(exposure)), control = glm.control(1e-8, 25))`                              |
| safe logit   | `stats::glm(y_sl ~ x, family = binomial(), ...)` (safelogitfit without separation is glm.fit with binomial)    |
| multinom     | `nnet::multinom(y_mn ~ x, trace = FALSE, maxit = 1000, reltol = 1e-16, abstol = 1e-4)`, multinomfit's settings |

`refit_call` in each leaf is the exact expression, evaluated with `leaf` the
data frame of the leaf's rows in `rows` order. The generator stops if the
refit's coefficients differ from the tree's by more than 1e-10 relative. In
every leaf they are **bit-identical** (`refit_vs_tree_coefficients_max_rel` 0),
and so are the refit's `sandwich::estfun` and the tree's stored `estfun`
(`refit_estfun_vs_tree_estfun_max_rel` 0), including the multinomial (nnet
starts from zero weights, so the refit repeats multinomfit's fit exactly) and
the negative binomial (formula interface against `Y ~ 0 + X`).

### 1.3 What is recorded per leaf

`rows` (0-based), `n` (rows), `k`, `n_clusters` (G, distinct school ids in the
leaf), `clusters_le_k`, the tree's coefficients, `vcov_model` (`stats::vcov` of
the refit: sigma^2 (X'X)^-1 for lm, dispersion 1 for binomial, poisson and
negbin, `ginv` of nnet's analytic Hessian for multinom), `bread`
(`sandwich::bread`, n times the inverse Hessian), `hessian` (by hand, below),
and `vcovcl`: eight matrices,

```r
sandwich::vcovCL(fit, cluster = leaf$school, type = t, cadjust = a)
sandwich::vcovCL(fit, cluster = NULL,        type = t, cadjust = a)
```

for `t` in HC0, HC1 and `a` in TRUE, FALSE, always passed explicitly. With
`cluster = NULL` meatCL uses `1:n`, so G = n there (`n_clusters` in the entry).
Each entry also has the eigenvalues. Matrices are row-major, as `estfun` in the
tree fixtures. R's `crossprod` results are symmetric only to rounding (the last
bit can differ between `[i, j]` and `[j, i]`).

### 1.4 The check by hand

For every matrix the generator computes the sandwich of docs/api-design.md 5.1
from the **tree fixture's own `estfun`** and a Hessian written out by hand, not
taken from sandwich:

- lm: `X'X`;
- glm, negbin and safe logit: `X' diag(w) X` with `w` the working weights of the
  last IRLS iteration (`fit$weights`; for negbin at the theta of that
  iteration), which is what `summary.glm`'s `cov.unscaled` uses;
- multinom: `sum_i w_i (diag(p_i) - p_i p_i') kron x_i x_i'` over the
  non-reference classes, class-major, at the fitted probabilities.

`V = H^-1 (c sum_g s_g s_g') H^-1` with `c = G / (G - 1)` under cadjust and
`(n - 1) / (n - k)` under HC1. The generator stops above 1e-8 relative to the
largest entry; the largest difference over all leaves is below 1e-14.
`bread / n` equals `H^-1` to about 1e-15 in every leaf.

### 1.5 Weights and zero weights

All seven trees are **unweighted**, so the refits have unit weights and no
zero-weight rows. That is deliberate: with case weights, sandwich and the design
differ, so a weighted fixture would not pass the check by hand.

- sandwich's `n` is `NROW(estfun)`, the number of rows; the design's `n` is the
  leaf's `nobs`, the sum of case weights. The HC1 factor `(n - 1) / (n - k)`
  therefore differs.
- Without clusters, sandwich makes each row its own cluster and sums
  `estfun_i estfun_i'`. The design divides by `w_i` (the HC0 meat of the rows
  replicated `w_i` times, as `mob_grow_fluctests` divides by `sqrt(weights)`).
  With clusters the two agree, as replicated rows stay in their cluster, apart
  from the HC1 `n`.
- Zero-weight rows: `lm` and `glm` keep them in `estfun` (rows of zeros), so
  sandwich counts them in `n` and, without clusters, in G. The design drops them
  from both.

So the fixtures pin down the unweighted case exactly. A weighted check against R
would need the replicated-rows comparison of design section 9, item 6, which
needs no fixture.

### 1.6 G <= k, and G = 1

- `sim_cluster_few_lmtree_cluster`, node 3: G = 3 schools, k = 4. R returns the
  matrix **without error or warning**. Since the summed scores are zero at the
  fit, `sum_g s_g s_g'` has rank at most G - 1 = 2: the recorded eigenvalues are
  two positive values and two of order 1e-18 (one slightly negative). The
  design's `TooFewClustersWarning` has no R counterpart.
- G = 1 is not a leaf of any tree.
  `vcovcl/sim_cluster_glmtree_binomial_cluster.json` has `single_cluster`: node
  2 refitted with every row in one cluster. sandwich 3.1-3 **stops** with
  "non-conformable arrays" for both cadjust values: meatCL aggregates with
  `apply(ef, 2, rowsum, cluster)`, which drops a one-row result to a vector, so
  the following `crossprod` has the wrong shape. It never reaches `G / (G - 1)`.
  The design's NaN for G = 1 is therefore not R's behaviour; R has no value to
  compare.

### 1.7 Not exposed

- partykit keeps no Hessian or covariance in the node info; `vcov =
  "opg"` is
  used only inside the tests. `refit.modelparty` refits from the model frame,
  which is what the generator does by hand.
- sandwich counts G as the number of factor levels when the cluster is a factor
  (`length(levels(...))`), so unused levels would count. The generator passes
  the integer school ids of the leaf's rows.
- sandwich's default `type` is HC1 only for objects whose first class is `lm`;
  every fixture passes `type` explicitly.

## 2. predict

### 2.1 Trees and rows

`tests/fixtures/predict/` has one file each for
`trees/sim_gaussian_lmtree_weights` (numeric, ordered and nominal splits, with
absent levels), `trees/sim_binomial_glmtree_multisplit` and
`trees/sim_poisson_glmtree_multisplit` (offset `log(exposure)`). The new rows
are built from the first training row of a node, so they reach that node unless
a change sends them elsewhere earlier. `target_node` names the node a case is
aimed at, and `reaches_target_node` says whether R's route passes through it.

### 2.2 Which rows R routes at random

partykit routes a row with `kidids_split` and, where that gives `NA`, draws a
kid with probability 1/2 (`kidids_node`, node.R:189-194; mob trees have no
surrogates and no `prob`). The generator marks a row `random` by walking the
tree with partykit's own `kidids_split` and no fallback, and checks the marking
with `predict(type = "node")` under seeds 1 to 25: deterministic rows never move
and every random row moves at least once. `NA` arises for:

- **a missing split variable** on the row's path. Off the path it does not
  matter (`missing_off_path` rows are deterministic), as the design says.
- **`-Inf` in a numeric split variable.**
  `.bincode(x, c(-Inf, b, Inf),
  right = TRUE)` uses intervals `(-Inf, b]` and
  `(b, Inf]`, so `-Inf` falls in neither and is routed at random. `Inf` goes to
  kid 2. This **differs from docs/api-port.md 11.4**, which sends `-Inf` to
  kid 1. The `minus_inf` rows are therefore random and have no expected values.
- **a level with no kid** (split index `NA`). In all three trees every such
  level is absent from its node because an ancestor already split on the same
  variable (for example node 14 of the lmtree splits z2 with b and d absent, but
  node 11 sends b and d to the other side). So the `absent_level` rows are
  deterministic: they never reach the node (`reaches_target_node` false) and R
  routes them by the ancestor's split. None of the existing tree fixtures has a
  reachable level without a kid, so the design's larger-kid rule has no R oracle
  here; R would route such a row at random.
- Absent levels of an **ordered** split variable route by the break, never at
  random (ordered splits have `breaks`, no `index`).

Rows at the exact break go to kid 1 (`right = TRUE`); one step above
(`b + |b| 2^-52`) to kid 2.

### 2.3 Expected values and failing calls

`expected.node`, `.response` and `.link` come from one `predict()` call on the
deterministic rows (`newdata[deterministic, ]`); random rows are null. A missing
regressor gives NA for response and link and still a node; a missing offset
variable (poisson) the same. The generator checks every response and link
against the leaf coefficients written out (`X beta + log(exposure)` and the
inverse link) and stops above 1e-12; they agree exactly.

Calls R does not answer:

- `type = "link"` for the **lmtree**: `predict.lmtree` passes the type to
  `predict.lm`, which stops ("'arg' should be one of ..."). The design's
  `"link"` for lmtree has no R value; it equals `"response"`.
- **Unseen levels** (`unseen_level_calls`): partykit checks the levels of the
  split variables only (`predict.party`, party.R:295-310). An extra level in a
  split variable sends it to `model.frame(xlev = ...)`, which stops with "factor
  z2 has new level zz_unseen". An extra level in a partitioning variable that no
  split uses (z3 in the binomial tree, z2 in the poisson tree) is **not
  checked**: R routes the row as before. The design's `unseen="error"` raises
  for both (docs/api-port.md 11.3), which is stricter than R for the second
  case.

### 2.4 Not exposed

partykit gives no way to ask whether a row was routed at random; the generator
reconstructs it with the unexported `kidids_split`. Newdata must have the
training factor levels and column classes exactly, or partykit goes through
`model.frame` with the default `na.action`, which would drop rows with missing
values and misalign the result; every fixture keeps both.

## 3. A multinomial tree with a reference-absent leaf

`reference_absent_trees/sim_multinom_noref_onesplit.json`, in the extension tree
format, on the new dataset `sim_multinom_noref` (seed 20261034, 600 rows):
classes lo < mid < hi with lo the reference, and no lo at all for z1 > 0.7 (mid
and hi there have their own coefficients). Grown by `partykit::mob` with
`multinomfit` (multinomfit's usual settings, minsize 60) and partitioning
variables z1, z2 (noise) and z3 (noise).

- **Split search.** Every z1 candidate at or above the last lo observation has a
  right child without lo. multinomfit fits such a child with the first present
  class (mid) as reference and returns its objective only (status
  `reference_absent`, all coefficients NA, no estfun). The candidate stays
  admissible, as docs/decisions.md says.
- **The tree.** The root splits on z1 exactly at the last lo observation
  (0.683...), the first reference-absent candidate. The right leaf (node 3, 185
  rows, mid and hi only) has no coefficients and no scores, so partykit skips
  its tests: `stop_reason` `no_estfun`, `model_info.status` `reference_absent`,
  objective 93.95. The left leaf is not significant.
- **`reference_absent`**, a field added to this fixture only: the root's
  split-search candidates (all z1 values; partykit records objectives for the
  chosen variable only) refitted side by side with multinomfit, with each side's
  classes, status and objective. The generator stops unless the two sides add up
  to the instrumented split-search objective to 1e-10 relative; they agree
  exactly, since partykit fits the same subsets. The field also lists the
  reference-absent leaves with their objective.

**Moved to `extension_trees/` in task 6e** (docs/api-port.md, section 14): the
five tests below were fixed and the file is now
`extension_trees/sim_multinom_noref_onesplit.json`. The rest of this section is
the reasoning at the time.

**Why not in `extension_trees/`.** With the file there, the port agrees with R
on everything the suite compares that the fixture can feed: the public tree
equals the array path, structure and splits, every split-search objective (the
126 reference-absent candidates included), the instability tests, the fitted
nodes, logLik and the leaf coefficient table. Five tests fail on assumptions
about the fixtures, not on a difference from R:

- `tests/test_multinom.py::test_fixture_lists` lists the multinomial trees by
  name.
- `tests/test_multinom.py::test_tree_nodes` and
  `tests/test_api_fixtures.py::test_coef_table_matches_the_fixture` call
  `_r_offset` with R's `estfun`, which is null for the reference-absent leaf
  (the first asserts `estfun is not None` for every node).
- `tests/test_api_fixtures.py::test_predict_equals_hand_computed_leaf_fits` and
  `tests/test_model_protocols.py::test_predict_matches_the_hand_computed_leaf_fit`
  compute an all-NaN leaf as if every non-reference class were absent
  (probability 1 for lo), while the port returns NaN for a fit without the
  reference class (docs/api-port.md 3.3, item 5).

The task does not allow changing the tests, so the tree is in its own directory
and the suite stays green. Once those tests skip `estfun` checks for `no_estfun`
nodes and handle the reference-absent leaf, the file can move to
`extension_trees/` (one line in r/generate_all.R).

partykit does not expose the per-side objectives of a candidate, only their sum
(and only through the instrumented copy of `mob_partynode`), so the side
objectives come from separate fits.
