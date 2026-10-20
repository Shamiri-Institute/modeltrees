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
  `trees/sim_cluster_few_lmtree_cluster` (lm, 2 leaves, one with G = 3 and
  k = 4).
- **New data** `sim_cluster_models` (seed 20261041): 36 schools of 8 to 40
  pupils, 830 rows, school-level random intercepts (sd 0.5; one per
  non-reference class for the multinomial), the school-level noise variables
  zs and zt, observation-level z1 (planted split at 5), z2 and z3, and an
  exposure. One response per model. Four trees, each with
  `cluster = school` and one split on z1 near 5:
  `sim_cluster_models_glmtree_poisson_cluster` (glmtree, poisson, offset
  `log(exposure)`), `sim_cluster_models_negbin_cluster` (negbinfit, same
  offset), `sim_cluster_models_safelogit_cluster` (safelogitfit, y_sl, no
  separation in either leaf; the generator checks it) and
  `sim_cluster_models_multinom_cluster` (multinomfit, 3 classes, k = 4).
- **Where the new trees live**: `tests/fixtures/cluster_trees/`, in the
  tree fixture format. They are not in `trees/` because
  `tests/test_node_models.py` asserts that `trees/` holds 32 files, and the
  task does not allow touching the Python tests. They are not in
  `extension_trees/` either, so the four trees stay together. A Python test
  of these trees has to glob `cluster_trees/` itself.

### 1.2 Leaf objects

partykit's leaf objects are `lm.fit` and `glm.fit` results with a patched
class and a call that cannot be re-evaluated, and safelogitfit and
multinomfit return lists. sandwich's methods need `model.matrix()`,
`weights(x, "working")` and `summary()` on the object. So **every leaf is
refitted** on its rows with a standard function:

| model | refit |
| --- | --- |
| lm | `stats::lm(y ~ x, data = leaf)` |
| glm binomial | `stats::glm(..., family = binomial(), control = glm.control(1e-8, 25))` |
| glm poisson | as binomial, `family = poisson()`, `offset = log(exposure)` |
| negbin | `MASS::glm.nb(y_nb ~ x + offset(log(exposure)), control = glm.control(1e-8, 25))` |
| safe logit | `stats::glm(y_sl ~ x, family = binomial(), ...)` (safelogitfit without separation is glm.fit with binomial) |
| multinom | `nnet::multinom(y_mn ~ x, trace = FALSE, maxit = 1000, reltol = 1e-16, abstol = 1e-4)`, multinomfit's settings |

`refit_call` in each leaf is the exact expression, evaluated with `leaf` the
data frame of the leaf's rows in `rows` order. The generator stops if the
refit's coefficients differ from the tree's by more than 1e-10 relative. In
every leaf they are **bit-identical** (`refit_vs_tree_coefficients_max_rel`
0), and so are the refit's `sandwich::estfun` and the tree's stored
`estfun` (`refit_estfun_vs_tree_estfun_max_rel` 0), including the
multinomial (nnet starts from zero weights, so the refit repeats
multinomfit's fit exactly) and the negative binomial (formula interface
against `Y ~ 0 + X`).

### 1.3 What is recorded per leaf

`rows` (0-based), `n` (rows), `k`, `n_clusters` (G, distinct school ids in
the leaf), `clusters_le_k`, the tree's coefficients, `vcov_model`
(`stats::vcov` of the refit: sigma^2 (X'X)^-1 for lm, dispersion 1 for
binomial, poisson and negbin, `ginv` of nnet's analytic Hessian for
multinom), `bread` (`sandwich::bread`, n times the inverse Hessian),
`hessian` (by hand, below), and `vcovcl`: eight matrices,

```r
sandwich::vcovCL(fit, cluster = leaf$school, type = t, cadjust = a)
sandwich::vcovCL(fit, cluster = NULL,        type = t, cadjust = a)
```

for `t` in HC0, HC1 and `a` in TRUE, FALSE, always passed explicitly. With
`cluster = NULL` meatCL uses `1:n`, so G = n there (`n_clusters` in the
entry). Each entry also has the eigenvalues. Matrices are row-major, as
`estfun` in the tree fixtures. R's `crossprod` results are symmetric only
to rounding (the last bit can differ between `[i, j]` and `[j, i]`).

### 1.4 The check by hand

For every matrix the generator computes the sandwich of
docs/api-design.md 5.1 from the **tree fixture's own `estfun`** and a
Hessian written out by hand, not taken from sandwich:

- lm: `X'X`;
- glm, negbin and safe logit: `X' diag(w) X` with `w` the working weights
  of the last IRLS iteration (`fit$weights`; for negbin at the theta of
  that iteration), which is what `summary.glm`'s `cov.unscaled` uses;
- multinom: `sum_i w_i (diag(p_i) - p_i p_i') kron x_i x_i'` over the
  non-reference classes, class-major, at the fitted probabilities.

`V = H^-1 (c sum_g s_g s_g') H^-1` with `c = G / (G - 1)` under cadjust and
`(n - 1) / (n - k)` under HC1. The generator stops above 1e-8 relative to
the largest entry; the largest difference over all leaves is below 1e-14.
`bread / n` equals `H^-1` to about 1e-15 in every leaf.

### 1.5 Weights and zero weights

All seven trees are **unweighted**, so the refits have unit weights and no
zero-weight rows. That is deliberate: with case weights, sandwich and the
design differ, so a weighted fixture would not pass the check by hand.

- sandwich's `n` is `NROW(estfun)`, the number of rows; the design's `n` is
  the leaf's `nobs`, the sum of case weights. The HC1 factor
  `(n - 1) / (n - k)` therefore differs.
- Without clusters, sandwich makes each row its own cluster and sums
  `estfun_i estfun_i'`. The design divides by `w_i` (the HC0 meat of the
  rows replicated `w_i` times, as `mob_grow_fluctests` divides by
  `sqrt(weights)`). With clusters the two agree, as replicated rows stay in
  their cluster, apart from the HC1 `n`.
- Zero-weight rows: `lm` and `glm` keep them in `estfun` (rows of zeros),
  so sandwich counts them in `n` and, without clusters, in G. The design
  drops them from both.

So the fixtures pin down the unweighted case exactly. A weighted check
against R would need the replicated-rows comparison of design section 9,
item 6, which needs no fixture.

### 1.6 G <= k, and G = 1

- `sim_cluster_few_lmtree_cluster`, node 3: G = 3 schools, k = 4. R
  returns the matrix **without error or warning**. Since the summed scores
  are zero at the fit, `sum_g s_g s_g'` has rank at most G - 1 = 2: the
  recorded eigenvalues are two positive values and two of order 1e-18 (one
  slightly negative). The design's `TooFewClustersWarning` has no R
  counterpart.
- G = 1 is not a leaf of any tree. `vcovcl/sim_cluster_glmtree_binomial_cluster.json`
  has `single_cluster`: node 2 refitted with every row in one cluster.
  sandwich 3.1-3 **stops** with "non-conformable arrays" for both cadjust
  values: meatCL aggregates with `apply(ef, 2, rowsum, cluster)`, which
  drops a one-row result to a vector, so the following `crossprod` has the
  wrong shape. It never reaches `G / (G - 1)`. The design's NaN for G = 1 is
  therefore not R's behaviour; R has no value to compare.

### 1.7 Not exposed

- partykit keeps no Hessian or covariance in the node info; `vcov =
  "opg"` is used only inside the tests. `refit.modelparty` refits from the
  model frame, which is what the generator does by hand.
- sandwich counts G as the number of factor levels when the cluster is a
  factor (`length(levels(...))`), so unused levels would count. The
  generator passes the integer school ids of the leaf's rows.
- sandwich's default `type` is HC1 only for objects whose first class is
  `lm`; every fixture passes `type` explicitly.
