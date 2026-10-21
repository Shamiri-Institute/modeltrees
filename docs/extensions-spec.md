# Extension node models (phase 4)

This document specifies the three node models that modeltrees adds to partykit's
mob: negative binomial with unknown shape, safe logit, and multinomial logit.
The R implementations in `r/extensions/` are the oracle. They are partykit
`mob()` type 1 fit functions (docs/model-interface.md, section 1.1). The Python
port must reproduce them.

The fixtures are `tests/fixtures/node_fits/` (single fits with verified estfun),
`tests/fixtures/extension_trees/` (trees in the tree-fixture format) and
`tests/fixtures/thesis_quine_crosscheck.json`. tests/fixtures/README.md
describes them.

Line numbers `trusch.md:N` refer to `reference/trusch.md`. `MOB.R:N` and
`MOB-Utils.R:N` refer to `reference/party/R/`. Section 7 lists open questions.

## 1. What the thesis provides

- The thesis lists three experimental node models: negbinModel, safelogitModel,
  and mlogitModel, an interface to the mlogit package for multinomial and
  related choice models (trusch.md:935-941). They were distributed in the
  mobtools package (trusch.md:954, 2133-2134).
- Only negbinModel's code is reproduced: the StatModel object
  (trusch.md:1101-1122), its fit slot (trusch.md:1188-1205), the reweight method
  (trusch.md:1296-1298) and the other S3 methods (trusch.md:1323-1327). The body
  of the underlying fitter `glm.nb_fit` is elided as `"<body>"`
  (trusch.md:1037). The text says it is "Venables and Ripley's code from glm.nb"
  adapted to take a design matrix (trusch.md:1029-1031).
- The scores come from sandwich's `estfun.glm`, printed at trusch.md:1056-1066.
  Weights come from `weights.negbin` (trusch.md:1071-1074).
- safelogitModel and mlogitModel are only named. The text does not describe
  their fitting, starting values or fallbacks.
- mobtools is not on CRAN, the CRAN archive, R-Forge or the GitHub CRAN mirror
  (checked 2026-10-20). Its code cannot be consulted.
- The worked example is a negative binomial tree on MASS's quine data
  (trusch.md:1339-1470).

So the negative binomial model is ported from the thesis code. The safe logit
and multinomial models are specified here from the thesis' description of their
purpose. Those specifications are this project's decisions, marked
**provisional**.

## 2. Common interface

All three fit functions have the signature

```r
fit(y, x = NULL, start = NULL, weights = NULL, offset = NULL, cluster = NULL, ...,
    estfun = FALSE, object = FALSE, <model arguments>)
```

and return `coefficients`, `objfun`, `estfun` (only when `estfun = TRUE`, and
`NULL` when the model has no valid scores) and `object` (only when
`object = TRUE`). They also return model-specific diagnostics as extra list
elements. `mob()` keeps these in the node info, and the tree fixtures record
them as `model_info`. In Python they belong in a model-specific result next to
`FitResult`, not in the core.

Shared rules:

- `x = NULL` becomes a one-column intercept matrix, as in partykit's glmfit.
- `weights` are prior weights in the likelihood. Integer weights act as
  frequency weights. None of the three models has a dispersion parameter in its
  scores, so partykit's `caseweights` setting changes nothing inside the fit. It
  still changes `w2n` and the instability tests in mob.
- `cluster` is ignored by the fitters. mob uses it only in the tests.
- When the scores do not exist (safe logit under complete separation, or
  multinomial with one class or no reference class), `estfun` is `NULL`.
  partykit then makes the node terminal and stores `test = NA`
  (modelparty.R:550, 569-573). Fixtures mark such nodes with
  `stop_reason = "no_estfun"`.
- Coefficients that a node cannot identify are `NA`. They still count in
  `minsize` and the pruning df and have no estfun column, as with aliased
  coefficients (docs/decisions.md).

## 3. Negative binomial: `negbinfit` (r/extensions/negbin.R)

**Model.** Y | x ~ NB(mu, theta) with log link: log mu = x'beta + offset and
Var(Y) = mu + mu^2 / theta. Each node estimates its own beta and theta.

**Fitter.**
`MASS::glm.nb(Y ~ 0 + X, weights, offset, start,
control = glm.control(epsilon, maxit))`.
The defaults are `epsilon = 1e-8` and `maxit = 25`, which match glm.control and
the thesis (`control =
glm.control()`, trusch.md:1105, 1113). The algorithm, as
in MASS 7.3:

1. Fit a Poisson GLM with glm.fit (`start` if given; mustart and etastart not
   given).
2. Set theta = `theta.ml(y, mu, sum(w), w, limit = maxit)` at the Poisson mu.
   theta.ml runs Newton steps from `sum(w) / sum(w * (y / mu - 1)^2)` with
   tolerance `.Machine$double.eps^0.25` on the step. It warns "iteration limit
   reached" or "estimate truncated at zero".
3. Alternate up to `maxit` times. First refit beta by glm.fit with
   `negative.binomial(theta)` and `etastart = log(mu)`. Then re-estimate theta
   at the previous mu, and update mu. Stop when
   `|Lm0 - Lm| / d1 + |theta_old - theta| <= epsilon`, where
   `d1 = sqrt(2 * max(1, df.residual of the Poisson fit))` and Lm is the log
   likelihood. If the limit is reached it warns "alternation limit reached".

**Parameter vector.** `coefficients` is beta only (p entries, named by the
columns of x). theta is returned separately as `theta`, with `theta_se`,
`theta_warning`, `converged` and `iter`.

**Objective.** The negative log-likelihood at (beta, theta): `-twologlik / 2`,
where

    l_i = w_i [lgamma(theta + y_i) - lgamma(theta) - lgamma(y_i + 1) + theta log theta
               + y_i log(mu_i + 1{y_i = 0}) - (theta + y_i) log(theta + mu_i)].

This equals partykit glmfit's `aic/2 - df` convention. With
`objective = "deviance"` the objective is the residual deviance at the node's
own theta. That is party 1.0-1's default split criterion (`objfun = deviance`,
MOB.R:91-92, MOB-Utils.R:365-376) and is needed to reproduce the thesis tree
(section 6). The two objectives choose different splits on quine. Deviances
computed with different thetas are not on a common scale, so loglik is the
default.

**estfun.** sandwich's `estfun.glm`, as in the thesis (trusch.md:1053-1066). The
dispersion is 1 because the family name starts with "Negative Binomial":

    estfun_i = working_residual_i * working_weight_i * x_i
             ≈ w_i (y_i - mu_i) / (1 + mu_i / theta) * x_i.

The ≈ matters for an exact port, for two reasons:

- glm.fit's working weights are computed from the linear predictor before its
  final update. So `estfun` equals the analytic score at the returned
  coefficients only to the IRLS tolerance (about 2e-8 relative on quine). This
  is the same quantity partykit's glmfit uses for every GLM.
- The final IRLS fit uses the theta from the previous alternation. glm.nb then
  updates theta once more. The scores therefore belong to `theta_irls` (also
  returned), while `objfun` uses `theta`. When the alternation converges they
  differ by at most `epsilon`.

The Python port should copy glm.nb's loop exactly and take the scores from the
last glm.fit call, as R does.

**Is theta tested? No.** The instability tests use only the beta columns.
Evidence:

1. The thesis takes its scores from `estfun.glm`, which has one column per
   regression coefficient. It treats "Negative Binomial" as known-dispersion
   (trusch.md:1059-1060). `coef()` of a negbin object excludes theta.
2. The thesis' quine statistics (Eth 11.36 and Age 10.796, trusch.md:1359-1360)
   are reproduced exactly by an intercept-only model with beta-only scores (k =
   1, section 6). With theta in the score vector, k = 2 and the statistics would
   differ.
3. partykit's glmfit also treats "Negative Binomial" as known-dispersion
   (glmtree.R:99). With fixed theta,
   `glmtree(family = negative.binomial(theta))` tests beta only.

Consequences:

- `minsize` defaults to `ceiling(10 * p)`, not `10 * (p + 1)`.
- A change in overdispersion alone is not targeted by the tests
  (`extension_trees/sim_negbin_theta_only.json` does not split).
- The theta score `w_i [digamma(theta + y_i) - digamma(theta) + log theta + 1
  - log(theta + mu_i) - (y_i + theta) / (mu_i +
    theta)]`is recorded in the
  node fit fixtures (`theta_score_sum`), so this
    decision can be revisited without regenerating data.
- The interface needs no change.
- `minsize` uses `len(coefficients)` = p, which matches the thesis code. party
  1.0-1 itself used a fixed `minsplit = 20` (MOB.R:91), not a multiple of the
  parameter count.

**Starting values.** `start` (passed by mob's split search when
`restart = FALSE`) is used for the initial Poisson fit only. theta has no
starting value: the thesis passes no `init.theta` (trusch.md:1105-1113).

**Fallbacks.** None inside the fit, as in the thesis. glm.nb's warnings
propagate, and the theta warnings are also kept in `theta_warning`. In the R
oracle an error stops `mob()`. In the Python port an error is a failed fit: the
candidate split is inadmissible, or the node becomes terminal with a warning
(Q6).

With near-Poisson data, theta.ml can stop at its iteration limit with a very
large theta, and the warning is kept in `theta_warning`. Two leaves of
`extension_trees/sim_negbin_weights.json` show this (theta about 1.7e4 and
8.4e4). The value of theta there depends on the exact Newton path. A port that
copies theta.ml reproduces it, but tests should compare such thetas loosely. The
coefficients and objective barely depend on theta at that size.

## 4. Safe logit: `safelogitfit` (r/extensions/safelogit.R), provisional

**What the thesis says.** safelogitModel is "binary logistic regression in
presence of quasi-complete separation" (trusch.md:938-939). In the LORET
application one terminal node holds people who never voted, where "we find
(quasi-)complete separation" (trusch.md:2607). Its footnote says: "In this node
the ML estimator does not exist. The algorithm has the positive effect of
separating these observations from the rest, hence estimation in other nodes
works well" (trusch.md:2651). Separation appears again at trusch.md:5209.
Nothing else is said, so "safe" is defined below from these two requirements:

- a separated subgroup can be split off from its parent;
- a node whose ML estimator does not exist ends the recursion.

**Model, fitter, objective, parameters.** Binary logistic regression, fitted by
`glm.fit(x, y, weights, start, offset, family = binomial("logit"),
control = glm.control(epsilon = 1e-8, maxit = 25))`.
A factor response is coded 0 for its first level and 1 otherwise. Proportions
are rejected. The objective is the negative log-likelihood (`aic/2 - rank`, as
in glmfit). The parameter vector is beta. estfun is sandwich's `estfun.glm`,
`w_i (y_i - p_i) x_i` up to the working-weight lag described in section 3.
Without complete separation the whole fit is identical to partykit's glmfit with
`binomial()`. The generator checks this for a node fit and for a whole tree
(`sim_separation_safelogit_plain`).

**What "safe" adds.**

1. Warnings from glm.fit ("algorithm did not converge", "fitted probabilities
   numerically 0 or 1 occurred") are caught and returned in `warnings`. They are
   not raised.
2. If glm.fit raises an error, the fit returns NA coefficients, `objfun = Inf`
   and no estfun, with `status = "error"`. In split search the candidate is then
   inadmissible. At a node fit the node becomes terminal. There is no fixture
   for this case (see Q2).
3. Separation is classified on the positive-weight observations from glm.fit's
   fitted values. An observation is "at the boundary" if
   `|y_i - p_i| < separation_tol` (default `1e-6`).
   - `complete`: every observation is at the boundary. This includes a constant
     response. The ML estimator does not exist, so **no estfun is returned** and
     mob makes the node terminal without testing. `coefficients` and `objfun`
     are glm.fit's values. objfun is within about 1e-9 of its supremum 0. The
     coefficients are where IRLS stopped, not estimates, so tests should compare
     them only loosely.
   - `quasi`: some observations are at the boundary. The fit and its scores are
     kept, so the parent can be tested and the separated observations split off.
     The scores of boundary observations are about 1e-9. The score along the
     diverging coefficient sums to about -2e-6 instead of 0
     (`node_fits/safelogit_quasi.json`).
   - `none`: no observation is at the boundary. `n_boundary` counts the boundary
     observations.

**Why plain glmtree is not enough.** On `sim_separation`, plain `glmtree` grows
the same root split and the same split at z1 = 0.649. It then splits the
constant-response group (z2 = g4, all y = 0, fitted probabilities about 3e-12)
twice on the noise variable z3, with p-values around 1e-83 and 1e-32. Those
scores are numerically zero, and the OPG decorrelation rescales them to unit
variance, so the tests report structure in rounding-level quantities. glmtree
also emits 89 non-convergence and 87 boundary warnings. safelogitfit grows the
same first two splits and leaves both separated nodes terminal with no warnings
(`extension_trees/sim_separation_safelogit.json`).

**Starting values and fallbacks.** `start` is passed to glm.fit. Without it,
glm.fit starts from binomial's `mustart = (w y + 0.5) / (w + 1)`. There is no
retry with other starting values.

## 5. Multinomial logit: `multinomfit` (r/extensions/multinom.R), provisional

**What the thesis says.** mlogitModel interfaced the mlogit package
(trusch.md:939-941). No code or example is given. As instructed, this port fits
with `nnet::multinom`. For a baseline-category logit with individual-specific
regressors, which is the only model in scope, the likelihood and the MLE are the
same as mlogit's. The variants mlogit supports (alternative-specific variables,
nested, rank-ordered) are out of scope (Q8).

**Model.** The response is a factor with levels l_1, ..., l_K, and l_1 is the
reference class.

    P(Y = l_k | x) = exp(x'beta_k) / sum_j exp(x'beta_j), with beta_1 = 0.

**Fitter.**
`nnet::multinom(Yf ~ 0 + X, weights, Wts, trace = FALSE,
maxit = 1000, reltol = 1e-16, abstol = 1e-4)`,
fitted on the positive-weight rows only. Zero-weight rows contribute nothing and
get zero scores. nnet's defaults (`maxit = 100`, `reltol = 1e-8`) leave
gradients around 1e-4. That is too loose for 1e-6 comparisons, so BFGS runs
until it makes no further progress. The fixtures' column sums of estfun are at
most about 2e-8 of the absolute column sums.

**Parameter vector.** (K - 1) p coefficients named `"<level>:<regressor>"` in
class-major order: all regressors of l_2, then l_3, and so on. This is the order
of `vcov(multinom)` and of sandwich's `estfun.multinom`. As noted in
docs/decisions.md, the order does not affect the test statistics. The full
length is kept in every node.

**Objective.** The negative log-likelihood `-sum_i w_i log P(Y = y_i | x_i)`,
which is nnet's `value` with decay 0. The generator checks it against the
log-likelihood computed independently.

**estfun.** sandwich's `estfun.multinom`, which equals the analytic score
exactly:

    column (k, j), row i: w_i (1{y_i = l_k} - p_ik) x_ij.

**Starting values.** By default, zero for all coefficients (multinom uses nnet
with `rang = 0`). With `start`, which mob passes in split search when
`restart = FALSE`, the coefficients are mapped into nnet's weight vector with NA
set to 0. Since the MLE is unique and the optimiser runs to convergence, the
starting values should not change the result beyond the convergence tolerance
(`node_fits/multinom_start.json`).

**Degenerate nodes.** A class counts as present if it has positive total weight
in the node.

- A non-reference class is absent: its coefficients are NA and it has no estfun
  columns. The fit uses the present classes. Its objective is the supremum of
  the full model's log-likelihood. With two classes left, multinom fits a binary
  logit (`node_fits/multinom_absent_class.json`).
- The reference class l_1 is absent (`status = "reference_absent"`): the
  remaining coefficients cannot be expressed relative to l_1. The objective is
  computed, but all coefficients are NA and there is no estfun (Q7).
- Only one class is present (`status = "single_class"`): the objective is 0, all
  coefficients are NA, and there is no estfun.
- An offset is not supported, because multinom cannot take a vector offset for K
  > 2. The fit stops with an error.
- Rank-deficient x is not detected: nnet fits it without dropping columns (Q9).

`minsize` defaults to `ceiling(10 (K - 1) p)`, because `NCOL(y)` is 1 for a
factor response.

## 6. Cross-check against the thesis implementation

party 1.0-1 does not install on R 4.6.1. Its C code uses the `Calloc` macro,
which R's headers no longer define, and compilation fails at
`Distributions.c:63`. renv also cannot parse the line break in its DESCRIPTION's
Depends field. The dependencies (modeltools, coin, vcd) installed into the renv
library. They are not in the lockfile and nothing uses them. As instructed, the
cross-check with party was skipped and no patched build was attempted.

Instead, the tree printed in the thesis (trusch.md:1356-1470) was compared with
partykit 1.3-0 and `negbinfit`. The comparison is in
`tests/fixtures/thesis_quine_crosscheck.json`, and the generator enforces it.
`mob(Days ~ 1 | Eth + Sex + Age + Lrn, quine, fit = negbinfit,
objective = "deviance", minsize = 20)`
reproduces the thesis tree exactly at the printed precision:

|              | thesis                                  | partykit + negbinfit (deviance) |
| ------------ | --------------------------------------- | ------------------------------- |
| root split   | Eth, statistic 11.36, 1 - p = 0.997     | Eth, 11.3605, 0.99700           |
| node 2 split | Age {F0, F1} vs {F2, F3}, 10.796, 0.962 | same, 10.7958, 0.96185          |
| terminal n   | 33, 36, 77                              | 33, 36, 77                      |
| intercepts   | 2.74, 3.28, 2.50                        | 2.7399, 3.2771, 2.4999          |
| theta        | 1.873, 1.500, 0.9186                    | 1.8726, 1.4998, 0.9186          |

The match confirms the thesis' score definition, which leaves theta out (section
3). It also confirms that partykit's chi-squared factor tests and its Bonferroni
adjustment agree with party's at these p-values. party adjusts with Sidak above
0.01 (MOB-Utils.R:70-72), partykit above 0.001, and both p-values here are above
0.01.

With the default log-likelihood objective
(`extension_trees/quine_negbin_intercept.json`), the tests are identical, but
node 2 is split into Age {F2} vs {F0, F1, F3}. On the Eth = A subset, the summed
child deviances are 75.610 for {F0, F1} | {F2, F3} and 75.697 for {F2} | {F0,
F1, F3}. The summed negative log-likelihoods are 275.124 and 274.385. The two
criteria rank the candidates differently.

## 7. Open questions

The answers were recorded on 2026-10-20 (task 5a). Q1 to Q3 and Q7 keep the
provisional specification above.

Q1. **Safe logit definition.** The thesis does not contain safelogitModel's
code, and mobtools is unavailable. Section 4 is a provisional definition:
complete separation means no scores and a terminal node, while quasi-separation
is only reported. Confirm it, or supply mobtools' source. The name could also
refer to the safeBinaryRegression package, which stops with an error on
separated data. That reading is speculative, and an error would conflict with
trusch.md:2651.

**Answer:** provisional as specified.

Q2. **Separation criterion.** `|y - p| < 1e-6` on glm.fit's fitted values is
deterministic and portable, but heuristic. An exact test for separation is a
linear program (Albert and Anderson 1984; Konis 2007). scipy's `linprog` could
run it in Python, but base R has no LP solver, so R could not serve as the
oracle. The glm.fit error path has no fixture, because no natural data trigger
it deterministically.

**Answer:** provisional as specified.

Q3. **Negative binomial objective default.** loglik (partykit's convention,
needed for AIC/BIC pruning) or deviance (party 1.0-1, needed to reproduce the
thesis tree). This spec uses loglik with a deviance option.

**Answer:** provisional as specified: loglik by default, deviance as an option.

Q4. **Pruning df for negative binomial.** mob's AIC/BIC pruning counts
`len(coefficients)` = p per node (modelparty.R:647), but `logLik.negbin` counts
p + 1 because of theta. Should pruning count theta? No fixture uses pruning with
negbinfit.

**Answer:** pruning counts coefficients only, as partykit does; theta counts in
the log-likelihood df (`tree.loglik()`, `aic()`, `bic()`) through the model
attribute `extra_df`. Task 5a first counted theta in pruning; the blind run of
task 6j found the R oracle (`blind_36_negbin`, node 18), and task 6k changed it
(docs/decisions.md).

Q5. **Optional theta test.** Should there be an option to add the theta score as
a tested column, to detect changes in overdispersion? The thesis does not do
this. The theta score is recorded in the node fixtures.

**Answer:** deferred.

Q6. **negbinfit errors.** glm.nb errors propagate (no fallback, as in the
thesis). safelogitfit instead turns errors into an inadmissible candidate.
Should negbinfit and multinomfit do the same?

**Answer:** for all models, a failed fit makes the candidate split inadmissible
(objective Inf), and a failed node fit makes the node terminal with a warning.
This includes negbinfit and multinomfit errors.

Q7. **Multinomial node without the reference class.** Provisional: no
coefficients and no scores. An alternative is to re-parameterize against the
first present class, which breaks the fixed coefficient layout.

**Answer:** as specified (no coefficients and no scores).

Q8. **mlogit features.** The thesis' mlogitModel also covered conditional,
nested and rank-ordered logits. Only the baseline-category multinomial logit is
in scope here.

**Answer:** out of scope.

Q9. **Multinomial offset and aliasing.** No offset, and no detection of
rank-deficient x. nnet would return arbitrary values for aliased columns.

**Answer:** a rank-deficient design is a failed fit for the multinomial model. R
has no oracle for this, since nnet fits it anyway; a unit test covers it. The
offset stays unsupported.

Q10. **Multinomial optimiser in Python.** R's coefficients are accurate to
roughly the size of the gradient (fixture gradients are about 1e-8 relative).
Python should use Newton's method or a tightly converged quasi-Newton method,
not a copy of nnet's BFGS. Coefficient tolerances should be set from the
recorded gradients.

**Answer:** Newton's method, with coefficient tolerances taken from the
gradients recorded in the fixtures.
