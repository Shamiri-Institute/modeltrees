# Design decisions

Answers to the open questions in docs/model-interface.md (section 5) and
notes/phase1-questions.md. Match R unless a decision says otherwise.

## Scope of v0.1

- Binary splits only. Multiway splits are deferred. Node and Split keep kids as
  a list so they can be added.
- vcov = "opg" only. "info" and "sandwich" raise NotImplementedError. The
  FitResult.vcov field stays.
- Options that need R's random numbers raise NotImplementedError: mtry < Inf,
  breakties = TRUE, ordinal = "L2", ordinal = "max". Ordered factors use the
  chi-squared test, which is R's default.
- cluster is in scope for v0.1 (applications often have grouped observations). It
  affects only the instability tests: scores are summed within clusters before
  the covariance estimate, as in mob_grow_fluctests (modelparty.R:240-245).
  Coefficient fits and split search ignore it. With fewer clusters than
  parameters the meat has rank G and is exactly singular, and whether
  partykit's chol() succeeds on it is decided by rounding. In the second
  blind run (task 6l) R failed at 68 of the 70 such nodes it tested and
  succeeded at 2, where it split on statistics of order 1e16. Python stops
  at G < k always, which is R's usual outcome: it warns and makes the node
  terminal. At exactly as many
  clusters as parameters the outcome depends on the weights: partykit sums
  `process / sqrt(weights)` within clusters (modelparty.R:244), and those sums
  add up to zero only when sqrt(weights) is constant over the node (unit or
  equal case weights, or caseweights = FALSE). Then the meat has rank k - 1,
  partykit's chol() fails or not by rounding, and Python warns and makes the
  node terminal. With unequal positive case weights the sums are
  unconstrained, the meat is regular and the tests run as in partykit (a
  failing Cholesky is still an InstabilityTestError). Evidence from the blind
  run (task 6j): at blind_03 node 20 (G = k = 3) and blind_13 node 13 (G = k =
  2) the meat built from R's own scores has smallest/largest eigenvalue 8.9e-4
  and 3.7e-3, and the port's tests reproduce R's statistics and log p-values
  to 17 digits. The cluster-robust vcov sums the scores themselves, whose
  cluster sums add up to zero under any weights, so its TooFewClustersWarning
  stays at G <= k (docs/api-port.md, section 19).
- Not implemented and without fixtures: parm, ytype and xtype other than the
  defaults.

## Behavior

- Coefficients keep NaN for aliased columns. minsize and the pruning df count
  them as R does, while estfun excludes them.
- Numeric partitioning variables are sorted with a stable sort (numpy argsort
  kind="stable"). Ties in split objectives go to the first candidate, as in R.
- Replicate R behavior that changes results, with a code comment on each: Sidak
  adjustment above 0.001 and Bonferroni below, a one-candidate factor split gets
  objective 1 without fitting, and mob BIC pruning uses log(n) of the full
  sample.
- Do not replicate R bugs: the cluster[zs] mistake in factor split search (use
  the complement), and zero case weights.
- Zero weights are handled by dropping zero-weight rows consistently inside the
  instability tests. Verify against R run on the same data with those rows
  deleted (fixture sim_gaussian_lmtree_zero_weights_deleted). Keep
  sim_gaussian_lmtree_zero_weights as a documented divergence from R.
- Split search takes candidate split values and factor levels from
  positive-weight rows only, as the instability tests do. This is tested in
  tests/test_tree.py. The related routing and pruning decisions for zero-weight
  rows are in docs/tree-growing.md (section 3).
- Speedups must not change results. The exact per-candidate path stays in the
  code as the reference implementation, and tests compare every faster path
  against it and against the R fixtures.
- The reference implementation is selected with `modeltrees.reference_path()`, a
  context manager: `find_split_reference`, full fits in split search and the
  dqrdc2 port for every least-squares solve. Outside it, `dqrls` takes a LAPACK
  path when every column keeps at least 1e-4 of its norm (1000 times R's
  tolerance), glm split-search fits skip warnings, null deviance and residuals,
  split search fits on the node's own arrays, and the binomial and poisson
  families use specialised formulas for their deviance residuals, links and AIC.
  The fast and reference paths agree to rounding (about 1e-15 relative), not bit
  for bit, because LAPACK and the port round differently. docs/performance.md
  has the details.
- Batch split objectives (docs/model-interface.md, section 4.1): a model may
  score all numeric candidates at once with an error bound. Split search refits
  exactly every candidate that could be the best given the bounds (within a
  relative margin of 1e-6, and at least 5), and NaN candidates, then takes the
  first minimum. The winner and its tie rule are the exact search's. The other
  candidates' objectives in `SplitSearch.objective` are approximations within
  their error bound. `LmModel` implements it, with rank-deficient or
  ill-conditioned sides (condition number of the scaled cross-products above
  1e8) fitted exactly. Not used with `restart = FALSE` or for factors.

## Interface

- The core receives numeric arrays. Categorical responses, such as a binomial
  factor, are converted at the glmtree-style boundary using R's rule: the first
  level is failure, the rest are success.
- caseweights is an explicit argument. Frequency-weight semantics are the
  default, as in partykit. caseweights=False is available.
- narwhals is the only runtime dependency beyond numpy and scipy.

## Extensions (phase 4)

The full specification, with thesis citations and open questions, is in
docs/extensions-spec.md.

- Negative binomial: theta is not among the tested parameters. The thesis scores
  come from estfun.glm (beta only), and the thesis' quine statistics are
  reproduced exactly with beta-only scores. coefficients is beta, so
  len(coefficients) = p, which matches the thesis code. theta is reported per
  node next to the FitResult. A change in overdispersion alone does not trigger
  a split.
- Negative binomial objective: the negative log-likelihood by default. objective
  = "deviance" reproduces the thesis (party 1.0-1 compares candidate splits by
  summed node deviance). Both are ported.
- Multinomial coefficients are flattened with names. Column order does not
  change the test statistics, since the statistic is a quadratic form that is
  invariant to column permutation. The order is class-major,
  "<level>:<regressor>", as in vcov(multinom) and sandwich's estfun.multinom, so
  fixtures compare directly.
- Safe logit and multinomial are our own designs. The thesis names them but has
  no code, and the mobtools package is unavailable. Docs and README must not
  present them as Rusch's implementation. Their specs are provisional.
- Safe logit: a node fitted to its 0/1 response within 1e-6 everywhere has no
  scores and is terminal. Quasi-separation is only reported. An exact LP check
  is deferred.
- Fit failures, all models: a candidate split whose child fit has status failed
  (a glm.fit error, or a rank-deficient multinomial design) is inadmissible with
  objective Inf, and the node records how many candidates failed. A fit with
  status no_estfun, such as a completely separated safe logit child or a
  single-class multinomial child, stays admissible with its objective, as in R,
  and makes the node terminal. A failed node fit makes the node terminal with a
  warning instead of raising.
- Pruning df counts coefficients only, as partykit's mob_prune does
  (modelparty.R:646-647), also for negative binomial: theta is not added. The
  blind run (task 6j) gave the oracle: R keeps the split of blind_36 node 18
  under AIC pruning, by 0.86 with R's own objectives and df (3, 7), where k + 1
  per node, df (4, 9), prunes it by 1.14. It is the convention partykit already
  uses for the gaussian dispersion, which logLik counts and pruning does not.
  The model attribute extra_df (1 for theta) stays and counts only in the
  log-likelihood df of tree.loglik(), aic() and bic(), as logLik.negbin counts
  theta. So aic() of a pruned negative binomial tree is not the quantity the
  pruning compared.
- Multinomial with a rank-deficient design: the fit counts as failed (so the
  candidate is inadmissible, or the node terminal). nnet would fit it and return
  arbitrary values for the aliased columns, so R has no oracle; a unit test
  covers it. An offset stays unsupported.
- Multinomial with a missing class: if the reference class is absent from a
  node, the fit has no coefficients (all NaN) and no scores, with the objective
  R computes (fitted with the first present class as reference). It has status
  no_estfun, so it stays admissible in split search and the node is terminal.
  Other absent classes get NaN coefficients, are dropped from the scores like
  aliased columns, and still count in minsize and pruning df. A rank-deficient
  design stays a failed fit.
- The glm.fit-based extension models copy glm.fit's early working weights in the
  scores (about 2e-8 from the exact derivative), as GlmModel does.

## glm.fit and log-likelihood

- Step halving has R fixtures (glm_poisson_step_halving and
  glm_poisson_step_halving_maxit) and a Python test cross-checked with
  statsmodels.
- The lmtree log-likelihood uses a per-leaf sigma, as logLik.lm does. It lives
  in LogLikModel and is not derived from objfun.
