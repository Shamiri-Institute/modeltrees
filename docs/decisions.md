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
- cluster is accepted by the interface but not implemented in v0.1.
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
  deleted (new fixture). Keep sim_gaussian_lmtree_zero_weights as a documented
  divergence from R.

- Split search must take candidate split values from positive-weight rows only,
  as the instability tests already do. A candidate from a zero-weight row could
  win a tie and shift the break. There is no split search yet, so nothing tests
  this; add the test with the split search.

## Interface

- The core receives numeric arrays. Categorical responses, such as a binomial
  factor, are converted at the glmtree-style boundary using R's rule: the first
  level is failure, the rest are success.
- caseweights is an explicit argument. Frequency-weight semantics are the
  default, as in partykit. caseweights=False is available.

## Extensions (phase 4)

- Whether theta is among the tested parameters for negative binomial is decided
  from reference/trusch.md in phase 4. The interface needs no change because
  estfun width is defined by the model. minsize uses len(coefficients), so check
  that count against the thesis code.
- Multinomial coefficients are flattened with names. Column order does not
  change the test statistics, since the statistic is a quadratic form that is
  invariant to column permutation. Use the order of R's flattened coef(multinom)
  so fixtures compare directly.

## glm.fit and log-likelihood

- Step halving gets a Python unit test cross-checked with statsmodels. An R
  fixture waits until non-canonical links are in scope (poisson with an identity
  link is the likely trigger).
- The lmtree log-likelihood uses a per-leaf sigma, as logLik.lm does. It lives
  in LogLikModel and is not derived from objfun.

## Extra fixtures wanted (phase 1b, after phase 2)

numsplit = "center", dfsplit = FALSE, AIC pruning, trim > 1, and the
zero-weights-deleted case.
