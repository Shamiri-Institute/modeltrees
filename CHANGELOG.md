# Changelog

All notable changes are listed here. Versions follow 0.x semantics: names in
`modeltrees.__all__` and `modeltrees.models` may still change in a minor release
(docs/api-design.md, section 8.4). A change that alters a grown tree gets an
entry in docs/decisions.md.

## 0.1.0 (unreleased)

First release.

- Model-based recursive partitioning ported from R's partykit 1.3-0: the
  parameter instability tests with partykit's supLM p-values and chi-squared
  factor tests, Bonferroni and Sidak adjustment, binary split search for
  numeric, ordered and nominal partitioning variables, `minsize`, `maxdepth`,
  `trim`, `restart`, `caseweights` and `numsplit`, and AIC and BIC post-pruning.
- Entry points `lmtree`, `glmtree` (gaussian, binomial and poisson with
  canonical links), `negbintree`, `safelogittree`, `multinomtree`, and `mob` for
  arrays and custom node models.
- Ports of R's `glm.fit`, `lm.fit`, `lm.wfit` and the LINPACK `dqrdc2` QR, and
  of MASS's `glm.nb` and `theta.ml`.
- Negative binomial node model following Rusch (2012), with the log-likelihood
  (default) or deviance split objective.
- Safe logit and multinomial logit node models, this project's own provisional
  designs (docs/extensions-spec.md).
- Formula interface (`y ~ x1 + x2 | z1 + z2`) on any eager dataframe that
  narwhals supports, or on a dict of arrays; levels, kinds and missing-data
  handling as in docs/api-design.md.
- `cluster` for the instability tests, and cluster-robust leaf covariances
  (`ModelTree.vcov`, HC0 and HC1 with `cadjust`) checked against
  `sandwich::vcovCL`.
- `predict` with types `node`, `link`, `response`, `prob` and `class`;
  `coef_table`, `summary`, `loglik`, `aic`, `bic` and a printed tree.
- Parallel split search with `n_jobs`, giving results identical to the serial
  search under single-threaded BLAS.
- `modeltrees.reference_path()`, which runs the exact reference implementation
  that the faster paths are tested against.
