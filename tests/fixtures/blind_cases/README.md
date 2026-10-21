# Blind differential run: the mismatching cases

Task 6j ran the port against partykit 1.3-0 on 60 cases (12 per model) that
nobody shaped during development: seeds from
`numpy.random.SeedSequence(20261022).generate_state(60) >> 1`, with the data
and the settings (n, partitioning variables, planted effect, clusters, case
weights, offset, `alpha`, `minsize`, `maxdepth`, `prune`, `numsplit`,
`bonferroni`) drawn in R from each case's seed. The R trees were grown once
with the fit functions of `r/extensions/` and extracted in the tree fixture
format by `r/fixtures/trees.R`. The port grew each case through its public
entry point from the CSV as a polars frame and was compared in the order of
CLAUDE.md, stopping at the first diverging component. 54 cases matched on
every component. The 6 below did not, and are copied here unchanged from
the run:

- `data/<case>.csv` and `.schema.json`: the dataset, written by
  `write_dataset` (`r/fixtures/common.R`);
- `trees/<case>.json`: R's tree, the format of `tests/fixtures/trees/`
  (`data_csv` is relative to this directory);
- `cases/<case>.json`: the case's drawn settings (planted effect, clusters,
  weights, offset, control), the R call and R's run time;
- `vcov/<case>.json`, clustered cases only: `sandwich::vcovCL(fit, cluster
  = school, type = HC0 or HC1, cadjust = TRUE)` per leaf of a refit by the
  standard R function. For weighted cases it is recorded but is not an
  oracle (docs/sandwich-fixtures.md 1.5).

Task 6k fixed or explained each one, and all six now match R on every
component. `tests/test_blind_cases.py` compares each case
(`tests/blind_compare.py`) as a standing test (`MATCHING`); a case that
diverged again would have to be marked xfail (strict) with its first
diverging component. The table gives the first divergence found in task 6j
and what task 6k did (docs/api-port.md, section 19).

| case | model | n | first diverging component (6j) | classification | task 6k |
| --- | --- | --- | --- | --- | --- |
| `blind_03_lm` | lmtree | 2953 | node sizes (node 21) | port defect: G = k with case weights | guard fixed: tests run at G = k under unequal case weights |
| `blind_13_glm` | glmtree binomial | 1015 | node sizes (node 14) | port defect: G = k with case weights | same |
| `blind_29_negbin` | negbinfit | 412 | coefficients (node 4 theta) | rounding: theta not identified | theta at 1e-6 under a weak-identification warning |
| `blind_36_negbin` | negbinfit | 961 | node sizes (node 19) | documented decision: theta in the pruning df | pruning counts coefficients only, as partykit; then as `blind_29` |
| `blind_43_safelogit` | safelogitfit | 828 | coefficients (node 3 slope) | rounding: separated leaf | linear predictor and fitted values instead of slopes |
| `blind_47_safelogit` | safelogitfit | 785 | coefficients (node 3 slope) | rounding: separated leaf | same |

The sections below are the analysis of task 6j; each ends with what task
6k changed.

## `blind_03_lm` and `blind_13_glm`: clusters with case weights at G = k

Both are clustered and case-weighted. R splits node 20 of `blind_03_lm` (3
coefficients, 3 clusters) and node 13 of `blind_13_glm` (2 coefficients, 2
clusters), with adjusted p-value 0; the port makes them terminal with a
`TooFewClustersWarning`, so the rows of the next node differ.
`src/modeltrees/instability.py` stops at `n_clusters <= k`, following
docs/decisions.md ("the covariance estimate is singular"). That holds when
the cluster sums of the scaled scores add up to zero, i.e. with unit
weights. With non-unit case weights partykit sums `process / sqrt(weights)`
within clusters (modelparty.R:244); those sums are unconstrained and the
meat has full rank at G = k (smallest/largest eigenvalue 8.9e-4 and 3.7e-3
on R's own scores). The port's tests with only the guard changed to
`n_clusters < k` reproduce R's statistics and log p-values at both nodes to
all digits, and both cases then match R on every component.

Task 6k: the guard stops at `n_clusters < k`, and at `n_clusters == k` only
when the cluster sums are constrained to sum to zero (`caseweights = FALSE`
or equal positive weights). The cluster-robust `vcov` keeps its warning at
G <= k, since its meat sums the scores themselves, whose cluster sums add
up to zero under any weights.

## `blind_29_negbin`: theta of a near-Poisson node

Node 4's theta is 4923.558164104566 against R's 4923.5579300528625 (4.7e-8
relative, tolerance 1e-8). theta's standard error there is 81787 and R's
glm.nb reports "alternation limit reached"; the standing tests loosen theta
only for theta.ml's "iteration limit reached". The coefficients agree to
2.6e-10, and the log-likelihoods at the two thetas differ by 4.7e-13
(1.1e-15 relative). With theta compared at 1e-6, the case matches R on
every component.

Task 6k: `tests/blind_compare.py` compares theta, theta_irls and theta_se
at 1e-6 when R or the port reports the alternation limit (theta.ml's
iteration limit keeps the standing 1e-3), and accepts a differing
alternation-limit flag when the final thetas agree to 1e-10. The standing
fixture tests are unchanged.

## `blind_36_negbin`: theta in the pruning df

R keeps the split of node 18 under AIC pruning and the port prunes it.
partykit's `mob_prune` counts `length(coefficients)` per node
(modelparty.R:646); the port counts theta too (docs/decisions.md, "Pruning
df counts theta for negative binomial"). With R's own objectives, R's df
(3, 7) keeps the split by 0.86 AIC and the port's (4, 9) prunes it by 1.14.
With the extra df removed, two rounding differences follow: theta_se at
node 86 (theta 272, "alternation limit reached", 1.4e-8 relative) and
glm.nb's alternation at node 112, where both R's and the port's theta
jitter by 1e-8 to 2e-7 around epsilon = 1e-8: R's step happened to fall to
4.3e-9 at alternation 24, the port's stayed above it through 25 and it
reports "alternation limit reached" (final thetas agree to 4.5e-15). With
all three allowed, the case matches R on every component.

Task 6k: pruning counts the coefficients only, as partykit does; theta
still counts in `loglik()`, `aic()` and `bic()`, as in `logLik.negbin`. The
two rounding differences are covered by the theta comparison of
`blind_29_negbin`.

## `blind_43_safelogit` and `blind_47_safelogit`: slopes of a separated leaf

Node 3 is a constant-response subgroup (y = 1). Both R and the port find
complete separation and stop it with `no_estfun`; glm.fit stops after 25
iterations with the intercept at 26.566. The slopes are not identified (their
effect on the fitted values is about 1e-19) and are rounding residue: x1 =
2.1e-16 against R's -7.7e-8 (`blind_43`), x2 = -5.1e-11 against R's -1.2e-9
(`blind_47`), with atol 1e-10. The port's `reference_path()` gives different
residue again, while the objective equals R's bit for bit. With node 3's
slopes exempt, both cases match R on every component.

Task 6k: for a completely separated leaf `tests/blind_compare.py` compares
the linear predictor on the leaf's rows at `SEPARATED_RTOL`, the fitted
means at the usual tolerance and the objective relatively (it is below the
absolute tolerance), instead of each coefficient. A 1e-6 relative change of
the objective or of one fitted mean is still caught.
