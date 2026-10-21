# Blind differential runs: the mismatching cases

Two blind runs are recorded here: task 6j (below) and task 6l, which
repeated it on new seeds after the fixes of task 6k (last section). Case
names number the cases within their run; `cases/<case>.json` gives the
seed.

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

## Second blind run (task 6l)

Task 6l repeated the run with the same harness, case grid, settings draws
and comparisons (`tests/blind_compare.py` as task 6k left it) on seeds from
`numpy.random.SeedSequence(20261023).generate_state(60) >> 1`, to check
that the task 6k fixes introduced no new difference. Two additions to the
grid: each model has at least two clustered cases with case weights (here
exactly two), and in those every partitioning variable is constant within
clusters, so nodes come down to G = k clusters for k coefficients and
below. Their R trees had 39 nodes with G = k (37 tested by R; the 31 in
the matching cases, 25 of them split, all match the port) and 94 with
G < k. The vcov of
weighted clustered leaves was compared against the formula of
docs/api-design.md 5.1 evaluated in R on R's own scores; their
`vcov/<case>.json` holds it as `formula` next to `vcovcl`, which is no
oracle under weights (for the multinomial not even for HC0: sandwich's
`estfun.multinom` drops the case weights of the refit).

57 of the 60 cases matched on every component. The 3 below did not; they
are copied here unchanged and are xfail (strict) in
`tests/test_blind_cases.py` with their first diverging component. Nothing
was fixed.

| case | model | n | first diverging component (6l) | classification |
| --- | --- | --- | --- | --- |
| `blind_22_glm` | glmtree poisson, clustered, weighted | 874 | whole tree (node 9 stop reason) | rounding: R's `chol()` of a singular meat at G = 2 < k = 3 succeeds |
| `blind_52_multinom` | multinomfit, clustered, weighted | 621 | node sizes (node 6) | rounding: same at G = 3 < k = 4 (node 6), then AIC prunes node 5 in the port |
| `blind_60_multinom` | multinomfit, weighted | 778 | split candidates and objectives (node 6) | documented decision: nnet stops at maxit, the port's Newton fit reaches the optimum |

### `blind_22_glm` and `blind_52_multinom`: R's Cholesky at G < k

R ran the instability tests at `blind_22_glm` node 9 (G = 2, k = 3,
weights 1 to 4) and `blind_52_multinom` node 6 (G = 3, k = 4). With
G < k the meat, the crossproduct of the G cluster sums of
`process / sqrt(weights)` (modelparty.R:244), has rank G under any
weights. On R's own scores its eigenvalues are 0.208, 0.0222, 1.1e-17 and
0.502, 0.0459, 0.00959, 8.3e-18: R's `chol()` happens to succeed, and the
statistics (9.8e15 to 2.5e17, p = 0) are the inverse of the rounding noise.
On copies with 1e-15 relative noise R's `chol()` succeeds about half the
time (99 and 112 of 199), and the port's `cho_factor` fails on R's meat
itself. Of the 70 weighted nodes with G < k that R tested in the run, its
`chol()` failed at 68, where the port (which stops at G < k,
docs/decisions.md) matches. In `blind_22_glm` R then prunes node 9 (BIC),
so only its stop reason differs; in `blind_52_multinom` R splits node 6
and keeps the split, while in the port node 6 is terminal and AIC prunes
node 5 by 0.072 (650.7901 against 650.8623 with R's own objectives). With
R's `chol.default` traced to stop when the smallest/largest eigenvalue is
below 1e-12, R grows trees the port matches on every component in both
cases. Not a port defect; docs/decisions.md's "partykit silently stops
splitting" at G < k holds only when rounding makes `chol()` fail.

### `blind_60_multinom`: nnet's iteration limit in split search

At node 6, 2 of 159 candidate objectives on z3 (133 and 134) differ by
4.3e-8 and 4.0e-8 relative (R higher by 1.8e-5); the other 157 agree to
4e-13, and the chosen split is the same. In both, the right child (25 and
24 rows, classes 4/14/3/4 and 4/13/3/4) stops in nnet with
`convergence = 1` (maxit 1000) and summed scores about 5e-4; the port's
Newton fit converges with summed scores 3e-11. R's own multinomial
log-likelihood is lower at the port's coefficients by exactly the
difference (1.83e-5 and 1.72e-5). With these two candidates exempt, the
case matches R on every component. docs/extensions-spec.md (Q10): R's
multinomial estimates are accurate only to about the gradient's size, and
the port uses Newton's method rather than nnet's BFGS.
