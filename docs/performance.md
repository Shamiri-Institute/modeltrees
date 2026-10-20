# Performance of `mob()`

Timings and profiles of the tree-growing loop: first the baseline, measured
before any optimization, then the speedups (Step 1, per-fit overhead; Step 2,
batch split objectives; Step 3, results and options for glm). Script:
`scripts/bench_tree.py` (`uv run python scripts/bench_tree.py`). The rule
for every speedup is that results do not change: the code it replaces stays
as the reference path (`modeltrees.reference_path()`, docs/decisions.md) and
the tests compare both paths with each other and with the R fixtures.

## Setup

- Data: 6 partitioning variables, `z1` to `z4` numeric (uniform on [-1, 1]),
  `z5` ordered with 5 levels, `z6` nominal with 5 levels. One regressor `x`
  plus the intercept. Seed `20261101 + n`.
- Planted structure: the coefficients change at `z1 <= 0`, then at `z5 <= 2nd
  level` on the left and at `z6 in {a, c, d}` on the right. The gaussian
  response has unit noise, the binary response uses the logistic link on the
  same linear predictor.
- Models: `LmModel()` and `GlmModel("binomial")`. Default `MobControl` (alpha
  0.05, Bonferroni, default `minsize` = 10 * 2 = 20, no `maxdepth`, no pruning).
- Machine: 12 cores, numpy 2.5.4, Python via `uv run`, single process.
  Timings are one machine on one day and are only comparable to each other.
- One untimed warm-up run, then 3 timed runs for n = 1000 and 5000, and 1 timed
  run for n = 20000 (the glm run takes 9 minutes). Profiles are from one extra
  run under `cProfile`, which adds 15 to 30 % overhead.

## Baseline wall time

| model | n | nodes | leaves | depth | min s | median s | max s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lm | 1000 | 7 | 4 | 3 | 0.300 | 0.302 | 0.302 |
| glm | 1000 | 7 | 4 | 3 | 2.540 | 2.582 | 2.652 |
| lm | 5000 | 7 | 4 | 3 | 2.236 | 2.246 | 2.264 |
| glm | 5000 | 7 | 4 | 3 | 22.924 | 22.957 | 23.057 |
| lm | 20000 | 7 | 4 | 3 | 49.702 | 49.702 | 49.702 |
| glm | 20000 | 7 | 4 | 3 | 549.081 | 549.081 | 549.081 |

Every run recovers the planted tree: 3 splits, 4 leaves, depth 3 (the root has
depth 1).

### R reference point

`scripts/bench_r.R` times partykit 1.3-0 (`lmtree` and `glmtree` with
`family = binomial()`, default settings, R 4.6.1) on the same data, which
`bench_tree.py --full --write-csv DIR` writes with 17 significant digits.
One warm-up, then 3 timed runs per cell, on the same machine:

| model | n | nodes | leaves | depth | min s | median s | max s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lm | 1000 | 7 | 4 | 3 | 0.219 | 0.219 | 0.245 |
| glm | 1000 | 7 | 4 | 3 | 1.601 | 1.620 | 1.791 |
| lm | 5000 | 7 | 4 | 3 | 3.150 | 3.184 | 3.185 |
| glm | 5000 | 7 | 4 | 3 | 28.491 | 28.557 | 28.564 |
| lm | 20000 | 7 | 4 | 3 | 42.791 | 43.454 | 44.040 |
| glm | 20000 | 7 | 4 | 3 | 443.191 | 449.882 | 458.993 |

R grows the same tree shape. The Python baseline above is within a factor
of 1.6 of R everywhere: slower at n = 1000, faster for lm at 5000 and
20000 and for glm at 5000, and 20 % slower for glm at 20000. partykit also
refits both sides of every candidate, so R scales the same way.

Scaling: lm grows 7.5x from n = 1000 to 5000 and 22x from 5000 to 20000 (n
grows 5x and 4x). glm grows 8.9x and 24x. That is roughly quadratic in n. The
glm tree costs about 8 to 11 times the lm tree at every size.

## Baseline profile

All of it is in the split search. At every size `find_split` accounts for 99.9
% of `mob()`'s time; the instability tests, routing and pruning are below the
noise. `find_split` is called 3 times (once per node that splits, the root and
its two children), and each call fits two models for every candidate split of
every numeric variable:

| run | model fits | fits / n |
| --- | --- | --- |
| lm, n = 20000 | 39967 | 2.0 |
| glm, n = 20000 | 39967 | 2.0 |
| glm, n = 5000 | 9967 | 2.0 |

The candidates are all distinct values of `z1` to `z4` (about n each) at the
root, and fewer in the children. Each fit costs a fixed small number of numpy
calls on arrays that grow with n, so total cost is about (number of fits) x
(cost per fit), where both scale with n.

### lm, n = 20000 (48.3 s under cProfile)

| function | calls | own s | cumulative s |
| --- | --- | --- | --- |
| `split_search.find_split` | 3 | 0.4 | 48.5 |
| `tree.side_fit` | 39960 | 4.5 | 42.7 |
| `tree.fit` (model call wrapper) | 39967 | 10.9 | 38.2 |
| `lm.fit` | 39967 | 1.9 | 27.3 |
| `lm_fit.lm_fit` | 39967 | 0.8 | 22.5 |
| `_qr.dqrls` | 39967 | 2.6 | 21.0 |
| `_qr.dqrdc2` | 39967 | 4.2 | 9.4 |
| `_qr._apply_householder` | 159868 | 6.4 | 7.3 |
| `split_search.admissible` | 20019 | 4.6 | 5.4 |

The R-style QR (`dqrls` plus `dqrdc2`) is 43 % of the time. The wrapper
`tree.fit` has the largest own time (10.9 s, 23 %): per-fit array handling
outside the model.

### glm (binomial), n = 20000 (723 s under cProfile)

| function | calls | own s | cumulative s |
| --- | --- | --- | --- |
| `split_search.find_split` | 3 | 0.6 | 724.0 |
| `glm_fit.glm_fit` | 39967 | 116.3 | 697.6 |
| `_qr.dqrls` | 159872 | 10.9 | 279.4 |
| `_qr.dqrdc2` | 159872 | 46.8 | 135.4 |
| `families.dev_resids` | 239806 | 133.0 | 134.8 |
| `_qr._apply_householder` | 639488 | 121.6 | 125.7 |
| `ndarray.dot` | 319744 | 69.9 | 69.9 |
| `families.linkinv` | 199839 | 38.0 | 39.7 |
| `families.mu_eta` | 199839 | 35.5 | 36.9 |
| `families._log_dbinom` (AIC) | 39967 | 34.2 | 34.2 |

Each `glm_fit` runs 4 IRLS iterations on average (159872 / 39967), so a glm
fit is four weighted least squares solves plus a deviance evaluation per
iteration. By own time: `dev_resids` 18 %, `_apply_householder` 17 %, the
`glm_fit` body 16 %, matrix products 10 %, `dqrdc2` 6 %, `linkinv` and `mu_eta`
10 % together, and the binomial log-likelihood used only for the AIC 5 %.
Cumulatively, the QR is 39 % and the deviance evaluation about 19 %.

### Smaller sizes

The profile at n = 5000 for glm (26.9 s) has the same shape: `glm_fit` 25.8 s
cumulative over 9967 fits, `dqrls` 6.0 s, `dev_resids` 5.4 s own,
`dqrdc2` 3.1 s, `linkinv` 1.8 s, `mu_eta` 1.6 s. The n = 1000 and lm n = 5000
profiles are in the script's output and look the same, with proportionally
less time.

## Baseline observations

- The split search refits from scratch for every candidate on the full node
  data, so the cost is quadratic in the node size. The Python overhead per fit
  (about 1 ms for lm, 18 ms for glm at n = 20000) is dominated by array work
  that scales with n, not by interpreter overhead.
- The `tree.fit` wrapper and `side_fit` spend 15 s of own time in the lm run on
  copying and slicing the node data for each candidate.
- The AIC computation (`_log_dbinom`) runs on every fit in the split search,
  although only the objective is used there.
- numpy's BLAS threading made the glm run use more than one core (the first
  attempt showed about 1000 % CPU); the timings above are with the default
  thread settings, not pinned.

These were measurements only; Steps 1 and 2 below act on them.

## Step 1: per-fit overhead

Measured after each change with `bench_tree.py`'s data and settings, at the
default sizes (n = 1000 and 5000), one warm-up and 3 timed runs per cell,
each configuration in a fresh process. The changes are cumulative in the
order a to d. Run-to-run noise between processes is about 5 to 10 %, so
differences below that are not effects.

### a. BLAS threads

The unchanged code (the reference path) with the thread variables set
before numpy is imported. Median seconds:

| setting | lm 1000 | lm 5000 | glm 1000 | glm 5000 |
| --- | --- | --- | --- | --- |
| default (12 threads) | 0.367 | 2.43 | 2.19 | 19.96 |
| `OMP_NUM_THREADS=1` | 0.324 | 2.62 | 2.20 | 19.34 |
| `OPENBLAS_NUM_THREADS=1` | 0.375 | 2.38 | 2.34 | 20.61 |
| both 1 | 0.340 | 2.70 | 2.52 | 22.29 |
| `OPENBLAS_NUM_THREADS=4` | 0.350 | 2.66 | 2.53 | 23.10 |

No setting is faster beyond the noise. The matrices in a node fit are n by
2, so OpenBLAS runs its products on one thread anyway, and process CPU use
stays near 100 to 120 % with the default (the 1000 % seen in the baseline
was at n = 20000). Decision: no change and no recommendation to pin threads.
Setting `OPENBLAS_NUM_THREADS=1` is harmless if mob runs inside a
multiprocess job.

### b to d. Changes and their effect

Median seconds, cumulative (`base` is the reference path):

| config | lm 1000 | lm 5000 | glm 1000 | glm 5000 |
| --- | --- | --- | --- | --- |
| base | 0.310 | 2.77 | 2.52 | 22.73 |
| + b, objective-only glm fits | 0.317 | 2.89 | 2.58 | 23.10 |
| + c, node-local data, vectorised admissibility | 0.300 | 3.05 | 2.63 | 22.60 |
| + d, LAPACK least squares | 0.257 | 2.29 | 2.16 | 21.24 |

Every configuration grows the same tree (same splits and breaks).

**b. Objective-only fits.** `GlmModel.fit` with `estfun=False` (split
search) calls `glm_objective`, which runs the same IRLS code as `glm_fit`
(`models/glm_fit.py`, `_irls`) in quiet mode: no warnings, no null deviance,
no working residuals or weights. The AIC is computed exactly as before,
since the objective is `aic / 2 - df`. Coefficient names stay, because
`FitResult` carries them and they cost a tuple. `LmModel` already skipped
the scores and has nothing else to skip. Tests check that the objective-only
fit equals the full fit bit for bit. The IRLS core also skips the copies
`x[good]` and friends when every row is good, which leaves every value
unchanged. Effect: none measurable. The skipped work (one deviance
evaluation and a few vector operations per fit, against about 4 IRLS
iterations of 6 to 8 vector operations and a solve each) is a few percent,
inside the noise.

**c. Node-local data and vectorised admissibility.** Split search now
extracts the node's `y`, `x`, weights, offset and clusters once per node
(`tree._Grower._node_fitters`) and masks those contiguous arrays per
candidate, instead of indexing the full data with `rows[mask]`. Admissibility
of all numeric candidates comes from one cumulative weight sum; candidates
within 1e-8 of `minsize` (relative to the total weight) are rechecked with
the reference's masked sums, so fractional weights cannot flip a candidate.
Effect: none measurable at these sizes; the fancy indexing and per-candidate
`w2n` were 15 to 25 % of the lm profile only at n = 20000.

*Ambiguity, decided: masks, not sorted views.* The task suggested sorting
once per variable and fitting on prefix and suffix slices (views). The data
is sorted once per variable (for Step 2), but the side fits still use masks
in the original row order. Measured on the root node at n = 5000 (4961
candidates, per-candidate fits): lm 2.28 s with masks, 1.30 s with views;
glm 23.0 to 26.6 s with masks, 23.9 to 25.9 s with views (no difference).
Views change the row order inside each fit, and with it the rounding: 27 %
of the candidate objectives differ in the last bits (up to 4e-16 relative)
from the reference. The lm gain disappears in Step 2, which replaces the
per-candidate lm fits, and glm does not gain, so the bit-identical masks
were kept.

**d. LAPACK least squares.** `_qr.dqrls` first tries `dgeqrf`, `dormqr`
and `dtrtrs` through `scipy.linalg.lapack` (`dqrls_lapack`) and keeps the
result only if every column keeps at least 1e-4 of its norm after the
earlier columns are projected out (`|R[j, j]| >= 1e-4 ||x_j||`, or 1000
times `tol` if that is larger). That ratio is exactly the quantity dqrdc2
compares with `tol` (1e-7 in `lm.fit`, 1e-11 in `glm.fit`), so the fast
path is taken only where dqrdc2 would certainly keep full rank and the
column order. Without pivoting dqrdc2 is Householder QR with LAPACK's sign
convention, so both compute the same factorisation, the residuals are formed
the same way (`Q (0, Q'y[p:])`), and they differ only in rounding. Anything
doubtful (rank deficiency, near-collinearity, a zero column, `n <= p`) goes
to the port. Per solve the fast path is 3 to 8 times faster (80 to 210 us
for the port at n = 40 to 10000, 10 to 73 us for LAPACK). Effect on the
trees: 15 to 20 % for lm and 7 to 15 % for glm.

Tests (`tests/test_fast_paths.py`): the fast path equals the port to 1e-10
(coefficients relative, residuals and effects relative to `||y||`) on random
data with scaled columns, weighted data (`sqrt(w)` scaling), tied values,
a duplicated column, collinear and near-collinear columns (offsets 0, 1e-9,
1e-6 decline the fast path, 1e-2 takes it), a column keeping 1e-5 of its
norm (dqrdc2 keeps it, the fast path declines and the port decides),
degenerate shapes, and `reference_path()` turning it off. The observed
agreement is about 1e-16 relative for coefficients.

### Profile after Step 1, glm, n = 5000 (25.4 s under cProfile)

| function | calls | own s | cumulative s |
| --- | --- | --- | --- |
| `glm_fit._irls` | 9967 | 3.4 | 21.7 |
| `families.dev_resids` | 49847 | 5.3 | 5.5 |
| `_qr.dqrls` (LAPACK path) | 39873 | 0.4 | 4.8 |
| `families.linkinv` | 49840 | 2.0 | 2.2 |
| `families.mu_eta` | 39880 | 1.4 | 1.6 |
| `families._log_dbinom` (AIC) | 9967 | 1.4 | 1.4 |
| `tree.side_fit` (masking) | 9960 | 1.0 | 25.4 |

The time is now the IRLS arithmetic itself: element-wise vector work on
each side's rows (deviance residuals 21 %, link functions 15 %, the IRLS
body 13 %) and the solve (19 %). Per-fit overhead outside the model is
about 4 %.

## Step 2: batch split objectives for lm

A model may now score all numeric split candidates at once
(`split_objectives`, docs/model-interface.md, section 4.1); `mob()` uses it
when the model has it and `restart = TRUE`, and otherwise fits every
candidate as before. `LmModel` implements it with prefix and suffix sums of
the weighted cross-products over the rows sorted by the split variable.

**Exactness.** The scorer returns an error bound with each approximation,
and split search refits exactly every candidate whose interval could reach
the best (widened by 1e-6 relative to the best), at least the 5 smallest,
and every candidate the scorer declines (NaN). The winner is the first
minimum over values in which every refitted candidate is exact and every
other candidate is provably worse, so the chosen split and the
first-candidate tie rule are the per-candidate search's. Sides that are
rank-deficient, have a zero column, or whose diagonal-scaled cross-product
matrix has a condition number above 1e8 are declined. The approximation
works on the residuals of the node's own fit rather than on `y`, which
leaves each side's RSS unchanged and keeps the sums on the RSS scale, so a
large mean of `y` does not cancel digits. The bound per side is
`64 eps (n + p) (cond r'Wr + sqrt(r'Wr y'Wy))`; the second term covers
responses whose residuals are tiny next to `y`, where even two exact
least-squares codes differ by about `eps sqrt(RSS y'Wy)`.

**Results.** Median seconds (n = 20000: one run after a warm-up for the
earlier columns, three for Step 2):

| n | baseline | after Step 1 | after Step 2 | R partykit |
| --- | --- | --- | --- | --- |
| 1000 | 0.302 | 0.257 | 0.017 | 0.219 |
| 5000 | 2.246 | 2.29 | 0.040 | 3.184 |
| 20000 | 49.7 | not run | 0.101 | 43.45 |

lm trees are 18 times faster at n = 1000, 56 times at 5000 and about 490
times at 20000, and now grow 13 to 430 times faster than partykit. The cost
of split search is now linear in n up to the sort.

**How much is refitted.** At the benchmark's root (n = 5000, 4961
candidates in `z1`) 5 candidates are refitted, 10 side fits instead of
9922. In 600 randomized searches (`n` from 40 to 600, ties, heavy-tailed
weights, `y` shifted by 1e8, exact fits, scales from 1e-6 to 1e6, and
near-collinear `x`) the winner always equalled the reference's. About 5
candidates were refitted per search, except with near-collinear `x` (`x`
keeps 1e-6 of its norm against the intercept), where every side is declined
and every candidate is fitted exactly, as before Step 2.

**Agreement.** Every existing test passes at unchanged tolerances, and all
31 oracle trees still match R. Grown under the fast and the reference
path, every fixture tree has the same nodes, rows, splits and stop reasons,
and the split-search objectives of the 13 lmtree fixtures agree to 2e-15
relative at worst (`sim_gaussian_lmtree_zero_weights_deleted`, node 1),
unrefitted approximations included. No fixture result changed, so there is
no diverging quantity to record.

Tests (`tests/test_fast_paths.py`): `find_split` with and without the batch
scorer against `find_split_reference` (same candidates, admissible set,
winner and split; objectives to 1e-10, plus the `eps sqrt(RSS y'Wy)` term)
on random data with ties, fractional, integer, zero and heavy-tailed
weights, a regressor constant on one side, a 0/1 regressor, tiny nodes
(`minsize` 2), `y + 1e8`, exact fits and near-collinear `x`, with case
weights on and off; every scored candidate within its error bound; declined
sides; the minsize boundary with weights of 0.1; a tie between mirrored
halves going to the first candidate with only a handful of refits;
`numsplit = "center"` and factors; and every fixture tree and two
simulated trees against `reference_path()`.

**Profile now** (lm, n = 5000, 0.047 s under cProfile): split search
0.024 s, of which 0.011 s is the two factor searches (19 candidates, fitted
one by one) and 0.008 s the batch scorer (half of it the batched
eigendecompositions); the instability tests 0.021 s; node fits the rest.

## Reproducing

```
uv run python scripts/bench_tree.py                       # n = 1000, 5000, 3 runs, profiles
uv run python scripts/bench_tree.py --full                # also n = 20000
uv run python scripts/bench_tree.py --sizes 1000 --runs 5 --no-profile
uv run python scripts/bench_tree.py --full --write-csv DIR
Rscript scripts/bench_r.R DIR 3 1000 5000 20000           # partykit on the same data
```

The default sizes are n = 1000 and 5000. With `--full`, the baseline run
took more than 40 minutes, almost all of it the glm runs at n = 20000; the
R script at n = 20000 takes 35 minutes.
