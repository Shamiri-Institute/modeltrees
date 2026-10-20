# Performance of `mob()`

Baseline timings and profiles of the tree-growing loop as it is. Nothing was
optimized or tuned for this document. Script: `scripts/bench_tree.py`
(`uv run python scripts/bench_tree.py`).

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

## Wall time

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

## Where the time goes

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

## Observations (not acted on)

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

These are measurements, not changes: no code in `src/` was touched for this
document.

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
