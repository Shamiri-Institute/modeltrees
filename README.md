# modeltrees

modeltrees fits model-based recursive partitioning (MOB) trees in Python.
MOB (Zeileis, Hothorn and Hornik 2008) fits a parametric model, tests whether
its coefficients are stable across a set of partitioning variables, splits
the data on the most unstable variable, and repeats in each part. The
result is a tree whose leaves are subgroups with their own model.

The core is a port of R's partykit 1.3-0 (`mob`, `lmtree`, `glmtree`) and is
tested against values computed by R. It adds a negative binomial model
ported from Rusch (2012) and MASS, and two models designed in this project:
a safe logit and a multinomial logit.

It is for people who would use partykit's `lmtree` or `glmtree` but work in
Python, and for people who need count or multiclass responses or
cluster-robust leaf covariances on grouped data. The dependencies are
numpy, scipy and narwhals. Input can be any eager dataframe that narwhals
supports (polars, pandas, pyarrow and others) or a dict of numpy arrays.

modeltrees is unrelated to the `py_mob` package on PyPI, which does
monotonic binning.

## Install

modeltrees needs Python 3.13 or later. Once it is released on PyPI:

```
pip install modeltrees
```

Until then, from a checkout: `uv sync`, or `pip install .`.

## Example

A simulated trial: pupils in 40 schools, a binary outcome, and a treatment
that helps only the older pupils in two regions. The tree looks for
subgroups in which the intercept and the treatment effect differ, and uses
the schools as clusters in its instability tests.

```python
import numpy as np
import polars as pl

import modeltrees as mt

rng = np.random.default_rng(42)
n = 1000
school = rng.integers(0, 40, size=n)
school_effect = rng.normal(scale=0.5, size=40)[school]
age = rng.uniform(10, 18, size=n)
region = rng.choice(["north", "south", "east", "west"], size=n)
treated = rng.integers(0, 2, size=n)
effect = np.where((age > 14) & np.isin(region, ["north", "east"]), 1.5, 0.0)
eta = -0.5 + effect * treated + school_effect
improved = rng.uniform(size=n) < 1 / (1 + np.exp(-eta))

df = pl.DataFrame(
    {
        "improved": improved,
        "treated": treated,
        "age": age,
        "region": region,
        "school": school,
    }
)

tree = mt.glmtree(
    "improved ~ treated | age + region",
    df,
    family="binomial",
    cluster="school",
)
print(tree)

new = pl.DataFrame(
    {"treated": [1, 1], "age": [16.0, 12.0], "region": ["north", "south"]}
)
print(tree.predict(new, type="node"))  # leaf ids
print(tree.predict(new, type="response"))  # probabilities

print(tree.coef_table(layout="long", se="robust"))

for node, cov in tree.vcov().items():
    print(node, cov.n_clusters, cov.std_errors)
```

The printed tree:

```
Generalized linear model tree (family: binomial)

Model formula:
improved ~ treated | age + region

Partitioning variables: age (numeric), region (nominal)
n = 1000

Fitted tree:
[1] root
|   [2] region in {east, north}
|   |   [3] age <= 15.42475: n = 348
|   |       (Intercept) treated
|   |           -0.7559  0.6104
|   |   [4] age > 15.42475: n = 162
|   |       (Intercept) treated
|   |          -0.02151   1.063
|   [5] region in {south, west}: n = 490
|       (Intercept) treated
|           -0.3489 -0.2222

Number of inner nodes:    2
Number of terminal nodes: 3
Number of parameters per node: 2
Objective function (negative log-likelihood): 660.0157
```

The formula is `response ~ regressors | partitioning variables`. A formula
with one part, `y ~ z1 + z2`, means an intercept-only model partitioned by
`z1` and `z2` (`y ~ 1 | z1 + z2`), as in partykit; the printed formula shows
the expanded form. Transformations and interactions are not supported in
formulas: compute the column in the dataframe first. The full API is in
docs/api-design.md.

## Models

| entry point | node model | provenance |
| --- | --- | --- |
| `lmtree` | linear regression | ported from partykit 1.3-0 (`lmtree`, `lmfit`) and R's `lm.fit` |
| `glmtree` | GLM: gaussian, binomial, poisson with canonical links | ported from partykit 1.3-0 (`glmtree`, `glmfit`) and R's `glm.fit` |
| `negbintree` | negative binomial with log link, theta estimated per node | ported from the thesis code (Rusch 2012) and MASS 7.3-66 (`glm.nb`, `theta.ml`) |
| `safelogittree` | binary logit that stops at separated nodes | this project's design, provisional rules |
| `multinomtree` | baseline-category multinomial logit | this project's design, provisional rules |

`mob` takes arrays and any object that implements the node-model protocol
(docs/model-interface.md). The protocol is provisional in 0.1.

Rusch's thesis names a safe logit and a multinomial model but gives no code
for them, and the package that held them (mobtools) is not available. Their
rules here (the separation threshold, terminal separated nodes, degenerate
classes, rank-deficient fits counted as failures) are this project's
choices, specified in docs/extensions-spec.md and docs/decisions.md. Their
R oracle is this project's own R code in `r/extensions/`, not a published
implementation, and the rules may change.

## How correctness is established

R is the oracle. Every behavior that R has is checked against values R
computed with the pinned packages (partykit 1.3-0, strucchange 1.6-0,
sandwich 3.1-3, MASS 7.3-66, nnet 7.3-21, R 4.6.1). The R scripts in `r/`
write the fixtures to `tests/fixtures/` as JSON and CSV with 17 significant
digits, and the fixtures are committed. The Python tests read them and
never call R.

The comparison goes component by component, so that a failure points at
the first part that diverges: scores, instability statistics, p-values,
split search, then whole trees. Tree structure, split variables and split
points must match exactly. Coefficients, objectives and statistics are
compared at `rtol = 1e-8` in most tests; the observed agreement is usually
1e-14 or better (docs/extensions-port.md, section 4). Where R itself is
less accurate, the tolerance is derived from a measured quantity instead of
chosen: nnet stops its optimisation with a gradient of about 1e-8, so the
multinomial tests use twice the coefficient offset that gradient explains
(docs/extensions-port.md, section 7). Tolerances are never loosened to make
a test pass.

The fixture families (tests/fixtures/README.md lists every file and the R
call that made it):

| fixtures | what they check |
| --- | --- |
| `pvalues.json` | partykit's supLM p-values from Hansen's table over 336 combinations of trimming and k, the chi-squared factor tests and the Bonferroni and Sidak adjustment |
| `glm/` | the `glm.fit` port: coefficients, deviance, AIC, iterations, convergence, rank deficiency, step halving, near separation |
| `trees/` | 32 lmtree and glmtree trees grown by partykit, with every node's tests, raw and adjusted p-values, every split-search objective, the scores and the fitted nodes; weights, zero weights, offsets, clusters, pruning, `restart = FALSE` |
| `node_fits/` | single fits of the extension models, with scores checked in R against analytic and finite-difference derivatives |
| `extension_trees/`, `cluster_trees/` | negative binomial, safe logit and multinomial trees grown by `partykit::mob` with this project's R fit functions |
| `thesis_quine_crosscheck.json` | the negative binomial tree printed in Rusch's thesis, reproduced at its printed precision |
| `vcovcl/` | `sandwich::vcovCL` on every leaf of seven clustered trees, HC0 and HC1, with and without `cadjust` and clusters |
| `predict/` | partykit's `predict` on new rows at split points, with absent levels and missing values |

Where R has no oracle (the safe logit and multinomial rules, failure
handling, theta in the pruning df, frequency weights in `vcov`), unit tests
check the behavior, some against statsmodels.

Faster code paths must give the same results. The exact per-candidate code
they replace stays in the library and runs under
`modeltrees.reference_path()`; the tests compare both paths with each other
and with R (docs/decisions.md).

## Differences from R

These are deliberate. Each is recorded in docs/decisions.md or
docs/api-port.md.

- **String order.** Levels of string columns are sorted by Unicode code
  point (Python's `sorted`), not by the locale as R does. Pass `levels=` or
  use an Enum or categorical dtype when the order matters, for example
  with mixed-case or accented labels.
- **Ordered regressors** are treatment-coded, not polynomial-coded as R's
  default contrasts do. Ordered partitioning variables behave as in R.
- **Missing split variables in `predict`.** A row with a missing value in a
  split variable on its path gets no prediction (node 0, NaN). R routes it
  to a random kid.
- **`-Inf`** in a numeric split variable goes to the left kid, as
  `-Inf <= break` says. R routes it at random.
- **Unseen levels.** A level not seen in training raises
  `UnseenLevelError`; `unseen="null"` gives those rows no prediction. A
  training level absent from a node goes to the kid with more observations.
- **Zero weights** act as deleted rows. partykit's result with zero weights
  differs from its result with the rows removed; modeltrees gives the
  second.
- **Depth** counts the root as 1, so that `maxdepth` bounds
  `ModelTree.depth`. partykit's `depth()` is one less.
- **Counts must be whole numbers** for poisson and negative binomial
  responses. R only warns for non-integer counts.
- **`vcov` defaults to HC1 with `cadjust=True` for every model.**
  `sandwich::vcovCL` uses HC1 only for `lm` fits and HC0 for every other
  model. To compare a glmtree, negbintree, safelogittree or multinomtree
  leaf with R, pass `type="HC0"`.
- **One cluster** in a leaf (G = 1) gives a NaN covariance with status
  `"unavailable"`. sandwich 3.1-3 stops with an error.

## Limitations

- **Negative binomial trees do not detect a change in overdispersion
  alone.** Only the regression coefficients are tested; theta is estimated
  in each node but has no score column, as in the thesis. An option to
  test theta is deferred (docs/extensions-spec.md, Q5). Leaf
  covariances are conditional on theta, as `vcov(glm.nb)` is, which
  understates the uncertainty in small leaves.
- **Quasi-separation is reported, not flagged by the covariance status.**
  For a quasi-separated safe logit leaf, `vcov()` warns and names the
  diverging coefficient in `LeafVcov.reason`, but `status` can still be
  `"ok"`, because the diverging direction need not make the scaled Hessian
  ill conditioned (docs/api-port.md, section 15.1, item 6). Complete
  separation makes a leaf terminal without scores, and its covariance is
  `"unavailable"`.
- **Inference after selection.** The leaves were chosen by tests on the
  same data, so any standard error, robust or not, is conditional on the
  tree and too small for inference about the subgroups. modeltrees reports
  standard errors but no p-values or confidence intervals for leaf
  coefficients. Growing the tree on one part of the data and estimating on
  the other is the honest route.
- **Scope.** Binary splits only; no surrogate splits or random routing of
  missing values; the instability tests use `vcov = "opg"` only; options
  that need R's random numbers (`mtry`, `breakties`, `ordinal = "L2"` and
  `"max"`) raise `NotImplementedError`; glmtree has gaussian, binomial and
  poisson with canonical links only.
- **Split search cost.** Every candidate split fits the model on both sides.
  lmtree uses a batch scorer and is close to linear in the node size; the
  other models are quadratic (next section).

## Performance and parallel split search

Timings from docs/performance.md (one machine with 12 cores, a tree with 3
planted splits, single-threaded BLAS):

| model | n | serial | 4 jobs | 8 jobs | R partykit |
| --- | --- | --- | --- | --- | --- |
| lm | 5000 | 0.034 s | 0.036 s | 0.037 s | 3.2 s |
| lm | 20000 | 0.100 s | 0.104 s | 0.104 s | 43 s |
| glm (binomial) | 5000 | 13.6 s | 5.1 s | 4.2 s | 28.6 s |
| glm (binomial) | 20000 | 128 s | 51.5 s | 42.1 s | 450 s |
| negative binomial | 5000 | 111 s | 38.8 s | 31.4 s | |

The R column is from an earlier run on the same machine with partykit's
default settings.

**BLAS threads.** Set `OPENBLAS_NUM_THREADS=1` (or `OMP_NUM_THREADS=1`)
before Python starts when nodes are large or when several fits run at once
(`n_jobs`, or modeltrees inside your own parallel jobs). The matrices are
narrow, and with the default threads the reference path ran 2.3 times
slower at n = 20000. Workers started by `n_jobs` pin themselves to one
thread; the parent process is yours to set.

**`n_jobs`.** `n_jobs=k` fits the candidate splits of large searches in k
worker processes; `n_jobs=-1` uses every CPU. A search runs in parallel
only when its candidate count times the node's rows exceeds 5e5, so small
trees start no processes. lmtree's numeric searches use its batch scorer
and stay serial, and `restart=False` runs serially with a warning.

The tree grown with any `n_jobs` is identical to the one grown with
`n_jobs=1`, bit for bit, in every stored number, provided the serial run
also uses single-threaded BLAS. The tests check this on every glm,
negative binomial, safe logit and multinomial fixture tree.

Workers are started with `forkserver` on Linux and `spawn` elsewhere. Both
import your main module in each worker, so a script that uses `n_jobs > 1`
needs the guard:

```
if __name__ == "__main__":
    tree = mt.glmtree(..., n_jobs=4)
```

Custom models must be importable by the workers (defined in a module, not
in an interactive `__main__`). While a pool is running, the three BLAS
variables are set to 1 in `os.environ`; the forkserver keeps the
environment it started with, so later forkserver children of the same
process also get one BLAS thread (docs/api-port.md, section 17).

## Citing

If you use modeltrees, please cite the method and the software
(CITATION.cff):

- Zeileis, A., Hothorn, T. and Hornik, K. (2008). Model-based recursive
  partitioning. Journal of Computational and Graphical Statistics 17(2),
  492-514. doi:10.1198/106186008X319331
- Rusch, T. (2012). Recursive Partitioning of Models of a Generalized
  Linear Model Type. Doctoral thesis, WU Vienna.
  doi:10.57938/f43bea09-fe60-44fb-a995-ba63733c8c69 (for the negative
  binomial model)

## License

GPL-3.0-only (LICENSE). The ported R code is licensed GPL-2 or GPL-3 by its
authors; THIRD_PARTY_NOTICES.md lists every upstream with its license.

## Development

```
uv sync                 # the package and the dev group
uv run pytest           # default run, slow tests excluded
uv run pytest -m ""     # every test
uv run pytest -m slow   # only the slow tests
uv run ruff check
uv run ruff format --check
uv run mypy src tests scripts
```

Tests marked `slow` take more than about 5 seconds each or as a
parametrized group: growing the larger negative binomial and multinomial
fixture trees, the parallel fixture trees and the benchmark tree. They are
excluded by default through `addopts` in pyproject.toml. On a 12-core
machine with single-threaded BLAS the default run takes about 3 minutes
(168 s) and the full run about 8 (474 s).

The fixtures are regenerated with R 4.6.1 and the renv environment in `r/`:

```
cd r && Rscript -e 'renv::restore()' && cd ..
Rscript r/generate_all.R
```

The output is deterministic. r/README.md describes the R setup, and
CONTRIBUTING.md the oracle workflow.

Commit messages are one line with no body, starting with a capital letter,
for example `Add glm.fit port with binomial family`.
