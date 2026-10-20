# Model interface for MOB

This document specifies what a node model must provide to the MOB algorithm,
how the algorithm uses each piece, and the tree data structures the Python port
needs. It is derived from partykit 1.3-0. Every behavioral claim cites the R
source in `reference/partykit/` by file and function. Line numbers refer to the
pinned 1.3-0 sources and are given as `file:line`.

Abbreviations used below:

- `modelparty.R` means `reference/partykit/R/modelparty.R`.
- `glmtree.R`, `lmtree.R`, `node.R`, `split.R` and `mob-pvalue.R` are in the
  same directory.
- `mob.Rnw` means `reference/partykit/vignettes/mob.Rnw`.
- `mob_grow`, `mob_grow_fluctests`, `mob_grow_findsplit` and `mob_prune` are
  closures defined inside `mob_partynode()` in `modelparty.R`. They are not
  reachable from outside the package.

## 1. What a model must provide to `mob()`

### 1.1 The two fit interfaces

`mob()` accepts a `fit` function in one of two forms (`mob.Rnw`, section
"Model fitting and parameter estimation"; `modelparty.R:4-36`, `mob()`).

**Type 1, full interface.**

```r
fit(y, x = NULL, start = NULL, weights = NULL, offset = NULL, cluster = NULL, ...,
    estfun = FALSE, object = FALSE)
```

It returns a list with four elements (`mob.Rnw`, same section):

| Element        | Meaning                                                         | Required                  |
| -------------- | --------------------------------------------------------------- | ------------------------- |
| `coefficients` | Estimated parameter vector theta-hat.                           | Always                    |
| `objfun`       | Minimized objective, the sum over observations of Psi(y, x, θ). | Always                    |
| `estfun`       | n by k matrix of per-observation score contributions.           | Only when `estfun = TRUE` |
| `object`       | A fitted model object for downstream methods.                   | Only when `object = TRUE` |

`mob()` checks the formals by name. It stops with "no suitable fitting
function specified" unless `y`, `x`, `start`, `weights` and `offset` are all
formals (`modelparty.R:5-8`). If `estfun` and `object` are both formals, the
function is used as type 1. If `cluster` is not a formal, `mob()` wraps the
function so that `cluster` is accepted and dropped (`modelparty.R:28-35`).

**Type 2, simple interface.**

```r
fit(y, x = NULL, start = NULL, weights = NULL, offset = NULL, ...)
```

It returns a classed model object. `mob()` wraps it into a type 1 function
`afit` that calls `coef()`, `-logLik()` and `sandwich::estfun()` on the result
(`modelparty.R:11-27`). The objective is therefore the negative log-likelihood
for type 2 fits. The vignette's `logit()` example uses this form (`mob.Rnw`,
chunk `logit`).

The Python port needs only the type 1 shape. Type 2 is an R convenience that
relies on S3 dispatch to `coef`, `logLik` and `estfun`. A Python model supplies
these directly.

### 1.2 The arguments mob passes in

| Argument  | What mob passes                                                                                                                                                                                                                                                                   | Source                                                         |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `y`       | Response for the node's rows. Shape depends on `control$ytype`: `"vector"` (default, a single model-frame column, possibly a factor), `"matrix"` (via `model.matrix(~ 0 + .)`), or `"data.frame"`.                                                                                | `modelparty.R:69-73`, `mob()`; subsetting in `suby`, `:177-179` |
| `x`       | Regressor matrix for the node's rows, built by `model.matrix()` on the first RHS part (`xtype = "matrix"`, default) or a data frame (`xtype = "data.frame"`). `NULL` when the formula has no regressor part. Carries attributes `formula`, `terms`, `offset`, `xlevels`, `contrasts`. | `modelparty.R:74-87`, `mob()`; `subx`, `:180-192`               |
| `start`   | `NULL` at every node fit. During split search, `NULL` when `restart = TRUE` (default), otherwise the previous candidate's coefficients.                                                                                                                                            | `mob_grow`, `:526`; `mob_grow_findsplit`, `:396-438`            |
| `weights` | Numeric vector of length n for the node, never `NULL`. Defaults to all 1. Zero weights are allowed.                                                                                                                                                                               | `mob()`, `:94-97`; `mob_partynode()`, `:163-164`                |
| `offset`  | Model offset subset to the node, or `NULL`. Only extracted when there is a regressor part.                                                                                                                                                                                        | `mob()`, `:98`                                                 |
| `cluster` | Cluster vector subset to the node, or `NULL`.                                                                                                                                                                                                                                     | `mob()`, `:99`                                                 |
| `...`     | Extra arguments given to `mob()`, for example `family`, `epsilon`, `maxit` from `glmtree()`.                                                                                                                                                                                      | `glmtree()`, `glmtree.R:37-48`                                 |
| `estfun`  | `TRUE` for the node fit, `FALSE` (default) in split search.                                                                                                                                                                                                                       | `mob_grow`, `:526-527`; `mob_grow_findsplit`, `:402-405`        |
| `object`  | `TRUE` for the node fit if `"object"` is in `control$terminal` or `control$vcov == "info"`. `FALSE` in split search.                                                                                                                                                              | `mob_grow`, `:527`                                              |

Two details matter for the port:

- `glmtree()` builds the regressor matrix with an intercept when the formula has
  only one RHS part, by rewriting `y ~ z1 + z2` into `y ~ 1 | z1 + z2`
  (`glmtree.R:11-18`, `lmtree.R:200-207`). Inside the fitters, `x = NULL` is
  replaced by a one-column intercept matrix (`glmfit`, `glmtree.R:74-75`;
  `lmfit`, `lmtree.R:230-231`).
- `glmtree()` passes `caseweights = control$caseweights` into its fitter
  through a wrapper, `glmfit0` (`glmtree.R:30-35`). The generic interface has
  no `caseweights` argument, so a model that needs it must receive it out of
  band. The Python interface passes it explicitly.

### 1.3 The objective

`objfun` must be additive over observations, because split search sums the
left and right objectives (`mob_grow_findsplit`, `:406`, `:436`, `:480`;
`mob.Rnw`, section "Splitting"). Smaller is better: split search takes
`which.min` (`:450-452`).

The two built-in models use different objectives:

- `lmfit` returns the weighted residual sum of squares,
  `sum(weights * residuals^2)` (`lmtree.R:244`).
- `glmfit` returns `aic/2 - df` for non-quasi families, where `df` is the rank
  plus one for gaussian, Gamma and inverse.gaussian (`glmtree.R:84-86`). For
  these families this equals the negative log-likelihood. For quasi families it
  returns the deviance.

Post-pruning compares objectives through a `prune` function (section 2.6).
`lmtree()` installs a different pruning rule from `mob()` because its
objective is an RSS, not a negative log-likelihood (`lmtree.R:178-195` versus
`mob()`, `modelparty.R:101-119`).

### 1.4 The scores, `estfun`

`estfun` is an n by k matrix with one row per observation in the node,
including zero-weight rows. Its columns are the non-aliased coefficients only:

- `lmfit`: `residuals * weights * x[, !is.na(coef)]` (`lmtree.R:251`).
- `glmfit`: `working_residuals * working_weights * x[, !is.na(coef)] /
  dispersion` (`glmtree.R:97-110`). The dispersion is 1 for poisson, binomial
  and "Negative Binomial". For other families it is
  `sum(wres^2 / weights) / sum(z$weights)` with case weights, or
  `sum(wres^2) / sum(z$weights)` without (`glmtree.R:99-108`).

The weights are already inside the scores. `mob_grow_fluctests` accounts for
this when it forms the variance estimate (section 2.3).

### 1.5 The optional pieces

| Piece                                        | When mob uses it                                                                                                                                                                                                        | Source                                                                                        |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `start` support                              | Only when `control$restart = FALSE`. The default `restart = TRUE` never passes starting values.                                                                                                                         | `mob_grow_findsplit`, `:396-438`                                                              |
| `object` with `vcov()`                       | When `control$vcov` is `"info"` or `"sandwich"`. The bread is `vcov(obj) * nobs`. If no object is available, `vcov` silently falls back to `"opg"`.                                                                     | `mob_grow_fluctests`, `:231-235`                                                              |
| `object` with `predict`, `fitted`, `logLik`, etc. | For methods on the finished tree. If `object` is not stored, `refit.modelparty` refits the node.                                                                                                                   | `apply_to_models`, `:894-905`; `refit.modelparty`, `:817-892`                                  |
| `cluster` support                            | Clustered variance in the OPG meat: scores are summed within clusters before the cross product.                                                                                                                        | `mob_grow_fluctests`, `:240-245`                                                              |
| `logLik` with a `df` attribute               | `logLik.modelparty` sums node log-likelihoods and their `df` plus `dfsplit` per inner node.                                                                                                                             | `logLik.modelparty`, `:907-920`                                                               |

## 2. How `mob()` uses each piece during tree growth

### 2.1 Setup

`mob()` builds a model frame with `drop.unused.levels = TRUE`
(`modelparty.R:41-63`). It extracts `Y`, `X`, the partitioning variables `Z`,
`weights`, `offset` and `cluster` (`:65-99`). Missing values are removed by the
model frame's `na.action` before any fitting. String pruning options are
converted to functions here (`:101-119`). Tree growth is done by
`mob_partynode()` (`:122-123`).

### 2.2 Node fit and stopping rules (`mob_grow`)

At each node, in this order (`mob_grow`, `modelparty.R:516-623`):

1. Fit the model on all rows with `estfun = TRUE` (`:526-527`).
2. Set `nobs = w2n(weights)`. `w2n` is `sum(w)` with case weights and
   `sum(w > 0)` otherwise (`:176`, `:529`).
3. If `minsize` is `NULL`, set it once, at the root, to
   `ceiling(10 * length(coefficients) / NCOL(y))` (`:533`). The length counts
   aliased coefficients. The assignment uses `<<-`, so every later node sees
   the root's value.
4. Stop if `w2n(weights) < 2 * minsize` or `depth >= maxdepth` (`:536-547`).
   The root has depth 1 (`:679`). A node stopped here has no `test` element.
5. Run the instability tests (`:550`). If they error, or if there is no
   `estfun`, the node becomes terminal with `test = NA` (`:569-573`).
6. Optionally adjust the p-values for multiplicity (section 2.4) and stop if
   the smallest adjusted p-value is above `alpha` (`:553-561`).
7. Search for a split in the selected variable (section 2.5). If no candidate
   is admissible, stop (`:581-588`).
8. Recurse into the children (`:596-616`).

### 2.3 Instability tests (`mob_grow_fluctests`)

Inputs are the node's `estfun`, the partitioning variables, the weights, the
fitted object and the clusters (`modelparty.R:209-381`).

**Scaling and decorrelation.** Rows with zero weight are dropped (`:221-224`).
`nobs` is `sum(weights)` if `caseweights` is set and any weight differs from 1,
otherwise the number of remaining rows (`:227`). The process is divided by
`sqrt(nobs)` (`:230`). The meat is `crossprod(process)`, or
`crossprod(process / sqrt(weights))` under case weights, because the weights
are already inside the scores (`:236-246`). With clusters the scaled process is
summed by cluster first (`:243-244`). The decorrelation matrix `J12` is the
symmetric square root (via `eigen`) of the inverse meat for `"opg"`, the bread
for `"info"`, or bread times meat times bread for `"sandwich"`
(`:195-203`, `:247-251`). The process becomes `process %*% t(J12)`
(`:252`). If `control$parm` is set, only those columns are kept (`:255-258`).

**Variables tested.** All variables if `mtry` is at least the number of
variables. Otherwise a random sample of `mtry` of them (`:218`). A variable
with fewer than two distinct values in the node is skipped and keeps
`stat = 0`, `pval = NA` (`:308`).

**Numeric variables, supLM.** (`:348-369`)

- Order rows by the variable (`order()`, stable for ties) or, with
  `breakties = TRUE`, by the variable plus uniform jitter (`:349-355`).
- Cumulate the process. The time index `tt0` is `1:n`, or the cumulative
  weights under case weights with non-unit weights (`:356-358`).
- Trimming: `from = ceiling(nobs * trim)` when `trim <= 1`, else `trim`;
  then `from = max(from, minsize)` and `to = nobs - from` (`:262-264`).
- The statistic is the maximum over `from <= tt0 <= to` of
  `rowSums(cumproc^2) / (t * (1 - t))` with `t = tt0 / nobs` (`:359-367`).
- The p-value is `logp.supLM(stat, k, lambda)` with
  `lambda = ((nobs - from) * to) / (from * (nobs - to))` (`:265`, `:368`).
  For `k <= 40` it uses Hansen's (1997) response-surface table
  `mob_beta_suplm` (`mob-pvalue.R`, 1000 by 3, 25 rows per k) with
  interpolation in `tau` (`:279-300`). For `k > 40` it uses Estrella (2003)
  with an ad hoc interpolation below `1.5 * k` (`:270-277`). This table is
  in partykit, not strucchange.

**Unordered factors, chi-squared.** (`:309-347`) Order by the factor, drop
unused levels in the node, compute per-level segment weights. The statistic is
`sum_j sum_levels (sum of process_j in level)^2 / segweight`, with
`k * (levels - 1)` degrees of freedom (`:345-346`).

**Ordered factors.** With the default `ordinal = "chisq"` they are tested
exactly like unordered factors (`:313`). `ordinal = "L2"` uses
`strucchange::ordL2BB`, which simulates critical values (`:340-342`).
`ordinal = "max"` uses `mvtnorm::pmvnorm` (`:331-338`). Both of these
depend on random numbers.

**Return value.** Raw p-values `exp(log p)`, the statistics, and `best`, the
index of the smallest log p-value with `NA` ignored and ties going to the first
variable (`:372-380`).

### 2.4 Multiplicity adjustment and the split decision (`mob_grow`)

With `bonferroni = TRUE` and `m` the number of non-`NA` p-values
(`modelparty.R:553-557`):

- `p_adj = 1 - (1 - p)^m` when `p > 0.001`;
- `p_adj = min(1, m * p)` otherwise.

So the "Bonferroni" adjustment is a Sidak correction above 0.001 and a
Bonferroni correction below it. The node splits only if the adjusted p-value
of `best` is at most `alpha` (`:560`). The stored `test` matrix holds the
statistics and the adjusted p-values. The raw p-values are not kept
(`:576`). `info$p.value` is the adjusted p-value of `best` (`:561`).

### 2.5 Split search (`mob_grow_findsplit`)

`minsize` is reinterpreted locally: a value in (0.5, 1) becomes `1 - minsize`,
and a value below 0.5 becomes `ceiling(w2n(weights) * minsize)`
(`modelparty.R:386-388`). A candidate is inadmissible (objective `Inf`) if
either side has `w2n < minsize`.

**Numeric variable.** Candidates are the sorted unique values `uz`. Candidate
`i` sends `z <= uz[i]` left (`:392-439`). Both sides are refit with
`estfun = FALSE` and the objectives summed. The chosen break is
`uz[which.min(dev)]`, or the midpoint of it and the next unique value if
`numsplit = "center"` (`:447-456`). Ties in `dev` go to the first candidate.

**Ordered factor.** Unused levels are dropped (`:466`). Candidates are the
first `nl - 1` levels, each sending levels up to and including it left
(`mob_grow_getlevels`, `:701-723`). The break is the position of the chosen
level among the original levels (`:492-497`).

**Unordered factor.** With `catsplit = "multiway"` every level gets its own
child, without search (`:460-462`). Otherwise all `2^(nl - 1) - 1` binary
partitions are enumerated with the last level always on the right
(`mob_grow_getlevels`, `:709-718`). The result is an index vector over the
original levels: 1 for the left kid, 2 for the right, `NA` for levels absent
in the node (`:498-507`).

Two quirks to replicate or flag:

- For a factor with exactly one candidate partition the objective is set to 1
  without fitting (`:475`).
- The right-hand fit for factors passes `cluster[zs]` instead of
  `cluster[!zs]` (`:479`). This only matters for cluster-aware fitters.

If every candidate is `Inf`, the split fails and the node becomes terminal
(`:441-447`, `:486-490`, `:585-587`).

### 2.6 Recursion, node ids and pruning

Children are numbered depth-first in preorder. The first child gets
`id + 1`. Each later child gets one more than the largest id in the previous
child's subtree (`mob_grow`, `:600-615`). Observations are routed with
`kidids_split` (`split.R:132-160`): numeric breaks use `.bincode` with
`right = TRUE`, so `z <= break` goes to kid 1. Ordered factors are routed
through their integer codes. Index splits map each level code through `index`.

Post-pruning (`mob_prune`, `modelparty.R:625-676`) runs only if
`control$prune` is a function. It repeatedly looks at inner nodes whose kids
are all terminal and calls `prune(objfun, df, nobs)` with:

- `objfun = c(parent objfun, sum of kids' objfun)`;
- `df = c(k, n_kids * k + as.integer(dfsplit))`, where `k` is the root's
  coefficient count;
- `nobs = c(parent nobs, n)`.

When it returns `TRUE` the kids are removed and node ids are renumbered.
`"aic"` and `"bic"` for `mob()` assume a negative log-likelihood
(`:102-114`). `lmtree()` uses `nobs * log(rss)` forms (`lmtree.R:178-195`).
Note that the `mob()` BIC rule uses `log(n)` of the whole sample, not
`nobs[2]`.

After pruning, `estfun` and `object` are dropped from node info unless listed
in `control$terminal` or `control$inner` (`:685-694`).

## 3. Tree data structures needed in Python

partykit represents a tree as nested `partynode` objects (`node.R:2-31`,
`partynode()`) holding `id`, `split`, `kids`, `surrogates` and `info`, wrapped
in a `party` object with the data and the fitted node per observation
(`mob()`, `modelparty.R:125-150`). MOB never builds surrogates.

A split is a `partysplit` (`split.R:2-74`, `partysplit()`) with `varid`,
`breaks`, `index`, `right`, `prob` and `info`. MOB uses `varid`, and either
`breaks` (numeric and ordered) or `index` (nominal). `right` is always
`TRUE`.

The node `info` for a modelparty holds `coefficients`, `objfun`, `object`,
`estfun`, `nobs`, `p.value` and `test` (`mob.Rnw`, section "Fitted party
objects"; `mob_grow`, `:526-576`).

Proposed Python structures:

```python
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class SplitKind(Enum):
    NUMERIC = "numeric"   # breaks: z <= break goes left
    ORDERED = "ordered"   # breaks: level code <= break goes left
    NOMINAL = "nominal"   # index: level -> kid number, or missing


@dataclass(frozen=True)
class Split:
    variable: int               # column index into the partitioning variables
    kind: SplitKind
    breaks: float | None        # numeric value, or 1-based level position for ORDERED
    index: tuple[int | None, ...] | None  # per original level: kid (1-based) or None


@dataclass(frozen=True)
class InstabilityTest:
    variables: tuple[str, ...]
    statistic: FloatArray       # 0.0 for untested variables, as in R
    p_value_raw: FloatArray     # NaN for untested variables
    p_value: FloatArray         # after the Bonferroni/Sidak adjustment
    best: int | None


@dataclass
class Node:
    id: int                     # 1-based, preorder, as in partykit
    depth: int                  # root is 1
    rows: NDArray[np.intp]      # row indices into the full data
    nobs: float                 # w2n(weights)
    fit: "FitResult"
    test: InstabilityTest | None
    p_value: float | None       # adjusted p-value of the selected variable
    split: Split | None = None
    kids: list["Node"] = field(default_factory=list)
```

The tree object also holds the `MobControl` settings, the terminal node id per
observation (`fitted` in `mob()`, `:126-131`), the partitioning variable
metadata (names, kind, original level labels), and the coefficient names.

## 4. Proposed Python interface

```python
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FitResult:
    coefficients: FloatArray          # shape (k,), NaN for aliased columns, as in R
    coef_names: tuple[str, ...]
    objfun: float                     # additive objective, smaller is better
    estfun: FloatArray | None         # shape (n, k_nonaliased), weights included
    vcov: FloatArray | None = None    # only needed for vcov "info" or "sandwich"
    converged: bool = True


class NodeModel(Protocol):
    """A model that MOB fits in every node.

    Mirrors the type 1 fit interface of partykit::mob (modelparty.R, mob()).
    """

    def fit(
        self,
        y: NDArray[np.generic],
        x: FloatArray | None,
        *,
        start: FloatArray | None,
        weights: FloatArray,
        offset: FloatArray | None,
        cluster: NDArray[np.generic] | None,
        caseweights: bool,
        estfun: bool,
        vcov: bool,
    ) -> FitResult: ...

    def n_response_columns(self, y: NDArray[np.generic]) -> int:
        """NCOL(y), used for the default minsize (mob_grow, modelparty.R:533)."""
        ...


class PredictingModel(NodeModel, Protocol):
    """Optional: prediction from a fitted node (predict.modelparty)."""

    def predict(
        self,
        result: FitResult,
        x: FloatArray | None,
        offset: FloatArray | None,
        kind: str,
    ) -> FloatArray: ...


class LogLikModel(NodeModel, Protocol):
    """Optional: log-likelihood and df (logLik.modelparty)."""

    def loglik(self, result: FitResult) -> tuple[float, int]: ...
```

Design notes:

- `fit` always receives a full weight vector, never `None`, matching
  `mob_partynode()` (`:163-164`). Zero weights are allowed.
- `caseweights` is an explicit argument because `glmfit` and the dispersion
  in its scores depend on it (`glmtree.R:30-35`, `:103-107`).
- `vcov=True` replaces R's `object = TRUE` request for the `"info"` and
  `"sandwich"` options. The model returns the covariance matrix itself.
  The full fitted object stays inside the model implementation.
- `FitResult` is immutable so fits can be cached and compared in tests.
- Split search calls `fit` with `estfun=False` and `vcov=False`, so a model
  can skip the expensive parts, as the vignette recommends.
- Core MOB code depends only on `NodeModel`. Prediction and log-likelihood are
  separate protocols so a minimal model stays small.

### 4.1 Optional: scoring all numeric split candidates at once

Refitting both sides of every candidate makes split search quadratic in the
node size (docs/performance.md). A model may provide a batch scorer, which
`mob()` uses for numeric split variables when it exists and
`restart = TRUE`:

```python
@dataclass(frozen=True)
class SplitObjectives:              # modeltrees.split_search
    left: FloatArray                # approximate objective of each left side
    right: FloatArray               # ... and right side
    error: FloatArray               # bound on |left + right - exact sum|


class BatchSplitModel(NodeModel, Protocol):
    def split_objectives(
        self,
        y: NDArray[np.generic],
        x: FloatArray | None,
        *,
        weights: FloatArray,
        offset: FloatArray | None,
        cluster: NDArray[np.generic] | None,
        caseweights: bool,
        order: NDArray[np.intp],    # stable sort of the node's rows by z
        cuts: NDArray[np.intp],     # candidate i sends order[:cuts[i]] left
    ) -> SplitObjectives | None: ...
```

The arguments are the node's data, as `fit` receives it, plus the sort
order of the split variable and one cut per admissible candidate. Rows with
zero weight are included and ride along with their `z` value. Returning
None declines, and every candidate is fitted as usual.

The contract:

- `left[i] + right[i]` approximates the summed objective that `fit` would
  give, and `error[i]` bounds the absolute difference, including the
  rounding of the exact fit itself. The bound must hold: split search relies
  on it to decide which candidates cannot win.
- NaN in any of the three for a candidate means "fit this one exactly".
  Use it for anything doubtful: rank-deficient or ill-conditioned sides,
  sides too small to estimate.

How split search uses it (`split_search.find_split`): the candidates whose
interval `approx ± error` reaches below the smallest upper bound
`min(approx + error)`, widened by `REFIT_MARGIN = 1e-6` relative to the best
approximation, plus at least the `REFIT_MIN = 5` smallest approximations,
plus every NaN candidate, are refitted with `fit`. The winner is the first
minimum of the resulting objective vector. Every candidate left unrefitted
has an approximation above every refitted candidate's exact value, so the
winner and the tie rule (first candidate) are those of the exact search.
`SplitSearch.objective` holds exact values for refitted candidates and the
approximations for the rest.

The batch scorer is not used with `restart = FALSE`, where each fit starts
from the previous candidate's coefficients, or for factor variables, which
have few candidates. Under `modeltrees.reference_path()` it is never used.

`LmModel` implements it with cumulative weighted cross-products
(`models/lm.py`, `LmModel.split_objectives`): per side, the RSS is
`r'Wr - b'A^-1 b` with `A = X'WX` and `b = X'Wr` from prefix sums (left) and
suffix sums (right) over the sorted rows, where `r` is the residual of the
node's own fit, so that the sums are on the scale of the RSS. `A` is scaled
by its diagonal and solved by a symmetric eigendecomposition; sides with a
condition number above 1e8 or a zero diagonal entry are NaN. `GlmModel` has
no batch scorer (docs/performance.md, Step 3, lists the options).

## 5. Open questions

1. **Raw p-values in fixtures.** partykit stores only the adjusted p-values in
   `info$test` (`mob_grow`, `:576`). Phase 1 must capture raw values some other
   way. The fixture generator patches a copy of `mob_partynode` to record them,
   and checks that the patched and unpatched trees are identical.
2. **Aliased coefficients.** `minsize` and the pruning `df` count all
   coefficients including `NA` ones (`:533`, `:647`), while `estfun` drops them
   (`glmtree.R:109`). Should Python keep `NaN` entries in `coefficients` to
   reproduce this, as proposed above?
3. **Response types.** Binomial responses arrive as factors and `glm.fit`
   converts them through `family$initialize`. Should the Python API accept
   categorical y and convert at the boundary, or require 0/1?
4. **RNG-dependent options.** `mtry < Inf`, `breakties = TRUE`,
   `ordinal = "L2"` and `ordinal = "max"` all consume R's random numbers
   (`:218`, `:352`, `:342`, `:333`). Exact tree matches against R are not
   possible for these. Should they be ported with statistical rather than
   exact tests, or deferred?
5. **Unknown-shape models.** For negative binomial with unknown theta, does
   theta belong in the tested parameter vector? `glmfit` treats
   "Negative Binomial" as a known-dispersion family and tests only the
   regression coefficients (`glmtree.R:99`). The thesis uses `party`'s
   `StatModel` interface (`trusch.md`, section on negbinModel) and should be
   checked before phase 3.
6. **Multinomial coefficients.** A multinomial model has a matrix of
   coefficients. The proposal flattens it to a vector with names, which fits
   the `estfun` contract. Is the flattening order (by category, then
   regressor) acceptable?
7. **The factor `cluster` quirk.** Should the right-hand fit in factor split
   search replicate `cluster[zs]` (`:479`), or use the corrected `!zs`? It only
   affects cluster-aware models.
8. **`vcov = "sandwich"` fallback.** If no object is requested, partykit
   silently uses `"opg"` (`:232`). With `vcov="sandwich"` and the default
   `terminal = "object"` an object is available, but with `terminal = NULL`
   it is not. Should Python replicate the silent fallback or raise?
9. **Multiway splits.** `catsplit = "multiway"` creates more than two kids.
   Is that needed in the first port, or can splits be binary only?
