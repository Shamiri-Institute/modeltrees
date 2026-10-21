# Third-party notices

modeltrees is distributed under GPL-3.0-only (see LICENSE). Parts of it are
ports of, or are derived from, the R code and data listed below. Each entry
gives the upstream version the port follows, its license string exactly as
written in the upstream DESCRIPTION file (read from `reference/` and from the
renv library of `r/renv.lock`), and the files here that derive from it. All
ported code comes from upstreams licensed `GPL-2 | GPL-3` or GPL-2 or later.

No upstream source file is shipped in the wheel. The R scripts in `r/` call
the packages below to generate the test fixtures; they do not copy their code.

## Code

### partykit 1.3-0

- Authors: Torsten Hothorn, Achim Zeileis (authors); Heidi Seibold
  (contributor).
- License (DESCRIPTION): `GPL-2 | GPL-3`.
- Ported: the MOB algorithm (`mob`, `mob_control`, `mob_grow`,
  `mob_grow_fluctests`, `mob_grow_findsplit`, `mob_grow_getlevels`,
  `mob_prune`, `logp.supLM`), `lmfit` and `glmfit` of `lmtree` and `glmtree`,
  `logLik.modelparty`, and the layout of `print.modelparty`.
- Files: `src/modeltrees/tree.py`, `instability.py`, `pvalues.py`,
  `split_search.py`, `pruning.py`, `_loglik.py`, `_report.py`,
  `models/lm.py`, `models/glm.py`.
- `src/modeltrees/_hansen_table.py` holds the response-surface table
  `mob_beta_suplm` from partykit's `R/mob-pvalue.R`, copied value by value.
  The table is Hansen's (1997) approximation for supLM p-values: B. E.
  Hansen, "Approximate asymptotic p values for structural-change tests",
  Journal of Business and Economic Statistics 15(1), 60-67, 1997.

### strucchange 1.6-0

- Authors: Achim Zeileis, Friedrich Leisch, Kurt Hornik, Christian Kleiber
  (authors); Bruce E. Hansen, Edgar C. Merkle, Nikolaus Umlauf
  (contributors).
- License (DESCRIPTION): `GPL-2 | GPL-3`.
- Applies only through partykit: `root.matrix`, which partykit copies from
  strucchange (`R/matrix.R`), is ported as `instability.root_matrix`.
  strucchange's own p-value code (`ordinal = "L2"`) is not ported.

### R 4.6.1: stats and LINPACK

- License: R's `doc/COPYRIGHTS` says "R as a whole is distributed under GPL
  version 2 or 3: most source files contain a copyright statement allowing
  use of that file under GPL version 2 or later". The stats package's
  DESCRIPTION says `License: Part of R 4.6.1`. `src/library/stats/R/glm.R`
  carries "Copyright (C) 1995-2020 The R Core Team" under "GPL version 2 of
  the License, or (at your option) any later version".
- Ported: `stats::glm.fit`, `lm.fit`, `lm.wfit` and the `gaussian`,
  `binomial` and `poisson` families with their C helpers in `family.c`
  (`src/modeltrees/models/glm_fit.py`, `lm_fit.py`, `families.py`).
- Ported: `Cdqrls` (`src/library/stats/src/lm.c`) and the Fortran routines
  `dqrls`, `dqrdc2` and `dqrsl` (`src/appl/`) in
  `src/modeltrees/models/_qr.py`. `dqrdc2` is R's modification of LINPACK's
  `dqrdc`. R's `doc/COPYRIGHTS` says: "The status of the included LINPACK
  routines is unclear: they are said to be in the public domain in the USA."

### MASS 7.3-66

- Authors: Brian Ripley, Bill Venables (authors, copyright holders), with
  contributors and translators listed in its DESCRIPTION.
- License (DESCRIPTION): `GPL-2 | GPL-3`. `R/negbin.R` carries "copyright
  (C) 1994-2023 W. N. Venables and B. D. Ripley".
- Ported: `glm.nb`, `theta.ml`, `negative.binomial` and glm.nb's internal
  log-likelihood (`src/modeltrees/models/negbin.py`, the `NegativeBinomial`
  family in `models/families.py`).

### sandwich 3.1-3

- Authors: Achim Zeileis, Thomas Lumley (authors); Nathaniel Graham, Susanne
  Koell (contributors).
- License (DESCRIPTION): `GPL-2 | GPL-3`.
- Derived: the clustered meat and its adjustments from `meatCL` and `vcovCL`
  (`type` HC0 and HC1, `cadjust`) in `src/modeltrees/_vcov.py`; the score
  definitions of `estfun.glm` and `estfun.multinom` used by the node models.
  The formulas are reimplemented in numpy; no code is copied.

### Rusch (2012), the negative binomial node model

- T. Rusch, "Recursive Partitioning of Models of a Generalized Linear Model
  Type", doctoral thesis, WU Vienna, 2012,
  doi:10.57938/f43bea09-fe60-44fb-a995-ba63733c8c69. The repository's copy
  records "Document License: Unspecified".
- The negative binomial node model (`negbinModel` of the thesis) follows the
  thesis' description and code listing: beta-only scores from `estfun.glm`,
  and glm.nb as the fitter. It is reimplemented in `r/extensions/negbin.R`
  and `src/modeltrees/models/negbin.py`; the thesis text and code are not
  reproduced, apart from a few short quotations in the docs, each attributed
  with its line reference in the thesis.
- The safe logit and multinomial models (`models/safelogit.py`,
  `models/multinom.py`, `r/extensions/safelogit.R`, `r/extensions/multinom.R`)
  are this project's designs. The thesis names models with these purposes
  but gives no code for them.

## Data used in the test fixtures

### MASS 7.3-66: `quine`

- `tests/fixtures/data/quine.csv` is `MASS::quine`.
- License of the package (DESCRIPTION): `GPL-2 | GPL-3`.
- Source as given in its help page: S. Quine, quoted in Aitkin, M. (1978)
  The analysis of unbalanced cross classifications (with discussion).
  Journal of the Royal Statistical Society series A 141, 195-223.

### nnet 7.3-21

- Authors: Brian Ripley (author, copyright holder), William Venables
  (copyright holder).
- License (DESCRIPTION): `GPL-2 | GPL-3`.
- Used only to generate the multinomial fixtures (`nnet::multinom` in
  `r/extensions/multinom.R`). The Python multinomial model is fitted by
  Newton's method and contains no nnet code.

### Other R packages used by the fixture generators

`jsonlite` 2.0.0 (`MIT + file LICENSE`) writes the fixture files. The
remaining packages in `r/renv.lock` are dependencies of the packages above.
None of their code is ported.

## Python runtime dependencies

numpy, scipy and narwhals are dependencies, not bundled. Their licenses
are distributed with them.
