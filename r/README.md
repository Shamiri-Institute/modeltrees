# R reference oracle

This directory contains the reproducible R environment used as a reference
oracle for the Python port of model-based recursive partitioning with
`partykit::mob()` and `partykit::glmtree()`.

## Host and R installation

The host was detected as Ubuntu 20.04.6 LTS (Focal), x86-64. CRAN no longer
publishes current R binaries for Focal, so R 4.6.1 was compiled from the
official CRAN source archive.

The required compilers and system libraries were installed with:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential gfortran \
  libreadline-dev libpcre2-dev \
  libcurl4-openssl-dev libssl-dev \
  libbz2-dev liblzma-dev zlib1g-dev \
  libicu-dev libpng-dev libjpeg-dev libtiff5-dev libcairo2-dev \
  libblas-dev liblapack-dev
```

R was configured as a headless build with system BLAS and LAPACK, then installed
without `sudo` under the user account:

```text
~/.local/lib/R/4.6.1
```

The commands `R` and `Rscript` are exposed through symlinks in `~/.local/bin`,
which is already on this host's `PATH`.

Verify the installation with:

```bash
Rscript --version
Rscript -e 'cat(R.version.string, "\n")'
```

Expected version:

```text
R version 4.6.1 (2026-06-24)
```

The build supports HTTPS, PCRE2, readline, BLAS, LAPACK, PNG, JPEG, TIFF, Cairo,
and ICU. Missing Java, `pdflatex`, and `texi2any` produced warnings while
building optional manuals and vignettes; they are not needed for this oracle.

## renv project

The renv project is rooted in this `r/` directory, rather than the repository
root. It was initialized with `bare = TRUE`, so renv does not create R project
files at the repository root.

renv uses its normal shared cache at:

```text
~/.cache/R/renv
```

The project uses explicit snapshots. `DESCRIPTION` declares the six direct
dependencies so that `renv::status()` recognizes all of them as intentional:

- `partykit`
- `strucchange`
- `sandwich`
- `MASS`
- `nnet`
- `jsonlite`

The lockfile also contains `renv` and the dependency closure required by those
packages: `Formula`, `inum`, `lattice`, `libcoin`, `Matrix`, `mvtnorm`, `rpart`,
`survival`, and `zoo`. Unrelated packages bundled with the R installation were
removed from the lockfile.

The environment was verified with:

```bash
(
  cd r
  Rscript -e 'renv::status()'
)
```

The expected result is:

```text
No issues found -- the project is in a consistent state.
```

## Smoke test

`smoke.R` simulates a binomial data set whose intercept and slope change with
partitioning variable `z`. It fits:

```r
partykit::glmtree(y ~ x | z, family = binomial())
```

The script writes the following information as JSON:

- internal node IDs;
- split variable names and split points;
- terminal node IDs and their fitted coefficients.

Run it from the repository root:

```bash
Rscript r/smoke.R r/smoke.json
```

The script locates and activates the `r/` renv project itself, so this command
works even though the current directory is the repository root. The generated
`r/smoke.json` is optional: commit it only if it will serve as a golden fixture
for the Python implementation.

## Fixture generators

`generate_all.R` regenerates every fixture in `tests/fixtures/`, including its
README, which lists each fixture with the exact R call. Run it from the
repository root:

```bash
Rscript r/generate_all.R
```

The pieces live in `fixtures/`: `common.R` (output helpers and the
instrumented copy of `partykit:::mob_partynode`), `datasets.R`, `trees.R`,
`glm.R` and `pvalues.R`. Output is deterministic; two consecutive runs produce
byte-identical files.

`extensions/` holds the phase 4 node models as `partykit::mob()` fit
functions: `negbinfit` (MASS::glm.nb), `safelogitfit` (glm.fit) and
`multinomfit` (nnet::multinom). `fixtures/extensions.R` builds their datasets,
node-level fit fixtures with estfun checks, and tree specs.
docs/extensions-spec.md specifies them.

party 1.0-1 (the thesis implementation) does not compile on R 4.6.1, because
its C code uses the removed `Calloc` macro. It is not part of this
environment. Its dependencies modeltools, coin and vcd were installed into the
renv library during the attempt. They are not in `renv.lock` and nothing uses
them.

`scan_sim_diabetes_like.R` is the seed scan behind the simulated
`sim_diabetes_like` data (`fixtures/datasets.R`). It grows the four
`sim_diabetes_like_*` trees for a range of seeds and lists the seeds whose
default tree finds the two planted splits and whose BIC, AIC and AIC without
dfsplit pruned trees differ; the first one is `SIM_DIABETES_LIKE_SEED`.
`generate_all.R` checks the same properties on the generated fixtures:

```bash
Rscript r/scan_sim_diabetes_like.R 20261051 20
```

## Restore on a clean machine or in CI

First install a compatible R and the required operating-system libraries. renv
restores R packages, but it does not install R itself or system headers.

After cloning the repository, run from its root:

```bash
(
  cd r
  Rscript -e 'renv::restore(prompt = FALSE)'
)
```

Starting `Rscript` inside `r/` causes `r/.Rprofile` to source
`r/renv/activate.R`. That activation script bootstraps the recorded renv version
when necessary, and `renv::restore()` recreates the project library from
`r/renv.lock`.

Then verify the restored environment and oracle:

```bash
(
  cd r
  Rscript -e 'renv::status()'
)
Rscript r/smoke.R r/smoke.json
```

## Version-control policy

Commit the reproducibility metadata and source files:

```bash
git add \
  r/.Rprofile \
  r/DESCRIPTION \
  r/README.md \
  r/renv.lock \
  r/renv/.gitignore \
  r/renv/activate.R \
  r/renv/settings.json \
  r/smoke.R
```

Do not commit the machine-specific project library or transient R state. The
repository `.gitignore` already contains:

```gitignore
.Rhistory
.RData
.Rproj.user/
r/renv/library/
r/renv/staging/
r/renv/local/
```

The generated `r/renv/.gitignore` additionally excludes renv cache, lock,
sandbox, staging, and library directories beneath `r/renv/`.

## Problems encountered during setup

- Ubuntu 20.04 is no longer supported by CRAN's current Ubuntu binary
  repository, so R 4.6.1 was built from source.
- `sudo` required an interactive password, so the system-library command was run
  manually before the remaining setup continued.
- CRAN did not provide the anticipated `.tar.xz.md5` sidecar URL. The archive
  was instead checked for compression integrity with `xz -t` before use.
- The sandbox initially blocked renv from creating `~/.cache/R/renv`; access to
  the standard user cache was then explicitly allowed.
- An initial `type = "all"` snapshot included unrelated recommended packages
  bundled with R. It was replaced by an explicit snapshot based on
  `DESCRIPTION`.
- In partykit 1.3-0, `nodeids(tree, terminal = FALSE)` returns all nodes rather
  than only internal nodes. The smoke test now obtains internal nodes by
  subtracting terminal node IDs from all node IDs.
