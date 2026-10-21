# Contributing

Read CLAUDE.md and docs/decisions.md first. They hold the ground rules and the
decisions every change must respect. docs/README.md says which of the other
documents are design records and which are working notes.

## Setup

```
uv sync
uv run pytest            # default run, slow tests excluded
uv run pytest -m ""      # every test
uv run pytest -m slow    # only the slow tests
uv run ruff check
uv run ruff format --check
uv run mypy src tests scripts
```

The tests need no R. They read the committed fixtures in `tests/fixtures/`.

## R is the oracle

modeltrees is a port. For every behavior that R has, R decides what is correct,
and the Python code is tested against values R computed:

1. The R environment lives in `r/`, managed by renv (`r/renv.lock`, with the
   pinned partykit, strucchange, sandwich, MASS, nnet and jsonlite versions).
   r/README.md describes the R installation.
2. `r/fixtures/*.R` and `r/extensions/*.R` define the cases. Each tree fixture
   is grown twice in R, once by partykit unchanged and once with an instrumented
   copy of `mob_partynode` that also records raw p-values, split-search
   objectives and the effective `minsize`. Generation stops if the two trees
   differ.
3. `Rscript r/generate_all.R`, run from the repository root, writes every JSON
   and CSV file in `tests/fixtures/` and its README. The output is
   deterministic: two runs give byte-identical files. Numbers are written with
   17 significant digits.
4. The Python tests compare component by component: scores, instability
   statistics, p-values, split search, then whole trees. When a test fails, find
   the first component that diverges before changing anything.

The source of the pinned R packages is fetched into `reference/` (read-only, not
committed) by `scripts/fetch_reference.sh`. Port behavior from that source. If R
and a textbook disagree, match R.

### Adding or regenerating a fixture

- Install R 4.6.1 and restore the renv library:
  `cd r && Rscript -e
  'renv::restore()'`.
- Add the case to the relevant generator in `r/fixtures/`, regenerate with
  `Rscript r/generate_all.R`, and check with `git diff --stat` that only the
  files you expected changed. A change to an existing fixture needs an
  explanation in the commit and, if it changes a tree, an entry in
  docs/decisions.md.
- Commit the generator change and the regenerated files together.

Where R has no oracle (the safe logit and multinomial rules, theta in the
pruning df, failure handling), write a unit test and record the decision in
docs/decisions.md or the relevant porting document.

## Tolerances

Tree structure, split variables and split points must match R exactly.
Coefficients and statistics match to about 1e-6 or better; most tests use
`rtol = 1e-8`. Where a test uses a wider tolerance, the porting documents derive
it from a measured quantity (for example nnet's convergence offset,
docs/extensions-port.md, section 7).

**Tolerances are never loosened to make a test pass.** A failing comparison is a
finding: find the first diverging component, explain the difference, and record
it. If a tolerance turns out to be wrong, the change needs its own derivation in
the docs, reviewed separately from the code change that exposed it.

## Speedups

A faster path must give the same results. The code it replaces stays as the
reference implementation behind `modeltrees.reference_path()`, and tests compare
both paths with each other and with the R fixtures (docs/decisions.md,
docs/performance.md).

## Commits

One line, no body. The first letter is capitalized, the rest can be lower case,
for example:

```
Add glm.fit port with binomial family
```

One commit per logical step.
